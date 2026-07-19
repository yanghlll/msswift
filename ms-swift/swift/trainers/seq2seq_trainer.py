# Copyright (c) ModelScope Contributors. All rights reserved.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
import time
import torch
import torch.distributed as dist
from contextlib import contextmanager, nullcontext
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
from swift.sequence_parallel import sequence_parallel
from swift.utils import HfConfigFactory, JsonlWriter, Serializer, gc_collect, get_logger, unwrap_model_for_generation
from .arguments import Seq2SeqTrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin
from .utils import per_token_loss_func, per_token_loss_func_sp

# ---------------- 分段计时(STAGE_PROFILE=1): ViT/aligner-MLP/LLM 前反向 + 步级分解 ----
# 前 STAGE_PROFILE_STEPS 步逐步计时并写 log(STREAM_PROFILE_DIR 或 output_dir 下
# stage_profile_rank{r}.log), 测完自动卸 hook, 之后零开销。
_STAGE_PROFILE = os.environ.get('STAGE_PROFILE', '') not in ('', '0', 'false', 'False')
_STAGE_PROFILE_STEPS = int(os.environ.get('STAGE_PROFILE_STEPS', '8'))


def _sp_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

logger = get_logger()


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.template.model_accepts_loss_kwargs is not None:
            self.model_accepts_loss_kwargs = self.template.model_accepts_loss_kwargs
        if self.args.predict_with_generate:
            self.infer_engine = TransformersEngine(
                self.model, template=self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_data_collator = self.data_collator
        self.data_collator = self._predict_data_collator
        packing = self.template.packing
        padding_free = self.template.padding_free
        self.template.packing = False
        self.template.padding_free = False
        try:
            yield
        finally:
            self.template.packing = packing
            self.template.padding_free = padding_free
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            with self.template.forward_context(self.model, inputs):
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation), self.template.generate_context():
            resp_list = self.infer_engine.infer(
                data_list,
                RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
                use_tqdm=False,
            )

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def _prepare_inputs(self, inputs):
        args = self.args
        inputs = super()._prepare_inputs(inputs)
        if self.template.sequence_parallel_size > 1:
            sequence_parallel.prepare_inputs(inputs)

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            if args.tuner_backend == 'unsloth' and isinstance(inputs['logits_to_keep'], torch.Tensor):
                inputs['logits_to_keep'] = int(inputs['logits_to_keep'].sum())

        base_model = self.template.get_base_model(self.model)
        forward_params = inspect.signature(base_model.forward).parameters
        if self.model.model_info.is_moe_model and any(key in forward_params
                                                      for key in ['output_router_logits', 'kwargs']):
            HfConfigFactory.set_config_attr(base_model.config, 'router_aux_loss_coef', args.router_aux_loss_coef)
            base_model.router_aux_loss_coef = args.router_aux_loss_coef
            logger.info_once(f'router_aux_loss_coef: {args.router_aux_loss_coef}')
            if args.router_aux_loss_coef > 0:
                inputs['output_router_logits'] = True
        inputs['compute_loss_func'] = self.compute_loss_func
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        _sp = getattr(self, '_sp', None)
        if _sp is not None and _sp['active']:
            _sp_sync()
            _sp_t0 = time.perf_counter()
        labels = None
        compute_loss_func: Callable = inputs.pop('compute_loss_func', None)
        loss_scale = inputs.pop('loss_scale', None)
        text_position_ids = inputs.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = inputs.get('position_ids')
        channels = inputs.pop('channel', None)

        if (self.label_smoother is not None or compute_loss_func is not None or loss_scale is not None
                or self.args.enable_dft_loss or self.args.enable_channel_loss
                or self.template.sequence_parallel_size > 1) and 'labels' in inputs:
            if self.args.use_liger_kernel:
                # per-token loss_scale 需要逐 token 的 loss, 与 Liger fused-linear-CE(只出标量
                # loss、不落 logits)原理冲突, CE 必走非融合路径; 其余 Liger kernel 仍生效。
                # 开了 use_logits_to_keep 时 lm_head 只对被监督位置算 logits, 显存影响可忽略,
                # 不必再吓人; 没开时才警告(此时会 materialize 全长 logits, 显存确实会涨)。
                if self.args.use_logits_to_keep:
                    logger.info_once('Liger CE 不生效(loss_scale 需逐 token loss), 但 use_logits_to_keep '
                                     '已开, logits 仅算被监督位置, 显存影响可忽略; 其余 Liger kernel 正常。')
                else:
                    logger.warning_once('The cross_entropy loss function defined in Liger Kernel will not '
                                        'take effect, potentially leading to increased GPU memory consumption. '
                                        '建议加 --use_logits_to_keep true 以只对被监督位置 materialize logits。')
            labels = inputs.pop('labels')
        outputs = self.template.compute_sft_loss(model, inputs, num_items_in_batch=num_items_in_batch, trainer=self)
        mode = 'train' if self.model.training else 'eval'
        if getattr(outputs, 'aux_loss', None) is not None:
            self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if hasattr(self.args, 'past_index') and self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            outputs.loss = None
            if (self.args.enable_dft_loss or loss_scale is not None or self.args.enable_channel_loss
                    or self.template.sequence_parallel_size > 1):
                if self.template.sequence_parallel_size > 1:
                    outputs.loss = per_token_loss_func_sp(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)
                else:
                    outputs.loss = per_token_loss_func(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)

                if loss_scale is not None:
                    loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
                    outputs.loss = outputs.loss * loss_scale

                if self.args.enable_channel_loss:
                    metrics = self.custom_metrics[mode]
                    masks = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
                    if self.template.padding_free:
                        cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
                    else:
                        cu_seqlens = torch.arange(0, labels.shape[0] + 1) * labels.shape[1]
                    for i in range(cu_seqlens.shape[0] - 1):
                        channel = None if channels is None else channels[i]
                        slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                        metrics[f'loss_{channel}'].update(outputs.loss[slice_][masks[slice_]])

            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(
                    outputs, labels, num_items_in_batch=num_items_in_batch, loss_scale=loss_scale, trainer=self)
            elif self.label_smoother is None:
                # Handle the outputs.loss generated by loss_scale.
                if num_items_in_batch is None:
                    # https://github.com/huggingface/transformers/blob/9dff7ca5c9693f4c02cdd2a9c2abc4772fcea5da/src/transformers/trainer.py#L2137
                    num_items_in_batch = (labels != -100).sum()  # compat SP
                    if self.template.sequence_parallel_size > 1:
                        # labels are sharded by SP; outputs.loss was gathered
                        # to full length via GatherLoss. Reduce the denominator
                        # across the SP group so it matches the gathered loss.
                        dist.all_reduce(num_items_in_batch, op=dist.ReduceOp.SUM)
                loss = outputs.loss.sum() / num_items_in_batch
            else:
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)

            if self.model.model_info.is_moe_model and self.args.router_aux_loss_coef is not None:
                aux_loss = outputs.get('aux_loss')
                if aux_loss is not None:
                    if num_items_in_batch is not None:
                        aux_loss = aux_loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)
                    loss = loss + self.args.router_aux_loss_coef * aux_loss.to(loss.device)

        if getattr(self.args, 'average_tokens_across_devices',
                   False) and self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes
            if mode == 'eval' and self.template.sequence_parallel_size > 1:
                loss /= self.template.sequence_parallel_size

        if (outputs.logits is not None and labels is not None and self.args.tuner_backend != 'unsloth'):
            cu_seqlens = None
            if self.template.padding_free and self.args.acc_strategy == 'seq':
                cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
            # Liger does not have logits
            # Unsloth has a bug with output logits
            self._compute_acc(outputs, labels, cu_seqlens=cu_seqlens)
        if _sp is not None and _sp['active']:
            _sp_sync()
            _sp['cur']['fwd_total'] = _sp['cur'].get('fwd_total', 0.) + time.perf_counter() - _sp_t0
        return (loss, outputs) if return_outputs else loss

    def _stage_profile_setup(self, model):
        """定位 visual / visual.merger / language_model 三个子模块并挂前反向计时 hook。"""
        st = {'active': True, 'cur': {}, 'rows': [], 'hooks': [], 'last_end': None, 'bwd_hook_ok': False}
        self._sp = st
        m = model
        while hasattr(m, 'module'):
            m = m.module
        base = getattr(m, 'model', None)
        mods = {}
        if base is not None:
            if getattr(base, 'visual', None) is not None:
                mods['vision(ViT+MLP)'] = base.visual
                if getattr(base.visual, 'merger', None) is not None:
                    mods['aligner_mlp'] = base.visual.merger
            if getattr(base, 'language_model', None) is not None:
                mods['llm'] = base.language_model

        def mk_fwd(name):
            def pre(mod, args, kwargs=None):
                _sp_sync()
                st['cur'][name + '@t0'] = time.perf_counter()
            def post(mod, args, output):
                _sp_sync()
                t0 = st['cur'].pop(name + '@t0', None)
                if t0 is not None:
                    st['cur'][name + '.fwd'] = st['cur'].get(name + '.fwd', 0.) + time.perf_counter() - t0
            return pre, post

        def mk_bwd(name):
            def pre(mod, grad_output):
                _sp_sync()
                st['cur'][name + '@bt0'] = time.perf_counter()
            def post(mod, grad_input, grad_output):
                _sp_sync()
                t0 = st['cur'].pop(name + '@bt0', None)
                if t0 is not None:
                    st['cur'][name + '.bwd'] = st['cur'].get(name + '.bwd', 0.) + time.perf_counter() - t0
                    st['bwd_hook_ok'] = True
            return pre, post

        for name, sub in mods.items():
            f_pre, f_post = mk_fwd(name)
            st['hooks'] += [sub.register_forward_pre_hook(f_pre), sub.register_forward_hook(f_post)]
            if name != 'vision(ViT+MLP)':      # ViT 冻结无反向; 只对 mlp/llm 试挂反向 hook
                try:
                    b_pre, b_post = mk_bwd(name)
                    st['hooks'] += [sub.register_full_backward_pre_hook(b_pre),
                                    sub.register_full_backward_hook(b_post)]
                except Exception:
                    pass
        logger.info(f'[STAGE_PROFILE] 已挂 hook: {list(mods)}; 测前 {_STAGE_PROFILE_STEPS} 步后写 log 并卸载')

    def _stage_profile_flush(self):
        st = self._sp
        st['active'] = False
        for h in st['hooks']:
            h.remove()
        st['hooks'] = []
        rows = st['rows']
        keys = sorted({k for r in rows for k in r})
        mean = {k: sum(r.get(k, 0.) for r in rows) / len(rows) for k in keys}
        # 派生: 纯 ViT 前向 = vision 总前向 - aligner MLP 前向
        if 'vision(ViT+MLP).fwd' in mean:
            mean['vit_only.fwd'] = mean['vision(ViT+MLP).fwd'] - mean.get('aligner_mlp.fwd', 0.)
        mean['other_fwd(embed/lm_head/loss等)'] = mean.get('fwd_total', 0.) \
            - mean.get('vision(ViT+MLP).fwd', 0.) - mean.get('llm.fwd', 0.)
        lines = [f'[STAGE_PROFILE rank={self.args.process_index}] {len(rows)} 步分段均值 (秒/步):']
        for k in ('gap', 'fwd_total', 'vision(ViT+MLP).fwd', 'vit_only.fwd', 'aligner_mlp.fwd',
                  'llm.fwd', 'other_fwd(embed/lm_head/loss等)', 'bwd_total', 'llm.bwd', 'aligner_mlp.bwd',
                  'step_total'):
            if k in mean:
                lines.append(f'  {k:36s} {mean[k]:8.3f}s')
        lines.append('  说明: gap=数据等待+优化器step+日志(在步间); ViT 冻结故无反向;')
        lines.append('       bwd_total 含梯度检查点重算; ' + (
            'llm.bwd 为 hook 实测。' if st['bwd_hook_ok'] else
            '反向 hook 未生效(ModelOutput 非张量输出), llm.bwd≈bwd_total(MLP 反向极小)。'))
        lines.append('  逐步明细:')
        for i, r in enumerate(rows):
            lines.append('    step%d: ' % i + ' '.join(f'{k}={v:.3f}' for k, v in sorted(r.items())))
        text = '\n'.join(lines)
        out_dir = os.environ.get('STREAM_PROFILE_DIR') or self.args.output_dir or '.'
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'stage_profile_rank{self.args.process_index}.log')
        with open(path, 'w') as f:
            f.write(text + '\n')
        print(text, flush=True)
        print(f'[STAGE_PROFILE rank={self.args.process_index}] 已写入 {path}, hook 已卸载, 恢复零开销', flush=True)

    def training_step(self, model, inputs, *args, **kwargs):
        if _STAGE_PROFILE and not hasattr(self, '_sp'):
            self._stage_profile_setup(model)
        sp = getattr(self, '_sp', None)
        if sp is None or not sp['active']:
            with self.template.forward_context(self.model, inputs):
                return super().training_step(model, inputs, *args, **kwargs)
        _sp_sync()
        t0 = time.perf_counter()
        sp['cur']['gap'] = 0. if sp['last_end'] is None else t0 - sp['last_end']
        with self.template.forward_context(self.model, inputs):
            loss = super().training_step(model, inputs, *args, **kwargs)
        _sp_sync()
        t1 = time.perf_counter()
        sp['last_end'] = t1
        cur, sp['cur'] = sp['cur'], {}
        cur = {k: v for k, v in cur.items() if not k.endswith('@t0') and not k.endswith('@bt0')}
        cur['step_total'] = t1 - t0
        cur['bwd_total'] = cur['step_total'] - cur.get('fwd_total', 0.)
        sp['rows'].append(cur)
        if len(sp['rows']) >= _STAGE_PROFILE_STEPS:
            self._stage_profile_flush()
        return loss
