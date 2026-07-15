# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import transformers
from dataclasses import dataclass, field
from packaging import version
from typing import Any, Dict, List, Literal, Optional

from swift.utils import get_env_args
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_video_llava
from .llama import Llama3TemplateMeta
from .qwen import QwenTemplateMeta
from .utils import ChatmlTemplateMeta


class LlavaHfTemplate(Template):
    placeholder_tokens = ['<image>']

    @property
    def image_token_index(self):
        if not hasattr(self, '_image_token_index'):
            self._image_token_index = self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        return self._image_token_index

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        if images:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                encoded['image_sizes'] = image_inputs['image_sizes']
            if version.parse(transformers.__version__) >= version.parse('4.47'):
                input_ids = encoded['input_ids']
                labels = encoded['labels']
                idx_list = findall(input_ids, self.image_token_index)  # <image>
                height, width = image_inputs['pixel_values'][0].shape[-2:]
                added_tokens_len = 0
                for i, idx in enumerate(idx_list):
                    if 'image_sizes' in image_inputs:
                        orig_height, orig_width = image_inputs['image_sizes'][i].tolist()
                        num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height,
                                                                                  width)
                    else:
                        num_image_tokens = (height // self.processor.patch_size) * (
                            width // self.processor.patch_size) + self.processor.num_additional_image_tokens
                    if self.processor.vision_feature_select_strategy == 'default':
                        num_image_tokens -= 1
                    input_ids = input_ids[:added_tokens_len + idx] + [self.image_token_index] * num_image_tokens \
                        + input_ids[added_tokens_len + idx + 1:]
                    if labels is not None:
                        labels = labels[:added_tokens_len + idx] + [-100] * num_image_tokens \
                            + labels[added_tokens_len + idx + 1:]
                    added_tokens_len += num_image_tokens - 1
                encoded['input_ids'] = input_ids
                encoded['labels'] = labels
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.llava1_5_hf,
        prefix=['<s>'],
        prompt=['USER: {{QUERY}}\nASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        system_prefix=['<s>{{SYSTEM}}\n'],
        template_cls=LlavaHfTemplate,
    ))


class LlavaVideoHfTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<image>\n']
        assert media_type == 'video'
        media_file = inputs.videos[index]
        if media_file.rsplit('.', 1)[-1] in {'jpg', 'png'}:
            return ['<image>\n']
        else:
            inputs.videos[index] = load_video_llava(inputs.videos[index])
            return ['<video>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        videos = inputs.videos or []
        if len(videos) > 0:
            video_processor = self.processor.video_processor
            video_inputs = video_processor(videos, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values_videos'] = video_inputs['pixel_values_videos']
        if len(images) > 0:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['image_sizes'] = image_inputs['image_sizes']
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.llava_next_video_hf,
        prefix=['{{SYSTEM}} '],
        prompt=['USER: {{QUERY}} ASSISTANT:'],
        chat_sep=[' '],
        suffix=[['eos_token_id']],
        template_cls=LlavaVideoHfTemplate,
        auto_add_bos=True,
    ))


class Llava1_6HfTemplate(LlavaHfTemplate):

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        for b in batch:
            pixel_values = b.get('pixel_values')
            if pixel_values is not None:
                b['pixel_values'] = pixel_values.squeeze(0)  # 5d -> 4d
        res = super()._data_collator(batch, padding_to=padding_to)
        return res


@dataclass
class LlavaMistralTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<s>[INST] '])
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}} [/INST]'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['</s>[INST] '])
    suffix: Prompt = field(default_factory=lambda: ['</s>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])


register_template(LlavaMistralTemplateMeta(MLLMTemplateType.llava1_6_mistral_hf, template_cls=Llava1_6HfTemplate))

register_template(
    TemplateMeta(
        MLLMTemplateType.llava1_6_vicuna_hf,
        prefix=['<s>'],
        prompt=['USER: {{QUERY}} ASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        default_system=('A chat between a curious human and an artificial intelligence assistant. '
                        "The assistant gives helpful, detailed, and polite answers to the human's questions."),
        system_prefix=['<s>{{SYSTEM}} '],
        template_cls=Llava1_6HfTemplate))


class LLava1_6YiHfTemplate(Llava1_6HfTemplate):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            return [[64000], '\n']
        else:
            return super().replace_tag(media_type, index, inputs)


register_template(ChatmlTemplateMeta(
    MLLMTemplateType.llava1_6_yi_hf,
    template_cls=LLava1_6YiHfTemplate,
))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llama3_llava_next_hf,
        template_cls=Llava1_6HfTemplate,
        agent_template=None,
    ))

register_template(
    QwenTemplateMeta(MLLMTemplateType.llava_next_qwen_hf, template_cls=Llava1_6HfTemplate, agent_template=None))


class LlavaOneVisionHfTemplate(Llava1_6HfTemplate):

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, 151646)  # <image>
        processor = self.processor
        if images:
            image_processor = processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
            height, width = image_inputs['pixel_values'][0].shape[-2:]
            added_tokens_len = 0
            for idx, pixel_v, image_size in zip(idx_list, image_inputs['pixel_values'], image_inputs['image_sizes']):
                if isinstance(image_size, torch.Tensor):
                    image_size = image_size.tolist()
                orig_height, orig_width = image_size
                num_image_tokens = processor._get_number_of_features(orig_height, orig_width, height, width)
                input_ids = input_ids[:added_tokens_len
                                      + idx] + [151646] * num_image_tokens + input_ids[added_tokens_len + idx + 1:]
                if labels is not None:
                    labels = labels[:added_tokens_len + idx] + [-100] * num_image_tokens + labels[added_tokens_len + idx
                                                                                                  + 1:]
                added_tokens_len += num_image_tokens - 1
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                encoded['image_sizes'] = image_inputs['image_sizes']
        return encoded


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.llava_onevision_hf,
        default_system=None,
        template_cls=LlavaOneVisionHfTemplate,
        agent_template=None,
    ))


class LlavaLlama3_1HfTemplate(LlavaHfTemplate):
    # DaozeZhang
    system = ('You are a helpful language and vision assistant. '
              'You are able to understand the visual content that the user provides, '
              'and assist the user with a variety of tasks using natural language.')

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if len(encoded['pixel_values'].shape) == 5:  # (1, num_patch, 3, H/W, W/H)
            encoded['pixel_values'] = torch.squeeze(encoded['pixel_values'], dim=0)  # (num_patch, 3, H/W, W/H)
        return encoded


register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llava_llama3_1_hf,
        default_system=LlavaLlama3_1HfTemplate.system,
        template_cls=LlavaLlama3_1HfTemplate,
        agent_template=None,
    ))


class LLavaLlama3HfTemplate(Template):
    # xtuner
    image_placeholder = ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        raw_image = inputs.images
        if raw_image:
            pixel_values = self.processor.image_processor(raw_image, return_tensors='pt')['pixel_values']
            encoded['pixel_values'] = pixel_values.to(self.model_info.torch_dtype)
        return encoded


register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llava_llama3_hf,
        template_cls=LLavaLlama3HfTemplate,
        agent_template=None,
    ))


class LLavaTemplate(Template):
    skip_prompt = False
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        image_sizes = [x.size for x in images]
        from llava.mm_utils import process_images
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        if images:
            images_tensor = process_images(images, image_processor, model.config)
            encoded['images'] = images_tensor.to(model.dtype).squeeze(0)
            encoded['image_sizes'] = image_sizes
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = images
            res['image_sizes'] = sum([b['image_sizes'] for b in batch if 'image_sizes' in b], start=[])
        return res


register_template(LlavaMistralTemplateMeta(MLLMTemplateType.llava1_6_mistral, template_cls=LLavaTemplate))

register_template(ChatmlTemplateMeta(MLLMTemplateType.llava1_6_yi, template_cls=LLavaTemplate))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llama3_llava_next,
        template_cls=LLavaTemplate,
        default_system=('You are a helpful language and vision assistant. '
                        'You are able to understand the visual content that the user provides, '
                        'and assist the user with a variety of tasks using natural language.'),
        agent_template=None,
    ))

register_template(QwenTemplateMeta(MLLMTemplateType.llava_next_qwen, template_cls=LLavaTemplate, agent_template=None))


class LLavaOneVision1_5Template(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    use_model = True
    support_padding_free = True

    def init_env_args(self):
        super().init_env_args()
        self.bbox_format = get_env_args('QWENVL_BBOX_FORMAT', str, 'legacy')

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            video, video_kwargs = fetch_video({'video': video}, return_video_sample_fps=True)
            inputs.mm_processor_kwargs.setdefault('fps', []).append(video_kwargs)
            tokens = ['<|vision_start|><|video_pad|><|vision_end|>']
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return tokens

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        else:
            return [ref]

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        else:
            return [str(bbox)]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    kwargs = {}
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                        kwargs['images'] = None
                    media_inputs = processor_func(videos=mm_data, return_tensors='pt', do_resize=False, **kwargs)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
        return {'inputs_embeds': inputs_embeds}


register_template(
    QwenTemplateMeta(MLLMTemplateType.llava_onevision1_5, template_cls=LLavaOneVision1_5Template, agent_template=None))


class LLavaOneVision2Template(Template):
    # token id 与官方 config.json 一致（151652/151653/151655/151656）
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    vision_end_token_id = 151653
    # 两个 pad 都受截断保护: video_pad 在 splice 前短暂存在于 token 序列中（V3）
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    use_model = True
    support_padding_free = False
    # chat_template.jinja 为每个视频产出的占位块。codec 后端把整块重写为
    # <X.X seconds><|vision_start|><|image_pad|>*n<|vision_end|>\n 序列
    VIDEO_BLOCK = '<|vision_start|><|video_pad|><|vision_end|>'
 
    # ------------------------------------------------------------------ env
 
    def init_env_args(self):
        super().init_env_args()
        self.bbox_format = get_env_args('QWENVL_BBOX_FORMAT', str, 'legacy')
        # None = 使用 checkpoint config 默认值（max_pixels=4000000 等）
        self.max_pixels: Optional[int] = get_env_args('MAX_PIXELS', int, None)
        self.num_frames: Optional[int] = get_env_args('NUM_FRAMES', int, None)
        self.max_frames: Optional[int] = get_env_args('VIDEO_MAX_FRAMES', int, None)
        self.target_fps: Optional[float] = get_env_args('FPS', float, None)
        # 与官方 processor 的 video_backend 参数同名同默认值:
        # frames = 帧采样 VideoProcessor; codec = cv-preinfer 码流选帧
        self.video_backend: str = get_env_args('VIDEO_BACKEND', str, 'frames').lower()
        assert self.video_backend in {'frames', 'codec'}, \
            f'VIDEO_BACKEND 仅支持 frames / codec, 得到 {self.video_backend!r}'
 
    # -------------------------------------------------- 官方模块函数的获取
 
    def _official(self, name: str):
        """从 checkpoint 自带模块取函数（processing / video_processing）。
        找不到即报错——宁可失败, 不静默用重写版偏离官方格式。"""
        import sys
        proc_module = sys.modules[type(self.processor).__module__]
        vp_module = sys.modules[type(self.processor.video_processor).__module__]
        for mod in (proc_module, vp_module):
            fn = getattr(mod, name, None)
            if fn is not None:
                return fn
        raise RuntimeError(f'checkpoint 模块中未找到 {name}, 请核对 processing 文件版本')

    def _codec_module(self):
        """import checkpoint 自带的 codec 模块。镜像官方的 相对导入→顶层导入 双回退,
        并适配 transformers 动态模块的包前缀（processing 模块同包的兄弟文件）。"""
        import importlib
        proc_mod_name = type(self.processor).__module__
        candidates = []
        if '.' in proc_mod_name:
            candidates.append(
                proc_mod_name.rsplit('.', 1)[0] + '.codec_video_processing_llava_onevision2')
        candidates.append('codec_video_processing_llava_onevision2')
        last_err = None
        for name in candidates:
            try:
                return importlib.import_module(name)
            except ImportError as e:
                last_err = e
        raise RuntimeError(
            'codec 模块导入失败。codec 后端需要 checkpoint 目录中的 '
            'codec_video_processing_llava_onevision2.py, 且已安装 '
            'codec-video-prep + opencv-python 并保证 ffmpeg 在 PATH 上。'
            f'原始错误: {last_err}')
    
    # ------------------------------------------------------ stage 1: replace_tag
    # 与官方 chat_template.jinja 逐字符一致: 只放结构占位, 不做计算。
 
    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type in {'image', 'video'}, f'llava_onevision2 不支持 {media_type}'
        if media_type == 'image':
            return ['<|vision_start|><|image_pad|><|vision_end|>']
        # 混排防线一（另一半在 _encode 的 IMAGE PATH assert）:
        # 官方 IMAGE PATH 按出现顺序消费 image_pad, 视频重写产生的 image_pad 会被误消费
        assert not inputs.images, 'llava_onevision2 v1: 暂不支持同一样本中图像与视频混用'
        return [self.VIDEO_BLOCK]
 
    # -------------------------------------------------- grounding（沿用 1.5;
    # 注意 grounding 训练属 v1 未验证功能: 裸调 image_processor 后 resize 发生在
    # _encode 内部, bbox 坐标缩放缺少挂载点, 使用前需单独适配）
 
    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        return [ref]
 
    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        return [str(bbox)]
 
    # --------------------------------------------------------- token 级 splice
 
    @staticmethod
    def _splice(input_ids, labels, loss_scale, start: int, end: int, new_tokens: List[int]):
        """input_ids[start:end] → new_tokens; labels 填 -100, loss_scale 填 0。
        官方在字符串上 pattern.sub, 此处为其 token 级等价物。"""
        n = len(new_tokens)
        input_ids = input_ids[:start] + new_tokens + input_ids[end:]
        if labels is not None:
            labels = labels[:start] + [-100] * n + labels[end:]
        if loss_scale is not None:
            loss_scale = loss_scale[:start] + [0.] * n + loss_scale[end:]
        return input_ids, labels, loss_scale
 
    # --------------------------------------------------------- stage 2: _encode
    # 段落顺序与官方 __call__ 一致: VIDEO PATH → IMAGE PATH → 张量装配。
 
    def _splice_video_blocks(self, input_ids, labels, loss_scale, expanded_texts: List[str]):
        """把第 i 个 <start><video_pad><end> 三元组替换为 expanded_texts[i] 的 token。
        frames / codec 两后端共用; 逆序 splice 免下标位移。"""
        idx_list = findall(input_ids, self.video_token_id)
        assert len(idx_list) == len(expanded_texts), (
            f'<|video_pad|> 占位数 {len(idx_list)} != 视频数 {len(expanded_texts)}')
        tokenizer = self.processor.tokenizer
        for i in range(len(idx_list) - 1, -1, -1):
            idx = idx_list[i]
            assert input_ids[idx - 1] == self.vision_start_token_id \
                and input_ids[idx + 1] == self.vision_end_token_id, (
                    'video_pad 未被 vision_start/end 包裹, 与 chat_template 不符')
            new_tokens = tokenizer(expanded_texts[i], add_special_tokens=False)['input_ids']
            input_ids, labels, loss_scale = self._splice(
                input_ids, labels, loss_scale, idx - 1, idx + 2, new_tokens)
        return input_ids, labels, loss_scale
 
    # -------------------------------------------------- 视频后端一: frames（默认）
 
    def _encode_video_frames(self, inputs: StdTemplateInputs):
        """镜像官方 VIDEO PATH。返回 (expanded_texts, tensors)。"""
        processor = self.processor
        sms = int(processor.spatial_merge_size)
        vp = processor.video_processor
        saved = (vp.fixed_num_frames, vp.max_frames, vp.target_fps)
        try:
            if self.num_frames is not None:
                vp.fixed_num_frames = int(self.num_frames)
            if self.max_frames is not None:
                vp.max_frames = int(self.max_frames)
            if self.target_fps is not None:
                vp.target_fps = float(self.target_fps)
            video_outputs = vp(videos=list(inputs.videos), return_tensors='pt')
        finally:
            vp.fixed_num_frames, vp.max_frames, vp.target_fps = saved
 
        video_grid_thw = video_outputs['video_grid_thw']           # [num_videos, 3]
        frame_timestamps = video_outputs['frame_timestamps']
        _expand = self._official('_expand_video_block_for_frames')
 
        expanded_texts = []
        for video_idx in range(video_grid_thw.shape[0]):
            t_eff = int(video_grid_thw[video_idx, 0].item())
            h_p = int(video_grid_thw[video_idx, 1].item())
            w_p = int(video_grid_thw[video_idx, 2].item())
            n_per_frame = (h_p * w_p) // (sms * sms)
            frame_seconds = list(frame_timestamps[video_idx])      # 官方防御性对齐
            if len(frame_seconds) < t_eff:
                frame_seconds += [frame_seconds[-1] if frame_seconds else 0.0] \
                    * (t_eff - len(frame_seconds))
            else:
                frame_seconds = frame_seconds[:t_eff]
            expanded_texts.append(_expand(n_per_frame, frame_seconds).rstrip('\n'))
 
        # frames 方向: 拆行 [T,H,W] → T×[1,H,W]（V4）
        expanded_rows = []
        for row in video_grid_thw:
            t_v, h_v, w_v = int(row[0]), int(row[1]), int(row[2])
            expanded_rows.extend([[1, h_v, w_v]] * t_v)
        tensors = {
            'pixel_values': video_outputs['pixel_values_videos'],
            'image_grid_thw': torch.tensor(expanded_rows, dtype=video_grid_thw.dtype),
            'patch_positions': video_outputs['patch_positions'],
        }
        return expanded_texts, tensors
 
    # -------------------------------------------------- 视频后端二: codec
 
    def _encode_video_codec(self, inputs: StdTemplateInputs):
        """镜像官方 CODEC VIDEO BACKEND 段（V12）。返回 (expanded_texts, tensors)。"""
        codec = self._codec_module()
        processor = self.processor
        ip = processor.image_processor
 
        # 有效配置: preprocessor_config 的 codec 字段(处理器加载时存入) < env MAX_PIXELS
        # 像素预算统一逻辑与官方一致: max_pixels 参数 > cfg > image_processor.max_pixels
        cfg_kwargs = dict(getattr(processor, '_codec_config_defaults', None) or {})
        effective_max_pixels = int(
            self.max_pixels if self.max_pixels is not None
            else cfg_kwargs.get('max_pixels', getattr(ip, 'max_pixels', 150000)))
        cfg_kwargs['max_pixels'] = effective_max_pixels
        cfg = codec.CodecConfig(**cfg_kwargs)
 
        expanded_texts = []
        pv_list, grid_rows, pp_list = [], [], []
        for video_url in inputs.videos:
            assert isinstance(video_url, str), (
                'codec 后端要求视频为文件路径/URL（需解析码流）, '
                f'得到 {type(video_url)}; 已解码帧请改用 VIDEO_BACKEND=frames')
            payload = codec.process_codec_video(video_url, cfg)
            imgs, src_positions, _ = codec.drop_padding_canvases(
                payload['images'], payload['src_positions'])
            if not imgs:
                raise RuntimeError(f'codec 未能为 {video_url} 产出可用 canvas')
            image_data = codec.codec_image_processor_outputs(
                ip, imgs, max_pixels=effective_max_pixels)
            image_grid_thw = image_data['image_grid_thw']
            patch_positions = codec.codec_positions_for_processor(
                src_positions, image_grid_thw, device=image_grid_thw.device)
            # P1: 官方对整段 prompt 调用, 此处喂裸占位块取回展开文本（按块替换应等价,
            # codec token diff 若在块边界分叉优先查此处）
            expanded = codec.rewrite_text_with_codec_positions(
                self.VIDEO_BLOCK, patch_positions,
                fps=float(payload['fps']), decimals=1)
            expanded_texts.append(expanded)
 
            pv_list.append(image_data['pixel_values'])
            pp_list.append(patch_positions)
            # codec 方向: 合并 N×[1,H,W] → [[N,H,W]]（V13, 官方原文照抄——
            # 使 _build_cu_seqlens(fixed_t=4) 将 canvas 分入 4 帧注意力窗口）
            grid = image_grid_thw
            if (grid.shape[0] > 1
                    and bool(torch.all(grid[:, 1] == grid[0, 1]).item())
                    and bool(torch.all(grid[:, 2] == grid[0, 2]).item())):
                grid_rows.append(torch.tensor(
                    [[int(grid.shape[0]), int(grid[0, 1]), int(grid[0, 2])]],
                    dtype=grid.dtype, device=grid.device))
            else:
                grid_rows.append(grid)
 
        tensors = {
            'pixel_values': torch.cat(pv_list, dim=0),
            'image_grid_thw': torch.cat(grid_rows, dim=0),
            'patch_positions': torch.cat(pp_list, dim=0),
        }
        return expanded_texts, tensors
 
    # --------------------------------------------------------- stage 2: _encode
 
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        sms = int(processor.spatial_merge_size)
        if self.max_pixels is not None:
            ip = processor.image_processor
            ip.max_pixels = self.max_pixels
            if isinstance(getattr(ip, 'size', None), dict):
                ip.size['longest_edge'] = self.max_pixels
 
        out: Dict[str, Any] = {}
 
        # ================= VIDEO（codec / frames 二选一, 产物同构）=================
        if inputs.videos:
            if self.video_backend == 'codec':
                expanded_texts, out = self._encode_video_codec(inputs)
            else:
                expanded_texts, out = self._encode_video_frames(inputs)
            input_ids, labels, loss_scale = self._splice_video_blocks(
                input_ids, labels, loss_scale, expanded_texts)
 
        # ================= IMAGE PATH =================
        if inputs.images:
            image_outputs = processor.image_processor(images=inputs.images, return_tensors='pt')
            image_grid_thw = image_outputs['image_grid_thw']
            idx_list = findall(input_ids, self.image_token_id)
            assert len(idx_list) == image_grid_thw.shape[0], (
                f'<|image_pad|> 占位数 {len(idx_list)} != 图像数 {image_grid_thw.shape[0]}')
 
            merge_factor = sms * sms
            image_token_counts = (
                (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
                // merge_factor).tolist()
 
            def _get_new_tokens(i):
                return [self.image_token_id] * int(image_token_counts[i])
 
            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens)
 
            build_patch_positions = self._official('build_patch_positions')
            image_pp = build_patch_positions(image_grid_thw, spatial_merge_size=sms)
            if 'pixel_values' in out:      # 混排分支因 F8 不可达, 保留官方同构写法
                out['pixel_values'] = torch.cat([out['pixel_values'],
                                                 image_outputs['pixel_values']], dim=0)
                out['image_grid_thw'] = torch.cat([out['image_grid_thw'], image_grid_thw], dim=0)
                out['patch_positions'] = torch.cat([out['patch_positions'], image_pp], dim=0)
            else:
                out['pixel_values'] = image_outputs['pixel_values']
                out['image_grid_thw'] = image_grid_thw
                out['patch_positions'] = image_pp
 
        # ================= 终检 + 汇入 =================
        assert self.video_token_id not in input_ids, \
            'video_pad 泄漏到最终序列, VIDEO 重写未生效'
        encoded.update(out)
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded
 
    # ------------------------------------------------------------- collator
 
    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        patch_positions = [b['patch_positions'] for b in batch
                           if b.get('patch_positions') is not None]
        if len(patch_positions) > 0:
            res['patch_positions'] = torch.concat(patch_positions)
        return res
 
    # ---------------------------------------------------------- _post_encode
 
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
 
        pixel_values = inputs.get('pixel_values')
        if pixel_values is None:
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            media_inputs = self.processor.image_processor(images=images, return_tensors='pt')
            media_inputs = to_device(media_inputs, input_ids.device)
            build_pp = self._official('build_patch_positions')
            dummy_pp = to_device(
                build_pp(media_inputs['image_grid_thw'].cpu(),
                         spatial_merge_size=int(self.processor.spatial_merge_size)),
                input_ids.device)
            image_embeds = self._call_visual(
                base_model, media_inputs['pixel_values'],
                media_inputs['image_grid_thw'], dummy_pp)
            inputs_embeds = inputs_embeds + image_embeds.mean().to(inputs_embeds.device) * 0.
        else:
            image_embeds = self._call_visual(
                base_model, pixel_values, inputs['image_grid_thw'], inputs['patch_positions'])
            image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = image_mask.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return {'inputs_embeds': inputs_embeds}
 
    @staticmethod
    def _call_visual(base_model, pixel_values, image_grid_thw, patch_positions):
        """V6 陷阱: 必须调内层 base_model.model 的 get_image_features
        （顶层同名方法丢弃 patch_positions）。内层签名已核对, 返回逐图 list。"""
        image_embeds = base_model.model.get_image_features(
            pixel_values, image_grid_thw, patch_positions=patch_positions)
        return torch.cat(list(image_embeds), dim=0)
 
 
register_template(
    QwenTemplateMeta(
        MLLMTemplateType.llava_onevision2,
        template_cls=LLavaOneVision2Template,
        agent_template=None,
    ))