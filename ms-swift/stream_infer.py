#!/usr/bin/env python3
"""JoyAI streaming video understanding —— 训练后的流式推理。

逐秒把视频帧喂给模型, 每秒模型输出:
  </silence>            -> 这一秒保持沉默
  </response> <文本>    -> 这一秒该说话, 输出事件/回答
可在指定秒注入用户问题(--question "..." --question-time 4)。

⚠️ 关键: MAX_PIXELS 必须和训练时一致(决定每帧 token 数, 否则和训练分布对不上)。

用法:
    
HF_HUB_OFFLINE=1  python3 stream_infer.py --model /root/zlx_workspace/test/msswift-docker/ms-swift/output/joy_streaming/v6-20260720-164122/checkpoint-1500 --video /data1/zlx/cache/huggingface/hub/vl-interaction/videos_pool/PerceptionTest/video_68.mp4 \
    --fps 1 --max-pixels 50176 \
    --question "whati is the people doing?" --question-time 0 \
    --topk 10 --max-seconds 3


本版为【稳妥版】: 每秒对「至今所有帧+历史」重跑 generate, 保证和训练格式一致、一定能跑。
代价是 O(T^2)(每秒重编码历史)。视频长/要提速 -> 见文件末尾 "KV-cache 加速" 注释。
"""
import argparse
import os
import sys


def decode_frames(video, fps):
    """按 fps 抽帧 -> list[PIL.Image], 以及每帧对应的秒(int)。"""
    import decord
    from PIL import Image
    try:
        decord.logging.set_level(decord.logging.QUIET)
    except Exception:
        pass
    vr = decord.VideoReader(video, num_threads=1)   # 单线程稳(避免坏 h264 崩)
    n = len(vr)
    src_fps = vr.get_avg_fps() or 30.0
    dur = n / src_fps
    secs = list(range(int(dur)))                    # 每秒一帧
    idx = [min(int(s * src_fps), n - 1) for s in secs]
    arr = vr.get_batch(idx).asnumpy()               # [T,H,W,3] RGB
    frames = [Image.fromarray(a) for a in arr]
    return frames, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='训练后的 checkpoint 目录')
    ap.add_argument('--video', required=True)
    ap.add_argument('--fps', type=float, default=1.0, help='抽帧率(和训练自适应fps对应; 长视频=1)')
    ap.add_argument('--max-pixels', type=int, default=50176, help='!! 必须和训练一致 !!')
    ap.add_argument('--question', default=None)
    ap.add_argument('--question-time', type=int, default=None, help='问题注入在第几秒')
    ap.add_argument('--max-new-tokens', type=int, default=64)
    ap.add_argument('--system', default=None,
                    help='system prompt。默认自动对齐训练模板的 STREAMING_SYSTEM_PROMPT '
                         '(从 swift 导入)。!! 必须和训练时一致, 否则 train/infer 漂移。'
                         '传 "" 表示不加 system。')
    ap.add_argument('--response-threshold', type=float, default=0.0,
                    help='0=纯贪心(易全沉默); >0: P(</response>)>=阈值就开口(0.1~0.5 常用, 越小越爱说)')
    ap.add_argument('--topk', type=int, default=0,
                    help='调试: 打印每秒第一个 token 的前 N 名(看模型到底想输出啥)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--max-seconds', type=int, default=0, help='>0 只跑前 N 秒(调试)')
    args = ap.parse_args()

    os.environ['MAX_PIXELS'] = str(args.max_pixels)
    os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print('加载模型...', flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    # 让每帧 token 数与训练一致
    if hasattr(processor, 'image_processor'):
        processor.image_processor.max_pixels = args.max_pixels
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, dtype=getattr(torch, args.dtype), device_map=args.device)
    model.eval()
    tok = processor.tokenizer
    # 关键: 模型把 </silence>/</response> 当【多子词序列】输出(都以 '</' 开头), 不是单
    # token。所以【绝不加单 special token】(加了会让历史里的控制 token 变成训练没见过的
    # 单 token), 判定一律走【字符串解析】。算出子词序列 + 分叉点(阈值触发看这里的分布)。
    sil_seq = tok.encode('</silence>', add_special_tokens=False)
    resp_seq = tok.encode('</response>', add_special_tokens=False)
    di = 0
    while di < min(len(sil_seq), len(resp_seq)) and sil_seq[di] == resp_seq[di]:
        di += 1
    sil_branch, resp_branch = sil_seq[di], resp_seq[di]
    print(f'  </silence> -> {sil_seq}  </response> -> {resp_seq}')
    print(f'  分叉@第{di}个token: silence={sil_branch}({tok.decode([sil_branch])!r}) '
          f'vs response={resp_branch}({tok.decode([resp_branch])!r})')

    frames, secs = decode_frames(args.video, args.fps)
    if args.max_seconds:
        frames, secs = frames[:args.max_seconds], secs[:args.max_seconds]
    print(f'视频 {len(secs)} 秒, 逐秒流式推理 (max_pixels={args.max_pixels})\n', flush=True)

    pad_id = tok.pad_token_id or tok.eos_token_id
    im_end = tok.convert_tokens_to_ids('<|im_end|>')

    # system prompt: 必须和训练时【完全一致】。默认从 swift 导入训练模板用的那句
    # (STREAMING_SYSTEM_PROMPT), 保证永不漂移; 导入失败(没配 PYTHONPATH)则回退硬编码副本。
    if args.system is None:
        try:
            from swift.template.templates.llava import STREAMING_SYSTEM_PROMPT as _SYS
        except Exception:
            _SYS = (
                'You are a real-time video streaming assistant observing a continuous camera feed '
                'frame by frame. The last frame represents the current moment.\n'
                '## Action Format\n'
                'At every inference step you MUST choose exactly one of the following two actions:\n'
                '**Stay silent** — output ONLY:\n</silence>\n'
                'Choose this when nothing noteworthy has changed in the scene, no user query is '
                'pending, or there is nothing useful to say.\n'
                '**Speak** — output the token followed by a concise reply:\n</response> Your reply here.\n'
                'Choose this when you observe something worth reporting or a significant state '
                'change, or when you can answer a user question based on available evidence.')
        args.system = _SYS
    messages = []          # 累积的对话历史
    if args.system:
        messages.append({'role': 'system', 'content': args.system})
        print(f'system prompt: {args.system[:60]}...  (len={len(args.system)})', flush=True)
    else:
        print('system prompt: <无>', flush=True)
    events = []            # (秒, 回答文本)

    for i, sec in enumerate(secs):
        # 1) 组这一秒的 user turn: 结构化 content(chat_template 只认 {'type':'image'} 的
        #    dict, 字符串里的 <image> 不会展开)。text 项自带 \n, 拼出训练的
        #    "question\n<sec.0 seconds>\n<vision块>" 结构。
        content = []
        if args.question_time is not None and sec == args.question_time and args.question:
            content.append({'type': 'text', 'text': args.question + '\n'})
        content.append({'type': 'text', 'text': f'<{sec}.0 seconds>\n'})
        content.append({'type': 'image'})
        messages.append({'role': 'user', 'content': content})

        # 2) 至今所有帧 + 历史 -> 文本
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=frames[:i + 1], return_tensors='pt', padding=True)
        inputs = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        prompt_len = inputs['input_ids'].shape[-1]

        # 3) 贪心生成这一秒的 assistant turn(控制 token 是多子词, 靠字符串解析判定)。
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                  num_beams=1, use_cache=True, eos_token_id=im_end, pad_token_id=pad_id,
                                  output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0][prompt_len:]
        resp_str = tok.decode(seq, skip_special_tokens=False)
        resp_str = resp_str.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
        is_resp = resp_str.startswith('</response>')
        # 分叉点(</ 之后)的 response 概率: 阈值触发 + 诊断
        p_resp = 0.0
        if len(out.scores) > di:
            bp = torch.softmax(out.scores[di][0].float(), dim=-1)
            p_resp = bp[resp_branch].item()
        if args.topk > 0:
            p0 = torch.softmax(out.scores[0][0].float(), dim=-1)
            tv, ti = p0.topk(args.topk)
            print(f'  [{sec:>3}s] top首token: '
                  f'{[(int(j), repr(tok.decode([int(j)])), round(float(v),3)) for v, j in zip(tv, ti)]}')
            if len(out.scores) > di:
                print(f'          分叉点: response={p_resp:.4f} silence={bp[sil_branch].item():.4f}')

        # 阈值触发: 模型本来沉默, 但分叉点 response 概率 >= 阈值 -> 强制走 response 路径重生成
        if args.response_threshold > 0 and not is_resp and p_resp >= args.response_threshold:
            dev = inputs['input_ids'].device
            forced = torch.cat([inputs['input_ids'], torch.tensor([resp_seq], device=dev)], dim=1)
            gi = dict(inputs)
            gi['input_ids'] = forced
            if gi.get('attention_mask') is not None:
                gi['attention_mask'] = torch.cat(
                    [gi['attention_mask'], torch.ones((1, len(resp_seq)), dtype=gi['attention_mask'].dtype,
                                                      device=dev)], dim=1)
            with torch.inference_mode():
                o2 = model.generate(**gi, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    num_beams=1, use_cache=True, eos_token_id=im_end, pad_token_id=pad_id)
            tail = tok.decode(o2[0][forced.shape[1]:], skip_special_tokens=False)
            resp_str = '</response> ' + tail.replace('<|im_end|>', '').strip()
            is_resp = True

        # 记历史 + 打印
        if is_resp:
            ans = resp_str[len('</response>'):].strip()
            messages.append({'role': 'assistant', 'content': resp_str})
            events.append((sec, ans))
            print(f'  [{sec:>3}s] 🗣  {ans}   (P响应={p_resp:.3f})', flush=True)
        else:
            messages.append({'role': 'assistant', 'content': '</silence>'})   # 非 response 一律记沉默
            tag = '...' if resp_str.startswith('</silence>') else f'⚠非预期:{resp_str[:40]!r}'
            print(f'  [{sec:>3}s] {tag}   (P响应={p_resp:.3f})', flush=True)

    print(f'\n=== 事件汇总({len(events)} 条)===')
    for sec, ans in events:
        print(f'  {sec}s: {ans}')


if __name__ == '__main__':
    sys.exit(main())


# =============================================================================
# KV-cache 加速(把 O(T^2) 降到 O(T), 真·流式):
# 上面每秒都对「至今所有帧」重跑 generate(处理器每秒重编码全部历史帧)。真流式应
# 复用 KV-cache, 每秒只处理【新帧 + 新生成的 token】。核心改法:
#
#   cache = None
#   for i, sec in enumerate(secs):
#       # 只 tokenize 这一秒的【新增段】(不含历史), 首秒才带 system 前缀:
#       seg = f"<|im_start|>user\n{q}<{sec}.0 seconds>\n<image><|im_end|>\n<|im_start|>assistant\n"
#       si = processor(text=[seg], images=[frames[i]], return_tensors='pt')  # 只当前帧
#       out = model(**si, past_key_values=cache, use_cache=True)             # 模型内部注入视觉+扩cache
#       cache = out.past_key_values
#       nxt = out.logits[:, -1].argmax(-1)
#       gen = []
#       while nxt.item() != im_end and len(gen) < max_new:                   # 逐 token 贪心, 复用cache
#           gen.append(nxt.item())
#           out = model(input_ids=nxt[None], past_key_values=cache, use_cache=True)  # 纯文本, 无pixel
#           cache = out.past_key_values
#           nxt = out.logits[:, -1].argmax(-1)
#       # 把生成的 assistant + "<|im_end|>\n" 也 forward 进 cache, 供下一秒接续
#       ...解析 gen 的 </silence>/</response>...
#
# 注意点(上机验证): ① 模型 forward 是否直接吃 input_ids+pixel_values+past_key_values
#   (OV2 modeling forward 已确认有 past_key_values/patch_positions 参数); ② position_ids
#   在带 cache 增量前向时是否需手动传(多数 HF 模型会自动按 cache 长度推); ③ 生成的
#   assistant token 要显式喂回 cache 才算进历史。先用上面稳妥版验证【输出正确】, 再换
#   cache 版验证【输出一致 + 变快】。
# =============================================================================
