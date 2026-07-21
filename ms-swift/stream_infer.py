#!/usr/bin/env python3
"""JoyAI streaming video understanding —— 训练后的流式推理。

逐秒把视频帧喂给模型, 每秒模型输出:
  </silence>            -> 这一秒保持沉默
  </response> <文本>    -> 这一秒该说话, 输出事件/回答
可在指定秒注入用户问题(--question "..." --question-time 4)。

⚠️ 关键: MAX_PIXELS 必须和训练时一致(决定每帧 token 数, 否则和训练分布对不上)。

用法:
  python3 stream_infer.py --model /path/to/trained_ckpt --video /path/to/clip.mp4 \\
      --fps 1 --max-pixels 50176 \\
      [--question "count the reps" --question-time 4]

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
    # 确认控制 token 在词表里(训练时 --new_special_tokens 加的; 存 checkpoint 时应已保存)
    for t in ('</silence>', '</response>'):
        assert tok.convert_tokens_to_ids(t) != tok.unk_token_id, \
            f'{t} 不在词表! 确认 checkpoint 保存了 new_special_tokens + resize 后的 embedding'

    frames, secs = decode_frames(args.video, args.fps)
    if args.max_seconds:
        frames, secs = frames[:args.max_seconds], secs[:args.max_seconds]
    print(f'视频 {len(secs)} 秒, 逐秒流式推理 (max_pixels={args.max_pixels})\n', flush=True)

    pad_id = tok.pad_token_id or tok.eos_token_id
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    messages = []          # 累积的对话历史
    events = []            # (秒, 回答文本)

    for i, sec in enumerate(secs):
        # 1) 组这一秒的 user turn(和训练 preprocessor 完全同构)
        parts = []
        if args.question_time is not None and sec == args.question_time and args.question:
            parts.append(args.question)
        parts.append(f'<{sec}.0 seconds>')
        parts.append('<image>')
        messages.append({'role': 'user', 'content': '\n'.join(parts)})

        # 2) 至今所有帧 + 历史 -> 文本, 让模型生成这一秒的 assistant turn
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=frames[:i + 1], return_tensors='pt', padding=True)
        inputs = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        prompt_len = inputs['input_ids'].shape[-1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                  num_beams=1, use_cache=True, eos_token_id=im_end, pad_token_id=pad_id)
        resp = tok.decode(out[0][prompt_len:], skip_special_tokens=False)
        resp = resp.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()

        # 3) 解析 silence / response
        messages.append({'role': 'assistant', 'content': resp})
        if resp.startswith('</response>'):
            ans = resp[len('</response>'):].strip()
            events.append((sec, ans))
            print(f'  [{sec:>3}s] 🗣  {ans}', flush=True)
        elif resp.startswith('</silence>'):
            print(f'  [{sec:>3}s] ...', flush=True)               # 沉默
        else:
            print(f'  [{sec:>3}s] ⚠ 非预期输出: {resp[:80]!r}', flush=True)

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
