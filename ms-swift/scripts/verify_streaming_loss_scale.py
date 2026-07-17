# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end sanity check for the JoyAI streaming loss_scale.

It drives a real JoyAI-style sample (per-second </silence> / </response> turns,
one <image> per second) through the *actual* ms-swift encode path and prints the
input_ids / labels / loss_scale columns token-by-token, so you can eyeball that:

  - control tokens carry w_response / w_silence_first / w_silence_repeated,
  - the response body and the <|im_end|> suffix carry 1.0,
  - prompt + vision tokens carry 0.0 (label == -100),
  - the offline (turn-based) sample reduces to plain SFT (all weights 1.0).

Two modes:

  # 1) fast, tokenizer-only (no vision weights loaded, <image> uses a stub token)
  python scripts/verify_streaming_loss_scale.py --model_dir /path/to/OV2 --no_vision

  # 2) full multimodal encode with real frames (needs the images to exist on disk)
  python scripts/verify_streaming_loss_scale.py --model_dir /path/to/OV2 \
      --data /path/to/processed.jsonl --index 0

Register the special tokens the same way training does with --new_special_tokens;
pass --raw_tokens to see what happens WITHOUT registering them (the control tokens
get shredded into subwords, which is exactly the failure we want to avoid).
"""
import argparse
import json
import os

os.environ.setdefault('W_SILENCE_FIRST', '1.0')
os.environ.setdefault('W_SILENCE_REPEATED', '0.4')
os.environ.setdefault('W_RESPONSE', '1.5')


def build_synthetic_sample(n_seconds=6, response_at=(3, )):
    """Mirror datasets/convert_data.py: one user/assistant pair per second."""
    messages = []
    images = []
    for sec in range(n_seconds):
        messages.append({'role': 'user', 'content': f'<{sec}.0 seconds>\n<image>'})
        images.append(f'frame_{sec:06d}.jpg')  # placeholder path; only used with --data
        if sec in response_at:
            messages.append({'role': 'assistant', 'content': '</response> a person walks in'})
        else:
            messages.append({'role': 'assistant', 'content': '</silence>'})
    return {'messages': messages, 'images': images}


OFFLINE_SAMPLE = {
    'messages': [
        {'role': 'user', 'content': 'What is 2 + 2?'},
        {'role': 'assistant', 'content': 'It is 4.'},
    ]
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True, help='local OV2 snapshot dir')
    p.add_argument('--data', default=None, help='optional processed JoyAI jsonl; uses --index row')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--loss_scale', default='joy_streaming')
    p.add_argument('--max_length', type=int, default=8192)
    p.add_argument('--no_vision', action='store_true',
                   help='skip loading the vision tower; replace <image> with a single stub token')
    p.add_argument('--raw_tokens', action='store_true',
                   help='do NOT register </silence></response> as special tokens (show the breakage)')
    args = p.parse_args()

    from swift.model import get_processor
    from swift.template import get_template, TemplateInputs

    processor = get_processor(args.model_dir, use_hf=True)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    ctrl_tokens = ['</silence>', '</response>']
    if not args.raw_tokens:
        added = tokenizer.add_special_tokens({'additional_special_tokens': ctrl_tokens})
        print(f'[setup] added {added} special tokens: {ctrl_tokens}')
    else:
        print('[setup] --raw_tokens: control tokens NOT registered')
    for t in ctrl_tokens:
        print(f'[setup] tokenize({t!r}) -> {tokenizer.tokenize(t)}')

    template = get_template(
        processor, max_length=args.max_length, truncation_strategy='right', loss_scale=args.loss_scale)
    template.set_mode('train')

    # In --no_vision mode we drop the frames entirely: the <image> tag then stays as
    # literal text inside the (weight-0) user turn, so it never touches |A| and the
    # loss_scale columns we are verifying are unaffected. This avoids loading the
    # vision tower and reading real frames off disk.

    def encode(sample, title):
        if args.no_vision:
            sample = {
                'messages': [{
                    'role': m['role'],
                    'content': m['content'].replace('\n<image>', '').replace('<image>', ''),
                } for m in sample['messages']]
            }
        inputs = TemplateInputs.from_dict(sample)
        encoded = template.encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale')
        print(f'\n===== {title} =====')
        print(f'len(input_ids)={len(input_ids)}  '
              f'supervised |A|={sum(1 for x in labels if x != -100)}  '
              f'loss_scale is {"None (binary!)" if loss_scale is None else "present"}')
        if loss_scale is None:
            print('!! loss_scale is None -> is_binary path taken; weights lost. Check is_binary=False.')
            return
        # collapse the long <|image_pad|> runs for readability
        print(f'{"tok":>7} {"w":>5} {"lbl":>7}  text')
        prev_img = False
        img_run = 0
        for tid, lb, w in zip(input_ids, labels, loss_scale):
            piece = tokenizer.decode([tid])
            is_img = tid == tokenizer.convert_tokens_to_ids('<|image_pad|>')
            if is_img:
                img_run += 1
                prev_img = True
                continue
            if prev_img:
                print(f'{"...":>7} {"0":>5} {"-100":>7}  <|image_pad|> x{img_run}')
                img_run = 0
                prev_img = False
            mark = '  <== CTRL' if w not in (0.0, 1.0) else ''
            print(f'{tid:>7} {w:>5} {lb:>7}  {piece!r}{mark}')
        if prev_img:
            print(f'{"...":>7} {"0":>5} {"-100":>7}  <|image_pad|> x{img_run}')

    if args.data:
        with open(args.data, 'r', encoding='utf-8') as f:
            rows = [json.loads(x) for x in f] if args.data.endswith('.jsonl') else json.load(f)
        sample = rows[args.index]
        sample = {k: sample[k] for k in ('messages', 'images') if k in sample}
        encode(sample, f'REAL sample #{args.index} from {os.path.basename(args.data)}')
    else:
        encode(build_synthetic_sample(), 'SYNTHETIC streaming sample')
    encode(OFFLINE_SAMPLE, 'OFFLINE turn-based sample (should be all 1.0)')


if __name__ == '__main__':
    main()
