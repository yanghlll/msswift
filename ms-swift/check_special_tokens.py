#!/usr/bin/env python3
"""训练前预检: 确认 </silence> </response> 会被当【单 special token】加进词表。

复刻 ms-swift 的加 token 路径(model_args._init_new_special_tokens +
register._add_new_special_tokens), 30 秒内验完, 不用真跑训练。

用法:
    python3 check_special_tokens.py /path/to/model
    # 或验已训练的 checkpoint(检查它有没有把 token 存进去):
    python3 check_special_tokens.py /path/to/checkpoint-1500 --check-only
"""
import argparse
import sys


def init_new_special_tokens(raw):
    """复刻 swift/arguments/base_args/model_args.py:_init_new_special_tokens"""
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for token in raw:
        if token.endswith('.txt'):
            with open(token, encoding='utf-8') as f:
                out += f.read().split()
        else:
            out.append(token)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--new-special-tokens', nargs='+', default=['</silence>', '</response>'],
                    help='和训练脚本 --new_special_tokens 写法一致(空格分开!)')
    ap.add_argument('--check-only', action='store_true',
                    help='只检查 model 目录的 tokenizer 里现在有没有这些单 token(验已训练的 ckpt)')
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    unk = tok.unk_token_id
    want = ['</silence>', '</response>']

    print(f'词表大小(加之前): {len(tok)}')
    print('=' * 60)

    if not args.check_only:
        # 复刻训练时的解析 + 添加
        parsed = init_new_special_tokens(args.new_special_tokens)
        print(f'CLI 传入: {args.new_special_tokens}')
        print(f'解析后要加的 token: {parsed}')
        if len(parsed) != 2 or any(',' in t for t in parsed):
            print(f'\n❌❌ 解析出的不是两个干净 token! 检查 --new_special_tokens 写法:')
            print(f'   对: --new_special_tokens \'</silence>\' \'</response>\'  (空格分)')
            print(f'   错: --new_special_tokens \'</silence>,</response>\'      (逗号=一个怪token)')
            return 1
        n = tok.add_special_tokens({'additional_special_tokens': parsed})
        print(f'add_special_tokens 返回 num_new_tokens = {n}   (训练日志会打 "Added {n} new special tokens")')
        if n != 2:
            print(f'❌ 期望新增 2 个, 实际 {n} 个')

    print('=' * 60)
    ok = True
    for t in want:
        tid = tok.convert_tokens_to_ids(t)
        ids = tok.encode(t, add_special_tokens=False)
        single = (tid != unk) and (len(ids) == 1)
        mark = '✅' if single else '❌'
        print(f'{mark} {t:14s} -> id={tid}  encode={ids}  {"(单token)" if single else "(被切成子词!)"}')
        ok = ok and single

    print('=' * 60)
    if ok:
        print('✅ 通过: 两个控制 token 都是单 token, 训练会正确学到它们的 embedding。')
        print('   推理时可直接按 token id 判定 </silence>/</response>。')
        return 0
    else:
        if args.check_only:
            print('❌ 这个 checkpoint 的 tokenizer 里没有单 token —— 训练时没加对(逗号 bug),')
            print('   模型学的是子词, 推理只能走字符串解析。要单 token 得修脚本重训。')
        else:
            print('❌ 没通过。')
        return 1


if __name__ == '__main__':
    sys.exit(main())
