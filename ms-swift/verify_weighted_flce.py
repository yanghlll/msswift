"""训练节点跑: 验证 Liger FLCE 两种加权写法与参照实现的 loss/梯度一致性。

用法:  python verify_weighted_flce.py        # 需要 GPU + liger-kernel

三种实现对比(相同输入):
  A. 参照: 完整 logits + F.cross_entropy(reduction='none') × weights   [现行 swift/LF 路径]
  B. FLCE reduction='none': loss向量 × weights                          [最简, 任意权重]
  C. FLCE reduction='sum' 按权重档分组求和                              [离散权重, 最保守]
B/C 均不 materialize 完整 logits。CPU 端数学等价性(float64, 30组含边界)已另行验证通过。
"""
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), '需要 GPU(liger 是 Triton kernel)'
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

dev = 'cuda'
torch.manual_seed(0)
N, H, V = 4096, 1024, 151936          # 接近真实规模: 4k 监督位 × Qwen 词表
dtype = torch.bfloat16

hidden = torch.randn(N, H, device=dev, dtype=dtype)
W = torch.randn(V, H, device=dev, dtype=dtype) * 0.02
labels = torch.randint(0, V, (N,), device=dev)
labels[torch.rand(N, device=dev) < 0.5] = -100
weights = torch.tensor([1.0, 0.4, 1.5, 1.0], device=dev)[torch.randint(0, 4, (N,), device=dev)]
valid = labels != -100
denom = weights[valid].sum()


def run(fn):
    h = hidden.detach().clone().requires_grad_(True)
    w = W.detach().clone().requires_grad_(True)
    loss = fn(h, w)
    loss.backward()
    return loss.detach().float(), h.grad.detach().float(), w.grad.detach().float()


def ref(h, w):                         # A 参照
    logits = (h @ w.T).float()
    ce = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')
    return (ce * weights)[valid].sum() / denom


def flce_none(h, w):                   # B
    flce = LigerFusedLinearCrossEntropyLoss(reduction='none')
    vec = flce(w, h, labels)
    return (vec * weights).sum() / denom


def flce_group(h, w):                  # C
    flce = LigerFusedLinearCrossEntropyLoss(reduction='sum')
    total = h.new_zeros((), dtype=torch.float32)
    for wv in weights[valid].unique():
        m = valid & (weights == wv)
        total = total + wv * flce(w, h[m], labels[m])
    return total / denom


la, ha, wa = run(ref)
peak_ref = torch.cuda.max_memory_allocated() / 2**30
for name, fn in (('B: FLCE none×weights', flce_none), ('C: FLCE 分组sum', flce_group)):
    torch.cuda.reset_peak_memory_stats()
    lb, hb, wb = run(fn)
    peak = torch.cuda.max_memory_allocated() / 2**30
    # bf16 kernel 与 fp32 参照存在正常数值差, 用相对误差
    dl = (la - lb).abs() / la.abs()
    dh = (ha - hb).abs().max() / ha.abs().max().clamp_min(1e-8)
    dw = (wa - wb).abs().max() / wa.abs().max().clamp_min(1e-8)
    ok = dl < 5e-3 and dh < 5e-2 and dw < 5e-2
    print(f'{name}: loss相对差={dl:.2e} hidden梯度相对差={dh:.2e} '
          f'lm_head梯度相对差={dw:.2e} 峰值显存={peak:.2f}G(参照 {peak_ref:.2f}G) '
          f'-> {"PASS" if ok else "FAIL"}')
print('\n两项 PASS 即可放心接入; B 更简洁(任意权重), C 只用最久经考验的 sum 路径。')
