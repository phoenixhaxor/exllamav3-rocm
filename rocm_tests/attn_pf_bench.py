"""Prefill attention microbenchmark: one 2048-query chunk at the end of a <ctx> context, fp16 paged cache,
Qwen3.8 geometry (24 q heads, 4 kv heads, head_dim 256). usage: attn_pf_bench.py <ctx> [block_m block_n warps stages]"""
import sys, time, torch
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
ctx = int(sys.argv[1]); q_len = 2048
cfg = [int(x) for x in sys.argv[2:6]] if len(sys.argv) > 5 else [None] * 4
P = 256; nq, nkv, hd = 24, 4, 256
pages = (ctx + P - 1) // P
kc = torch.randn(pages, P, nkv, hd, dtype = torch.half, device = "cuda") * 0.5
vc = torch.randn(pages, P, nkv, hd, dtype = torch.half, device = "cuda") * 0.5
bt = torch.arange(pages, dtype = torch.int32, device = "cuda").view(1, pages)
seqlens = torch.tensor([ctx - q_len], dtype = torch.int32, device = "cuda")
q = torch.randn(1, q_len, nq, hd, dtype = torch.half, device = "cuda")
def run():
    return paged_attn_triton_prefill(q, None, None, kc, vc, bt, seqlens, causal = True, pre_appended_len = q_len,
                                     block_m = cfg[0], block_n = cfg[1], num_warps = cfg[2], num_stages = cfg[3])
o = run(); torch.cuda.synchronize()
t0 = time.perf_counter(); n = 5
for _ in range(n): o = run()
torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / n
flop = 4 * q_len * nq * hd * (ctx - q_len / 2)
# reference on a subset of heads (torch SDPA, causal with offset)
import torch.nn.functional as F
past = ctx - q_len
kk = kc.view(-1, nkv, hd)[:ctx].float(); vv = vc.view(-1, nkv, hd)[:ctx].float()
err = 0.0
for h in (0, 7, 23):
    kh = kk[:, h // (nq // nkv)]; vh = vv[:, h // (nq // nkv)]
    qq = q[0, :, h].float()
    sc = (qq @ kh.T) / hd ** 0.5
    mask = torch.arange(ctx, device = "cuda")[None, :] > (past + torch.arange(q_len, device = "cuda"))[:, None]
    sc.masked_fill_(mask, float("-inf"))
    ref = torch.softmax(sc, -1) @ vh
    err = max(err, ((o[0, :, h].float() - ref).norm() / ref.norm()).item())
print(f"ctx {ctx} cfg {cfg}: {dt * 1000:.1f} ms  {flop / dt / 1e12:.1f} TFLOPS  rel err {err:.2e}")
