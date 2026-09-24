"""
Correctness and timing of the fused multi-matrix EXL3 matmul on ROCm (exl3_mgemm, RDNA3 path)
against separate ext.exl3_gemm calls: MultiLinear (MLP gate + up) and SlicedMultiLinear
(GDN qkv + z, attention q / k / v)

usage: python test_rdna3_mgemm.py <model_dir>
"""
import sys, json, os, time
from types import SimpleNamespace
import torch
from safetensors import safe_open
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.modules.multilinear import MultiLinear, SlicedMultiLinear

torch.manual_seed(0)
model_dir = sys.argv[1]
index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
pre = "model.language_model.layers"

def load(prefix):
    t = {}
    for sub in ("trellis", "suh", "svh", "su", "sv", "mul1", "mcg", "bias"):
        key = f"{prefix}.{sub}"
        if key in index:
            with safe_open(os.path.join(model_dir, index[key]), "pt", device = "cuda:0") as f:
                t[sub] = f.get_tensor(key)
    kt, nt, _ = t["trellis"].shape
    lin = LinearEXL3(
        None, kt * 16, nt * 16,
        su = t.get("su"), sv = t.get("sv"), suh = t.get("suh"), svh = t.get("svh"),
        trellis = t["trellis"], mcg = t.get("mcg"), mul1 = t.get("mul1"), bias = t.get("bias"),
    )
    return SimpleNamespace(quant_type = "exl3", inner = lin, in_features = kt * 16, out_features = nt * 16,
                           softcap = 0.0, post_scale = 1.0)

def separate(x, lins, c_dtype):
    outs = []
    for l in lins:
        y = torch.empty((x.shape[0], l.out_features), dtype = c_dtype, device = x.device)
        ext.exl3_gemm(x, l.inner.trellis, y, l.inner.suh, torch.empty_like(x), l.inner.svh, -1, l.inner.mcg, l.inner.mul1, 0)
        outs.append(y)
    return outs

def fused_multi(x, ml, c_dtype):
    m = x.shape[0]
    y = torch.empty((ml.num_linears, m, ml.out_features), dtype = c_dtype, device = x.device)
    xh = torch.empty((ml.num_linears, m, ml.in_features), dtype = torch.half, device = x.device)
    ext.exl3_mgemm(x.unsqueeze(0), ml.ptrs_trellis, y, ml.ptrs_suh, xh, ml.ptrs_svh, None, None,
                   ml.K, -1, ml.mcg, ml.mul1, -1, -1, 0, 1, None, None)
    return list(y.unbind(0))

def fused_sliced(x, sl, c_dtype, outs = None, c_ptrs = None):
    m = x.shape[0]
    if outs is None:
        outs = [torch.empty((m, l.out_features), dtype = c_dtype, device = x.device) for l in sl.linears]
    if c_ptrs is None:
        c_ptrs = sl.c_ptrs(outs)
    carrier = torch.empty((1, 1, sl.width), dtype = c_dtype, device = x.device).expand(sl.num_slices, m, sl.width)
    xh = torch.empty((sl.num_src, m, sl.in_features), dtype = torch.half, device = x.device)
    ext.exl3_mgemm(x.unsqueeze(0), sl.ptrs_trellis, carrier, sl.ptrs_suh, xh, sl.ptrs_svh, None, None,
                   sl.K, -1, sl.mcg, sl.mul1, -1, -1, 0, 1, sl.size_n_list, c_ptrs,
                   sl.n_stride_list, sl.had_src_list, sl.num_src)
    return outs

def timeit(fn, n_it = 100):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_it): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_it * 1e6

cases = [
    ("mlp gate+up", "multi", [f"{pre}.0.mlp.gate_proj", f"{pre}.0.mlp.up_proj"]),
    ("gdn qkv+z", "sliced", [f"{pre}.0.linear_attn.in_proj_qkv", f"{pre}.0.linear_attn.in_proj_z"]),
    ("attn q/k/v", "sliced", [f"{pre}.3.self_attn.q_proj", f"{pre}.3.self_attn.k_proj", f"{pre}.3.self_attn.v_proj"]),
]

fails = 0
for name, mode, prefixes in cases:
    lins = [load(p) for p in prefixes]
    b = MultiLinear("cuda:0", lins) if mode == "multi" else SlicedMultiLinear("cuda:0", lins)
    desc = f"{b.num_linears} matrices" if mode == "multi" else f"{b.num_slices} slices of {b.width}"
    print(f"{name}: k={lins[0].in_features} n={[l.out_features for l in lins]} K={lins[0].inner.K} ({desc})")
    run = fused_multi if mode == "multi" else fused_sliced
    for m in (1, 2, 3, 5, 8, 16, 32):
        x = torch.randn((m, lins[0].in_features), dtype = torch.half, device = "cuda:0")
        for c_dtype in (torch.half, torch.float):
            ref = separate(x, lins, c_dtype)
            got = run(x, b, c_dtype)
            torch.cuda.synchronize()
            err = max(((g.float() - r.float()).norm() / r.float().norm()).item() for g, r in zip(got, ref))
            ok = err < 1e-3 and all(torch.isfinite(g).all().item() for g in got)
            fails += not ok
            if not ok or m in (1, 8):
                print(f"  m={m:2d} {str(c_dtype)[6:]:7s} max rel diff vs separate {err:.2e} {'OK' if ok else 'FAIL'}")
    for m in (1, 5, 8):
        x = torch.randn((m, lins[0].in_features), dtype = torch.half, device = "cuda:0")
        t_sep = timeit(lambda: separate(x, lins, torch.half))
        if mode == "multi":
            t_fus = timeit(lambda: run(x, b, torch.half))
        else:
            outs = [torch.empty((m, l.out_features), dtype = torch.half, device = x.device) for l in lins]
            cp = b.c_ptrs(outs)
            t_fus = timeit(lambda: run(x, b, torch.half, outs, cp))
        print(f"  timing m={m}: separate {t_sep:7.1f} us, fused {t_fus:7.1f} us")

# Fused prologues vs. the unfused kernels: silu(gate) * up -> down, gated RMSNorm -> o_proj
down = load(f"{pre}.0.mlp.down_proj")
oproj = load(f"{pre}.0.linear_attn.out_proj")
for m in (1, 5, 8, 16):
    kd = down.in_features
    g = torch.randn((m, kd), dtype = torch.half, device = "cuda:0")
    u = torch.randn((m, kd), dtype = torch.half, device = "cuda:0")
    for c_dtype in (torch.half, torch.float):
        a = torch.empty_like(g)
        ext.silu_mul(g, u, a, 0.0)
        ref = torch.empty((m, down.out_features), dtype = c_dtype, device = "cuda:0")
        ext.exl3_gemm(a, down.inner.trellis, ref, down.inner.suh, torch.empty_like(a), down.inner.svh, -1, down.inner.mcg, down.inner.mul1, 0)
        got = torch.empty_like(ref)
        ok = ext.exl3_gemm_silu(g, u, down.inner.trellis, got, down.inner.suh, down.inner.svh, down.inner.mcg, down.inner.mul1)
        torch.cuda.synchronize()
        err = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        good = ok and err < 1e-6
        fails += not good
        print(f"  silu+down m={m:2d} {str(c_dtype)[6:]:7s} launched={ok} rel diff {err:.2e} {'OK' if good else 'FAIL'}")
    ko = oproj.in_features
    heads = ko // 128
    x = torch.randn((m, heads, 128), dtype = torch.bfloat16, device = "cuda:0")
    for w_dtype, z_dtype, act in ((torch.float, torch.float, 0), (torch.bfloat16, torch.bfloat16, 0), (torch.float, torch.float, 1)):
        w = (torch.rand(128, device = "cuda:0") + 0.5).to(w_dtype)
        z = torch.randn((m, heads, 128), dtype = z_dtype, device = "cuda:0")
        y = torch.empty((m, heads, 128), dtype = torch.half, device = "cuda:0")
        ext.gated_rms_norm(x, w, y, z, 1e-6, 0.0, 1, False, act)
        ref = torch.empty((m, oproj.out_features), dtype = torch.half, device = "cuda:0")
        y2 = y.view(m, ko)
        ext.exl3_gemm(y2, oproj.inner.trellis, ref, oproj.inner.suh, torch.empty_like(y2), oproj.inner.svh, -1, oproj.inner.mcg, oproj.inner.mul1, 0)
        got = torch.empty_like(ref)
        ok = ext.exl3_gemm_gnorm(x, z, w, 1e-6, 0.0, act == 1, oproj.inner.trellis, got, oproj.inner.suh, oproj.inner.svh, oproj.inner.mcg, oproj.inner.mul1)
        torch.cuda.synchronize()
        err = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        good = ok and err < 1e-3
        fails += not good
        print(f"  gnorm+o_proj m={m:2d} w={str(w_dtype)[6:]} z={str(z_dtype)[6:]} act={act} launched={ok} rel diff {err:.2e} {'OK' if good else 'FAIL'}")

aoproj = load(f"{pre}.3.self_attn.o_proj")
for m in (1, 5, 8, 16):
    ka = aoproj.in_features
    o = torch.randn((m, ka), dtype = torch.half, device = "cuda:0")
    gt = torch.randn((m, ka), dtype = torch.half, device = "cuda:0")
    for c_dtype in (torch.half, torch.float):
        og = o.clone()
        ext.mul_sigmoid_(og, gt)
        ref = torch.empty((m, aoproj.out_features), dtype = c_dtype, device = "cuda:0")
        ext.exl3_gemm(og, aoproj.inner.trellis, ref, aoproj.inner.suh, torch.empty_like(og), aoproj.inner.svh, -1, aoproj.inner.mcg, aoproj.inner.mul1, 0)
        got = torch.empty_like(ref)
        ok = ext.exl3_gemm_sigmoid_gate(o, gt, aoproj.inner.trellis, got, aoproj.inner.suh, aoproj.inner.svh, aoproj.inner.mcg, aoproj.inner.mul1)
        torch.cuda.synchronize()
        err = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        good = ok and err < 1e-6
        fails += not good
        print(f"  gate+o_proj m={m:2d} {str(c_dtype)[6:]:7s} launched={ok} rel diff {err:.2e} {'OK' if good else 'FAIL'}")

print("FAILS:", fails)
sys.exit(1 if fails else 0)
