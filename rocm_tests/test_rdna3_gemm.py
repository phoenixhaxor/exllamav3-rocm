"""
Correctness check for the EXL3 matmul on ROCm against two references:

1. an independent numpy decode of the trellis (integer bitrates, mul1 codebook) for a few tiles,
   checked against ext.reconstruct
2. x @ W with W = LinearEXL3.get_weight_tensor() (reconstruct + torch Hadamards), checked against
   ext.exl3_gemm for a sweep of m

usage: python test_rdna3_gemm.py <model_dir> [tensor_prefix ...]
"""
import sys, json, os, time
import numpy as np
import torch
from safetensors import safe_open
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3

torch.manual_seed(0)
model_dir = sys.argv[1]
prefixes = sys.argv[2:] or [
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.3.self_attn.o_proj",
    "lm_head",
]

index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]

def load(prefix):
    t = {}
    for sub in ("trellis", "suh", "svh", "su", "sv", "mul1", "mcg", "bias"):
        key = f"{prefix}.{sub}"
        if key in index:
            with safe_open(os.path.join(model_dir, index[key]), "pt", device = "cuda:0") as f:
                t[sub] = f.get_tensor(key)
    return t


def ref_decode_tile(tile_u16: np.ndarray, K: int) -> np.ndarray:
    """Decode one 16x16 tile (mul1 codebook, integer K) to a (k, n) fp16 array"""
    words = tile_u16.view(np.uint32)
    W = K * 256 // 32
    out = np.zeros((16, 16), dtype = np.float16)
    k_inv = np.frombuffer(np.uint16(0x1eee).tobytes(), dtype = np.float16)[0]
    k_bias = np.frombuffer(np.uint16(0xc931).tobytes(), dtype = np.float16)[0]
    for p in range(256):
        b0 = p * K + K - 16 + 256 * K
        b1 = b0 + 16
        i0 = b0 // 32
        i1 = (b1 - 1) // 32
        s0 = (i1 + 1) * 32 - b1
        a = int(words[i0 % W]); b = int(words[i1 % W])
        w = ((((a << 32) | b) >> (s0 & 31)) & 0xffffffff) & 0xffff
        x = (w * 0x83DCD12D) & 0xffffffff
        s = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + ((x >> 24) & 0xff) + 0x6400
        h = np.frombuffer(np.uint16(s & 0xffff).tobytes(), dtype = np.float16)[0]
        v = np.float16(np.float32(h) * np.float32(k_inv) + np.float32(k_bias))  # hfma, one rounding
        lane, j = p // 8, p % 8
        n = (lane >> 2) + 8 * (j >> 2)
        k = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1)
        out[k, n] = v
    return out


fails = 0
for prefix in prefixes:
    t = load(prefix)
    trellis = t["trellis"]
    kt, nt, tw = trellis.shape
    in_f, out_f = kt * 16, nt * 16
    lin = LinearEXL3(
        None, in_f, out_f,
        su = t.get("su"), sv = t.get("sv"), suh = t.get("suh"), svh = t.get("svh"),
        trellis = trellis, mcg = t.get("mcg"), mul1 = t.get("mul1"), bias = t.get("bias"),
    )
    K = lin.K
    print(f"{prefix}: k={in_f} n={out_f} K={K} mul1={lin.mul1} mcg={lin.mcg}")

    # 1. reconstruct vs numpy decode, a few tiles
    if lin.mul1 and isinstance(K, int):
        w_inner = lin.get_inner_weight_tensor().cpu().numpy()
        tr = trellis.cpu().numpy()
        worst = 0.0
        for (ti, tj) in [(0, 0), (1, 3), (kt - 1, nt - 1), (kt // 2, nt // 3)]:
            ref = ref_decode_tile(tr[ti, tj].copy(), K)
            got = w_inner[ti * 16:(ti + 1) * 16, tj * 16:(tj + 1) * 16]
            worst = max(worst, float(np.abs(ref.astype(np.float32) - got.astype(np.float32)).max()))
        ok = worst < 1e-3
        fails += not ok
        print(f"  reconstruct vs numpy decode: max abs diff {worst:.2e} {'OK' if ok else 'FAIL'}")

    # 2. gemm vs x @ W
    W = lin.get_weight_tensor().float()
    for m in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 32, 64, 144]:
        x = torch.randn((m, in_f), dtype = torch.half, device = "cuda:0")
        ref = x.float() @ W
        for c_fp32 in (False, True):
            y = torch.empty((m, out_f), dtype = torch.float if c_fp32 else torch.half, device = "cuda:0")
            xh = torch.empty_like(x)
            ext.exl3_gemm(x, trellis, y, lin.suh, xh, lin.svh, -1, lin.mcg, lin.mul1, 0)
            torch.cuda.synchronize()
            err = (y.float() - ref).norm() / ref.norm()
            ok = err.item() < 2e-3 and torch.isfinite(y).all().item()
            fails += not ok
            if not ok or m in (1, 16, 144):
                print(f"  m={m:3d} fp32={int(c_fp32)} rel err {err.item():.2e} {'OK' if ok else 'FAIL'}")

    # 3. timing, m = 1 / 4 / 16
    for m in (1, 4, 8, 16):
        x = torch.randn((m, in_f), dtype = torch.half, device = "cuda:0")
        y = torch.empty((m, out_f), dtype = torch.half, device = "cuda:0")
        xh = torch.empty_like(x)
        for _ in range(3): ext.exl3_gemm(x, trellis, y, lin.suh, xh, lin.svh, -1, lin.mcg, lin.mul1, 0)
        torch.cuda.synchronize()
        n_it = 50
        t0 = time.perf_counter()
        for _ in range(n_it): ext.exl3_gemm(x, trellis, y, lin.suh, xh, lin.svh, -1, lin.mcg, lin.mul1, 0)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_it
        gbs = trellis.numel() * 2 / dt / 1e9
        print(f"  timing m={m:2d}: {dt * 1e6:8.1f} us  {gbs:6.1f} GB/s trellis")

print("FAILS:", fails)
sys.exit(1 if fails else 0)
