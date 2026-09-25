"""
Runtime directional ablation ("abliteration") on top of unmodified quantized weights.

For a sublayer output projection W (out x in, rows = output features) with unit direction v and
weight lam, the abliterated projection (as in Heretic, per row_normalization mode) is

    none:  W' = (I - lam v v^T) W
    pre:   W' = W - lam (n * v) (v^T Wn)            Wn = diag(1/n) W,  n = row norms of W
    full:  W' = diag(n / m) (I - lam v v^T) Wn      m  = row norms of (I - lam v v^T) Wn

In all three cases W' x = y * s - b <y, a> with y = W x, i.e. the ablation is a function of the
projection's output only. It is applied exactly (no low-rank approximation, no requantization) to
the output of the original quantized projection by TransformerBlock (see apply_ablation).

Ablation vectors live in a sidecar file next to the model, keyed by transformer block:
    {block_key}.abl_attn.{a,b,s}   (attention / linear-attention output projection)
    {block_key}.abl_mlp.{a,b,s}    (MLP down projection)
s is omitted when it is all ones. Set EXL3_ABLATION=0 to ignore the sidecar.
"""
from __future__ import annotations
import os
import torch

SIDECAR = "abliteration.safetensors"

_sidecar_cache = {}


def _sidecar_path(config) -> str | None:
    if os.environ.get("EXL3_ABLATION", "1") == "0":
        return None
    directory = getattr(config, "directory", None)
    if not directory:
        return None
    path = os.path.join(directory, SIDECAR)
    return path if os.path.exists(path) else None


def _read_sidecar(path: str) -> dict:
    if path not in _sidecar_cache:
        from safetensors.torch import load_file
        _sidecar_cache[path] = load_file(path, device = "cpu")
    return _sidecar_cache[path]


def load_block_ablation(config, block_key: str, device) -> tuple:
    """(abl_attn, abl_mlp) for a transformer block, each (a, b, s) fp32 on device or None"""
    path = _sidecar_path(config)
    if path is None:
        return None, None
    tensors = _read_sidecar(path)

    def get(sub):
        a = tensors.get(f"{block_key}.{sub}.a")
        if a is None:
            return None
        b = tensors[f"{block_key}.{sub}.b"]
        s = tensors.get(f"{block_key}.{sub}.s")
        f = lambda t: None if t is None else t.to(device = device, dtype = torch.float).contiguous()
        return f(a), f(b), f(s)

    return get("abl_attn"), get("abl_mlp")


def projection_weight(linear) -> torch.Tensor:
    """Effective weight of an (EXL3 or fp16) Linear as fp32 (in_features, out_features), y = x @ w"""
    w = linear.inner.get_weight_tensor()
    w = w[:linear.in_features_unpadded, :linear.out_features_unpadded]
    return w.float()


def weight_stats(linear, directions: torch.Tensor) -> dict:
    """
    Per-projection statistics for a basis of k unit directions (k, out) that make the ablation
    parameters for any unit combination v = directions^T c cheap to compute:
        n: row norms of W (out)
        P: Wn Wn^T directions^T (out, k)
        K: directions Wn Wn^T directions^T (k, k)
    (heretic orientation: Wn = rows of W normalized, rows = output features)
    """
    w = projection_weight(linear)                       # (in, out) = W^T
    d = directions.to(device = w.device, dtype = torch.float)
    n = w.norm(dim = 0)
    wn = w / n.clamp_min(1e-12)                         # Wn^T
    q = wn @ d.T                                        # (in, k): columns are Wn^T d_j
    p = wn.T @ q                                        # (out, k)
    k = q.T @ q                                         # (k, k)
    return {"n": n, "P": p, "K": k}


def compute_ablation(
    stats: dict,
    directions: torch.Tensor,
    coeffs: torch.Tensor,
    lam: float,
    mode: str,
) -> tuple:
    """
    (a, b, s) such that y * s - b <y, a> equals the output of the abliterated projection, for
    unit direction v = directions^T coeffs and ablation weight lam. mode: "none", "pre", "full"
    """
    n = stats["n"]
    dev = n.device
    c = coeffs.to(device = dev, dtype = torch.float)
    v = directions.to(device = dev, dtype = torch.float).T @ c

    if mode == "none":
        return v, lam * v, None

    nc = n.clamp_min(1e-12)
    a = v / nc
    if mode == "pre":
        return a, lam * n * v, None

    assert mode == "full"
    wq = stats["P"].to(dev) @ c                         # Wn (Wn^T v)
    qq = c @ stats["K"].to(dev) @ c                     # |Wn^T v|^2
    m2 = 1.0 - 2.0 * lam * v * wq + (lam * v) ** 2 * qq
    m = m2.clamp_min(1e-24).sqrt()
    s = torch.where(n > 0, 1.0 / m, torch.ones_like(m))
    b = lam * n * v / m
    return a, b, s
