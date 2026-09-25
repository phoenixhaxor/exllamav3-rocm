"""Logit differences between ablation implementations (torch reference, standalone kernel, fused into
the residual norm) to tell a bug from summation-order noise. usage: test_abl_noise.py -m <heretic model>"""
import argparse, torch
from exllamav3 import Config, Model, Tokenizer
import exllamav3.modules.transformer as T
ap = argparse.ArgumentParser(); ap.add_argument("-m", required = True); args = ap.parse_args()
config = Config.from_directory(args.m); model = Model.from_config(config); model.load(progressbar = False)
tok = Tokenizer.from_config(config)
text = open(__file__).read() * 2
ids = tok.encode(text)[:, :512]
def run(kernel, fuse):
    T._abl_kernel = kernel; T._abl_fuse = fuse
    return model.forward(ids, {"attn_mode": "flash_attn_nc"}).float()
ref = run(False, False)          # torch matmul reduction
ker = run(True, False)           # standalone kernel
fus = run(True, True)            # fused into rms_norm
ref2 = run(False, False)
def cmp(a, b, name):
    d = (a - b).abs()
    lp = torch.log_softmax(a, -1); lq = torch.log_softmax(b, -1)
    kl = (lp.exp() * (lp - lq)).sum(-1).mean().item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"{name:22s} max|d| {d.max().item():.4f}  mean|d| {d.mean().item():.5f}  KL {kl:.2e}  top1 agree {agree * 100:.2f}%")
cmp(ref, ref2, "torch vs torch (rerun)")
cmp(ref, ker, "torch vs kernel")
cmp(ref, fus, "torch vs fused")
cmp(ker, fus, "kernel vs fused")
T._abl_kernel = True; T._abl_fuse = True
import os; os.environ["EXL3_ABLATION"] = "0"
