"""Fused ablation (in the residual RMSNorm) vs standalone ablate kernel: logits and greedy decode.
usage: test_abl_fuse.py -m <heretic model dir> [-dm <draft>]"""
import argparse, torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler
import exllamav3.modules.transformer as T
ap = argparse.ArgumentParser(); ap.add_argument("-m", required = True); ap.add_argument("-dm")
ap.add_argument("--tokens", type = int, default = 300)
args = ap.parse_args()
config = Config.from_directory(args.m); model = Model.from_config(config)
dm = Model.from_config(Config.from_directory(args.dm)) if args.dm else None
cache = Cache(model, max_num_tokens = 8192, max_history = 8 if dm else 0, max_batch_size = 1); model.load(progressbar = False)
dc = None
if dm: dc = Cache(dm, max_num_tokens = 8192); dm.load(progressbar = False)
tok = Tokenizer.from_config(config)
nabl = sum(1 for m in model.modules if getattr(m, "abl_attn", None) is not None or getattr(m, "abl_mlp", None) is not None)
print("blocks with ablation:", nabl)

prompt = "<|im_start|>user\nExplain how a hash map handles collisions, with a short Python example.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tok.encode(prompt, encode_special_tokens = True)

def logits(fuse, n):
    T._abl_fuse = fuse
    x = ids[:, :n]
    return model.forward(x, {"attn_mode": "flash_attn_nc"}).float()

for n in (8, ids.shape[-1]):
    a = logits(False, n); b = logits(True, n)
    d = (a - b).abs().max().item()
    print(f"forward {n} tokens: max |dlogit| = {d:.4g}, argmax equal: {bool((a.argmax(-1) == b.argmax(-1)).all())}")

def greedy(fuse):
    T._abl_fuse = fuse
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dm, draft_cache = dc)
    return gen.generate(prompt = prompt, max_new_tokens = args.tokens, sampler = ComboSampler(temperature = 0.0),
                        encode_special_tokens = True, add_bos = False, completion_only = True)
a = greedy(False); b = greedy(True)
print("greedy decode identical:", a == b, len(a), len(b))
if a != b:
    k = next(i for i in range(min(len(a), len(b))) if a[i] != b[i]) if any(x != y for x, y in zip(a, b)) else min(len(a), len(b))
    print("first diff at char", k, repr(a[max(0, k - 60):k + 40]), "|", repr(b[max(0, k - 60):k + 40]))
