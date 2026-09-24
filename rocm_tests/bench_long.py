"""Decode speed after a long prompt (code-like filler), optional draft. usage: bench_long.py <ctx_tokens,...> [-dm draft | --mtp]"""
import os
import argparse, time, random, torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler
from exllamav3.cache import CacheLayer_quant

ap = argparse.ArgumentParser()
ap.add_argument("ctx"); ap.add_argument("-m", default = os.environ.get("EXL3_MODEL_DIR", "models/Qwen3.8-27B-EXL3-3.5bpw"))
ap.add_argument("-dm", default = None); ap.add_argument("--mtp", action = "store_true")
ap.add_argument("--cache", type = int, default = 131072); ap.add_argument("--dkv", type = int, default = 4)
ap.add_argument("--tokens", type = int, default = 256); ap.add_argument("--show", type = int, default = 0)
args = ap.parse_args()
config = Config.from_directory(args.m); model = Model.from_config(config)
dm = Model.from_config(config, component = "mtp") if args.mtp else (Model.from_config(Config.from_directory(args.dm)) if args.dm else None)
mh = dm.caps.get("default_draft_size", 4) if dm else 0
cache = Cache(model, max_num_tokens = args.cache, max_history = mh, max_batch_size = 1, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
model.load(progressbar = False)
dc = None
if dm:
    dc = Cache(dm, max_num_tokens = args.cache, layer_type = CacheLayer_quant, k_bits = args.dkv, v_bits = args.dkv); dm.load(progressbar = False)
tok = Tokenizer.from_config(config)
gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dm, draft_cache = dc)
random.seed(0)
names = ["user", "order", "cache", "config", "session", "item", "index", "buffer", "token", "event"]
def filler(n_chars):
    out = []; i = 0
    while sum(map(len, out)) < n_chars:
        a, b = random.choice(names), random.choice(names)
        out.append(f"def {a}_{b}_{i}(x, y):\n    '''Update {a} with {b}.'''\n    if x > {i}:\n        return {a}(x) + {b}(y)\n    return None\n\n"); i += 1
    return "".join(out)
for ctx in [int(c) for c in args.ctx.split(",")]:
    body = filler(ctx * 3)
    ids = tok.encode(body)[:, :ctx - 60]
    tail = tok.encode("\n# Now write a new function merge_sessions(a, b) that combines two session dicts, with docstring and tests:\n")
    prompt = torch.cat([tok.encode("<|im_start|>user\n", encode_special_tokens = True), ids, tail,
                        tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", encode_special_tokens = True)], dim = -1)
    job = Job(input_ids = prompt, max_new_tokens = args.tokens, sampler = ComboSampler(temperature = 0.0), stop_conditions = [])
    gen.enqueue(job); t0 = time.perf_counter(); first = None; last = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("text") and first is None: first = time.perf_counter()
            if r.get("eos"): last = r
    t1 = time.perf_counter(); n = last["new_tokens"]
    if args.show: print("TEXT:", repr(last.get("full_completion", ""))[:args.show])
    print(f"ctx {prompt.shape[-1]}: prefill {(first - t0):.1f} s ({prompt.shape[-1] / (first - t0):.0f} tok/s), decode {(n - 1) / (t1 - first):.1f} tok/s, "
          f"draft {last.get('accepted_draft_tokens', 0)}/{last.get('accepted_draft_tokens', 0) + last.get('rejected_draft_tokens', 0)}")
