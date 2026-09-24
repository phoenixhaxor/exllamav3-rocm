"""Kernel breakdown of decode at long context. usage: prof_long.py <ctx> [-dm draft]"""
import os
import argparse, torch
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler
from exllamav3.cache import CacheLayer_quant
ap = argparse.ArgumentParser(); ap.add_argument("ctx", type = int); ap.add_argument("-dm", default = None); ap.add_argument("--mtp", action = "store_true"); ap.add_argument("--steps", type = int, default = 20); ap.add_argument("--kv", type = int, default = 8)
args = ap.parse_args()
config = Config.from_directory(os.environ.get("EXL3_MODEL_DIR", "models/Qwen3.8-27B-EXL3-3.5bpw")); model = Model.from_config(config)
dm = Model.from_config(config, component = "mtp") if args.mtp else (Model.from_config(Config.from_directory(args.dm)) if args.dm else None)
mh = dm.caps.get("default_draft_size", 4) if dm else 0
cache = Cache(model, max_num_tokens = args.ctx + 4096, max_history = mh, max_batch_size = 1, **(dict(layer_type = CacheLayer_quant, k_bits = args.kv, v_bits = args.kv) if args.kv else {}))
model.load(progressbar = False); dc = None
if dm: dc = Cache(dm, max_num_tokens = args.ctx + 4096, layer_type = CacheLayer_quant, k_bits = 4, v_bits = 4); dm.load(progressbar = False)
tok = Tokenizer.from_config(config)
gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dm, draft_cache = dc)
ids = torch.randint(1000, 100000, (1, args.ctx))
gen.enqueue(Job(input_ids = ids, max_new_tokens = 2000, sampler = ComboSampler(temperature = 0.0), stop_conditions = []))
while True:
    r = gen.iterate()
    if any(x.get("text") for x in r): break
for _ in range(10): gen.iterate()
torch.cuda.synchronize()
with profile(activities = [ProfilerActivity.CUDA]) as prof:
    for _ in range(args.steps): gen.iterate()
    torch.cuda.synchronize()
for e in sorted(prof.key_averages(), key = lambda e: -e.self_device_time_total)[:14]:
    print(f"{e.self_device_time_total / 1000 / args.steps:8.2f} ms/iter {e.count / args.steps:7.1f}x  {e.key[:110]}")
print(f"total {sum(e.self_device_time_total for e in prof.key_averages()) / 1000 / args.steps:.2f} ms/iter")
