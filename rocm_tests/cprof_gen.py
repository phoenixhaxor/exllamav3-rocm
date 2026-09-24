"""CPU-side profile of steady-state generation (cProfile), plus wall vs. GPU-busy estimate."""
import argparse, time, cProfile, pstats, io
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True)
ap.add_argument("-dm", "--draft", default = None)
ap.add_argument("--mtp", action = "store_true")
ap.add_argument("--steps", type = int, default = 60)
args = ap.parse_args()

config = Config.from_directory(args.model)
model = Model.from_config(config)
draft_model = None
if args.mtp: draft_model = Model.from_config(config, component = "mtp")
elif args.draft: draft_model = Model.from_config(Config.from_directory(args.draft))
max_history = draft_model.caps.get("default_draft_size", 4) if draft_model else 0
cache = Cache(model, max_num_tokens = 16384, max_history = max_history, max_batch_size = 1)
model.load(progressbar = False)
draft_cache = None
if draft_model:
    draft_cache = Cache(draft_model, max_num_tokens = 16384)
    draft_model.load(progressbar = False)
tokenizer = Tokenizer.from_config(config)
generator = Generator(model = model, cache = cache, tokenizer = tokenizer, draft_model = draft_model, draft_cache = draft_cache)
prompt = "<|im_start|>user\nWrite a complete Python module implementing a thread-safe LRU cache with TTL.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tokenizer.encode(prompt, encode_special_tokens = True)
job = Job(input_ids = ids, max_new_tokens = 4000, sampler = ComboSampler(temperature = 0.0), stop_conditions = [])
generator.enqueue(job)
for _ in range(15): generator.iterate()
torch.cuda.synchronize()
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
ntok = 0
for _ in range(args.steps):
    for r in generator.iterate():
        ntok += len(r.get("token_ids", [])) if isinstance(r.get("token_ids"), list) else r.get("token_ids").numel() if r.get("token_ids") is not None else 0
pr.disable()
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"{args.steps} iterations, {dt / args.steps * 1000:.2f} ms/iter (cProfile on), tokens {ntok}")
s = io.StringIO()
pstats.Stats(pr, stream = s).sort_stats("tottime").print_stats(30)
print(s.getvalue()[:6000])
