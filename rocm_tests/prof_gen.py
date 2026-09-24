"""
Kernel-time breakdown of steady-state generation (torch.profiler / roctracer).

usage: python prof_gen.py -m <model_dir> [--mtp | -dm <draft_dir>] [--steps 40]
"""
import argparse, time
import torch
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True)
ap.add_argument("-dm", "--draft", default = None)
ap.add_argument("--mtp", action = "store_true")
ap.add_argument("--steps", type = int, default = 40)
ap.add_argument("--rows", type = int, default = 40)
args = ap.parse_args()

config = Config.from_directory(args.model)
model = Model.from_config(config)
draft_model = None
if args.mtp:
    draft_model = Model.from_config(config, component = "mtp")
elif args.draft:
    draft_model = Model.from_config(Config.from_directory(args.draft))
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

def start():
    job = Job(input_ids = ids, max_new_tokens = 2000, sampler = ComboSampler(temperature = 0.0), stop_conditions = [])
    generator.enqueue(job)
    # prefill + warm graphs
    for _ in range(12): generator.iterate()
    return job

job = start()
torch.cuda.synchronize()
t0 = time.perf_counter()
n0 = job.new_tokens if hasattr(job, "new_tokens") else None
with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(args.steps):
        generator.iterate()
    torch.cuda.synchronize()
dt = time.perf_counter() - t0
generator.clear_queue() if hasattr(generator, "clear_queue") else None
print(f"{args.steps} iterations in {dt * 1000:.1f} ms ({dt / args.steps * 1000:.2f} ms/iter, profiled)")
print(prof.key_averages().table(sort_by = "self_device_time_total", row_limit = args.rows, max_name_column_width = 90))
