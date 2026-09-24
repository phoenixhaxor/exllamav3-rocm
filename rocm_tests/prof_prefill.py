"""Kernel-time breakdown of prompt prefill (torch.profiler)."""
import sys, time, torch, random
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler
from exllamav3.cache import CacheLayer_quant

model_dir = sys.argv[1]
n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
config = Config.from_directory(model_dir)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = n_tok + 1024, max_batch_size = 1, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8)
model.load(progressbar = False)
tokenizer = Tokenizer.from_config(config)
generator = Generator(model = model, cache = cache, tokenizer = tokenizer)
random.seed(0)
ids = torch.randint(1000, 100000, (1, n_tok))

def run():
    job = Job(input_ids = ids, max_new_tokens = 1, sampler = ComboSampler(temperature = 0.0))
    generator.enqueue(job)
    while generator.num_remaining_jobs(): generator.iterate()

ids_w = torch.randint(1000, 100000, (1, 4096)); job = Job(input_ids = ids_w, max_new_tokens = 1); generator.enqueue(job)
while generator.num_remaining_jobs(): generator.iterate()
torch.cuda.synchronize(); t0 = time.perf_counter(); run(); torch.cuda.synchronize(); dt = time.perf_counter() - t0
print(f"prefill {n_tok} tokens: {dt:.2f} s, {n_tok / dt:.0f} tok/s")
ids = torch.randint(1000, 100000, (1, n_tok))
with profile(activities = [ProfilerActivity.CUDA]) as prof:
    run(); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by = "self_device_time_total", row_limit = 22, max_name_column_width = 80))
