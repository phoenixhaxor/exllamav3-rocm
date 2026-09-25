"""In-process A/B of decode-loop options at fixed (greedy) acceptance: ms per speculative round and tok/s.
Variants toggle module flags; each variant runs the same prompts, interleaved over several passes.
usage: bench_ab.py -m <model> -dm <draft> [--tokens 400] [--passes 3] [--variants base,all,...]"""
import argparse, time, torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler
import exllamav3.generator.generator as G
import exllamav3.modules.transformer as T
import exllamav3.modules.gated_delta_net as GDN
ap = argparse.ArgumentParser(); ap.add_argument("-m", required = True); ap.add_argument("-dm", required = True)
ap.add_argument("--tokens", type = int, default = 400); ap.add_argument("--passes", type = int, default = 3)
ap.add_argument("--variants", default = "base,abl,abl+ps,all"); ap.add_argument("--temp", type = float, default = 0.0)
args = ap.parse_args()
FLAGS = {
    "base":   dict(abl = False, ps = False, nb = False, rc = False, mk = False),
    "abl":    dict(abl = True,  ps = False, nb = False, rc = False, mk = False),
    "abl+ps": dict(abl = True,  ps = True,  nb = False, rc = False, mk = False),
    "host":   dict(abl = True,  ps = True,  nb = True,  rc = True,  mk = False),
    "all":    dict(abl = True,  ps = True,  nb = True,  rc = True,  mk = True),
    "nomk":   dict(abl = True,  ps = True,  nb = True,  rc = True,  mk = False),
}
from exllamav3.ext import exllamav3_ext as ext
def setf(f):
    T._abl_fuse = f["abl"]; G._presample_enable = f["ps"]; G._draft_nb_enable = f["nb"]; GDN._rewind_cache_enable = f["rc"]
    if hasattr(ext, "gdn_mk_set"): ext.gdn_mk_set(1 if f["mk"] else 0)
config = Config.from_directory(args.m); model = Model.from_config(config)
dm = Model.from_config(Config.from_directory(args.dm))
cache = Cache(model, max_num_tokens = 16384, max_history = 8, max_batch_size = 1); model.load(progressbar = False)
dc = Cache(dm, max_num_tokens = 16384); dm.load(progressbar = False)
tok = Tokenizer.from_config(config)
prompts = [
    "Write a complete Python module implementing an LRU cache with TTL expiry, thread safety and pytest tests.",
    "Write a vivid short story about a lighthouse keeper who discovers the lamp is signaling something at sea.",
    "Explain step by step how a CPU pipeline handles branch misprediction, with concrete examples.",
]
ids = [tok.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", encode_special_tokens = True) for p in prompts]
gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dm, draft_cache = dc)
def run_one(i):
    job = Job(input_ids = ids[i], max_new_tokens = args.tokens, sampler = ComboSampler(temperature = args.temp), stop_conditions = [], seed = 1)
    gen.enqueue(job)
    rounds = 0; t0 = None; ntok = 0
    while gen.num_remaining_jobs():
        r = gen.iterate()
        for x in r:
            if x.get("stage") == "streaming":
                if t0 is None and x.get("token_ids") is not None:
                    torch.cuda.synchronize(); t0 = time.perf_counter(); rounds = 0; ntok = 0; continue
                if t0 is not None:
                    ntok += x["token_ids"].shape[-1] if x.get("token_ids") is not None else 0
        if t0 is not None: rounds += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t0, rounds, ntok
names = args.variants.split(",")
import statistics
per = {n: [] for n in names}
tot = {n: [0.0, 0, 0] for n in names}
setf(FLAGS[names[0]]); run_one(0)   # warmup
for p in range(args.passes):
    for i in range(len(ids)):
        order = names if (p + i) % 2 == 0 else names[::-1]
        for n in order:
            setf(FLAGS[n])
            dt, r, nt = run_one(i)
            per[n].append(1000 * dt / r)
            tot[n][0] += dt; tot[n][1] += r; tot[n][2] += nt
for n in names:
    dt, r, nt = tot[n]
    print(f"{n:8s}: median {statistics.median(per[n]):6.2f} ms/round, mean {1000 * dt / r:6.2f}  {nt / dt:7.2f} tok/s  ({r} rounds, {nt} tok)")
