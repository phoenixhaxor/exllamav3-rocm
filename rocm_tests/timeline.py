"""Per-round GPU timeline around the large idle gaps (host turnaround), from a torch.profiler trace.
usage: timeline.py -m <model> -dm <draft> [--steps 30] [--gap 20] [--temp 0.6]"""
import argparse, json, time, torch, tempfile, collections
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler
ap = argparse.ArgumentParser(); ap.add_argument("-m", required = True); ap.add_argument("-dm")
ap.add_argument("--steps", type = int, default = 30); ap.add_argument("--gap", type = float, default = 20)
ap.add_argument("--temp", type = float, default = 0.6); ap.add_argument("--rounds", type = int, default = 3); ap.add_argument("--stack", action = "store_true"); ap.add_argument("--ops", action = "store_true"); ap.add_argument("--big", action = "store_true")
args = ap.parse_args()
config = Config.from_directory(args.m); model = Model.from_config(config)
dm = Model.from_config(Config.from_directory(args.dm)) if args.dm else None
mh = dm.caps.get("default_draft_size", 4) if dm else 0
cache = Cache(model, max_num_tokens = 16384, max_history = mh, max_batch_size = 1); model.load(progressbar = False)
dc = None
if dm: dc = Cache(dm, max_num_tokens = 16384); dm.load(progressbar = False)
tok = Tokenizer.from_config(config)
gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dm, draft_cache = dc)
ids = tok.encode("<|im_start|>user\nWrite a long story about a lighthouse keeper.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", encode_special_tokens = True)
gen.enqueue(Job(input_ids = ids, max_new_tokens = 3000, sampler = ComboSampler(temperature = args.temp), stop_conditions = []))
for _ in range(20): gen.iterate()
torch.cuda.synchronize()
with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack = args.stack) as prof:
    marks = []
    for _ in range(args.steps):
        with torch.profiler.record_function("ROUND"):
            gen.iterate()
    torch.cuda.synchronize()
path = tempfile.mktemp(suffix = ".json"); prof.export_chrome_trace(path)
ev = json.load(open(path))["traceEvents"]
kn = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e)
rounds = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("name") == "ROUND" and e.get("cat") == "user_annotation")
cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("python_function", "cpu_op", "cuda_runtime") and "dur" in e]
def short(n): return n.split("(")[0].replace("void ", "")[:48]
tot = collections.Counter(); totn = collections.Counter()
for r, (rs, re_) in enumerate(rounds):
    ks = [k for k in kn if rs <= k[0] < re_ + 3000]
    if not ks: continue
    big = any(ks[i][0] - ks[i - 1][1] > 2000 for i in range(1, len(ks)))
    show = r >= len(rounds) - args.rounds or (big and args.big)
    if show: print(f"--- round {r}: cpu {(re_ - rs) / 1000:.2f} ms, {len(ks)} gpu ops")
    for i in range(1, len(ks)):
        g = ks[i][0] - ks[i - 1][1]
        if g >= args.gap:
            key = (short(ks[i - 1][2]), short(ks[i][2]))
            tot[key] += g; totn[key] += 1
            if show:
                # innermost python frames running at the gap midpoint
                mid = ks[i - 1][1] + g / 2
                act = sorted((e["dur"], e["name"][:80]) for e in cpu if e["ts"] <= mid <= e["ts"] + e["dur"])
                inner = [n for _, n in act[:4]]
                print(f"  {g:7.0f} us  #{i:4d}  {key[0]:48s} -> {key[1]:48s}")
                for n_ in inner: print(f"            cpu: {n_}")
    if show and args.ops:
        t0 = ks[0][0]
        for i, (s0, e0, nm) in enumerate(ks):
            if i < 130 or i > len(ks) - 60:
                print(f"    op #{i:4d} t={(s0 - t0) / 1000:8.3f} ms dur {e0 - s0:6.1f} us  {short(nm)}")
n = len(rounds)
print(f"=== gaps >= {args.gap} us, per round (avg over {n}):")
for k, v in tot.most_common(15):
    print(f"  {v / n:7.0f} us  x{totn[k] / n:.1f}  {k[0]} -> {k[1]}")
