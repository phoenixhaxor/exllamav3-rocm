"""GPU busy vs wall per generator iteration from a torch.profiler trace (kernel intervals merged).
usage: gaps.py -m <model> [--mtp | -dm <draft>] [--steps 30]"""
import argparse, json, time, torch, os, tempfile
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler
ap = argparse.ArgumentParser(); ap.add_argument("-m", required = True); ap.add_argument("-dm"); ap.add_argument("--mtp", action = "store_true"); ap.add_argument("--steps", type = int, default = 30); ap.add_argument("--stack", action = "store_true")
args = ap.parse_args()
config = Config.from_directory(args.m); model = Model.from_config(config)
dm = Model.from_config(config, component = "mtp") if args.mtp else (Model.from_config(Config.from_directory(args.dm)) if args.dm else None)
mh = dm.caps.get("default_draft_size", 4) if dm else 0
cache = Cache(model, max_num_tokens = 16384, max_history = mh, max_batch_size = 1); model.load(progressbar = False)
dc = None
if dm: dc = Cache(dm, max_num_tokens = 16384); dm.load(progressbar = False)
tok = Tokenizer.from_config(config)
gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dm, draft_cache = dc)
ids = tok.encode("<|im_start|>user\nWrite a long story about a lighthouse keeper.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", encode_special_tokens = True)
gen.enqueue(Job(input_ids = ids, max_new_tokens = 3000, sampler = ComboSampler(temperature = 0.0), stop_conditions = []))
for _ in range(20): gen.iterate()
torch.cuda.synchronize()
with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack = args.stack) as prof:
    t0 = time.perf_counter()
    for _ in range(args.steps): gen.iterate()
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
path = tempfile.mktemp(suffix = ".json"); prof.export_chrome_trace(path)
ev = json.load(open(path))["traceEvents"]
k = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e)
busy = 0; cur_s, cur_e = k[0]
for s, e in k[1:]:
    if s > cur_e: busy += cur_e - cur_s; cur_s, cur_e = s, e
    else: cur_e = max(cur_e, e)
busy += cur_e - cur_s
span = k[-1][1] - k[0][0]
print(f"{args.steps} iterations: GPU span {span / 1000 / args.steps:.2f} ms/iter, busy {busy / 1000 / args.steps:.2f} ms/iter, idle {(span - busy) / 1000 / args.steps:.2f} ms/iter ({100 * (span - busy) / span:.0f}%)")
# largest gaps
gaps = sorted(((k[i + 1][0] - k[i][1]), i) for i in range(len(k) - 1) if k[i + 1][0] > k[i][1])[-8:]
names = sorted(((e["ts"], e["name"]) for e in ev if e.get("cat") == "kernel" and "dur" in e))
print("largest gaps (us):", [round(g) for g, _ in gaps])
import collections
allg = [k[i + 1][0] - k[i][1] for i in range(len(k) - 1) if k[i + 1][0] > k[i][1]]
for lo, hi in [(0, 10), (10, 50), (50, 150), (150, 400), (400, 1e9)]:
    sel = [g for g in allg if lo <= g < hi]
    print(f"  gaps {lo}-{hi} us: {len(sel) / args.steps:.1f}/iter, {sum(sel) / 1000 / args.steps:.2f} ms/iter")
rt = collections.defaultdict(lambda: [0, 0.0])
for e in ev:
    if e.get("cat") in ("cuda_runtime", "cuda_driver") and "dur" in e:
        rt[e["name"]][0] += 1; rt[e["name"]][1] += e["dur"]
for n, (c, d) in sorted(rt.items(), key = lambda x: -x[1][1])[:6]:
    print(f"  cpu {n}: {c / args.steps:.1f}/iter, {d / 1000 / args.steps:.2f} ms/iter, {d / max(c, 1):.1f} us avg")
# which kernel precedes the big gaps
kn = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev if e.get("cat") == "kernel" and "dur" in e)
prev = collections.Counter()
for i in range(len(kn) - 1):
    if kn[i + 1][0] - kn[i][1] > 150: prev[(kn[i][2][:50], kn[i + 1][2][:50])] += 1
for (a, b), c in prev.most_common(6): print(f"  gap>150us after [{a}] before [{b}]: {c / args.steps:.1f}/iter")
cnt = collections.Counter(n.split("(")[0].split("<")[0][:40] for _, _, n in kn)
print("kernels per iter:", round(len(kn) / args.steps))
for n, c in cnt.most_common(14): print(f"  {c / args.steps:7.1f}  {n}")
tt = collections.defaultdict(float)
for s, e, n in kn: tt[n.split("(")[0][:60]] += e - s
print("GPU time per iter by kernel:")
for n, d in sorted(tt.items(), key = lambda x: -x[1])[:16]: print(f"  {d / 1000 / args.steps:6.2f} ms  {n}")

if args.stack:
    # CPU activity inside large GPU gaps: time per Python function / op overlapping the gaps
    big = [(kn[i][1], kn[i + 1][0]) for i in range(len(kn) - 1) if kn[i + 1][0] - kn[i][1] > 100]
    cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("python_function", "cpu_op", "cuda_runtime") and "dur" in e]
    acc = collections.defaultdict(float)
    for gs, ge in big:
        for e in cpu:
            s0, e0 = e["ts"], e["ts"] + e["dur"]
            ov = min(e0, ge) - max(s0, gs)
            if ov > 0: acc[(e["cat"][:6], e["name"][:90])] += ov
    tot = sum(ge - gs for gs, ge in big)
    print(f"large gaps (>100 us): {len(big) / args.steps:.1f}/iter, {tot / 1000 / args.steps:.2f} ms/iter; CPU time inside them by event (inclusive):")
    for (c, n), d in sorted(acc.items(), key = lambda x: -x[1])[:45]:
        print(f"  {d / 1000 / args.steps:6.3f} ms  [{c}] {n}")
