"""Memory-bandwidth utilization: practical peak (read / copy) vs the EXL3 matmuls of a model at 1 and 8
rows, per layer type, and the weight bytes one speculative round has to stream.
usage: bw.py -m <model> [-dm <draft>]"""
import argparse, collections, torch
from exllamav3 import Config, Model
from exllamav3.modules import Linear
ap = argparse.ArgumentParser(); ap.add_argument("-m", required = True); ap.add_argument("-dm"); ap.add_argument("--rows", action = "store_true")
args = ap.parse_args()
dev = torch.device("cuda:0")

def evtime(fn, n = 20, warm = 200):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
    e0.record()
    for _ in range(n): fn()
    e1.record(); e1.synchronize()
    return e0.elapsed_time(e1) / 1000 / n

# Practical peak: large read-only reduction and device copy
big = torch.empty(1 << 29, dtype = torch.half, device = dev).normal_()   # 1 GiB
dst = torch.empty_like(big)
t_read = evtime(lambda: big.sum(), 10, 5)
t_copy = evtime(lambda: dst.copy_(big), 10, 5)
peak_read = big.nbytes / t_read / 1e9
peak_copy = 2 * big.nbytes / t_copy / 1e9
print(f"practical peak: read {peak_read:.0f} GB/s, copy (r+w) {peak_copy:.0f} GB/s  (spec 960 GB/s)")
del big, dst
torch.cuda.empty_cache()

def linears(model):
    def walk(m):
        yield m
        for s in getattr(m, "modules", None) or []:
            yield from walk(s)
    for top in model.modules:
        for s in walk(top):
            if isinstance(s, Linear) and getattr(getattr(s, "inner", None), "trellis", None) is not None:
                yield s

def wbytes(lin):
    i = lin.inner
    return sum(t.nbytes for t in (i.trellis, i.suh, i.svh) if t is not None)

def measure(path, label):
    config = Config.from_directory(path); model = Model.from_config(config); model.load(progressbar = False)
    stats = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])   # kind -> count, bytes, t1, t8
    tot_b = tot_1 = tot_8 = 0.0
    seen = {}
    for lin in linears(model):
        key = (lin.in_features, lin.out_features, lin.inner.K if hasattr(lin.inner, "K") else 0)
        kind = lin.key.split(".")[-1]
        if key not in seen:
            x1 = torch.randn(1, 1, lin.in_features, device = dev).half()
            x8 = torch.randn(1, 8, lin.in_features, device = dev).half()
            seen[key] = (evtime(lambda: lin.forward(x1, {})), evtime(lambda: lin.forward(x8, {})))
        t1, t8 = seen[key]
        b = wbytes(lin)
        st = stats[kind]; st[0] += 1; st[1] += b; st[2] += t1; st[3] += t8
        tot_b += b; tot_1 += t1; tot_8 += t8
    print(f"\n{label}: EXL3 weights {tot_b / 1e9:.2f} GB; one pass 1 row {tot_1 * 1e3:.2f} ms "
          f"({tot_b / tot_1 / 1e9:.0f} GB/s), 8 rows {tot_8 * 1e3:.2f} ms ({tot_b / tot_8 / 1e9:.0f} GB/s)")
    print(f"  {'kind':14s} {'n':>4s} {'GB':>6s} {'1-row GB/s':>11s} {'8-row GB/s':>11s} {'8-row ms':>9s}")
    for kind, (n, b, t1, t8) in sorted(stats.items(), key = lambda x: -x[1][3]):
        print(f"  {kind:14s} {n:4d} {b / 1e9:6.2f} {b / t1 / 1e9:11.0f} {b / t8 / 1e9:11.0f} {t8 * 1e3:9.2f}")
    if args.rows:
        # Whole-pass scaling with the number of rows (distinct shapes timed once, weighted by count)
        shapes = collections.Counter()
        lin_of = {}
        for lin in linears(model):
            k = (lin.in_features, lin.out_features, getattr(lin.inner, "K", 0))
            shapes[k] += 1; lin_of[k] = lin
        print(f"  rows scaling (one pass over all {sum(shapes.values())} matmuls):")
        for r in (1, 2, 4, 8, 16):
            t = 0.0
            for k, n in shapes.items():
                lin = lin_of[k]; x = torch.randn(1, r, lin.in_features, device = dev).half()
                t += n * evtime(lambda: lin.forward(x, {}), 10, 30)
            print(f"    {r:2d} rows: {t * 1e3:6.2f} ms  {tot_b / t / 1e9:4.0f} GB/s  {t * 1e3 / r:6.2f} ms per row")
    model.unload()
    torch.cuda.empty_cache()
    return tot_b, tot_8

tb, t8 = measure(args.m, "target")
if args.dm:
    db, d8 = measure(args.dm, "draft (own layers; its lm_head is the target's)")
