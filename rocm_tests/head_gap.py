"""Is there GPU idle after the lm_head matmul itself? lm_head(8 rows) + tiny op, repeated, profiled."""
import sys, json, tempfile, torch
from torch.profiler import profile, ProfilerActivity
from exllamav3 import Config, Model
config = Config.from_directory(sys.argv[1]); model = Model.from_config(config); model.load(progressbar = False)
lm = model.modules[model.logit_layer_idx]
print("head:", type(lm).__name__, getattr(lm, "key", ""), [type(m).__name__ for m in getattr(lm, "modules", [])])
x = torch.randn(1, 8, config.hidden_size, device = "cuda:0").half()
t = torch.zeros(16, device = "cuda:0")
def run(n):
    for _ in range(n):
        y = lm.forward(x, {})
        t.add_(1)
    return y
y = run(5); torch.cuda.synchronize()
print("out", y.shape, y.dtype)
with profile(activities = [ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    run(10); torch.cuda.synchronize()
p = tempfile.mktemp(suffix = ".json"); prof.export_chrome_trace(p)
ev = json.load(open(p))["traceEvents"]
kn = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e)
for i in range(1, min(len(kn), 16)):
    print(f"  gap {kn[i][0] - kn[i-1][1]:7.1f} us  dur {kn[i][1] - kn[i][0]:7.1f}  {kn[i][2][:60]}")

# Event timing (no profiler): lm_head alone vs lm_head + tiny op, per row count
def timed(fn, n = 20):
    fn(); torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
    e0.record()
    for _ in range(n): fn()
    e1.record(); e1.synchronize()
    return e0.elapsed_time(e1) * 1000 / n
for rows in (1, 2, 8, 16):
    xr = torch.randn(1, rows, config.hidden_size, device = "cuda:0").half()
    a = timed(lambda: lm.forward(xr, {}))
    b = timed(lambda: (lm.forward(xr, {}), t.add_(1)))
    print(f"rows {rows:2d}: lm_head {a:8.1f} us, lm_head + tiny {b:8.1f} us (+{b - a:6.1f})")
