"""VRAM per component: main + cache, draft + cache, vision (usage: vram.py <ctx> <kv_bits> <draft_kv_bits>)"""
import os
import sys, torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant
G = lambda: torch.cuda.memory_allocated() / 2**30
ctx = int(sys.argv[1]); kb = int(sys.argv[2]); dkb = int(sys.argv[3])
M = os.environ.get("EXL3_MODEL_DIR", "models/Qwen3.8-27B-EXL3-3.5bpw"); D = os.environ.get("EXL3_DRAFT_DIR", "models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw")
q = lambda b: dict(layer_type = CacheLayer_quant, k_bits = b, v_bits = b) if b else {}
cfg = Config.from_directory(M); m = Model.from_config(cfg); dm = Model.from_config(Config.from_directory(D))
c = Cache(m, max_num_tokens = ctx, max_history = 7, max_batch_size = 1, **q(kb))
m.load(progressbar = False); a = G(); print("main + cache:", round(a, 2))
dc = Cache(dm, max_num_tokens = ctx, **q(dkb))
dm.load(progressbar = False); b = G(); print("draft + cache:", round(b - a, 2))
v = Model.from_config(cfg, component = "vision"); v.load(progressbar = False); print("vision:", round(G() - b, 2))
free, tot = torch.cuda.mem_get_info(); print("total alloc", round(G(), 2), "free", round(free / 2**30, 2), "of", round(tot / 2**30, 2))
