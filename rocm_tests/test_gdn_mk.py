"""GDN core megakernel vs the four separate kernels: speculative greedy decode (verification with
history) and plain greedy decode (no draft, no history) must produce identical text.
usage: test_gdn_mk.py -m <model> -dm <draft>"""
import argparse, torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.generator.sampler import ComboSampler
from exllamav3.ext import exllamav3_ext as ext
ap = argparse.ArgumentParser(); ap.add_argument("-m", required = True); ap.add_argument("-dm", required = True)
args = ap.parse_args()
config = Config.from_directory(args.m); model = Model.from_config(config)
dm = Model.from_config(Config.from_directory(args.dm))
cache = Cache(model, max_num_tokens = 8192, max_history = 8, max_batch_size = 1); model.load(progressbar = False)
dc = Cache(dm, max_num_tokens = 8192); dm.load(progressbar = False)
tok = Tokenizer.from_config(config)
prompts = [
    "<|im_start|>user\nWrite a Python function that merges overlapping intervals, with tests.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<|im_start|>user\nDescribe a thunderstorm over the sea in two paragraphs.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
]
def run(mk, draft):
    ext.gdn_mk_set(mk)
    if hasattr(ext, "act_epi_set"): ext.act_epi_set(mk)
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dm if draft else None, draft_cache = dc if draft else None)
    return [gen.generate(prompt = p, max_new_tokens = 250, sampler = ComboSampler(temperature = 0.0),
                         encode_special_tokens = True, completion_only = True) for p in prompts]
for draft in (True, False):
    a = run(0, draft); b = run(1, draft)
    for i in range(len(prompts)):
        same = a[i] == b[i]
        print(f"draft={draft} prompt {i}: identical {same} ({len(b[i])} chars)")
        if not same:
            k = next((j for j in range(min(len(a[i]), len(b[i]))) if a[i][j] != b[i][j]), min(len(a[i]), len(b[i])))
            print("   diff at", k, repr(a[i][max(0, k - 40):k + 30]), "|", repr(b[i][max(0, k - 40):k + 30]))
ext.gdn_mk_set(-1)
