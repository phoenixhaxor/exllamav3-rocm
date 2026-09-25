"""Batched verification sampling vs per-position sampling: greedy output must match exactly.
usage: test_presample.py -m <model> -dm <draft>"""
import argparse, torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.generator.sampler import ComboSampler
import exllamav3.generator.generator as G
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
out = {}
for flag in (False, True):
    G._presample_enable = flag
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dm, draft_cache = dc)
    out[flag] = [gen.generate(prompt = p, max_new_tokens = 300, sampler = ComboSampler(temperature = 0.0),
                              encode_special_tokens = True, completion_only = True) for p in prompts]
    # with penalties set to neutral values (as TabbyAPI builds its stack) the fast path must engage too
    s = ComboSampler(temperature = 0.6, rep_p = 1.0, pres_p = 0.0, freq_p = 0.0)
    print(f"presample={flag}: sampler reqs_past_ids={s.reqs_past_ids}")
for i in range(len(prompts)):
    print(f"prompt {i}: greedy identical: {out[False][i] == out[True][i]} ({len(out[True][i])} chars)")
