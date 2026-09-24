"""
End-to-end generation benchmark: plain decode, MTP drafting or a DFlash2 draft model.

usage: python bench_gen.py -m <model_dir> [--mtp | -dm <draft_dir>] [-ndt N] [--cache 32768]
                           [--tokens 512] [--prompts prose,code] [--image path]
"""
import argparse, time, sys
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler
from exllamav3.cache import CacheLayer_quant

PROMPTS = {
    "prose": "Write a detailed, vivid short story (about 600 words) about a lighthouse keeper who discovers "
             "that the lighthouse's lamp has been signaling to something far out at sea.",
    "code": "Write a complete, well-structured Python module implementing an LRU cache with TTL expiry, "
            "thread safety, type hints, docstrings, and a set of pytest unit tests covering edge cases.",
    "explain": "Explain step by step how a CPU pipeline handles branch misprediction, including the roles of "
               "the branch predictor, the reorder buffer and speculative execution. Use concrete examples.",
}

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True)
ap.add_argument("-dm", "--draft", default = None)
ap.add_argument("--mtp", action = "store_true")
ap.add_argument("-ndt", "--num_draft_tokens", type = int, default = None)
ap.add_argument("--cache", type = int, default = 32768)
ap.add_argument("--kv_bits", type = int, default = 0, help = "quantized KV cache bits (0 = fp16)")
ap.add_argument("--dkv_bits", type = int, default = -1, help = "draft KV cache bits (-1 = same as --kv_bits)")
ap.add_argument("--tokens", type = int, default = 512)
ap.add_argument("--bsz", type = int, default = 1)
ap.add_argument("--prompts", default = "prose,code")
ap.add_argument("--think", action = "store_true", help = "leave thinking enabled")
ap.add_argument("--temp", type = float, default = 0.6)
ap.add_argument("--image", default = None)
ap.add_argument("--show", action = "store_true")
ap.add_argument("--dds", action = "store_true", help = "dynamic draft tokens")
ap.add_argument("--dc", type = float, default = 0.4, help = "draft confidence for --dds")
args = ap.parse_args()

config = Config.from_directory(args.model)
model = Model.from_config(config)

draft_model = draft_cache = None
if args.mtp:
    draft_model = Model.from_config(config, component = "mtp")
elif args.draft:
    dconfig = Config.from_directory(args.draft)
    draft_model = Model.from_config(dconfig)

# Recurrent (GDN) layers keep one state per drafted position for rewinds
max_history = max(
    draft_model.caps.get("default_draft_size", 4) if draft_model else 0,
    args.num_draft_tokens or 0,
)
cache_kw = dict(layer_type = CacheLayer_quant, k_bits = args.kv_bits, v_bits = args.kv_bits) if args.kv_bits else {}
cache = Cache(model, max_num_tokens = args.cache, max_history = max_history, max_batch_size = args.bsz, **cache_kw)
t0 = time.time()
model.load(progressbar = False)
print(f"main model loaded in {time.time() - t0:.1f}s, max_history {max_history}")
tokenizer = Tokenizer.from_config(config)

if draft_model is not None:
    dkb = args.kv_bits if args.dkv_bits < 0 else args.dkv_bits
    dcache_kw = dict(layer_type = CacheLayer_quant, k_bits = dkb, v_bits = dkb) if dkb else {}
    draft_cache = Cache(draft_model, max_num_tokens = args.cache, **dcache_kw)
    draft_model.load(progressbar = False)
    print("draft loaded:", type(draft_model).__name__, draft_model.caps.get("default_draft_size"))

vision_model = None
image_embeddings = None
if args.image:
    from PIL import Image
    vision_model = Model.from_config(config, component = "vision")
    vision_model.load(progressbar = False)
    image_embeddings = [vision_model.get_image_embeddings(tokenizer = tokenizer, image = Image.open(args.image))]

print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB, reserved {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

generator = Generator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
    draft_model = draft_model,
    draft_cache = draft_cache,
    num_draft_tokens = args.num_draft_tokens,
    dynamic_draft_tokens = args.dds,
    draft_confidence = args.dc,
)

def format_prompt(text):
    # Qwen3.x chat format; empty think block disables reasoning
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n" + ("<think>\n" if args.think else "<think>\n\n</think>\n\n")

def run(name, text, embeddings = None):
    if embeddings:
        text = "\n".join(e.text_alias for e in embeddings) + "\n" + text
    prompt = format_prompt(text)
    input_ids = tokenizer.encode(prompt, encode_special_tokens = True, embeddings = embeddings)
    job = Job(
        input_ids = input_ids,
        max_new_tokens = args.tokens,
        sampler = ComboSampler(temperature = args.temp, top_p = 0.95, top_k = 20),
        stop_conditions = [tokenizer.eos_token_id, "<|im_end|>"],
        embeddings = embeddings,
    )
    generator.enqueue(job)
    out = []
    first = None
    t_start = time.perf_counter()
    last = None
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r.get("text"):
                if first is None: first = time.perf_counter()
                out.append(r["text"])
            if r.get("eos"):
                last = r
    t_end = time.perf_counter()
    n = last.get("new_tokens", 0) if last else 0
    dec = (n - 1) / (t_end - first) if first and n > 1 else 0
    acc = last.get("accepted_draft_tokens", 0) if last else 0
    rej = last.get("rejected_draft_tokens", 0) if last else 0
    ar = acc / (acc + rej) if acc + rej else 0
    print(f"[{name}] prompt {input_ids.shape[-1]} tok, TTFT {(first - t_start) * 1000 if first else 0:.0f} ms, "
          f"{n} new tok, decode {dec:.1f} tok/s, draft accept {acc}/{acc + rej} ({ar:.2f})")
    text_out = "".join(out)
    if args.show: print(text_out)
    else: print("   ", text_out[:300].replace("\n", " "), "...")
    return dec

# warmup
run("warmup", "Say hi.")
results = []
for p in args.prompts.split(","):
    if p: results.append(run(p, PROMPTS[p]))
if image_embeddings:
    run("vision", "Describe this image in detail.", image_embeddings)
print("MEAN decode tok/s:", sum(results) / max(len(results), 1))
