# exllamav3 on AMD RDNA3 (ROCm port)

A ROCm/HIP port of [exllamav3](https://github.com/turboderp-org/exllamav3) for RDNA3 GPUs (tested on a
Radeon RX 7900 XTX, gfx1100), with RDNA3-specific kernels for the EXL3 matmul and for decode attention.
It runs Qwen3.8-27B EXL3 with **DFlash2** or **MTP** speculative decoding, **vision**, reasoning and an
**8-bit KV cache at 192K-256K context** on a single 24 GB card, served through a lightly patched
[TabbyAPI](https://github.com/theroyallab/tabbyAPI).

Upstream exllamav3 is CUDA-only: its EXL3 kernels are built on `mma.sync`, `ldmatrix`, `cp.async` and
cooperative launches, and TabbyAPI refuses AMD GPUs. This fork keeps the upstream Python stack and model
format unchanged and replaces the pieces that do not map to RDNA3.

The original upstream README is kept as [README.exllamav3.md](README.exllamav3.md).

---

## Results (RX 7900 XTX 24 GB, ROCm 7.2.4, PyTorch 2.13.0+rocm7.2)

Decode speed, greedy, code-style prompt, 8-bit KV cache (DFlash2 draft KV 4-bit), measured with
`rocm_tests/bench_long.py` after a prompt of the given length:

| Context | DFlash2 (default) | MTP | No draft |
|---|---|---|---|
| 2K | **135 tok/s** | 99 tok/s | 38.5 tok/s |
| 32K | **131 tok/s** | 88 tok/s | 36.3 tok/s |
| 99K | **101 tok/s** | 64 tok/s | - |

(No-draft column measured before the kernel-fusion round.)

Short prompts, 512 generated tokens, greedy (`rocm_tests/bench_gen.py`):

| Workload | DFlash2 | MTP (3 draft tokens) |
|---|---|---|
| Code | 122-125 tok/s | 97 tok/s |
| Explanation | 90-92 tok/s | 77 tok/s |
| Prose / story | 66-67 tok/s | 71 tok/s |

Run-to-run variance is noticeable (occasional runs 10-20% slower); profiling shows the extra time is host
side (GPU idle between launches), not in the kernels.

Through the TabbyAPI OpenAI endpoint (temperature 0.6, 400 tokens): code 94-109 tok/s, prose 63-66 tok/s.

Prompt processing (prefill): ~1180-1200 tok/s at 32K, ~1090 tok/s at 64K, ~945 tok/s at 99K. Long-context
retrieval (needle in a haystack through the API): 184,656-token prompt with the DFlash2 / 192K profile
and 247,056-token prompt with the MTP / 256K profile, both answered correctly.

For reference, the same GPU with llama.cpp (Qwen3.8-27B IQ3_XXS GGUF + DFlash2 Q8 draft) decoded at
40-53 tok/s.

Speculative decoding speed depends on the text: drafts are accepted far more often on code than on free
prose. All numbers are single-stream (batch size 1).

## Models

| Role | Repository | Size | Contents |
|---|---|---|---|
| Main | [Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw) | 15.3 GB | Qwen3.8-27B, EXL3 3.5 bpw (module-adaptive), `mul1` codebook |
| Draft | [Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw) | 1.47 GB | DFlash2 block-diffusion drafter, EXL3 5.0 bpw |

Main model details (read from the checkpoint):

- 64 layers: 48 Gated DeltaNet (linear attention) + 16 full attention (24 q heads, 4 kv heads, head dim 256).
- Per-module bitrates: MLP 3-bit (most layers) or 4-bit, attention / DeltaNet projections 4-bit, one
  5-bit `o_proj`, `lm_head` 6-bit; embeddings bf16 (kept in system RAM by exllamav3).
- **Built-in MTP head** (4-bit), used by `draft_mode: mtp`.
- **Vision tower included** (27 blocks, bf16, 0.78 GB). The model card says "no vision tower", but the
  weights are in the checkpoint and work.

DFlash2 draft: 5 sliding-window (2048) layers conditioned on target hidden states, drafts 7 tokens per
round (verification batch of 8).

Download:

```bash
rocm/scripts/download_models.sh models
```

## Serving profiles

| Profile | File | Draft | Context | KV cache | VRAM (idle) |
|---|---|---|---|---|---|
| Default, fastest | `rocm/tabbyapi/config.dflash2-192k.yml` | DFlash2 | 196,608 | Q8 (draft Q4) | ~23.6 GB |
| Full context | `rocm/tabbyapi/config.mtp-256k.yml` | MTP | 262,144 | Q8 | ~23.5 GB |

Measured VRAM per component (8-bit KV):

| Component | 8K ctx | 128K ctx | 192K ctx |
|---|---|---|---|
| Main model + cache + recurrent state | 12.45 GB | 16.43 GB | 18.56 GB |
| DFlash2 draft + cache (Q8 / Q4) | 1.55 GB | 2.80 GB | 3.46 / 2.52 GB |
| Vision tower | 0.78 GB | 0.78 GB | offloaded to RAM |

Main KV cost is ~33.9 KB per token at Q8 (16 attention layers x 4 kv heads x 256). DFlash2 with a 256K
context does not fit in 24 GB (the draft cache is allocated for the full context although the draft only
attends to the last 2048 tokens); use the MTP profile for 256K.

The draft KV cache at Q4 gives the same acceptance as FP16 in greedy tests, so the default profile keeps
the main cache at Q8 and quantizes only the draft cache further.

## Quick start

Requirements: Linux, ROCm 7.x (tested 7.2.4) with an RDNA3 GPU, ~20 GB disk for models, 32 GB+ RAM.

```bash
git clone <this repository> exllamav3-rocm && cd exllamav3-rocm

rocm/scripts/setup_env.sh .venv-rocm          # Python 3.12 + torch 2.13.0 (ROCm 7.2 wheels) + deps
source .venv-rocm/bin/activate                 # or: conda activate ./.venv-rocm
ROCM_HOME=/opt/rocm rocm/scripts/build.sh      # builds exllamav3_ext for gfx1100 (~10 min)

rocm/scripts/download_models.sh models
rocm/scripts/install_tabbyapi.sh ../tabbyAPI models
rocm/scripts/run_tabbyapi.sh config.yml ../tabbyAPI
```

The server listens on port 8096 (OpenAI-compatible `/v1/chat/completions`, streaming, tools, images).
Authentication is enabled in the shipped configs: TabbyAPI writes the keys to `api_tokens.yml` on first
start. Set `disable_auth: true` only if the port is not reachable from untrusted machines.

Full-context profile:

```bash
cp rocm/tabbyapi/config.mtp-256k.yml ../tabbyAPI/
rocm/scripts/run_tabbyapi.sh config.mtp-256k.yml ../tabbyAPI
```

Without TabbyAPI (Python API), see `rocm_tests/bench_gen.py`: load `Model.from_config(config)` for the
main model, `Model.from_config(config, component = "mtp")` or the DFlash2 directory for the draft,
`Model.from_config(config, component = "vision")` for images, and create the main `Cache` with
`max_history` equal to the draft length (the DeltaNet layers keep one state per drafted position).

## What changed (vs. upstream exllamav3)

### EXL3 matmul for small batches: `quant/exl3_rdna3*.cu`

Used for every EXL3 linear with up to 144 rows (decode, draft verification, MTP), replacing the NVIDIA
GEMV/GEMM kernels:

- Input transform (sign flips + 128-point Hadamard) runs once per matmul in a small kernel that writes
  the activations in the main kernel's LDS layout.
- One wave per 16x16 weight tile column, k-split across blocks, split-K reduction by the last block to
  arrive (atomic counter), output Hadamard in the epilogue. A single graph-patchable launch pair.
- Trellis words stream through a register prefetch ring built on `raw_buffer_load` and pinned with
  `sched_barrier` (plain loads get folded into load-at-use by InstCombine, which serializes the stream).
- `mul1` codebook decode tuned for RDNA3 instruction rates: two 24-bit multiplies instead of the
  1/5-rate `v_mul_lo_u32`, `v_sad_u8` byte sums instead of the half-rate `v_dot4`, `v_alignbit`
  window extraction; the codebook's affine map is folded into the epilogue so weights enter
  `v_dot2_f32_f16` raw.
- Instantiated for 1-8 bit (and x.5 with `mul1`), rows per pass 1/2/3/4/5/6/8/12/16.
- Multi-matrix mode (`exl3_rdna3_mgemm`, blockIdx.z per entry): projections that share an input run as
  one input-transform launch plus one matmul launch, driven by the existing `MultiLinear` /
  `SlicedMultiLinear` pointer tables: MLP gate + up, Gated DeltaNet qkv + z (8 slices of 2048),
  attention q / k / v (14 slices of 1024). 3.3% less time per speculative round than separate matmuls
  (~320 fewer kernels per round; q/k/v alone is 26-28% faster at 5-8 rows).
- Input-transform prologues, each replacing a separate elementwise kernel: `silu(gate) * up` before
  the MLP down projection (bit-identical to `silu_mul`), the attention output gate `o * sigmoid(g)`
  before `o_proj` (bit-identical to `mul_sigmoid_`), and the Gated DeltaNet gated RMSNorm before
  `out_proj` (one head = one 128-element Hadamard block; matches `gated_rms_norm` to fp16 rounding).
- `mul1` pair packing with `v_sad_hi_u8` (the second byte sum lands in the high half directly): one
  VALU op less per weight pair, ~2-4% faster decode matmuls.
- Grid sizing: the k-split targets one residency wave (6 blocks per WGP = 288, LDS-limited); a partial second
  wave of blocks roughly doubles the kernel tail. 3-10% faster per matmul in isolation (`kbench`),
  neutral end-to-end.

Achieved bandwidth on 4-bit tensors: ~700-770 GB/s at 1 row, ~450-530 GB/s at 8 rows (VALU-bound).

### Decode and verification attention: `rdna3_attn.cu`

Drop-in replacements for the Triton flash-decoding split kernel on the graphed decode path (same
arguments, same partial layout, the Triton combine kernel is reused), for 8-bit and FP16 caches:

- `q_len == 1`: lane-per-token scores with `v_dot2`, online softmax, lane-per-dimension values.
- `q_len 2..8` (draft verification, 8-bit cache): `v_wmma_f32_16x16x16_f16` for both Q·K^T and P·V,
  K/V tiles staged in LDS once per kv head (V transposed so a WMMA B fragment is contiguous), register
  prefetch of the next tile.

At 32K context this took decode attention from ~10 ms to ~2.7 ms per token; at 100K the DFlash2
verification from ~45 ms to ~14 ms per round.

### Other changes

- HIP compatibility across the extension: 64-bit warp masks, `dp4a` / `lop3` / atomics / cache-hinted
  load shims, `__nanosleep`, `__grid_constant__`, driver-API graph calls, Triton `hsaco` loading.
- `hgemm`: rocBLAS on gfx11 has no WMMA solution for fp16 x fp16 -> fp32 output (~16 vs ~80 TFLOPS),
  so fp32-output products of 64+ rows run as fp16 output plus a widening copy (prefill 2.3x faster).
- Triton paged prefill: 128x64 tiles, one stage on HIP for head dim 256 (~23 -> ~57 TFLOPS).
- Triton decode split kernel: skips keys before the sliding window (the DFlash2 draft scanned the whole
  context: 13 ms -> 1.2 ms per round at 100K).
- `gdn_ba_gemv`: 16-byte loads and `v_dot2` with independent accumulators.
- Gated DeltaNet recurrence (`gdn.cu`): for 128x128 heads the state slice of each thread stays in
  registers across the drafted tokens, instead of being read from memory twice per token (the state
  is 3 MB per layer). 2.5 -> 1.7 ms per speculative round, bit-identical; it is now bound by the
  per-token history writes needed for rollback.
- Residual adds: a transformer block hands its final `x += mlp(x)` to the next block's input RMSNorm
  (`rms_norm_res_in`, already used between attention and MLP), removing ~58 elementwise kernels per
  round.

## Environment switches

| Variable | Default | Effect |
|---|---|---|
| `EXL3_RDNA3_GEMM` | 1 | 0 = fall back to the (emulated) upstream EXL3 kernels |
| `EXL3_RDNA3_ATTN` | 1 | 0 = Triton decode attention |
| `EXL3_RDNA3_ATTN_SPLIT_MULT` | 16 | kv splits per CU for the HIP attention kernels (cap 128) |
| `EXL3_RDNA3_TARGET_BLOCKS` | 6 x WGPs (288) | grid size target for the EXL3 matmul k-split |
| `EXL3_HIP_MGEMM` | 1 | 0 = run bundled projections (gate/up, qkv/z, q/k/v) as separate matmuls |
| `EXL3_FUSE_ACT` | 1 | 0 = separate `silu_mul` / `mul_sigmoid_` kernels before the MLP down projection / attention `o_proj` |
| `EXL3_FUSE_GNORM` | 1 | 0 = separate gated RMSNorm kernel before the DeltaNet `out_proj` |
| `EXL3_GDN_REG` | 1 | 0 = original DeltaNet recurrence kernel (state re-read from memory per token) |
| `EXL3_RESID_DEFER` | 1 | 0 = no residual-add folding into the next block's input norm |
| `EXL3_NOGRAPH` | - (`mlp,gdn` in `run_tabbyapi.sh`) | modules (`mlp`, `gdn`, `attn`) that decode eagerly instead of through a HIP graph |
| `EXL3_PF_BLOCK_M`, `EXL3_PF_BLOCK_N`, `EXL3_PF_WARPS` | - | Triton prefill tile overrides |

## Tests and benchmarks (`rocm_tests/`)

| Script | Purpose |
|---|---|
| `test_rdna3_gemm.py <model_dir> [tensor ...]` | EXL3 matmul vs. reconstructed weights (m = 1..144, fp16/fp32 out) and an independent numpy trellis decoder |
| `test_rdna3_mgemm.py <model_dir>` | multi-matrix matmul (gate/up, sliced qkv/z and q/k/v) and the fused prologues (silu, output gate, gated norm) vs. the unfused kernels |
| `gaps.py -m <model> [-dm <draft>] [--stack]` | GPU busy/idle per speculative round, gap histogram, kernel counts and times; `--stack`: CPU activity inside large GPU gaps |
| `bench_gen.py -m <model> [-dm <draft> \| --mtp]` | short-prompt generation speed, draft acceptance, `--image` for vision |
| `bench_long.py 2000,32000,99000 [-dm <draft> \| --mtp]` | decode speed after long prompts (greedy, prints output with `--show`) |
| `prof_gen.py`, `prof_long.py`, `prof_prefill.py` | kernel-time breakdowns (torch.profiler) |
| `kbench.cc` | standalone EXL3 matmul timing harness (`hipcc -x hip`; warms the clocks first; `-DKB_TRACE` prints a per-block timeline) |
| `attn_pf_bench.py` | prefill attention microbenchmark with a torch reference |
| `api_test.py <url> <image>`, `needle_api.py <n> <depth>` | OpenAI API smoke test, long-context retrieval |
| `vram.py <ctx> <kv_bits> <draft_kv_bits>` | VRAM per component |

Model paths default to `models/...` or `EXL3_MODEL_DIR` / `EXL3_DRAFT_DIR`. Greedy speculative decoding
produces the same text as plain decoding for the DFlash2 path in these tests.

## Limitations

- Tested on gfx1100 only. The kernels assume wave32 (RDNA3); RDNA2 lacks the dot/WMMA instructions used,
  CDNA (wave64) is not supported.
- Tested models use the `mul1` codebook with integer bitrates; the half-integer (x.5 bpw) and
  `mcg` / 3INST codebook paths compile but are not validated.
- MoE / block-sparse models: the fused expert kernels are not ported (graph parameter patching for
  per-expert weights is not supported by the RDNA3 matmul).
- HIP attention kernels cover causal full attention without softcap or sinks, head dim 128/256,
  q_len up to 16 (WMMA verification up to 8, 8-bit cache); other shapes use the Triton kernels.
- Quantization (conversion) kernels compile but are untested on ROCm.
- The 8-row matmul (codebook decode + FMA) is ~78% of GPU time per speculative round. About 16-18% of
  each round is GPU idle time: ~3 us per kernel boundary (graph or eager alike) and the host-side
  turnaround after each verification. HIP graphs save little on ROCm: `hipGraphLaunch` costs CPU time
  per node like eager launches, and each graph launch adds ~8 us of GPU idle. Replacing the per-module
  graphs with eager launches (`EXL3_NOGRAPH=mlp,gdn`) is ~0.5% faster; merging attention + MLP graphs
  per layer would save at most ~0.5 ms per ~41 ms round, so it was not done.

## Notes on published RTX 3090 / Arc B70 numbers

The ~144 tok/s Qwen3.8-27B figure on an RTX 3090 comes from SGLang with the `sglang-exl3` plugin and
NEXTN (MTP) drafting on the 3.0 bpw quant (prose ~99, code ~143 tok/s), not from exllamav3 + DFlash2;
TabbyAPI/exllamav3 runs in the same registry reach 58-69 tok/s on a 3090. The Arc B70 figure of 60 tok/s
is single-stream at 4 bpw (330 tok/s is the 16-stream aggregate).

## License and credits

- exllamav3 by turboderp and contributors, MIT License (see [LICENSE](LICENSE)); this fork keeps it.
- TabbyAPI (AGPL-3.0) is not included; `rocm/tabbyapi/` only contains a patch and config files.
- Models by the Qwen team, EXL3 quants and DFlash2 draft by Mia-AiLab (see their model cards for licenses).
