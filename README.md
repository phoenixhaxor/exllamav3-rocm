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
| 2K | **157 tok/s** | 108 tok/s | 38.5 tok/s |
| 32K | **147 tok/s** | 92 tok/s | 36.3 tok/s |
| 99K | **107 tok/s** | 66-68 tok/s | - |

(No-draft and MTP columns measured before the latest rounds; DFlash2 re-measured with direct tile-word loads.)

Short prompts, 512 generated tokens, greedy (`rocm_tests/bench_gen.py`):

| Workload | DFlash2 | MTP (3 draft tokens) |
|---|---|---|
| Code | 135 tok/s | 101-108 tok/s |
| Explanation | 98 tok/s | 81-82 tok/s |
| Prose / story | 72 tok/s | 70-73 tok/s |

Run-to-run variance is noticeable (occasional runs 10-20% slower); profiling shows the extra time is host
side (GPU idle between launches), not in the kernels.

Through the TabbyAPI OpenAI endpoint, greedy, generating to EOS: code ~186 tok/s, prose ~78 tok/s
(abliterated build with the runtime-ablation sidecar; earlier sampled runs at temperature 0.6 / 400 tokens:
code 94-109, prose 63-66 tok/s).

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
- RMSNorm feeding a projection bundle (MLP gate/up, DeltaNet qkv/z) writes that bundle's matmul input
  transform in its tail (`rms_norm_had`), so the matmul skips its input kernel (eager path; the
  hand-off is matched on input pointer, suh table and shape, and dropped by any other matmul).
- Prefetch ring depth 4 k-slices (was 8): 1-4% faster at 1-8 rows.
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
| `EXL3_FUSE_NORM_HAD` | 1 | 0 = no matmul input transform in the RMSNorm tail |
| `EXL3_RESID_DEFER` | 1 | 0 = no residual-add folding into the next block's input norm |
| `EXL3_NOGRAPH` | - (`mlp,gdn` in `run_tabbyapi.sh`) | modules (`mlp`, `gdn`, `attn`) that decode eagerly instead of through a HIP graph |
| `EXL3_PF_BLOCK_M`, `EXL3_PF_BLOCK_N`, `EXL3_PF_WARPS` | - | Triton prefill tile overrides |

## Tests and benchmarks (`rocm_tests/`)

| Script | Purpose |
|---|---|
| `test_rdna3_gemm.py <model_dir> [tensor ...]` | EXL3 matmul vs. reconstructed weights (m = 1..144, fp16/fp32 out) and an independent numpy trellis decoder |
| `test_rdna3_mgemm.py <model_dir>` | multi-matrix matmul (gate/up, sliced qkv/z and q/k/v) and the fused prologues (silu, output gate, gated norm) vs. the unfused kernels |
| `bench_ab.py -m <model> -dm <draft> [--variants ...]` | in-process A/B of decode-loop options at fixed greedy acceptance: median ms per speculative round (the reliable speed metric) |
| `micro/*.cc` | standalone HIP microbenchmarks: peak read bandwidth (`peakbw`), EXL3-like strided access (`stridebw`), kernel boundary vs grid barrier (`gridbar`), idle after large kernels (`tailgap`) |
| `bw.py -m <model> [-dm <draft>] [--rows]` | practical peak bandwidth vs every EXL3 matmul at 1 / 8 rows, per projection kind; `--rows`: whole-pass scaling 1-16 rows |
| `timeline.py -m <model> -dm <draft> [--stack] [--ops]` | GPU ops and large gaps inside individual rounds, with the CPU frames running during each gap |
| `test_gdn_mk.py`, `test_presample.py`, `test_abl_fuse.py`, `test_abl_noise.py` | bit-identity / numerics checks for the megakernel, batched sampling and runtime ablation |
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

## Performance analysis and open ideas

### Where the time goes

One DFlash2 speculative round (draft forward + 8-token verification) takes ~34.7 ms on the 7900 XTX
after the rounds below (it was ~39 ms before the megakernel / host-side round; profile with
`rocm_tests/gaps.py`, wall-clock A/B with `rocm_tests/bench_ab.py`):

| Part | Time per round | Notes |
|---|---|---|
| EXL3 matmuls | ~25 ms | verification (8 rows) + draft + two lm_head passes; decode-VALU-bound at 8 rows (see below) |
| Gated DeltaNet core (`gdn_core_mk`) | ~2.3 ms | 48 layers; recurrence latency-bound, plus per-token history writes for rollback |
| Attention, norms, other kernels | ~4 ms | |
| GPU idle | ~3.4 ms (~10%) | ~690 dispatch gaps of ~3-4.5 us between dependent kernels (~718 kernels per round) |

The 8-row matmul costs ~1.4x a 1-row matmul. The `mul1` codebook decode (a hash plus byte sum per
weight, ~5 VALU ops) and the 8-row FMA (`v_dot2`, dual-issued) share one issue port, and the loop is
also latency-bound (~54% VALU utilization). Decoding costs the same per weight at any bitrate, so a
3.0 bpw model is not faster than 3.5 bpw. A 4-row matmul is ~20% cheaper than an 8-row one, but
shortening the verified block (dynamic draft length) lost more in accepted tokens than it saved.

HIP graphs save little on ROCm: `hipGraphLaunch` spends CPU time per node like eager launches, and
each graph launch adds ~8 us of GPU idle. Replacing the per-module graphs with eager launches
(`EXL3_NOGRAPH=mlp,gdn`) is ~0.5% faster. Runtime knobs (`HIP_FORCE_DEV_KERNARG`,
`HSA_ENABLE_INTERRUPT`, `GPU_MAX_HW_QUEUES`) do not change the ~3.3 us per-kernel dispatch cost.

### Tried and dropped (measured, no gain)

| Idea | Result |
|---|---|
| WMMA for the matmul FMA | half of the 16 rows wasted at 8 rows; slower than dual-issued `v_dot2` |
| Hadamard input transform inside the matmul | redundant per block, ~13% more VALU work at 8 rows |
| Persistent / work-queue matmul scheduling | no gain once clocks are warm; block-time spread is contention, not imbalance |
| Smaller or larger x chunks in LDS (more waves) | no gain or slower |
| 2 tiles per wave, split accumulators | slower / no gain |
| Merging attention + MLP graphs per layer | at most ~0.5 ms per round; eager MLP/DeltaNet gets the same |
| Prefetching the DeltaNet inputs into registers | slower (loads were already overlapped) |
| DeltaNet rollback by replay instead of per-token history | the replay kernel cost more than the saved writes (the recurrence is latency-bound); also needs per-layer copies of the inputs, which share scratch buffers across layers |
| GPU-side embedding gather from pinned host memory | the CPU lookup is only ~0.1 ms per round |
| 3.0 bpw main model | not faster (decode cost is per weight) and lower quality |
| Dynamic draft length, hot-token (vocabulary-pruned) draft head | slower end-to-end |
| Cooperative launch (`grid.sync`) for the DeltaNet megakernel | +10-20 us per launch on ROCm, ate the gain; replaced by a normal launch with an atomic epoch barrier |
| "Host turnaround" after the verification sync | mostly a profiler artifact (long kernels are under-reported); batched sampling and non-blocking readbacks recovered only ~0.3 ms |
| ROCm 10.0 | same speed as 7.2.4 (see below) |
| mul1 product as `v_mad_u32_u24((x << 8), 0x830000, lo)` (shift pairable with `dot2` in VOPD) | the shift never paired, `dot2` pairing got worse; 8 rows +9%, 1 row +27% |
| mul1 / byte-sum constants in SGPRs instead of 32-bit literals (smaller instructions) | no gain (front-end is not the limit), 4-bit 1 row +6% |
| Four accumulators per tile (`KB_NACC=4`, better `dot2` pairing, 70 instead of 78 issue slots) | 3-bit -1.7%, 4-bit +1.4%, 5 / 6-bit equal; not bit-identical |
| Deeper weight prefetch (`KB_PF` 8-16), smaller x chunk (more waves), other grid targets | equal or slower |
| Adaptive verification window (per-position acceptance EMA, window maximizing tokens / round time) | prose +4-5%, code -3%; kept off (`EXL3_ADAPT_WINDOW=1`) |
| Continuous weight ring across x chunks + double-buffered x staging (one barrier per half chunk) | 3-bit +6%, 5-bit +4%, 4-bit +1% at 8 rows (more barriers, more VGPRs) |
| CU mode (`-mcumode`), `amdgpu_waves_per_eu` 8-16 | CU mode +6%; waves-per-EU no effect |
| Hardware counters (rocprofv3 from the ROCm 10 wheel) | the profiler aborts on gfx1100; component costs measured with the kbench switches instead |

### Memory bandwidth utilization

Measured with `rocm_tests/bw.py` (event timing, each EXL3 matmul of the model run in isolation) and the
microbenchmarks in `rocm_tests/micro/` (build with `hipcc -O3 --offload-arch=gfx1100 -x hip <file>.cc`):

| | Bandwidth | Share of peak |
|---|---|---|
| Peak read, `micro/peakbw.cc` (16-byte non-temporal loads, 1536 blocks x 256 threads, 4 in flight) | 952 GB/s | 99% of the 960 GB/s spec |
| EXL3-like pattern, `micro/stridebw.cc` (1 KB per block per k-slice, 139 KB stride, 272 blocks) | 856 GB/s | 90% |
| `torch.sum` / device copy (what `bw.py` reports as "practical peak") | 750 / 701 GB/s | 79% / 74% |
| All target matmuls, 1 row (plain decode) | 663 GB/s | 70% |
| All target matmuls, 8 rows (speculative verification) | 446-460 GB/s | ~48% |
| Whole round: ~13.6 GB of weights (target 11.6, draft 1.1, draft-side lm_head 0.95) per ~34.7 ms | ~390 GB/s | ~41% |

A full pass over the 401 target matmuls, by rows:

| Rows | Pass | Per row | Effective bandwidth |
|---|---|---|---|
| 1 | 17.5 ms | 17.5 ms | 663 GB/s |
| 2 | 18.5 ms | 9.2 ms | 628 GB/s |
| 4 | 21.1 ms | 5.3 ms | 551 GB/s |
| 8 | 26.0 ms | 3.3 ms | 446 GB/s |
| 16 | 38.8 ms | 2.4 ms | 299 GB/s |

Up to two rows the matmul streams at close to what its access pattern allows; beyond that it is VALU-bound.
Isolating the parts of the 8-row 4-bit kernel with the `kbench` experiment switches (gate/up shape,
k 5120, n 17408):

| Variant | Time |
|---|---|
| Full kernel | 85.3 us |
| x from registers instead of LDS | 78.6 us (LDS reads ~7 us) |
| No LDS, no decode | 62.5 us (memory floor for this kernel) |
| No LDS, MAC over one row only | 62.6 us |
| Single-row kernel | 56.6 us |

The decode alone and the 8-row MACs alone each fit under the memory time; together they need ~60 VALU
issue cycles per 16x16 tile slice per wave (41 decode, ~19.5 for 32 `v_dot2` of which 26 dual-issue) and
the SIMDs reach only ~57% VALU utilization, so the kernel ends ~35% above its memory floor. A purely
bandwidth-bound round would take ~16 ms (13.6 GB at ~856 GB/s) against ~34.7 ms now: ~8 ms is the 8-row
decode compute, ~3.4 ms dispatch gaps, ~6 ms non-matmul kernels.

#### Why the 8-row kernel is not faster

Measured on the gate/up shape (4-bit, k 5120, n 17408, 8 rows) with `rocm_tests/kbench.cc` (numbers include
the ~6 us input-transform launch and gap of the bench loop). The shader clock under this load is ~2.5 GHz
(`rocm-smi` during a 60k-iteration run), so the 61 VALU issue cycles per tile slice amount to 44 us of pure
VALU over the 192 SIMDs:

| Variant | Time | Meaning |
|---|---|---|
| Full kernel | 84.9 us | |
| No weight loads (words perturbed per slice), x from LDS | 71.3 us | the memory stream costs ~13.5 us on top of compute |
| No weight loads, x from registers | 61.9 us | compute only: ~56 us kernel for 44 us of VALU |
| Weights streamed, no decode, MAC over one row (x from registers) | 62.5 us | memory only: ~54 us kernel = ~830 GB/s |
| 3-bit instead of 4-bit (25% fewer bytes) | 81.5 us | only 4% faster: not bandwidth-bound |

Marginal cost of 16 extra independent instructions per slice: `v_xor`, `v_mul_u32_u24`, `v_sad_u8`,
`v_sad_hi_u8`, `v_bfe_u32`, `v_lshl_add_u32`, `v_add_nc_u32`, `v_mov_b16` each +9.1-10.5 us (~0.9 cycles),
two-source ops +13.7-14.2 us; 8 extra SALU ops ~+2 us (co-issued); each extra LDS read ~+1.7 us. So no
instruction in the decode is unexpectedly slow, the SIMDs are close to issue-saturated inside the loop, and
the kernel's 79 us (without the bench overhead) sits between the ~56 us each of compute-only and memory-only
and the 110 us of running them back to back: memory and compute are both near their limits and only
partly overlap. Staging bubbles are small (skipping x re-staging after the first chunk: -2 us).

#### Direct tile-word loads (5 / 6-bit)

Each lane decodes eight 16-bit windows of a 16x16 tile and needs only 2 (up to 4 bits) or 4 (5 / 6 bits)
of the tile's words, at lane-constant indices. The kernel used to load one word per lane and gather the
needed ones with `ds_bpermute`; tiles over 32 words (5 / 6 bits) need two gathers plus a select per word:
8 `ds_bpermute`, 4 `v_cndmask` and extra `s_waitcnt` per k-slice. Now each lane loads its words directly
(the wave still reads the same tile bytes, coalesced; the word indices come from `windows8` itself), which
is bit-identical (`test_rdna3_gemm.py`: 0 failures; greedy output and logits identical to the old build):

| Kernel, 8 rows | Gather | Direct |
|---|---|---|
| 5-bit 5120 x 17408 | 113.9 us | 87.3 us (-23%) |
| 5-bit 5120 x 12288 | 81.3 us | 59.6 us (-27%) |
| 6-bit lm_head 5120 x 248320 | 1599 us (596 GB/s) | 1390 us (686 GB/s, -13%) |
| 3-bit 5120 x 17408 | 83.4 us | 81.3 us (-2.5%) |
| 4-bit 5120 x 17408 | 85.4 us | 84.9 us |

4-bit single-row matmuls keep the gather path (direct was ~2% slower there); `EXL3_RDNA3_NO_DIRECT`
(compile-time) restores the old path. Round time 35.9 -> 34.7 ms (-3.5%), draft layers 455 -> 561 GB/s.

### Megakernel and host-side round

Measured in-process with `rocm_tests/bench_ab.py` (greedy, variants interleaved per prompt, median
ms per speculative round): **38.0 -> 35.9 ms per round (-5.5%) with the abliterated (runtime-ablation)
model, 36.7 -> 35.9 ms (-2%) with the plain model**, which never ran the ablation kernel. All kernel changes are bit-identical to the
kernels they replace (`rocm_tests/test_gdn_mk.py`, `test_presample.py`).

| Change | Effect | Switch |
|---|---|---|
| Runtime ablation sidecar: float4 kernel, ablation folded into the residual RMSNorm | ~-1.4 ms (heretic models only) | `EXL3_ABL_FUSE=0` |
| Batched verification sampling (one launch and one sync for the whole window), stream-ordered draft readback, memoized recurrent rewind jobs | ~-0.3 ms | `EXL3_PRESAMPLE=0`, `EXL3_DRAFT_NB=0`, `EXL3_REWIND_CACHE=0` |
| `gdn_core_mk`: b/a GEMV + conv update, grid barrier, recurrence in one launch (was four kernels per DeltaNet layer); the last of each head's four blocks also writes o_proj's gated-norm input transform | ~-0.45 ms | `EXL3_GDN_MK=0` |
| Gated MLP: the gate/up matmul's second block per column group writes the down projection's input transform of silu(gate) * up | ~-0.15 ms | `EXL3_ACT_EPI=0` |

Kernels per round went from ~1085 to ~718. Notes from this round:

- torch.profiler under-reports long kernels on ROCm: the ~270 us "gap" after every lm_head matmul is
  the lm_head itself (~1.55 ms, event-timed). Most of the apparent host turnaround was this artifact;
  judge changes with wall-clock A/B, not the profiler's idle time.
- A grid barrier inside a persistent kernel costs ~0.4-2.5 us against ~3-6 us for a kernel boundary,
  but `hipLaunchCooperativeKernel` adds 10-20 us per launch; the megakernel uses a normal launch with
  an atomic epoch barrier (its grid is far below one residency wave).
- Epilogues that need no barrier use a per-tile counter: the last block to finish a unit does the
  dependent work (no deadlock risk, no co-residency assumption).

### ROCm 10

ROCm 10.0 (TheRock packaging, `stable.repo.amd.com`, torch `2.13.0+rocm10.0.0` wheels) builds and runs
this port with two fixes: the Triton-kernel launcher now takes the already-loaded HIP runtime
(`dlopen` of the bare `libamdhip64.so` picked the system ROCm next to the pip SDK and crashed), and
`OMP_NUM_THREADS` must be set (the wheel defaults to one OpenMP thread per logical CPU, which made
the small CPU ops around each forward take ~17 ms). With those, a round takes the same time as on
7.2.4 (35.9 vs 36.0 ms) and the per-kernel dispatch cost is unchanged (2.94 us).

### Open ideas (not done)

Estimated gains are per speculative round (~36 ms); each is small, which is why they were left out.

| Idea | Estimated gain | Notes |
|---|---|---|
| Residual RMSNorm + input transform as a prologue of the next matmul | ~0.3-0.5 ms (-128 kernels) | GEMM blocks may only wait on lower block ids; needs a pending-norm hand-off with a flush for non-RDNA3 consumers of the norm output |
| Lower-bit copy of lm_head for the draft only | up to ~0.8 ms | the draft needs only the top-k per row over the full vocabulary; a 3-bit head (~0.45 GB instead of 0.95 GB) keeps the ordering mostly; verification stays exact, the cost is some acceptance. Needs requantizing the head (ROCm quantization kernels untested) |
| Better memory / compute overlap in the 8-row matmul | up to ~20% of the verification matmuls (~5 ms) at perfect overlap, realistically less | compute-only and memory-only are each ~56 us for the gate/up shape against 79 us combined (see "Why the 8-row kernel is not faster"); the next structural step would be loader waves streaming weights into LDS so compute waves never wait on memory, at the price of an LDS write + read per word (an LDS read costs about 2 VALU cycles here) |
| 4-bit single-row path at ~790 GB/s | ~0.5 ms for plain decode only | the access pattern allows ~856 GB/s (`micro/stridebw.cc`) |
| Per-layer megakernel including the EXL3 matmuls | up to ~1.5 ms | `exl3_rdna3_unit` is already a device function; needs its LDS in one union and 256- vs 128-thread stages reconciled, at the risk of slowing the matmuls |
| Parallelize the RMSNorm input-transform tail | ~0.3 ms | `rms_norm_had` runs one block per row (8 blocks at 8 rows) and got ~2 us slower per call |
| Attention: fuse RoPE + paged KV update + q/g deinterleave | ~0.1-0.15 ms (-32..48 kernels) | 16 attention layers |
| RMSNorm input-transform hand-off for the attention qkv bundle | ~0.05 ms | the attention graph is still captured; the graph would need a variant without its input kernel |
| Cheaper draft lm_head | up to ~1.3 ms | DFlash2 runs the full 6-bit lm_head (950 MB) over all 8 block rows every round; a pruned head lost acceptance, a dedicated small head would need training |
| DeltaNet history in 16-bit | ~0.4 ms | halves the rollback-history writes; risks accuracy of the carried state |
| MoE models: routed multi-matrix matmul on RDNA3 | functionality | expert routing (indices / weights) is not ported; MoE models fall back to the upstream path |

## Notes on published RTX 3090 / Arc B70 numbers

The ~144 tok/s Qwen3.8-27B figure on an RTX 3090 comes from SGLang with the `sglang-exl3` plugin and
NEXTN (MTP) drafting on the 3.0 bpw quant (prose ~99, code ~143 tok/s), not from exllamav3 + DFlash2;
TabbyAPI/exllamav3 runs in the same registry reach 58-69 tok/s on a 3090. The Arc B70 figure of 60 tok/s
is single-stream at 4 bpw (330 tok/s is the 16-stream aggregate).

## License and credits

- exllamav3 by turboderp and contributors, MIT License (see [LICENSE](LICENSE)); this fork keeps it.
- TabbyAPI (AGPL-3.0) is not included; `rocm/tabbyapi/` only contains a patch and config files.
- Models by the Qwen team, EXL3 quants and DFlash2 draft by Mia-AiLab (see their model cards for licenses).
