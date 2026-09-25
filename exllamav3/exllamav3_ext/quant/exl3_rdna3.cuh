#pragma once

// EXL3 small-m matmul for AMD RDNA3 (gfx11, wave32), used in place of the NVIDIA GEMV/GEMM
// kernels (mma.sync, cp.async, cooperative launch) for m <= AUTO_RECONSTRUCT_THRESHOLD.
//
// One launch per matmul, same graph-patchable argument positions as exl3_gemm_kernel (A = 0,
// B = 1, C = 2):
//
// - each block owns 128 output columns (8 n-tiles, one wave per tile) and a k-range (split)
// - a small kernel applies suh and the 128-point Hadamard to x once, in the main kernel's LDS layout
// - each block copies its x rows into LDS
// - trellis words stream straight to registers (coalesced, prefetch ring), are staged through
//   wave-private LDS and decoded with the standard dq_dispatch; products accumulate in fp32 via
//   v_dot2_f32_f16
// - k-splits reduce through a fp32 workspace; the last block per column group (atomic counter)
//   sums the partials, applies the output Hadamard and svh and writes C

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define EXL3_RDNA3_NT 8
#ifndef EXL3_RDNA3_TPW
#define EXL3_RDNA3_TPW 1        // n-tiles per wave
#endif
#define EXL3_RDNA3_THREADS (EXL3_RDNA3_NT / EXL3_RDNA3_TPW * 32)
#define EXL3_RDNA3_WS_BYTES (64ull << 20)
#define EXL3_RDNA3_MAX_COUNTERS (1 << 16)
#define EXL3_RDNA3_NUM_MR 9   // rows per pass: 1, 2, 4, 8, 16, 3, 5, 6, 12 (mr_idx order)

#define EXL3_RDNA3_XH_BYTES (16ull << 20)
#define EXL3_RDNA3_XCS_FLOATS (1 << 18)

// Multi-matrix mode (blockIdx.z = entry): per-entry trellis / svh / output pointers, output row
// stride and input source. A null b table selects plain single-matrix mode
struct Exl3Rdna3MTab
{
    const uint64_t* b;         // trellis pointers (pre-offset to the entry's first column)
    const uint64_t* svh;       // svh pointers (pre-offset)
    const uint64_t* c;         // output pointers, or null: C + e * size_m * size_n
    const int* n_stride;       // row stride of trellis and output, or null: size_n
    const int* src;            // input source (xh / xcs slab), or null: e
    // Optional gated-MLP epilogue (two entries gate, up with fp16 outputs): the second block to finish a
    // column group writes silu(gate) * up of that group, transformed as the down projection's input
    const half* act_g;         // null: no epilogue
    const half* act_u;
    const half* act_suh;       // down projection suh
    uint2* act_xh;
    float* act_xcs;
    int* act_cnt;              // per (row chunk, group), zero between launches
    int act_k;                 // gate/up width = down input width
};

// Gated RMSNorm folded into the input transform (GatedDeltaNet output -> o_proj): each 128-element
// input block is one head, y = norm(x) * (w + bias) * act(g), matching gated_rms_norm's arithmetic.
// Input x is bf16; flags: GN_ACTIVE, GN_W_BF16 (else fp32 weight), GN_G_BF16 (else fp32 gate), GN_SIGMOID
// GN_UP_SIGMOID: with A_up, the input is A * sigmoid(A_up) (attention output gate) instead of silu(A) * A_up
enum { GN_ACTIVE = 1, GN_W_BF16 = 2, GN_G_BF16 = 4, GN_SIGMOID = 8, GN_UP_SIGMOID = 16 };
struct Exl3Rdna3GNorm
{
    const void* w;
    const void* g;
    float eps;
    float bias;
    int flags;
};

typedef void (*fp_exl3_rdna3_kernel)
(
    const uint2* __restrict__ xh,
    const uint32_t* __restrict__ B,
    void* __restrict__ C,
    int size_m,
    int size_k,
    int size_n,
    int* __restrict__ counters,
    const float* __restrict__ xcs,
    float* __restrict__ ws,
    const half* __restrict__ svh,
    int splits,
    int ks_per_split,
    Exl3Rdna3MTab mt
);

fp_exl3_rdna3_kernel exl3_rdna3_get_k1(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k2(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k3(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k4(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k5(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k6(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k7(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k8(bool half_k, int cb, int mr_idx, bool c_fp32);

class Graph;

__global__ void exl3_rdna3_had_kernel
(
    const half* __restrict__ A,
    const half* __restrict__ suh,
    uint2* __restrict__ xh,
    float* __restrict__ xcs,
    int size_m,
    int size_k,
    const uint64_t* __restrict__ suh_tab,
    const half* __restrict__ A_up,
    Exl3Rdna3GNorm gn
);

// Returns true if the matmul was launched
bool exl3_rdna3_gemm
(
    const half* A,
    const uint16_t* B,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    const half* suh,
    const half* svh,
    int device,
    cudaStream_t stream,
    Graph* graph,
    const half* A_up = nullptr,  // gated MLP down projection: input is silu(A) * A_up
    const Exl3Rdna3GNorm* gn = nullptr   // gated RMSNorm prologue (A is then bf16)
);

// Several matrices sharing one input (MultiLinear / SlicedMultiLinear tables), one launch pair
bool exl3_rdna3_mgemm
(
    const half* A,
    const uint64_t* b_tab,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    const uint64_t* suh_tab,
    const uint64_t* svh_tab,
    const uint64_t* c_tab,
    const int* n_stride_tab,
    const int* src_tab,
    int num_entries,
    int num_src,
    int device,
    cudaStream_t stream,
    Graph* graph
);

// Producers that write the transformed input themselves (rms_norm_had): returns the workspace and
// records it as prepared for (A, suh_tab, m, k, num_src); the next exl3_rdna3_mgemm with exactly that
// input skips its input kernel (eager launches only), any other RDNA3 matmul drops the record
bool exl3_rdna3_prepare_input(int device, const void* A, const void* suh_tab, int m, int k, int num_src, uint2** xh, float** xcs);

// Gated MLP: arm the silu(gate) * up input-transform epilogue for the next exl3_rdna3_mgemm (gate/up pair)
// on this device; the matching down projection (exl3_rdna3_gemm on the gate output with A_up) then skips its
// input kernel. Disarm after the mgemm call, whether or not it consumed the arming
void exl3_rdna3_arm_act_epilogue(int device, const void* down_suh);
void exl3_rdna3_disarm_act_epilogue(int device);

void exl3_rdna3_prepare(int device);
bool exl3_rdna3_enabled();
