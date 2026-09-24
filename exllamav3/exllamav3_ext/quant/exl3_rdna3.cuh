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
    int ks_per_split
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
    int size_k
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
    Graph* graph
);

void exl3_rdna3_prepare(int device);
bool exl3_rdna3_enabled();
