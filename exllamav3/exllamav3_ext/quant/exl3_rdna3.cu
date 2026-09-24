#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <mutex>
#include "../util.h"
#include "../util.cuh"
#include "../graph.cuh"
#include "exl3_devctx.cuh"
#include "exl3_rdna3.cuh"

namespace
{
    std::mutex g_mtx;
    float* g_ws[MAX_DEVICES] = {};
    int* g_counters[MAX_DEVICES] = {};
    uint2* g_xh[MAX_DEVICES] = {};
    float* g_xcs[MAX_DEVICES] = {};
    int g_enabled = -1;
    int g_target_blocks = -1;
}

bool exl3_rdna3_enabled()
{
    #ifdef __HIP_PLATFORM_AMD__
        if (g_enabled < 0)
        {
            const char* e = std::getenv("EXL3_RDNA3_GEMM");
            g_enabled = !(e && e[0] == '0');
        }
        return g_enabled;
    #else
        return false;
    #endif
}

// Workspace and counters are allocated once per device, outside of any graph capture (prepare_ctx
// runs at model load), and never move, so captured graphs keep valid pointers
void exl3_rdna3_prepare(int device)
{
    #ifdef __HIP_PLATFORM_AMD__
        std::lock_guard<std::mutex> lock(g_mtx);
        if (g_ws[device]) return;
        c10::cuda::CUDAGuard guard(device);
        cuda_check(cudaMalloc(&g_ws[device], EXL3_RDNA3_WS_BYTES));
        cuda_check(cudaMalloc(&g_counters[device], EXL3_RDNA3_MAX_COUNTERS * sizeof(int)));
        cuda_check(cudaMemset(g_counters[device], 0, EXL3_RDNA3_MAX_COUNTERS * sizeof(int)));
        cuda_check(cudaMalloc(&g_xh[device], EXL3_RDNA3_XH_BYTES));
        cuda_check(cudaMalloc(&g_xcs[device], EXL3_RDNA3_XCS_FLOATS * sizeof(float)));
        cuda_check(cudaDeviceSynchronize());
    #endif
}

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
)
{
    #ifdef __HIP_PLATFORM_AMD__
        if (!exl3_rdna3_enabled()) return false;
        if (!suh || !svh) return false;
        if (size_k % 128 || size_n % 128) return false;
        if (K < 1 || K > 8) return false;

        if (!g_ws[device]) exl3_rdna3_prepare(device);

        // Rows per pass: smallest instantiated MR >= m (MR 16 with multiple passes above that)
        int mr_idx, mr;
        if      (size_m <= 1)  { mr_idx = 0; mr = 1; }
        else if (size_m <= 2)  { mr_idx = 1; mr = 2; }
        else if (size_m <= 3)  { mr_idx = 5; mr = 3; }
        else if (size_m <= 4)  { mr_idx = 2; mr = 4; }
        else if (size_m <= 5)  { mr_idx = 6; mr = 5; }
        else if (size_m <= 6)  { mr_idx = 7; mr = 6; }
        else if (size_m <= 8)  { mr_idx = 3; mr = 8; }
        else if (size_m <= 12) { mr_idx = 8; mr = 12; }
        else                   { mr_idx = 4; mr = 16; }

        fp_exl3_rdna3_kernel kernel = nullptr;
        switch (K)
        {
            case 1: kernel = exl3_rdna3_get_k1(half_k, cb, mr_idx, c_fp32); break;
            case 2: kernel = exl3_rdna3_get_k2(half_k, cb, mr_idx, c_fp32); break;
            case 3: kernel = exl3_rdna3_get_k3(half_k, cb, mr_idx, c_fp32); break;
            case 4: kernel = exl3_rdna3_get_k4(half_k, cb, mr_idx, c_fp32); break;
            case 5: kernel = exl3_rdna3_get_k5(half_k, cb, mr_idx, c_fp32); break;
            case 6: kernel = exl3_rdna3_get_k6(half_k, cb, mr_idx, c_fp32); break;
            case 7: kernel = exl3_rdna3_get_k7(half_k, cb, mr_idx, c_fp32); break;
            case 8: kernel = exl3_rdna3_get_k8(half_k, cb, mr_idx, c_fp32); break;
        }
        if (!kernel) return false;

        const int groups = size_n / 128;
        const int kblocks = size_k / 128;

        if (g_target_blocks < 0)
        {
            const char* e = std::getenv("EXL3_RDNA3_TARGET_BLOCKS");
            g_target_blocks = e ? atoi(e) : 4 * DevCtx::instance().get_num_sms(device);
        }

        // Rows per launch pair, bounded by the transformed-input workspace
        const int max_rows = (int) MIN((size_t) EXL3_RDNA3_XH_BYTES / ((size_t) size_k * 2), (size_t) EXL3_RDNA3_XCS_FLOATS / kblocks);
        TORCH_CHECK(max_rows >= 1, "exl3_rdna3_gemm: k too large for the input workspace");

        const uint32_t* B32 = (const uint32_t*) B;
        int* counters = g_counters[device];
        float* ws = g_ws[device];
        uint2* xh = g_xh[device];
        float* xcs = g_xcs[device];
        const size_t c_elem = c_fp32 ? 4 : 2;

        for (int r0 = 0; r0 < size_m; r0 += max_rows)
        {
            const int m = MIN(max_rows, size_m - r0);
            const half* A_r = A + (size_t) r0 * size_k;
            void* C_r = (void*) ((char*) C + (size_t) r0 * size_n * c_elem);
            const int row_chunks = (m + mr - 1) / mr;

            // k-split so the grid covers the device; splits are whole 128-element Hadamard blocks
            int splits = (g_target_blocks + groups * row_chunks - 1) / (groups * row_chunks);
            splits = MAX(1, MIN(splits, kblocks));
            while (splits > 1 && (size_t) splits * m * size_n * sizeof(float) > EXL3_RDNA3_WS_BYTES) splits--;
            if (splits > 1 && groups * row_chunks > EXL3_RDNA3_MAX_COUNTERS) splits = 1;
            int kb_per_split = (kblocks + splits - 1) / splits;
            splits = (kblocks + kb_per_split - 1) / kb_per_split;
            int ks_per_split = kb_per_split * 8;

            const int had_tasks = m * kblocks;
            exl3_rdna3_had_kernel<<<(had_tasks + 7) / 8, 256, 0, stream>>>(A_r, suh, xh, xcs, m, size_k);
            kernel<<<dim3(groups * splits, row_chunks), EXL3_RDNA3_THREADS, 0, stream>>>
            (
                xh, B32, C_r, m, size_k, size_n, counters, xcs, ws, svh, splits, ks_per_split
            );

            // Graph patching: callers bind the input (A) and output (C) pointers; the transformed
            // input, workspace and counters are static per device
            if (graph && r0 == 0)
            {
                graph->record_param((void*) exl3_rdna3_had_kernel, GP_gemm_A, 0);
                graph->record_param((void*) exl3_rdna3_had_kernel, GP_end, 0);
                graph->record_param((void*) kernel, GP_gemm_C, 2);
                graph->record_param((void*) kernel, GP_end, 0);
            }
            TORCH_CHECK(!graph || size_m <= max_rows, "exl3_rdna3_gemm: graphed matmul exceeds the input workspace");
        }
        cuda_check(cudaPeekAtLastError());
        return true;
    #else
        return false;
    #endif
}
