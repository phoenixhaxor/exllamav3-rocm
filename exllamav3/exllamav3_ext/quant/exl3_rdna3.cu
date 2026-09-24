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
    struct Prepared { const void* A; const void* suh_tab; int m, k, num_src; };
    Prepared g_prepared[MAX_DEVICES] = {};
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

#ifdef __HIP_PLATFORM_AMD__
// Rows per pass: smallest instantiated MR >= m (MR 16 with multiple passes above that)
static fp_exl3_rdna3_kernel select_kernel(int size_m, int K, bool half_k, int cb, bool c_fp32, int& mr)
{
    int mr_idx;
    if      (size_m <= 1)  { mr_idx = 0; mr = 1; }
    else if (size_m <= 2)  { mr_idx = 1; mr = 2; }
    else if (size_m <= 3)  { mr_idx = 5; mr = 3; }
    else if (size_m <= 4)  { mr_idx = 2; mr = 4; }
    else if (size_m <= 5)  { mr_idx = 6; mr = 5; }
    else if (size_m <= 6)  { mr_idx = 7; mr = 6; }
    else if (size_m <= 8)  { mr_idx = 3; mr = 8; }
    else if (size_m <= 12) { mr_idx = 8; mr = 12; }
    else                   { mr_idx = 4; mr = 16; }

    switch (K)
    {
        case 1: return exl3_rdna3_get_k1(half_k, cb, mr_idx, c_fp32);
        case 2: return exl3_rdna3_get_k2(half_k, cb, mr_idx, c_fp32);
        case 3: return exl3_rdna3_get_k3(half_k, cb, mr_idx, c_fp32);
        case 4: return exl3_rdna3_get_k4(half_k, cb, mr_idx, c_fp32);
        case 5: return exl3_rdna3_get_k5(half_k, cb, mr_idx, c_fp32);
        case 6: return exl3_rdna3_get_k6(half_k, cb, mr_idx, c_fp32);
        case 7: return exl3_rdna3_get_k7(half_k, cb, mr_idx, c_fp32);
        case 8: return exl3_rdna3_get_k8(half_k, cb, mr_idx, c_fp32);
    }
    return nullptr;
}

static int target_blocks(int device)
{
    if (g_target_blocks < 0)
    {
        const char* e = std::getenv("EXL3_RDNA3_TARGET_BLOCKS");
        // Default: one full residency wave. The multiprocessor count is WGPs on gfx11 (48 on a
        // 7900 XTX) and LDS allows 6 blocks per WGP; a partial second wave roughly doubles the tail
        g_target_blocks = e ? atoi(e) : 6 * DevCtx::instance().get_num_sms(device);
    }
    return g_target_blocks;
}

// k-split so the grid (items independent block columns) covers the device; splits are whole
// 128-element Hadamard blocks
static void choose_splits(int items, int kblocks, size_t ws_per_split, int device, int& splits, int& ks_per_split)
{
    const int tb = target_blocks(device);
    splits = (tb + items - 1) / items;
    splits = MAX(1, MIN(splits, kblocks));
    while (splits > 1 && (size_t) splits * ws_per_split > EXL3_RDNA3_WS_BYTES) splits--;
    if (splits > 1 && items > EXL3_RDNA3_MAX_COUNTERS) splits = 1;
    int kb_per_split = (kblocks + splits - 1) / splits;
    splits = (kblocks + kb_per_split - 1) / kb_per_split;
    ks_per_split = kb_per_split * 8;
}

#endif

bool exl3_rdna3_prepare_input(int device, const void* A, const void* suh_tab, int m, int k, int num_src, uint2** xh, float** xcs)
{
    #ifdef __HIP_PLATFORM_AMD__
        if (!exl3_rdna3_enabled()) return false;
        static const bool fuse = !(std::getenv("EXL3_FUSE_NORM_HAD") && std::getenv("EXL3_FUSE_NORM_HAD")[0] == '0');
        if (!fuse) return false;
        if (k % 128) return false;
        if ((size_t) num_src * m * k * 2 > EXL3_RDNA3_XH_BYTES) return false;
        if ((size_t) num_src * m * (k / 128) > EXL3_RDNA3_XCS_FLOATS) return false;
        if (!g_ws[device]) exl3_rdna3_prepare(device);
        g_prepared[device] = { A, suh_tab, m, k, num_src };
        *xh = g_xh[device];
        *xcs = g_xcs[device];
        return true;
    #else
        return false;
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
    Graph* graph,
    const half* A_up,
    const Exl3Rdna3GNorm* gn
)
{
    #ifdef __HIP_PLATFORM_AMD__
        if (!exl3_rdna3_enabled()) return false;
        g_prepared[device].A = nullptr;   // this launch overwrites the input workspace
        if (!suh || !svh) return false;
        if (size_k % 128 || size_n % 128) return false;
        if (K < 1 || K > 8) return false;

        if (!g_ws[device]) exl3_rdna3_prepare(device);

        int mr;
        fp_exl3_rdna3_kernel kernel = select_kernel(size_m, K, half_k, cb, c_fp32, mr);
        if (!kernel) return false;

        const int groups = size_n / 128;
        const int kblocks = size_k / 128;


        // Rows per launch pair, bounded by the transformed-input workspace
        const int max_rows = (int) MIN((size_t) EXL3_RDNA3_XH_BYTES / ((size_t) size_k * 2), (size_t) EXL3_RDNA3_XCS_FLOATS / kblocks);
        TORCH_CHECK(max_rows >= 1, "exl3_rdna3_gemm: k too large for the input workspace");
        if (gn && size_m > max_rows) return false;   // the gate pointer is not re-based per row chunk

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

            int splits, ks_per_split;
            choose_splits(groups * row_chunks, kblocks, (size_t) m * size_n * sizeof(float), device, splits, ks_per_split);

            const int had_tasks = m * kblocks;
            exl3_rdna3_had_kernel<<<(had_tasks + 7) / 8, 256, 0, stream>>>(A_r, suh, xh, xcs, m, size_k, nullptr, A_up ? A_up + (size_t) r0 * size_k : nullptr,
                                                                           gn ? *gn : Exl3Rdna3GNorm {});
            kernel<<<dim3(groups * splits, row_chunks), EXL3_RDNA3_THREADS, 0, stream>>>
            (
                xh, B32, C_r, m, size_k, size_n, counters, xcs, ws, svh, splits, ks_per_split, Exl3Rdna3MTab {}
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
)
{
    #ifdef __HIP_PLATFORM_AMD__
        if (!exl3_rdna3_enabled()) return false;
        if (size_k % 128 || size_n % 128) return false;
        if (K < 1 || K > 8) return false;
        if (num_entries < 1 || num_src < 1 || size_m < 1) return false;

        if (!g_ws[device]) exl3_rdna3_prepare(device);

        int mr;
        fp_exl3_rdna3_kernel kernel = select_kernel(size_m, K, half_k, cb, c_fp32, mr);
        if (!kernel) return false;

        // Single pass only: every source's transformed input has to fit the workspace
        const int kblocks = size_k / 128;
        if ((size_t) num_src * size_m * size_k * 2 > EXL3_RDNA3_XH_BYTES) return false;
        if ((size_t) num_src * size_m * kblocks > EXL3_RDNA3_XCS_FLOATS) return false;

        const int groups = size_n / 128;
        const int row_chunks = (size_m + mr - 1) / mr;
        const int items = groups * row_chunks * num_entries;
        if (items > EXL3_RDNA3_MAX_COUNTERS) return false;

        int splits, ks_per_split;
        choose_splits(items, kblocks, (size_t) num_entries * size_m * size_n * sizeof(float), device, splits, ks_per_split);

        uint2* xh = g_xh[device];
        float* xcs = g_xcs[device];
        const Prepared& pr = g_prepared[device];
        const bool prepared = !graph && pr.A == (const void*) A && pr.suh_tab == (const void*) suh_tab &&
                              pr.m == size_m && pr.k == size_k && pr.num_src == num_src;
        g_prepared[device].A = nullptr;
        const int had_tasks = size_m * kblocks;
        if (!prepared)
            exl3_rdna3_had_kernel<<<dim3((had_tasks + 7) / 8, num_src), 256, 0, stream>>>(A, nullptr, xh, xcs, size_m, size_k, suh_tab, nullptr, Exl3Rdna3GNorm {});

        Exl3Rdna3MTab mt { b_tab, svh_tab, c_tab, n_stride_tab, src_tab };
        kernel<<<dim3(groups * splits, row_chunks, num_entries), EXL3_RDNA3_THREADS, 0, stream>>>
        (
            xh, nullptr, C, size_m, size_k, size_n, g_counters[device], xcs, g_ws[device], nullptr,
            splits, ks_per_split, mt
        );

        // Graph patching: the input (A) only; output tables and pointers are static
        if (graph)
        {
            graph->record_param((void*) exl3_rdna3_had_kernel, GP_mgemm_A, 0);
            graph->record_param((void*) exl3_rdna3_had_kernel, GP_end, 0);
            graph->record_param((void*) kernel, GP_mgemm_C, 2);
            graph->record_param((void*) kernel, GP_end, 0);
        }
        cuda_check(cudaPeekAtLastError());
        return true;
    #else
        return false;
    #endif
}
