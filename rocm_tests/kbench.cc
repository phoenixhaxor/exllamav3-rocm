#include <algorithm>
// Standalone timing harness for exl3_rdna3_kernel (random trellis, correctness not checked)
// build: hipcc -O3 --offload-arch=gfx1100 -I<ext>/quant -DKB_BITS=4 -DKB_MR=8 kbench.hip
#include <hip/hip_runtime.h>
#define EXL3_RDNA3_DEFINE_HAD
#include "exl3_rdna3_kernel_hip.cuh"
#include <cstdio>
#include <vector>
#include <cstdlib>

#ifndef KB_BITS
#define KB_BITS 4
#endif
#ifndef KB_MR
#define KB_MR 8
#endif

int main(int argc, char** argv)
{
    int k = argc > 1 ? atoi(argv[1]) : 5120;
    int n = argc > 2 ? atoi(argv[2]) : 17408;
    int m = argc > 3 ? atoi(argv[3]) : KB_MR;
    int target = argc > 4 ? atoi(argv[4]) : 384;
    const int bits = KB_BITS;
    size_t tw = (size_t) bits * 8;
    size_t n_words = (size_t) (k / 16) * (n / 16) * tw;
    uint32_t* B; half* A; half* C; half* suh; half* svh; float* ws; int* counters; uint2* xh; float* xcs;
    hipMalloc(&B, n_words * 4); hipMalloc(&A, (size_t) m * k * 2); hipMalloc(&C, (size_t) m * n * 4);
    hipMalloc(&suh, k * 2); hipMalloc(&svh, n * 2); hipMalloc(&ws, 64 << 20); hipMalloc(&counters, 1 << 18); hipMalloc(&xh, 16 << 20); hipMalloc(&xcs, 1 << 20);
    hipMemset(counters, 0, 1 << 18);
    std::vector<uint32_t> hb(n_words); for (auto& x : hb) x = rand() * 2654435761u; hipMemcpy(B, hb.data(), n_words * 4, hipMemcpyHostToDevice);
    std::vector<uint16_t> ha((size_t) m * k, 0x3c00); hipMemcpy(A, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
    std::vector<uint16_t> hs(std::max(k, n), 0x3c00); hipMemcpy(suh, hs.data(), k * 2, hipMemcpyHostToDevice); hipMemcpy(svh, hs.data(), n * 2, hipMemcpyHostToDevice);

    int groups = n / 128, kblocks = k / 128, row_chunks = (m + KB_MR - 1) / KB_MR;
    int splits = std::max(1, std::min(kblocks, (target + groups * row_chunks - 1) / (groups * row_chunks)));
    int kbps = (kblocks + splits - 1) / splits; splits = (kblocks + kbps - 1) / kbps;
    auto kern = exl3_rdna3_kernel<bits, false, 2, KB_MR, false>;
    auto launch = [&] {
        exl3_rdna3_had_kernel<<<(m * kblocks + 7) / 8, 256, 0, 0>>>(A, suh, xh, xcs, m, k, nullptr, nullptr);
        kern<<<dim3(groups * splits, row_chunks), EXL3_RDNA3_THREADS, 0, 0>>>(xh, B, C, m, k, n, counters, xcs, ws, svh, splits, kbps * 8, Exl3Rdna3MTab {});
    };
    for (int i = 0; i < 5; ++i) launch();
    hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    int it = 200;
    hipEventRecord(e0); for (int i = 0; i < it; ++i) launch(); hipEventRecord(e1); hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms, e0, e1);
    double us = ms * 1000 / it;
    printf("bits=%d MR=%d m=%d k=%d n=%d splits=%d blocks=%d: %.1f us, %.1f GB/s\n", bits, KB_MR, m, k, n, splits, groups * splits * row_chunks, us, n_words * 4 / us / 1e3);
#ifdef KB_TRACE
    {
        // One isolated launch: block timeline relative to the earliest start (us, 100 MHz ticks)
        hipDeviceSynchronize();
        launch();
        hipDeviceSynchronize();
        int nb = groups * splits * row_chunks;
        std::vector<unsigned long long> t(3 * 8192);
        hipMemcpyFromSymbol(t.data(), HIP_SYMBOL(kb_trace), t.size() * 8);
        unsigned long long t0 = ~0ull, tend = 0;
        for (int b = 0; b < nb; ++b) { t0 = std::min(t0, t[3 * b]); tend = std::max(tend, t[3 * b + 2]); }
        auto us_ = [&] (unsigned long long v) { return (v - t0) / 100.0; };
        std::vector<double> st, lp, en;
        for (int b = 0; b < nb; ++b) { st.push_back(us_(t[3 * b])); lp.push_back((t[3 * b + 1] - t[3 * b]) / 100.0); en.push_back(us_(t[3 * b + 2])); }
        auto pct = [] (std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v[(size_t) (p * (v.size() - 1))]; };
        printf("  kernel span %.1f us | start p50 %.1f p90 %.1f max %.1f | loop p10 %.1f p50 %.1f p90 %.1f max %.1f | end p50 %.1f p90 %.1f max %.1f\n",
               us_(tend), pct(st, .5), pct(st, .9), pct(st, 1), pct(lp, .1), pct(lp, .5), pct(lp, .9), pct(lp, 1), pct(en, .5), pct(en, .9), pct(en, 1));
        int late = 0; for (double v : st) late += v > 2.0;
        printf("  blocks starting > 2 us late: %d of %d\n", late, nb);
    }
#endif
}
