#pragma once

#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_rdna3.cuh"

#ifdef __HIP_PLATFORM_AMD__

// Bench-only experiment switches (rocm_tests/kbench.cc): 1 = MAC over row 0 only, 2 = skip decode
#ifndef KB_EXP
#define KB_EXP 0
#endif

namespace exl3_rdna3_ns {

constexpr int NT = EXL3_RDNA3_NT;
constexpr int THREADS = EXL3_RDNA3_THREADS;
#ifndef EXL3_RDNA3_XS_BYTES
#define EXL3_RDNA3_XS_BYTES 16384
#endif
constexpr int XS_BYTES = EXL3_RDNA3_XS_BYTES;   // LDS budget for the staged x chunk

typedef _Float16 hv2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint32_t h2u(half2 h)
{
    return *reinterpret_cast<uint32_t*>(&h);
}

// v_dot2_f32_f16: c + a.x * b.x + a.y * b.y, fp32 accumulate
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c)
{
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(hv2, a), __builtin_bit_cast(hv2, b), c, false);
}

__device__ __forceinline__ void wave_sync()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// Natural-order 128-point Walsh-Hadamard across one wave, lane holds elements 4 * lane .. + 3
__device__ __forceinline__ void had128(float& h0, float& h1, float& h2, float& h3, int lane)
{
    float s0 = h0 + h1, d0 = h0 - h1, s1 = h2 + h3, d1 = h2 - h3;
    h0 = s0 + s1; h1 = d0 + d1; h2 = s0 - s1; h3 = d0 - d1;
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        float p0 = __shfl_xor(h0, i);
        float p1 = __shfl_xor(h1, i);
        float p2 = __shfl_xor(h2, i);
        float p3 = __shfl_xor(h3, i);
        bool hi = lane & i;
        h0 = hi ? p0 - h0 : h0 + p0;
        h1 = hi ? p1 - h1 : h1 + p1;
        h2 = hi ? p2 - h2 : h2 + p2;
        h3 = hi ? p3 - h3 : h3 + p3;
    }
}

// 64-bit funnel: (a:b) >> s, low 32 bits, 0 <= s < 64
__device__ __forceinline__ uint32_t fsh(uint32_t b, uint32_t a, int s)
{
    return (uint32_t) ((((uint64_t) a << 32) | (uint64_t) b) >> s);
}

// (a:b) >> (s + T) for a runtime s < 32 and a constant T, without the quarter-rate 64-bit shift:
// y = low word of (a:b) >> s (v_alignbit), hi = a >> s, then constant shifts
struct Funnel
{
    uint32_t y, hi;
    __device__ __forceinline__ Funnel(uint32_t b, uint32_t a, int s) :
        y(__builtin_amdgcn_alignbit(a, b, s)), hi(a >> s) {}
    template <int T> __device__ __forceinline__ uint32_t at() const
    {
        if constexpr (T == 0) return y;
        else if constexpr (T >= 32) return hi >> (T - 32);
        else return (y >> T) | (hi << (32 - T));
    }
};

// mul1 codebook, raw form: fp16 bit pattern of 1024 + bytesum(idx * 0x83DCD12D) for two 16-bit windows,
// packed as half2. The affine map to the codebook value (k_inv * h + k_bias) is applied to the dot products
// instead (see the kernel epilogue), saving the per-weight fma
__device__ __forceinline__ uint32_t mul1_raw_pair(uint32_t x0, uint32_t x1)
{
    // v_sad_u8 against zero is a full-rate byte sum (v_dot4_u32_u8 is half rate on RDNA3); v_sad_hi_u8
    // adds the second sum into the high half, so the pair packs without a separate shift/or
#ifdef KB_NOSADHI
    uint32_t s0 = __builtin_amdgcn_sad_u8(mul1_mul(x0), 0u, 0x6400u);
    uint32_t s1 = __builtin_amdgcn_sad_u8(mul1_mul(x1), 0u, 0x6400u);
    return s0 | (s1 << 16);
#else
    const uint32_t s = __builtin_amdgcn_sad_u8(mul1_mul(x0), 0u, 0x64006400u);
    return __builtin_amdgcn_sad_hi_u8(mul1_mul(x1), 0u, s);
#endif
}

// Eight 16-bit windows (positions t_offset .. +7 of the tile, t_offset = 8 * lane) for integer bitrates,
// from register-resident tile words. gather(j) returns tile word j. Window order matches dq_dispatch:
// frag0 = (w0, w1), (w2, w3); frag1 = (w4, w5), (w6, w7)
template <int bits, typename G>
__device__ __forceinline__ void windows8(G gather, int lane, uint32_t* w)
{
    constexpr int W = bits * 256 / 32;
    const int t = lane << 3;
    if constexpr (bits == 2)
    {
        const int i1 = t >> 4;
        uint32_t a = gather((i1 + 15) & 15);
        uint32_t b = gather(i1);
        b = __builtin_amdgcn_alignbit(a, b, ((~t) & 8) << 1);
        #pragma unroll
        for (int j = 0; j < 8; ++j) w[7 - j] = (b >> (2 * j)) & 0xffff;
    }
    else if constexpr (bits == 3)
    {
        const int b1 = (t + 257) * 3;
        const int b2 = b1 + 21;
        const int i0 = (b1 - 16) / 32;
        const int i2 = (b2 - 1) / 32;
        const int s2 = (i2 + 1) * 32 - b2;
        uint32_t a = gather(i0 % W);
        uint32_t b = gather(i2 % W);
        const Funnel f(b, a, s2);
        uint32_t w7 = f.at<0>();
        uint32_t w3 = f.at<12>();
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            w[7 - j] = (w7 >> (3 * j)) & 0xffff;
            w[3 - j] = (w3 >> (3 * j)) & 0xffff;
        }
    }
    else if constexpr (bits == 4)
    {
        const int i1 = t >> 3;
        uint32_t a = gather((i1 + 31) & 31);
        uint32_t b = gather(i1);
        uint32_t s = fsh(b, a, 20);
        #pragma unroll
        for (int j = 0; j < 5; ++j) w[7 - j] = (b >> (4 * j)) & 0xffff;
        #pragma unroll
        for (int j = 0; j < 3; ++j) w[2 - j] = (s >> (4 * j)) & 0xffff;
    }
    else  // 5, 6: two groups of four windows (dq4)
    {
        #pragma unroll
        for (int g = 0; g < 2; ++g)
        {
            const int b0 = (t + 4 * g + 257) * bits - 16;
            const int b2 = b0 + 3 * bits + 16;
            const int i0 = b0 / 32;
            const int i2 = (b2 - 1) / 32;
            const int s2 = (i2 + 1) * 32 - b2;
            uint32_t a = gather(i0 % W);
            uint32_t b = gather(i2 % W);
            const Funnel f(b, a, s2);
            w[4 * g + 3] = f.at<0>() & 0xffff;
            w[4 * g + 2] = f.at<bits>() & 0xffff;
            w[4 * g + 1] = f.at<2 * bits>() & 0xffff;
            w[4 * g + 0] = f.at<3 * bits>() & 0xffff;
        }
    }
}

}  // namespace exl3_rdna3_ns

#ifdef EXL3_RDNA3_DEFINE_HAD   // defined in exactly one TU (exl3_rdna3_k1.cu)
// Input transform, once per matmul: x * suh -> 128-point Hadamard -> 1/sqrt(128) -> fp16, written in the
// main kernel's LDS pair layout ([m][k/16][4] uint2: .x = x[16s + 2q .. +1], .y = x[16s + 8 + 2q .. +1]),
// plus the per-128-block sums of the rounded values (codebook bias term of the raw mul1 decode)
__global__ __launch_bounds__(256)
void exl3_rdna3_had_kernel
(
    const half* __restrict__ A,
    const half* __restrict__ suh,
    uint2* __restrict__ xh,
    float* __restrict__ xcs,
    int size_m,
    int size_k,
    const uint64_t* __restrict__ suh_tab,  // multi-source: suh per blockIdx.y, one xh / xcs slab each
    const half* __restrict__ A_up,         // gated MLP: input is silu(A) * A_up (act_mul_kernel_h rounding)
    Exl3Rdna3GNorm gn                      // gated RMSNorm prologue, one head per 128-block (A is bf16)
)
{
    using namespace exl3_rdna3_ns;
    const int lane = threadIdx.x & 31;
    const int kblocks = size_k / 128;
    if (suh_tab)
    {
        suh = (const half*) suh_tab[blockIdx.y];
        xh += (size_t) blockIdx.y * size_m * (size_k / 16) * 4;
        xcs += (size_t) blockIdx.y * size_m * kblocks;
    }
    const int task = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (task >= size_m * kblocks) return;
    const int m = task / kblocks;
    const int c = task % kblocks;
    const int kbase = c * 128;

    const half2* ap = (const half2*) (A + (size_t) m * size_k + kbase + lane * 4);
    const half2* sp = (const half2*) (suh + kbase + lane * 4);
    half2 x01, x23;
    if (gn.flags & GN_ACTIVE)
    {
        // gated_rms_norm (small path, one warp per head): fma sum of squares, xor-reduce 16 .. 1,
        // x * w * rmf, then * act(g), rounded to fp16
        auto bf = [] (uint32_t v, int hi) { return __uint_as_float(hi ? (v & 0xffff0000u) : (v << 16)); };
        const size_t off = (size_t) m * size_k + kbase + lane * 4;
        const uint2 xr = *(const uint2*) (((const uint16_t*) A) + off);
        float f[4] = { bf(xr.x, 0), bf(xr.x, 1), bf(xr.y, 0), bf(xr.y, 1) };
        float ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) ss = fmaf(f[i], f[i], ss);
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1) ss += __shfl_xor(ss, i);
        const float rmf = rsqrtf(ss / 128.0f + gn.eps);
        float w[4], g[4];
        if (gn.flags & GN_W_BF16)
        {
            const uint2 wr = *(const uint2*) (((const uint16_t*) gn.w) + lane * 4);
            w[0] = bf(wr.x, 0); w[1] = bf(wr.x, 1); w[2] = bf(wr.y, 0); w[3] = bf(wr.y, 1);
        }
        else
        {
            const float4 wr = *(const float4*) (((const float*) gn.w) + lane * 4);
            w[0] = wr.x; w[1] = wr.y; w[2] = wr.z; w[3] = wr.w;
        }
        if (gn.flags & GN_G_BF16)
        {
            const uint2 gr = *(const uint2*) (((const uint16_t*) gn.g) + off);
            g[0] = bf(gr.x, 0); g[1] = bf(gr.x, 1); g[2] = bf(gr.y, 0); g[3] = bf(gr.y, 1);
        }
        else
        {
            const float4 gr = *(const float4*) (((const float*) gn.g) + off);
            g[0] = gr.x; g[1] = gr.y; g[2] = gr.z; g[3] = gr.w;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const float wi = gn.bias != 0.0f ? w[i] + gn.bias : w[i];
            float v = f[i] * wi * rmf;
            const float r = __fdividef(1.0f, 1.0f + __expf(-g[i]));
            v *= (gn.flags & GN_SIGMOID) ? r : g[i] * r;
            f[i] = v;
        }
        x01 = __floats2half2_rn(f[0], f[1]);
        x23 = __floats2half2_rn(f[2], f[3]);
    }
    else
    {
        x01 = ap[0];
        x23 = ap[1];
    }
    if (A_up)
    {
        const half2* up = (const half2*) (A_up + (size_t) m * size_k + kbase + lane * 4);
        auto sigmoid2 = [] (half2 x) -> half2
        {
            const half2 one = __float2half2_rn(1.0f);
            return h2rcp(__hadd2(one, h2exp(__hneg2(x))));
        };
        if (gn.flags & GN_UP_SIGMOID)
        {
            // mul_sigmoid_kernel_h: x * sigmoid(g)
            x01 = __hmul2(x01, sigmoid2(up[0]));
            x23 = __hmul2(x23, sigmoid2(up[1]));
        }
        else
        {
            // act_mul_kernel_h<ACT_SILU>: silu(x) * u
            x01 = __hmul2(__hmul2(x01, sigmoid2(x01)), up[0]);
            x23 = __hmul2(__hmul2(x23, sigmoid2(x23)), up[1]);
        }
    }
    half2 a01 = __hmul2(x01, sp[0]);
    half2 a23 = __hmul2(x23, sp[1]);
    float h0 = __low2float(a01), h1 = __high2float(a01), h2 = __low2float(a23), h3 = __high2float(a23);
    had128(h0, h1, h2, h3, lane);
    const float r = 0.088388347648f;
    const half2 p0 = __floats2half2_rn(h0 * r, h1 * r);
    const half2 p1 = __floats2half2_rn(h2 * r, h3 * r);

    float s = __low2float(p0) + __high2float(p0) + __low2float(p1) + __high2float(p1);
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) s += __shfl_xor(s, i);
    if (lane == 0) xcs[(size_t) m * kblocks + c] = s;

    // Elements 4 * lane .. +3 of the block: k-slice c * 8 + lane / 4, offset j0 = 4 * (lane % 4)
    uint32_t* xrow = (uint32_t*) &xh[((size_t) m * (size_k / 16) + c * 8 + (lane >> 2)) * 4];
    const int j0 = (lane & 3) * 4;
    const int j1 = j0 + 2;
    xrow[((j0 & 7) >> 1) * 2 + (j0 >> 3)] = h2u(p0);
    xrow[((j1 & 7) >> 1) * 2 + (j1 >> 3)] = h2u(p1);
}
#endif

#ifndef KB_WPE
#define EXL3_RDNA3_WPE
#else
#define EXL3_RDNA3_WPE __attribute__((amdgpu_waves_per_eu(KB_WPE, 16)))
#endif

#ifdef KB_TRACE
// kbench timeline: per block [start, main loop done, finished] in steady-counter ticks (100 MHz)
__device__ unsigned long long kb_trace[3 * 8192];
#define KB_T(slot) do { if (threadIdx.x == 0) { const int b_ = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z); \
    if (b_ < 8192) kb_trace[3 * b_ + (slot)] = __builtin_readsteadycounter(); } } while (0)
#else
#define KB_T(slot) do {} while (0)
#endif

// One work unit: output columns group * 128 .. +127 of entry e, rows rc * MR .. +MR-1, k-slices of split
template <int bits, bool half_k, int cb, int MR, bool c_fp32>
__device__ __forceinline__ void exl3_rdna3_unit
(
    const int e,
    const int rc,
    const int group,
    const int split,
    const int row_chunks,
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
    const Exl3Rdna3MTab& mt
)
{
    using namespace exl3_rdna3_ns;

    int n_stride = size_n;
    if (mt.b)
    {
        const int src = mt.src ? mt.src[e] : e;
        B = (const uint32_t*) mt.b[e];
        svh = (const half*) mt.svh[e];
        if (mt.c) C = (void*) mt.c[e];
        else C = (void*) ((char*) C + (size_t) e * size_m * size_n * (c_fp32 ? 4 : 2));
        if (mt.n_stride) n_stride = mt.n_stride[e];
        xh += (size_t) src * size_m * (size_k / 16) * 4;
        xcs += (size_t) src * size_m * (size_k / 128);
        ws += (size_t) e * splits * size_m * size_n;
        counters += (size_t) e * row_chunks * (size_n / 128);
    }

    constexpr int TWORDS = half_k ? 4 * (2 * bits + 1) : 8 * bits;   // uint32 per 16x16 tile
    constexpr int LPT = (TWORDS + 31) / 32;                            // words per lane per tile
    constexpr int KCS = XS_BYTES / (MR * 2) / 16 / 8 * 8;              // k-slices per staged x chunk
    static_assert(KCS % 8 == 0, "x chunk must cover whole 128-element Hadamard blocks");
    // Register-resident tile words and raw mul1 decode (codebook affine map folded into the epilogue)
    constexpr bool RAW = cb == 2 && !half_k && bits >= 2 && bits <= 6;
#ifdef KB_PF
    constexpr int PF = KB_PF;
#else
    constexpr int PF = (LPT > 1 || MR >= 16) ? 4 : 8;                  // prefetch ring depth, k-slices
#endif

    // x chunk, [MR][KCS][4] pairs: .x = x[16 ks + 2q .. +1], .y = x[16 ks + 8 + 2q .. +1]
    __shared__ uint2 xs[MR * KCS * 4];
    __shared__ uint32_t stage[RAW ? 1 : EXL3_RDNA3_NT][2][TWORDS];
    __shared__ float red[MR][128];
    __shared__ float xsum[MR];
    __shared__ int last_flag;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q = lane & 3;

    const int ntiles = size_n / 16;
    const int groups = ntiles / NT;
    const int row0 = rc * MR;
    const int rows = min(MR, size_m - row0);
    const int kslices = size_k / 16;
    const int ks_begin = split * ks_per_split;
    const int ks_end = min(kslices, ks_begin + ks_per_split);
    constexpr int TPW = EXL3_RDNA3_TPW;               // n-tiles per wave
    constexpr int WAVES = NT / TPW;
    const int nt0 = group * NT + warp;               // tiles nt0 + j * WAVES
    const size_t sstride = (size_t) (n_stride / 16) * TWORDS;

    if constexpr (RAW)
    {
        // Codebook bias term: sum of the transformed input over this block's k-range, per row
        if (threadIdx.x < MR)
        {
            float s = 0.0f;
            if ((int) threadIdx.x < rows)
                for (int c = ks_begin / 8; c < ks_end / 8; ++c)
                    s += xcs[(size_t) (row0 + threadIdx.x) * (kslices / 8) + c];
            xsum[threadIdx.x] = s;
        }
    }

    // Raw buffer over the trellis (byte offsets; EXL3 tensors are well below 2 GB)
    const __amdgpu_buffer_rsrc_t brsrc = __builtin_amdgcn_make_buffer_rsrc((void*) B, (short) 0, 0x7fffffff, 0x31004000);

#ifdef KB_NACC
    constexpr int NACC = KB_NACC;   // 4: x / y halves in separate accumulators (no gain measured)
#else
    constexpr int NACC = 2;
#endif
    float acc[MR][TPW][NACC];
    #pragma unroll
    for (int m = 0; m < MR; ++m)
        #pragma unroll
        for (int j = 0; j < TPW; ++j)
            #pragma unroll
            for (int a = 0; a < NACC; ++a) acc[m][j][a] = 0.0f;

    for (int kc0 = ks_begin; kc0 < ks_end; kc0 += KCS)
    {
        const int nk = min(ks_end - kc0, KCS);   // multiple of 8
        const int nch = nk / 8;                  // 128-element blocks per row

        // Weight prefetch first, so the stream is in flight while x is staged. Main loop below: one
        // n-tile per wave, all k-slices of the chunk. nk is a multiple of PF (chunks
        // are whole 128-element blocks), so the unrolled body is branch-free; the prefetch past the
        // end re-reads the last slice instead of branching, which keeps vmcnt waits incremental
        uint32_t pf[PF][TPW][LPT];
        int lofs[LPT];
        #pragma unroll
        for (int l = 0; l < LPT; ++l) lofs[l] = min(l * 32 + lane, TWORDS - 1);

        // Per-lane byte offset within a k-slice row (vector) + k-slice offset (uniform, scalar), 32-bit
        uint32_t voff[TPW][LPT];
        #pragma unroll
        for (int j = 0; j < TPW; ++j)
            #pragma unroll
            for (int l = 0; l < LPT; ++l) voff[j][l] = (uint32_t) ((nt0 + j * WAVES) * TWORDS + lofs[l]) * 4u;
        const uint32_t sstride_b = (uint32_t) sstride * 4u;
        auto ldw = [&] (int slice, int j, int l) -> uint32_t
        {
            const uint32_t soff = (uint32_t) slice * sstride_b;
            return __builtin_amdgcn_raw_buffer_load_b32(brsrc, voff[j][l], soff, 6);   // slc | dlc: streaming
        };

        #pragma unroll
        for (int d = 0; d < PF; ++d)
            #pragma unroll
            for (int j = 0; j < TPW; ++j)
                #pragma unroll
                for (int l = 0; l < LPT; ++l)
                    pf[d][j][l] = ldw(kc0 + d, j, l);

        // Stage this chunk of the transformed input: nk * 32 contiguous bytes per row, 16-byte copies.
        // Rows past size_m are zero so the unrolled MAC loop needs no row guards
        __syncthreads();
        {
            const int per_row = nk * 2;   // uint4 per row
            for (int idx = threadIdx.x; idx < MR * per_row; idx += THREADS)
            {
                const int m = idx / per_row;
                const int j = idx % per_row;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (m < rows)
                    v = ((const uint4*) (xh + ((size_t) (row0 + m) * kslices + kc0) * 4))[j];
                ((uint4*) (xs + (size_t) m * KCS * 4))[j] = v;
            }
        }
        __syncthreads();

        for (int ib = 0; ib < nk; ib += PF)
        {
            #pragma unroll
            for (int d = 0; d < PF; ++d)
            {
                const int i = ib + d;

                uint32_t wd[TPW][4];
                #pragma unroll
                for (int j = 0; j < TPW; ++j)
                {
                    uint32_t w[LPT];
                    #pragma unroll
                    for (int l = 0; l < LPT; ++l) w[l] = pf[d][j][l];

                    // Refill slot d once its words are consumed (after the gathers below), so the load can
                    // target the same registers: loading earlier makes the compiler rename the ring and copy
                    // it back at the loop edge, which waits for every outstanding load (vmcnt(0))
                    const int inext = min(i + PF, nk - 1);
                    // Buffer-load intrinsic: with plain loads InstCombine folds phi(load, load) into
                    // load(phi(addr)), moving the prefetch to the point of use and serializing the stream
                    auto refill = [&] ()
                    {
                        #pragma unroll
                        for (int l = 0; l < LPT; ++l)
                            pf[d][j][l] = ldw(kc0 + inext, j, l);
                        // Pin the load here: the scheduler otherwise sinks the refills to the loop end in
                        // reverse order, and the next iteration's first use then waits for all of them
                        __builtin_amdgcn_sched_barrier(0);
                    };

                    uint32_t w00, w01, w10, w11;   // this tile's decoded weights (packed half2)
                    if constexpr (RAW && (KB_EXP & 2))
                    {
                        w00 = w[0]; w01 = w[0] ^ 0x3c003c00u; w10 = w[0] + 1; w11 = w[0] ^ 0x1234u;
                        refill();
                    }
                    else if constexpr (RAW)
                    {
                        // Tile words resolved in-wave (ds_bpermute), no LDS round trip
                        auto gather = [&] (int j) -> uint32_t
                        {
                            if constexpr (LPT == 1) return __shfl(w[0], j);
                            else
                            {
                                uint32_t lo = __shfl(w[0], j & 31);
                                uint32_t hi = __shfl(w[1], j & 31);
                                return j < 32 ? lo : hi;
                            }
                        };
                        uint32_t win[8];
                        windows8<bits>(gather, lane, win);
                        refill();
                        w00 = mul1_raw_pair(win[0], win[1]);
                        w01 = mul1_raw_pair(win[2], win[3]);
                        w10 = mul1_raw_pair(win[4], win[5]);
                        w11 = mul1_raw_pair(win[6], win[7]);
                    }
                    else
                    {
                        uint32_t* st = stage[warp * TPW + j][d & 1];
                        #pragma unroll
                        for (int l = 0; l < LPT; ++l)
                            if (l * 32 + lane < TWORDS)
                                st[l * 32 + lane] = w[l];
                        refill();
                        wave_sync();
                        FragB f0, f1;
                        dq_dispatch<bits, cb, half_k>(st, lane << 3, f0, f1);
                        w00 = h2u(f0[0]); w01 = h2u(f0[1]);
                        w10 = h2u(f1[0]); w11 = h2u(f1[1]);
                    }
                    wd[j][0] = w00; wd[j][1] = w01; wd[j][2] = w10; wd[j][3] = w11;
                }

                #pragma unroll
                for (int m = 0; m < ((KB_EXP & 1) ? 1 : MR); ++m)
                {
                    const uint2 xv = (KB_EXP & 16) ? make_uint2(0x3c003c00u + i + m, 0x3c003c00u ^ (i * m)) : xs[(m * KCS + i) * 4 + q];
                    #pragma unroll
                    for (int j = 0; j < TPW; ++j)
                    {
                        acc[m][j][0] = dot2(wd[j][0], xv.x, acc[m][j][0]);
                        acc[m][j][NACC - 2] = dot2(wd[j][1], xv.y, acc[m][j][NACC - 2]);
                        acc[m][j][1] = dot2(wd[j][2], xv.x, acc[m][j][1]);
                        acc[m][j][NACC - 1] = dot2(wd[j][3], xv.y, acc[m][j][NACC - 1]);
                    }
                }
            }
        }
    }

    if constexpr (NACC == 4)
    {
        #pragma unroll
        for (int m = 0; m < MR; ++m)
            #pragma unroll
            for (int j = 0; j < TPW; ++j)
            {
                acc[m][j][0] += acc[m][j][2];
                acc[m][j][1] += acc[m][j][3];
            }
    }

    // Reduce over the four lanes sharing a column; lane l holds columns l / 4 and 8 + l / 4 of its tile
    #pragma unroll
    for (int m = 0; m < MR; ++m)
    #pragma unroll
    for (int j = 0; j < TPW; ++j)
    {
        #pragma unroll
        for (int f = 0; f < 2; ++f)
        {
            float v = acc[m][j][f];
            v += __shfl_xor(v, 1);
            v += __shfl_xor(v, 2);
            if constexpr (RAW)
            {
                // Codebook value = k_inv * h + k_bias: sum(w * x) = k_inv * sum(h * x) + k_bias * sum(x)
                const float k_inv = __half2float(__ushort_as_half(0x1eee));
                const float k_bias = __half2float(__ushort_as_half(0xc931));
                v = fmaf(v, k_inv, k_bias * xsum[m]);
            }
            acc[m][j][f] = v;
        }
        if (q == 0)
        {
            const int tl = warp + j * WAVES;
            red[m][tl * 16 + (lane >> 2)] = acc[m][j][0];
            red[m][tl * 16 + 8 + (lane >> 2)] = acc[m][j][1];
        }
    }
    __syncthreads();

    // Cross-split reduction: last block to arrive for this (row chunk, column group) sums the partials
    if (splits > 1)
    {
        const int gidx = rc * groups + group;
        float* wp = ws + ((size_t) split * size_m + row0) * size_n + group * 128;
        for (int idx = threadIdx.x; idx < rows * 128; idx += THREADS)
        {
            const int m = idx >> 7, c = idx & 127;
            wp[(size_t) m * size_n + c] = red[m][c];
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int prev = atomicAdd(&counters[gidx], 1);
            last_flag = prev == splits - 1;
        }
        __syncthreads();
        if (!last_flag) return;
        __threadfence();

        for (int idx = threadIdx.x; idx < rows * 128; idx += THREADS)
        {
            const int m = idx >> 7, c = idx & 127;
            const float* rp = ws + (size_t) (row0 + m) * size_n + group * 128 + c;
            float s = 0.0f;
            for (int sp = 0; sp < splits; ++sp)
                s += __builtin_nontemporal_load(rp + (size_t) sp * size_m * size_n);
            red[m][c] = s;
        }
        if (threadIdx.x == 0) counters[gidx] = 0;
        __syncthreads();
    }

    // Output Hadamard, 1/sqrt(128), svh post-scale
    for (int m = warp; m < rows; m += WAVES)
    {
        float h0 = red[m][lane * 4 + 0];
        float h1 = red[m][lane * 4 + 1];
        float h2 = red[m][lane * 4 + 2];
        float h3 = red[m][lane * 4 + 3];
        had128(h0, h1, h2, h3, lane);
        const int col = group * 128 + lane * 4;
        const half2* sp = (const half2*) (svh + col);
        const half2 s01 = sp[0], s23 = sp[1];
        const float r = 0.088388347648f;
        h0 *= r * __low2float(s01);
        h1 *= r * __high2float(s01);
        h2 *= r * __low2float(s23);
        h3 *= r * __high2float(s23);
        const size_t o = (size_t) (row0 + m) * n_stride + col;
        if constexpr (c_fp32)
        {
            *((float4*) (((float*) C) + o)) = make_float4(h0, h1, h2, h3);
        }
        else
        {
            half2* cp = (half2*) (((half*) C) + o);
            cp[0] = __floats2half2_rn(h0, h1);
            cp[1] = __floats2half2_rn(h2, h3);
        }
    }
}

template <int bits, bool half_k, int cb, int MR, bool c_fp32>
__global__ __launch_bounds__(EXL3_RDNA3_THREADS) EXL3_RDNA3_WPE
void exl3_rdna3_kernel
(
    const uint2* __restrict__ xh,       // input after suh / Hadamard, pair layout (exl3_rdna3_had_kernel)
    const uint32_t* __restrict__ B,
    void* __restrict__ C,
    int size_m,
    int size_k,
    int size_n,
    int* __restrict__ counters,
    const float* __restrict__ xcs,      // per-128-block sums of xh
    float* __restrict__ ws,
    const half* __restrict__ svh,
    int splits,
    int ks_per_split,
    Exl3Rdna3MTab mt
)
{
    KB_T(0);
    const int groups = size_n / 128;
    exl3_rdna3_unit<bits, half_k, cb, MR, c_fp32>
    (
        blockIdx.z, blockIdx.y, blockIdx.x % groups, blockIdx.x / groups, gridDim.y,
        xh, B, C, size_m, size_k, size_n, counters, xcs, ws, svh, splits, ks_per_split, mt
    );
    KB_T(1);
    KB_T(2);
}

// Kernel table for one integer bitrate. Half-integer rates (bits + 0.5) exist for bits 1..3 with mul1 only
#define EXL3_RDNA3_MR_SWITCH(bits, hk, cb, fp32)                                           \
    switch (mr_idx)                                                                         \
    {                                                                                       \
        case 0: return exl3_rdna3_kernel<bits, hk, cb, 1, fp32>;                            \
        case 1: return exl3_rdna3_kernel<bits, hk, cb, 2, fp32>;                            \
        case 2: return exl3_rdna3_kernel<bits, hk, cb, 4, fp32>;                            \
        case 3: return exl3_rdna3_kernel<bits, hk, cb, 8, fp32>;                            \
        case 4: return exl3_rdna3_kernel<bits, hk, cb, 16, fp32>;                           \
        case 5: return exl3_rdna3_kernel<bits, hk, cb, 3, fp32>;                            \
        case 6: return exl3_rdna3_kernel<bits, hk, cb, 5, fp32>;                            \
        case 7: return exl3_rdna3_kernel<bits, hk, cb, 6, fp32>;                            \
        case 8: return exl3_rdna3_kernel<bits, hk, cb, 12, fp32>;                           \
    }                                                                                       \
    return nullptr;

#define EXL3_RDNA3_INT_BODY(bits)                                                           \
    if (cb == 0) { if (c_fp32) { EXL3_RDNA3_MR_SWITCH(bits, false, 0, true) } else { EXL3_RDNA3_MR_SWITCH(bits, false, 0, false) } } \
    if (cb == 1) { if (c_fp32) { EXL3_RDNA3_MR_SWITCH(bits, false, 1, true) } else { EXL3_RDNA3_MR_SWITCH(bits, false, 1, false) } } \
    if (cb == 2) { if (c_fp32) { EXL3_RDNA3_MR_SWITCH(bits, false, 2, true) } else { EXL3_RDNA3_MR_SWITCH(bits, false, 2, false) } } \
    return nullptr;

#define EXL3_RDNA3_GETTER(bits)                                                             \
fp_exl3_rdna3_kernel exl3_rdna3_get_k##bits(bool half_k, int cb, int mr_idx, bool c_fp32)   \
{                                                                                           \
    if (half_k) return nullptr;                                                             \
    EXL3_RDNA3_INT_BODY(bits)                                                               \
}

#define EXL3_RDNA3_GETTER_HALF(bits)                                                        \
fp_exl3_rdna3_kernel exl3_rdna3_get_k##bits(bool half_k, int cb, int mr_idx, bool c_fp32)   \
{                                                                                           \
    if (half_k)                                                                             \
    {                                                                                       \
        if (cb != 2) return nullptr;                                                        \
        if (c_fp32) { EXL3_RDNA3_MR_SWITCH(bits, true, 2, true) }                           \
        else        { EXL3_RDNA3_MR_SWITCH(bits, true, 2, false) }                          \
    }                                                                                       \
    EXL3_RDNA3_INT_BODY(bits)                                                               \
}

#else

#define EXL3_RDNA3_GETTER(bits) \
fp_exl3_rdna3_kernel exl3_rdna3_get_k##bits(bool, int, int, bool) { return nullptr; }
#define EXL3_RDNA3_GETTER_HALF(bits) EXL3_RDNA3_GETTER(bits)

#endif
