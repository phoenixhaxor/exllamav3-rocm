#pragma once
#include "util.cuh"
#ifndef __HIP_PLATFORM_AMD__
#include <cuda/atomic>
#endif

#ifdef __HIP_PLATFORM_AMD__
__device__ unsigned char rocm_anchor;  // static shared utk inverse cvta
__device__ __forceinline__ unsigned rocm_g2s(const void* p)
{ return (unsigned)(uintptr_t) p; }
__device__ __forceinline__ void* rocm_sh_ptr(unsigned addr32)
{
    unsigned base = rocm_g2s(&rocm_anchor);
    return (void*) ((unsigned char*) &rocm_anchor + (int)(addr32 - base));
}
#endif
// ======== ROCm HIP-PORTABLE LAYER (gfx1100) ========
#ifdef __HIP_PLATFORM_AMD__
__device__ __forceinline__ unsigned rocm_lane() { return threadIdx.x & 31; }
__device__ __forceinline__ unsigned rocm_shfl32(unsigned v, int src)
{ return __shfl_sync(EXL3_FULL_MASK, v, src); }
__device__ __forceinline__ float rocm_hlo(unsigned u) { return __half2float(__ushort_as_half((unsigned short)(u & 0xffff))); }
__device__ __forceinline__ float rocm_hhi(unsigned u) { return __half2float(__ushort_as_half((unsigned short)(u >> 16))); }
__device__ __forceinline__ unsigned rocm_pk2(float lo, float hi)
{ return ((unsigned)__half_as_ushort(__float2half(hi)) << 16) | (unsigned)__half_as_ushort(__float2half(lo)); }
__device__ __forceinline__ int rocm_dp4a_(unsigned a, unsigned b, int c)
{
    int r = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        r += (int)(signed char)((a >> (8 * i)) & 0xff) * (int)(signed char)((b >> (8 * i)) & 0xff);
    return r;
}
// ldmatrix.x4 emulasi: reg m milik lane l = word (l&3) dari alamat lane (8m + (l>>2))
__device__ __forceinline__ void rocm_ldsm4(unsigned* r, const void* p)
{
    unsigned long long a = (unsigned long long)(uintptr_t) p;
    unsigned lo32 = (unsigned)a, hi32 = (unsigned)(a >> 32);
    #pragma unroll
    for (int m = 0; m < 4; ++m)
    {
        int src = 8 * m + (rocm_lane() >> 2);
        unsigned long long rowptr =
            ((unsigned long long) rocm_shfl32(hi32, src) << 32) | rocm_shfl32(lo32, src);
        r[m] = ((const uint32_t*)(uintptr_t) rowptr)[rocm_lane() & 3];
    }
}
// ldmatrix.x4.trans emulasi: reg m lane l = {M[2(l&3)][l>>2], M[2(l&3)+1][l>>2]} (lo,hi)
__device__ __forceinline__ void rocm_ldsm4_t(unsigned* r, const void* p)
{
    unsigned long long a = (unsigned long long)(uintptr_t) p;
    unsigned lo32 = (unsigned)a, hi32 = (unsigned)(a >> 32);
    unsigned row = 2 * (rocm_lane() & 3), col = rocm_lane() >> 2;
    #pragma unroll
    for (int m = 0; m < 4; ++m)
    {
        unsigned long long p0 = ((unsigned long long) rocm_shfl32(hi32, 8 * m + row) << 32) | rocm_shfl32(lo32, 8 * m + row);
        unsigned long long p1 = ((unsigned long long) rocm_shfl32(hi32, 8 * m + row + 1) << 32) | rocm_shfl32(lo32, 8 * m + row + 1);
        unsigned short e0 = *(const unsigned short*)((const char*)(uintptr_t) p0 + col * 2);
        unsigned short e1 = *(const unsigned short*)((const char*)(uintptr_t) p1 + col * 2);
        r[m] = (unsigned)e0 | ((unsigned)e1 << 16);
    }
}
#endif

// Tensor core fragments

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;

// m8n8k4 tensor core matmul (emulated on Ampere and later), don't use
//
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m8n8k4-with-f16-floating-point-type

#ifdef __HIP_PLATFORM_AMD__
__device__ inline void ptx_mma_m8n8k4(const Vec<half2, 2>& frag_a, const Vec<half2, 2>& frag_b, Vec<float, 8>& frag_c)
{
    // ROCm: unused path (0 callers) — f32 emulasi sederhana
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    unsigned lane = rocm_lane(), r = lane >> 2, cc = lane & 3;
    float a0[2] = {rocm_hlo(a[0]), rocm_hhi(a[0])}, a1[2] = {rocm_hlo(a[1]), rocm_hhi(a[1])};
    float b0[2] = {rocm_hlo(b[0]), rocm_hhi(b[0])}, b1[2] = {rocm_hlo(b[1]), rocm_hhi(b[1])};
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        float acc = c[i * 4 + 0];
        acc = fmaf(a0[0], b0[0], acc); acc = fmaf(a0[1], b1[0], acc);
        acc = fmaf(a1[0], b0[1], acc); acc = fmaf(a1[1], b1[1], acc);
        c[i * 4 + 0] = acc;
        float acc1 = c[i * 4 + 1];
        acc1 = fmaf(a0[0], b0[0], acc1); acc1 = fmaf(a0[1], b1[0], acc1);
        acc1 = fmaf(a1[0], b0[1], acc1); acc1 = fmaf(a1[1], b1[1], acc1);
        c[i * 4 + 1] = acc1;
    }
}
#else
__device__ inline void ptx_mma_m8n8k4
(
    const Vec<half2, 2>& frag_a,
    const Vec<half2, 2>& frag_b,
    Vec<float, 8>& frag_c
)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    const float* d = reinterpret_cast<const float*>(&frag_c);

    asm
    (
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"

        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]),"=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])

        :  "r"(a[0]), "r"(a[1]),
           "r"(b[0]), "r"(b[1]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "f"(d[4]), "f"(d[5]), "f"(d[6]), "f"(d[7])
    );
}
#endif

// m16n8k16 tensor core matmul
//
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type

// FP16 @ FP16 + FP32 -> FP32
#ifdef __HIP_PLATFORM_AMD__
__device__ inline void ptx_mma_m16n8k16(const FragA& frag_a, const FragB& frag_b, FragC& frag_c)
{
    // ROCm emulasi m16n8k16 f32-acc via warp shuffle (PTX lane map)
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    unsigned lane = rocm_lane(), r = lane >> 2, cc = lane & 3;
    float acc0 = c[0], acc1 = c[1], acc2 = c[2], acc3 = c[3];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
    {
        unsigned ar0 = rocm_shfl32(a[0], 4 * r + t);      // A[r][2t..2t+1]
        unsigned ar1 = rocm_shfl32(a[1], 4 * r + t);      // A[r+8][2t..2t+1]
        unsigned ar2 = rocm_shfl32(a[2], 4 * r + t);      // A[r][8+2t..]
        unsigned ar3 = rocm_shfl32(a[3], 4 * r + t);      // A[r+8][8+2t..]
        unsigned j0_0 = rocm_shfl32(b[0], 8 * cc + t);    // B[2t..2t+1][2c]
        unsigned j0_1 = rocm_shfl32(b[1], 8 * cc + t);    // B[8+2t..][2c]
        unsigned j1_0 = rocm_shfl32(b[0], 8 * cc + 4 + t);// B[2t..2t+1][2c+1]
        unsigned j1_1 = rocm_shfl32(b[1], 8 * cc + 4 + t);
        acc0 = fmaf(rocm_hlo(ar0), rocm_hlo(j0_0), acc0);
        acc0 = fmaf(rocm_hhi(ar0), rocm_hhi(j0_0), acc0);
        acc0 = fmaf(rocm_hlo(ar2), rocm_hlo(j0_1), acc0);
        acc0 = fmaf(rocm_hhi(ar2), rocm_hhi(j0_1), acc0);
        acc1 = fmaf(rocm_hlo(ar0), rocm_hlo(j1_0), acc1);
        acc1 = fmaf(rocm_hhi(ar0), rocm_hhi(j1_0), acc1);
        acc1 = fmaf(rocm_hlo(ar2), rocm_hlo(j1_1), acc1);
        acc1 = fmaf(rocm_hhi(ar2), rocm_hhi(j1_1), acc1);
        acc2 = fmaf(rocm_hlo(ar1), rocm_hlo(j0_0), acc2);
        acc2 = fmaf(rocm_hhi(ar1), rocm_hhi(j0_0), acc2);
        acc2 = fmaf(rocm_hlo(ar3), rocm_hlo(j0_1), acc2);
        acc2 = fmaf(rocm_hhi(ar3), rocm_hhi(j0_1), acc2);
        acc3 = fmaf(rocm_hlo(ar1), rocm_hlo(j1_0), acc3);
        acc3 = fmaf(rocm_hhi(ar1), rocm_hhi(j1_0), acc3);
        acc3 = fmaf(rocm_hlo(ar3), rocm_hlo(j1_1), acc3);
        acc3 = fmaf(rocm_hhi(ar3), rocm_hhi(j1_1), acc3);
    }
    c[0] = acc0; c[1] = acc1; c[2] = acc2; c[3] = acc3;
}
#else
__device__ inline void ptx_mma_m16n8k16
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC& frag_c
)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    const float* d = reinterpret_cast<const float*>(&frag_c);

    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"

        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
           "r"(b[0]), "r"(b[1]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3])
    );
}
#endif

// FP16 @ FP16 + FP16 -> FP16
#ifdef __HIP_PLATFORM_AMD__
__device__ inline void ptx_mma_m16n8k16(const FragA& frag_a, const FragB& frag_b, FragC_h& frag_c)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
    float c0l = rocm_hlo(c[0]), c0h = rocm_hhi(c[0]), c1l = rocm_hlo(c[1]), c1h = rocm_hhi(c[1]);
    FragC tmp;
    float* tf = reinterpret_cast<float*>(&tmp);
    tf[0] = c0l; tf[1] = c0h; tf[2] = c1l; tf[3] = c1h;
    ptx_mma_m16n8k16(frag_a, frag_b, tmp);
    c[0] = rocm_pk2(tf[0], tf[1]);
    c[1] = rocm_pk2(tf[2], tf[3]);
}
#else
__device__ inline void ptx_mma_m16n8k16
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC_h& frag_c
)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
    const uint32_t* d = reinterpret_cast<const uint32_t*>(&frag_c);

    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"

        : "=r"(c[0]), "=r"(c[1])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
           "r"(b[0]), "r"(b[1]),
           "r"(d[0]), "r"(d[1])
    );
}
#endif

// Global barrier

#ifdef __HIP_PLATFORM_AMD__
__device__ inline void barrier_acquire(int* lock, int stage)
{
    if (threadIdx.x == 0)
    {
        int state = -1;
        do { state = __atomic_load_n(lock, __ATOMIC_ACQUIRE); } while (state != stage);
    }
    __syncthreads();
}
#else
__device__ inline void barrier_acquire
(
    int* lock,
    int stage
)
{
    if (threadIdx.x == 0)
    {
        volatile int state = -1;
        do
        {
            asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        }
        while (state != stage);
    }
    __syncthreads();
}
#endif

#ifdef __HIP_PLATFORM_AMD__
__device__ inline void barrier_release(int* lock, int val, bool reset)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (reset) { *lock = 0; __threadfence(); return; }
        __atomic_fetch_add(lock, val, __ATOMIC_ACQ_REL);
    }
}
#else
__device__ inline void barrier_release
(
    int* lock,
    int val,
    bool reset
)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (reset)
        {
            *lock = 0;
            return;
        }
        asm volatile ("fence.acq_rel.gpu;\n");
        asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
    }
}
#endif

// Load global to shared memory, predicated. Seems to produce incorrect code when compiling for Blackwell, but
// `if (...) cp_async(...)` compiles to a predicated instruction anyway

#ifdef __HIP_PLATFORM_AMD__
__device__ inline void cp_async_pred(void* smem_ptr, const void* glob_ptr, bool pred)
{
    if (pred) *(uint4*) smem_ptr = *(const uint4*) glob_ptr;
    else *(uint4*) smem_ptr = make_uint4(0u, 0u, 0u, 0u);
}
#else
__device__ inline void cp_async_pred(void* smem_ptr, const void* glob_ptr, bool pred = true)
{
    const int bytes = 16;
    uint32_t smem = rocm_g2s(smem_ptr);
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}
#endif

// Load global to shared memory

#ifdef __HIP_PLATFORM_AMD__
__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr)
{ *(uint4*) smem_ptr = *(const uint4*) glob_ptr; }
#else
__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr)
{
    const int bytes = 16;
    uint32_t smem = rocm_g2s(smem_ptr);
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}
#endif

// Load global to shared memory with cache hint to evict data from L2 ASAP

#ifdef __HIP_PLATFORM_AMD__
__device__ inline void cp_async_stream(void* smem_ptr, const void* glob_ptr)
{ *(uint4*) smem_ptr = *(const uint4*) glob_ptr; }
#else
__device__ inline void cp_async_stream(void* smem_ptr, const void* glob_ptr)
{
    uint32_t smem = rocm_g2s(smem_ptr);
    const int bytes = 16;
    asm volatile
    (
        "{\n"
        "   .reg .b64 p;\n"
        "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
        "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}
#endif

// Async copy fence, commit all pending async copies

#ifdef __HIP_PLATFORM_AMD__
__device__ inline void cp_async_fence() { }
#else
__device__ inline void cp_async_fence()
{
    asm volatile("cp.async.commit_group;\n" ::);
}
#endif

// Wait until at most n async groups are still pending.

#ifdef __HIP_PLATFORM_AMD__
template <int n>
__device__ inline void cp_async_wait()
{
    __syncwarp(0xffffffffull);
    __syncthreads();
}
#else
__device__ inline void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}
#endif

// Load 16x16 matrix fragment from shared memory, directly in tensor core layout

#ifdef __HIP_PLATFORM_AMD__
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr)
{
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    rocm_ldsm4(a, smem_ptr);
}
#else
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr)
{
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = rocm_g2s(smem_ptr);
    asm volatile
    (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
    );
}
#endif

#ifdef __HIP_PLATFORM_AMD__
__device__ inline uint32_t mul_lo_u32(uint32_t x, uint32_t y) { return x * y; }
#else
__device__ inline uint32_t mul_lo_u32(uint32_t x, uint32_t y)
{
    uint32_t w;
    asm volatile
    (
        "mul.lo.u32 %0, %1, %2;"
        : "=r"(w)
        :  "r"(x), "r"(y)
    );
    return w;
}
#endif

#ifdef __HIP_PLATFORM_AMD__
__device__ inline uint32_t mul_hi_u32(uint32_t x, uint32_t y)
{
    return (uint32_t)(((uint64_t) x * (uint64_t) y) >> 32);
}
#else
__device__ inline uint32_t mul_hi_u32(uint32_t x, uint32_t y)
{
    uint32_t w;
    asm volatile
    (
        "mul.hi.u32 %0, %1, %2;"
        : "=r"(w)
        :  "r"(x), "r"(y)
    );
    return w;
}
#endif

// Memory ops

#ifdef __HIP_PLATFORM_AMD__
__device__ __forceinline__ void stg_wt_u32(uint32_t* p, uint32_t v)
{ *(volatile uint32_t*) p = v; __threadfence_system(); }
#else
__device__ __forceinline__ void stg_wt_u32(uint32_t* p, uint32_t v)
{
    asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(p), "r"(v));
}
#endif

#ifdef __HIP_PLATFORM_AMD__
__device__ __forceinline__ void stg_wt_u128(uint4* p, const uint4& v)
{
    volatile unsigned* vp = (volatile unsigned*) p;
    vp[0] = v.x; vp[1] = v.y; vp[2] = v.z; vp[3] = v.w;
    __threadfence_system();
}
#else
__device__ __forceinline__ void stg_wt_u128(uint4* p, const uint4 v)
{
    asm volatile ("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
                  :: "l"(p),
                     "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}
#endif

#ifdef __HIP_PLATFORM_AMD__
__device__ __forceinline__ uint32_t ldg_cv_u32(const uint32_t* p)
{ return *(const volatile uint32_t*) p; }
#else
__device__ __forceinline__ uint32_t ldg_cv_u32(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
#endif

#ifdef __HIP_PLATFORM_AMD__
__device__ __forceinline__ uint4 ldg_cv_u128(const uint4* p)
{
    const volatile unsigned* vp = (const volatile unsigned*) p;
    return make_uint4(vp[0], vp[1], vp[2], vp[3]);
}
#else
__device__ __forceinline__ uint4 ldg_cv_u128(const uint4* p)
{
    uint4 v;
    asm volatile ("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                  : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                  : "l"(p));
    return v;
}
#endif

#ifdef __HIP_PLATFORM_AMD__
__device__ __forceinline__ uint32_t ldg_acquire_sys_u32(const uint32_t* p)
{ return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
#else
__device__ __forceinline__ uint32_t ldg_acquire_sys_u32(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.global.acquire.sys.u32 %0, [%1];"
                 : "=r"(v) : "l"(p) : "memory");
    return v;
}
#endif

#ifdef __HIP_PLATFORM_AMD__
__device__ __forceinline__ uint64_t ldg_acquire_sys_u64(const uint64_t* p)
{ return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
#else
__device__ __forceinline__ uint64_t ldg_acquire_sys_u64(const uint64_t* p)
{
    uint64_t v;
    asm volatile("ld.global.acquire.sys.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
#endif

#ifdef __HIP_PLATFORM_AMD__
__device__ __forceinline__ void stg_release_sys_u32(uint32_t* p, uint32_t v)
{ __atomic_store_n(p, v, __ATOMIC_RELEASE); }
#else
__device__ __forceinline__ void stg_release_sys_u32(uint32_t* p, uint32_t v)
{
    asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
#endif

#ifdef __HIP_PLATFORM_AMD__
__device__ __forceinline__ void stg_release_sys_u64(uint64_t* p, uint64_t v)
{ __atomic_store_n(p, v, __ATOMIC_RELEASE); }
#else
__device__ __forceinline__ void stg_release_sys_u64(uint64_t* p, uint64_t v)
{
    asm volatile("st.global.release.sys.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
#endif

// Global time in nanoseconds

#ifdef __HIP_PLATFORM_AMD__
__device__ __forceinline__ uint64_t globaltimer_ns()
{
    // ROCm: approx ns dari cycle counter (hanya utk timeout deteksi deadlock)
    return (uint64_t) (((unsigned long long) clock64() * 1000ull) / 2400ull);
}
#else
__device__ __forceinline__ uint64_t globaltimer_ns()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#endif

// Bitfield stuff

#ifdef __HIP_PLATFORM_AMD__
static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = ((uint64_t) hi << 32) | (uint64_t) lo;
    return (uint32_t) ((value >> offset) & ((length >= 64) ? ~0ull : ((1ull << length) - 1)));
}
#else
__device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    uint64_t result64;
    asm ("bfe.u64 %0, %1, %2, %3;"
         : "=l"(result64)
         : "l"(value), "r"(offset), "r"(length));
    return static_cast<uint32_t>(result64);
}
#endif

#ifdef __HIP_PLATFORM_AMD__
#define FSHF_IMM(dst, lo, hi, imm) (dst) = __funnelshift_r((lo), (hi), (imm))
#define BFE16_IMM(dst, src, imm) (dst) = ((src) >> (imm)) & 0xffffu
#else
#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))
#endif
// Inter-block barrier

#ifdef __HIP_PLATFORM_AMD__
__device__ inline void group_barrier(int group_id, int group_size, int* barrier_counters_sense)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int* counter = barrier_counters_sense + group_id * 2;
        int* sense = barrier_counters_sense + group_id * 2 + 1;
        int old_sense = __atomic_load_n(sense, __ATOMIC_RELAXED);
        int old = __atomic_fetch_add(counter, 1, __ATOMIC_ACQ_REL);
        if (old == group_size - 1)
        {
            __atomic_store_n(counter, 0, __ATOMIC_RELAXED);
            __atomic_store_n(sense, 1 - old_sense, __ATOMIC_RELEASE);
        }
        else
        {
            while (__atomic_load_n(sense, __ATOMIC_ACQUIRE) == old_sense) { }
        }
    }
    __syncthreads();
}
#else
__device__ inline void group_barrier
(
    int group_id,
    int group_size,
    int* barrier_counters_sense  // length 2*max(group_id). odd positions are flipped after sync (sense)
)
{
    __syncthreads();

    if (threadIdx.x == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> counter(barrier_counters_sense[group_id * 2]);
        cuda::atomic_ref<int, cuda::thread_scope_device> sense(barrier_counters_sense[group_id * 2 + 1]);

        int old_sense = sense.load(cuda::memory_order_relaxed);
        int old = counter.fetch_add(1, cuda::memory_order_acq_rel);

        if (old == group_size - 1)
        {
            counter.store(0, cuda::memory_order_relaxed);
            sense.store(1 - old_sense, cuda::memory_order_release);
        }
        else
        {
            while (sense.load(cuda::memory_order_acquire) == old_sense) __nanosleep(32);
        }
    }

    __syncthreads();
}
#endif
