#pragma once

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#endif

#if defined(__HIP_PLATFORM_AMD__)
#ifndef PHX_NANOSLEEP
#define PHX_NANOSLEEP 1
// HIP has no __nanosleep; s_sleep N waits ~64*N clocks
__device__ __forceinline__ void __nanosleep(unsigned ns)
{
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_s_sleep(1);
#endif
}
#endif
#endif

#ifdef __HIP_PLATFORM_AMD__
// ROCm: __halves2half2 tak selalu terlihat di host pass libtorch TU
__device__ __forceinline__ half2 rocm_h2(half a, half b)
{
    __half2 r; r.x = a; r.y = b; return r;
}
__device__ __forceinline__ __hip_bfloat162 rocm_bh2(__hip_bfloat16 a, __hip_bfloat16 b)
{
    __hip_bfloat162 r; r.x = a; r.y = b; return r;
}
#define __halves2half2(a, b) rocm_h2((a), (b))
#define __halves2bfloat162(a, b) rocm_bh2((a), (b))
#endif

#include <cstdio>

#ifdef __HIP_PLATFORM_AMD__
#define EXL3_FULL_MASK 0xffffffffull
#else
#define EXL3_FULL_MASK 0xffffffffu
#endif
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#ifdef __HIP_PLATFORM_AMD__
#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif
#ifndef __grid_constant__
#define __grid_constant__
#endif

// Cache-hinted loads: plain loads on AMD
template <typename T> __device__ __forceinline__ T __ldcs(const T* p) { return *p; }
template <typename T> __device__ __forceinline__ T __ldcg(const T* p) { return *p; }

// dp4a: v_dot4_u32_u8 for the unsigned form (codebook byte sums), v_dot4_i32_iu8 for the signed form
__device__ __forceinline__ unsigned int exl3_dp4a(unsigned int a, unsigned int b, unsigned int c)
{
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_udot4(a, b, c, false);
#else
    return 0;
#endif
}
__device__ __forceinline__ int exl3_dp4a(int a, int b, int c)
{
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    return 0;
#endif
}
#define __dp4a exl3_dp4a
#endif


typedef struct __align__(8) half4
{
    half2 x;
    half2 y;
    __device__ half4() = default;
    __device__ half4(half2 x_, half2 y_) : x(x_), y(y_) {}
    __device__ half4(half h0, half h1, half h2, half h3) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)) {}
}
half4;

typedef struct __align__(8) bfloat164
{
    __nv_bfloat162 x;
    __nv_bfloat162 y;
    __device__ bfloat164() = default;
    __device__ bfloat164(__nv_bfloat162 x_, __nv_bfloat162 y_): x(x_), y(y_) {}
    __device__ bfloat164(__nv_bfloat16 b0, __nv_bfloat16 b1, __nv_bfloat16 b2, __nv_bfloat16 b3) :
        x(__halves2bfloat162(b0, b1)),
        y(__halves2bfloat162(b2, b3)) {}
}
bfloat164;

typedef struct __align__(16) half8
{
    half2 x;
    half2 y;
    half2 z;
    half2 w;
     __device__ half8() = default;
     __device__ half8(half2 x_, half2 y_, half2 z_, half2 w_) : x(x_), y(y_), z(z_), w(w_) {}
     __device__ half8(half h0, half h1, half h2, half h3, half h4, half h5, half h6, half h7) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)),
         z(__halves2half2(h4, h5)),
         w(__halves2half2(h6, h7)) {}
}
half8;

struct Dim3
{
    int m;
    int k;
    int n;
    inline __device__ int numel_a() { return m * k; }
    inline __device__ int numel_b() { return k * n; }
    inline __device__ int numel_c() { return m * n; }
};

#define READ128(__x, __y) ((uint4*)&__x)[0] = ((uint4*)(__y))[0];
#define WRITE128(__x, __y) ((uint4*)__x)[0] = ((uint4*)(&__y))[0];
#define READ64(__x, __y) ((uint2*)&__x)[0] = ((uint2*)(__y))[0];
#define WRITE64(__x, __y) ((uint2*)__x)[0] = ((uint2*)(&__y))[0];

#define LOW_TO_FLOAT(__x) __half2float(__low2half(__x))
#define HIGH_TO_FLOAT(__x) __half2float(__high2half(__x))

#define LOW_TO_FLOAT(__x) __half2float(__low2half(__x))
#define HIGH_TO_FLOAT(__x) __half2float(__high2half(__x))

#define CLAMP(__x, __min, __max) fmaxf(__min, fminf(__x, __max))
#define CLAMP_FP16(__x) CLAMP(__x, -65504.0f, 65504.0f)

#define SWAP16(__x) __byte_perm(__x, 0, 0x1032)

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};

#define cuda_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:           return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:   return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:      return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:     return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:     return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:     return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:  return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:    return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:     return "CUBLAS_STATUS_NOT_SUPPORTED";
        default:                              return "Unknown cuBLAS status";
    }
}

#define cublas_check(ans) { cublas_assert((ans), __FILE__, __LINE__); }
inline void cublas_assert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS assert: %s %s %d\n",
                cublasGetErrorString(code), file, line);
        if (abort) exit(static_cast<int>(code));
    }
}

__device__ inline float fxor(float v, unsigned long long mask)
{
    uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
    *vi ^= mask;
    return v;
}

__device__ inline half2 h2xor(half2 v, unsigned long long mask)
{
    uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
    *vi ^= mask;
    return v;
}

#define NEG_INF_F16 __ushort_as_half(0xFC00)
#define POS_INF_F16 __ushort_as_half(0x7C00)
