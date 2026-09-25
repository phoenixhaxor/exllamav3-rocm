// Peak read bandwidth on gfx1100: 16-byte loads, UNR independent loads in flight per thread, grid sweep
#include <hip/hip_runtime.h>
#include <cstdio>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("err %s\n", hipGetErrorString(e)); exit(1); } } while (0)

template <int UNR, bool NT>
__global__ __launch_bounds__(256) void rd(const uint4* __restrict__ p, size_t n16, unsigned* out)
{
    unsigned acc = 0;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    for (; i + (UNR - 1) * stride < n16; i += UNR * stride)
    {
        uint4 v[UNR];
        #pragma unroll
        for (int u = 0; u < UNR; ++u)
        {
            if constexpr (NT) { typedef unsigned u4 __attribute__((ext_vector_type(4))); u4 t = __builtin_nontemporal_load((const u4*) &p[i + u * stride]); v[u] = make_uint4(t.x, t.y, t.z, t.w); }
            else v[u] = p[i + u * stride];
        }
        #pragma unroll
        for (int u = 0; u < UNR; ++u) acc += v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x12345678) out[0] = acc;
}

template <int UNR, bool NT>
void run(const uint4* p, size_t n16, unsigned* out, int grid, hipEvent_t e0, hipEvent_t e1)
{
    for (int w = 0; w < 30; ++w) rd<UNR, NT><<<grid, 256>>>(p, n16, out);
    CK(hipEventRecord(e0));
    const int it = 50;
    for (int w = 0; w < it; ++w) rd<UNR, NT><<<grid, 256>>>(p, n16, out);
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
    float ms; CK(hipEventElapsedTime(&ms, e0, e1));
    printf("UNR %d NT %d grid %5d: %6.1f GB/s\n", UNR, NT, grid, n16 * 16.0 * it / (ms * 1e-3) / 1e9);
}

int main()
{
    size_t bytes = (size_t) 2 << 30;
    uint4* p; unsigned* out;
    CK(hipMalloc(&p, bytes)); CK(hipMemset(p, 1, bytes)); CK(hipMalloc(&out, 4));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    size_t n16 = bytes / 16;
    for (int grid : {192, 384, 768, 1536, 3072, 6144})
    {
        run<1, false>(p, n16, out, grid, e0, e1);
        run<4, false>(p, n16, out, grid, e0, e1);
        run<8, false>(p, n16, out, grid, e0, e1);
        run<4, true>(p, n16, out, grid, e0, e1);
    }
    return 0;
}
