// Idle after a large streaming kernel: time(big; tiny) - time(big) for varying bytes read and grid size.
#include <hip/hip_runtime.h>
#include <cstdio>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("err %s line %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

__global__ void big(const uint4* __restrict__ src, size_t n16, float* out, int write_bytes_per_block)
{
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x)
    {
        uint4 v = src[i];
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    // optional output traffic
    for (int j = threadIdx.x; j < write_bytes_per_block / 4; j += blockDim.x)
        out[(size_t) blockIdx.x * (write_bytes_per_block / 4) + j] = (float) acc;
}
__global__ void tiny(float* t) { if (threadIdx.x == 0) t[0] += 1.0f; }

int main()
{
    size_t maxb = (size_t) 1100 << 20;
    uint4* src; float* out; float* t;
    CK(hipMalloc(&src, maxb)); CK(hipMemset(src, 1, maxb));
    CK(hipMalloc(&out, 64 << 20)); CK(hipMalloc(&t, 64));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const int IT = 20;
    for (size_t mb : {16, 128, 512, 1000})
    for (int G : {96, 2000})
    for (int wb : {0, 2048})
    {
        size_t n16 = (mb << 20) / 16;
        float ms_a, ms_b;
        for (int i = 0; i < 3; ++i) { big<<<G, 256>>>(src, n16, out, wb); tiny<<<1, 64>>>(t); }
        CK(hipEventRecord(e0));
        for (int i = 0; i < IT; ++i) big<<<G, 256>>>(src, n16, out, wb);
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms_a, e0, e1));
        CK(hipEventRecord(e0));
        for (int i = 0; i < IT; ++i) { big<<<G, 256>>>(src, n16, out, wb); tiny<<<1, 64>>>(t); }
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms_b, e0, e1));
        printf("read %5zu MB  grid %5d  write %4d B/blk: big %8.1f us, +tiny adds %7.1f us\n",
               mb, G, wb * (wb ? 1 : 0), ms_a * 1000 / IT, (ms_b - ms_a) * 1000 / IT);
    }
    return 0;
}
