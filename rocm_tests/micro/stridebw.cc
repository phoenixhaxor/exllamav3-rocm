// EXL3-like access: each block reads CHUNK contiguous bytes per "slice" (b32 per lane, 8 waves = 8 tiles),
// slices STRIDE bytes apart, like exl3_rdna3_kernel (tile row across n = one slice). Compare chunk widths.
#include <hip/hip_runtime.h>
#include <cstdio>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("err %s\n", hipGetErrorString(e)); exit(1); } } while (0)

// blocks: groups x splits; block reads slices [split*sps, +sps) of its column group (chunk_words dwords per slice)
template <int PF>
__global__ __launch_bounds__(256) void rd(const uint32_t* __restrict__ p, int slices, int stride_words, int chunk_words, int splits, unsigned* out)
{
    const int group = blockIdx.x % (gridDim.x / splits);
    const int split = blockIdx.x / (gridDim.x / splits);
    const int sps = slices / splits;
    unsigned acc = 0;
    const uint32_t* base = p + (size_t) group * chunk_words;
    for (int s = split * sps; s < (split + 1) * sps; s += PF)
    {
        uint32_t v[PF];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const uint32_t* q = base + (size_t) (s + d) * stride_words;
            v[d] = 0;
            for (int w = threadIdx.x; w < chunk_words; w += 256) v[d] ^= __builtin_nontemporal_load(q + w);
        }
        #pragma unroll
        for (int d = 0; d < PF; ++d) acc += v[d];
    }
    if (acc == 0x12345678) out[0] = acc;
}

int main()
{
    // gate_up-like 4-bit: k=5120 -> 320 slices; n=17408 -> 1088 tiles * 128 B = 139264 B per slice
    const int slices = 320;
    const int stride_bytes = 139264;
    size_t bytes = (size_t) slices * stride_bytes;
    uint32_t* p; unsigned* out;
    CK(hipMalloc(&p, bytes + (1 << 20))); CK(hipMemset(p, 1, bytes)); CK(hipMalloc(&out, 4));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    for (int chunk : {1024, 2048, 4096, 8192})
    for (int splits : {1, 2, 4})
    {
        int groups = stride_bytes / chunk;
        int grid = groups * splits;
        auto go = [&] { rd<4><<<grid, 256>>>(p, slices, stride_bytes / 4, chunk / 4, splits, out); };
        for (int w = 0; w < 50; ++w) go();
        CK(hipEventRecord(e0)); for (int w = 0; w < 200; ++w) go(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        printf("chunk %5d B splits %d grid %5d: %6.1f GB/s\n", chunk, splits, grid, bytes * 200.0 / (ms * 1e-3) / 1e9);
    }
    return 0;
}
