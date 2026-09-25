// Kernel-boundary gap vs in-kernel grid barrier on gfx1100.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <vector>
namespace cg = cooperative_groups;
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("err %s line %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

// Dependent stage: each block reads a value from a (different) block of the previous stage
__device__ __forceinline__ void stage(float* buf, int s, int G, int work)
{
    int b = blockIdx.x;
    const float* src = buf + ((s & 1) ? 0 : 4096);
    float* dst = buf + ((s & 1) ? 4096 : 0);
    float v = src[((b + 7) % G) * 32 + (threadIdx.x & 31)];
    for (int i = 0; i < work; ++i) v = v * 0.999f + 0.001f;
    if (threadIdx.x < 32) dst[b * 32 + threadIdx.x] = v + 1.0f;
}

__global__ void k_stage(float* buf, int s, int G, int work) { stage(buf, s, G, work); }

// Sense-reversing barrier on a global counter
__device__ __forceinline__ void grid_barrier(unsigned* bar, unsigned& gen, int G)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned my = gen + 1;
        __atomic_thread_fence(__ATOMIC_RELEASE);  // agent scope in practice
        unsigned old = __hip_atomic_fetch_add(bar, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        unsigned target = my * G;
        if (old + 1 != target)
            while (__hip_atomic_load(bar, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target)
                __builtin_amdgcn_s_sleep(0);
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
    }
    gen++;
    __syncthreads();
}

__global__ void k_persist(float* buf, unsigned* bar, int S, int G, int work)
{
    unsigned gen = 0;
    for (int s = 0; s < S; ++s)
    {
        stage(buf, s, G, work);
        grid_barrier(bar, gen, G);
    }
}

__global__ void k_coop(float* buf, int S, int G, int work)
{
    cg::grid_group g = cg::this_grid();
    for (int s = 0; s < S; ++s) { stage(buf, s, G, work); g.sync(); }
}

int main()
{
    float* buf; unsigned* bar;
    CK(hipMalloc(&buf, 8192 * 4 * 2)); CK(hipMalloc(&bar, 4));
    CK(hipMemset(buf, 0, 8192 * 8));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const int S = 2000;
    for (int work : {0, 200})
    for (int G : {8, 48, 96, 192})
    {
        int T = 256;
        float ms;
        // warmup
        for (int i = 0; i < 3000; ++i) k_stage<<<G, T>>>(buf, i, G, work);
        CK(hipEventRecord(e0));
        for (int i = 0; i < S; ++i) k_stage<<<G, T>>>(buf, i, G, work);
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1));
        float t_launch = ms * 1000 / S;

        CK(hipMemset(bar, 0, 4));
        k_persist<<<G, T>>>(buf, bar, 200, G, work);
        CK(hipMemset(bar, 0, 4));
        CK(hipEventRecord(e0));
        k_persist<<<G, T>>>(buf, bar, S, G, work);
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1));
        float t_persist = ms * 1000 / S;

        int SS = S; int GG = G; int WW = work;
        void* args[] = { &buf, &SS, &GG, &WW };
        CK(hipLaunchCooperativeKernel((void*) k_coop, dim3(G), dim3(T), args, 0, 0));
        CK(hipEventRecord(e0));
        CK(hipLaunchCooperativeKernel((void*) k_coop, dim3(G), dim3(T), args, 0, 0));
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1));
        float t_coop = ms * 1000 / S;
        printf("work %3d G %3d: kernel-per-stage %.2f us, persistent+atomic barrier %.2f us, cg grid.sync %.2f us\n",
               work, G, t_launch, t_persist, t_coop);
    }
    return 0;
}
