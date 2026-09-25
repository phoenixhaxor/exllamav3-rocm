#include <cuda_fp16.h>
#include "ablate.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

#define ABL_THREADS 1024
#define ABL_MAX_V4 2

// One block per row: t = <y, a>, then y <- y * s - b * t (s optional). Rows are read as float4 and
// kept in registers between the two phases (dim <= 4 * ABL_MAX_V4 * blockDim), otherwise re-read
__launch_bounds__(ABL_THREADS)
__global__ void ablate_kernel
(
    float* __restrict__ y,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ s,
    const int dim
)
{
    float4* yr = (float4*) (y + (uint64_t) blockIdx.x * dim);
    const float4* a4 = (const float4*) a;
    const float4* b4 = (const float4*) b;
    const float4* s4 = (const float4*) s;
    const int columns = dim / 4;
    const bool cached = columns <= ABL_MAX_V4 * (int) blockDim.x;

    float4 v[ABL_MAX_V4];
    float acc = 0.0f;
    if (cached)
    {
        #pragma unroll
        for (int j = 0; j < ABL_MAX_V4; ++j)
        {
            int c = threadIdx.x + j * blockDim.x;
            if (c < columns)
            {
                v[j] = yr[c];
                float4 w = a4[c];
                acc = fmaf(v[j].x, w.x, acc);
                acc = fmaf(v[j].y, w.y, acc);
                acc = fmaf(v[j].z, w.z, acc);
                acc = fmaf(v[j].w, w.w, acc);
            }
        }
    }
    else
    {
        for (int c = threadIdx.x; c < columns; c += blockDim.x)
        {
            float4 u = yr[c], w = a4[c];
            acc = fmaf(u.x, w.x, acc);
            acc = fmaf(u.y, w.y, acc);
            acc = fmaf(u.z, w.z, acc);
            acc = fmaf(u.w, w.w, acc);
        }
    }

    for (int o = warpSize / 2; o > 0; o >>= 1)
        acc += __shfl_xor_sync(EXL3_FULL_MASK, acc, o);

    __shared__ float red[ABL_THREADS / 32];
    const int lane = threadIdx.x % warpSize;
    const int wid = threadIdx.x / warpSize;
    const int nw = blockDim.x / warpSize;
    if (lane == 0) red[wid] = acc;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < nw; ++w) t += red[w];

    auto apply = [&] (float4 u, int c)
    {
        float4 q = b4[c];
        if (s)
        {
            float4 m = s4[c];
            u.x *= m.x; u.y *= m.y; u.z *= m.z; u.w *= m.w;
        }
        u.x = fmaf(-q.x, t, u.x);
        u.y = fmaf(-q.y, t, u.y);
        u.z = fmaf(-q.z, t, u.z);
        u.w = fmaf(-q.w, t, u.w);
        yr[c] = u;
    };

    if (cached)
    {
        #pragma unroll
        for (int j = 0; j < ABL_MAX_V4; ++j)
        {
            int c = threadIdx.x + j * blockDim.x;
            if (c < columns) apply(v[j], c);
        }
    }
    else
    {
        for (int c = threadIdx.x; c < columns; c += blockDim.x)
            apply(yr[c], c);
    }
}

/*
Directional ablation of residual-stream sublayer outputs, in place:
y <- y * s - b * <y, a> for every row of y (fp32, contiguous, last dim = hidden size)
*/

void ablate
(
    at::Tensor y,
    const at::Tensor& a,
    const at::Tensor& b,
    const c10::optional<at::Tensor>& s
)
{
    const at::cuda::OptionalCUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(y.dtype() == at::kFloat && y.is_contiguous(), "ablate: y must be contiguous fp32");
    TORCH_CHECK(a.dtype() == at::kFloat && b.dtype() == at::kFloat, "ablate: a, b must be fp32");
    const int dim = (int) y.size(-1);
    TORCH_CHECK(a.numel() == dim && b.numel() == dim, "ablate: shape mismatch");
    const float* s_ptr = nullptr;
    if (s.has_value())
    {
        TORCH_CHECK(s->dtype() == at::kFloat && s->numel() == dim, "ablate: bad s");
        s_ptr = (const float*) s->data_ptr();
    }
    TORCH_CHECK(dim % 4 == 0, "ablate: dim must be a multiple of 4");
    const int64_t rows = y.numel() / dim;
    if (rows == 0) return;

    // Size the block to the row (whole warps), at most ABL_THREADS
    int threads = ((dim / 4 + ABL_MAX_V4 - 1) / ABL_MAX_V4 + 31) / 32 * 32;
    threads = threads < 32 ? 32 : (threads > ABL_THREADS ? ABL_THREADS : threads);

    ablate_kernel<<<rows, threads, 0, stream>>>
    (
        (float*) y.data_ptr(),
        (const float*) a.data_ptr(),
        (const float*) b.data_ptr(),
        s_ptr,
        dim
    );
    cuda_check(cudaPeekAtLastError());
}
