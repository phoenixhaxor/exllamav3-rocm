#pragma once

#include <ATen/Tensor.h>

class Graph;

void rms_norm
(
    at::Tensor x,
    const c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    bool span_heads,
    bool add_residual,
    int w_groups = 1
);

void rms_norm_gr
(
    at::Tensor x,
    const c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    Graph* graph
);

bool rms_norm_had
(
    at::Tensor x,
    c10::optional<at::Tensor> w,
    at::Tensor y,
    c10::optional<at::Tensor> r,
    float epsilon,
    float constant_bias,
    float constant_scale,
    at::Tensor suh_tab,
    int num_src,
    c10::optional<at::Tensor> abl_a,
    c10::optional<at::Tensor> abl_b,
    c10::optional<at::Tensor> abl_s
);

void rms_norm_res_in
(
    at::Tensor x,
    c10::optional<at::Tensor> w,
    at::Tensor y,
    at::Tensor r,
    float epsilon,
    float constant_bias,
    float constant_scale,
    c10::optional<at::Tensor> abl_a,
    c10::optional<at::Tensor> abl_b,
    c10::optional<at::Tensor> abl_s
);

void gated_rms_norm
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    at::Tensor g,
    float epsilon,
    float constant_bias,
    int w_groups = 1,
    bool gate_first = false,
    int gate_act = 0
);

void gated_rms_norm_gr
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    at::Tensor g,
    float epsilon,
    float constant_bias,
    Graph* graph,
    int w_groups = 1,
    bool gate_first = false,
    int gate_act = 0
);

void rms_norm_gr
(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& w,
    at::Tensor& y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    bool span_heads,
    Graph* graph
);
