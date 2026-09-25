#pragma once

#include <ATen/Tensor.h>

void ablate
(
    at::Tensor y,
    const at::Tensor& a,
    const at::Tensor& b,
    const c10::optional<at::Tensor>& s
);
