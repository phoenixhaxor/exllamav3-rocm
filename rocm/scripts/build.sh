#!/bin/bash
# Build the exllamav3 extension for RDNA3 in place (run from the repository root, env activated).
# Host C++ goes through ROCm clang (g++ cannot parse the HIP bf16 headers).
set -euo pipefail
ROCM_HOME=${ROCM_HOME:-/opt/rocm}
export CC=${CC:-$ROCM_HOME/llvm/bin/clang} CXX=${CXX:-$ROCM_HOME/llvm/bin/clang++}
export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx1100} MAX_JOBS=${MAX_JOBS:-16} ROCM_HOME
# --force: setuptools does not see header-only changes. After editing a header, delete the matching
# *_hip.cuh / *.hip files (and those of the .cu files including it): ninja does not track header
# dependencies of the hipified sources
python setup.py build_ext --inplace --force
EXLLAMA_NOCOMPILE=1 pip install -e . --no-deps --no-build-isolation
python -c "from exllamav3.ext import exllamav3_ext; print('exllamav3_ext loaded')"
