#!/bin/bash
# Python environment for exllamav3-rocm: Python 3.12 + PyTorch ROCm 7.2 wheels + runtime deps.
# usage: rocm/scripts/setup_env.sh [env_dir]   (default: ./.venv-rocm, needs conda or python3.12)
set -euo pipefail
ENV_DIR=${1:-$(pwd)/.venv-rocm}
if command -v conda >/dev/null; then
    conda create -y -p "$ENV_DIR" python=3.12 >/dev/null
    source "$(conda info --base)/bin/activate" "$ENV_DIR"
else
    python3.12 -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
fi
pip install --upgrade pip
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install "huggingface_hub[hf_transfer]" tokenizers "numpy>=2.1" rich typing_extensions safetensors ninja \
            pillow pyyaml marisa_trie pydantic "llguidance>=1.7.0"
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
echo "environment ready: $ENV_DIR"
