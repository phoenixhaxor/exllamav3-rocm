#!/bin/bash
# Install TabbyAPI next to this repository, patched to accept RDNA3 GPUs, using the active environment.
# usage: rocm/scripts/install_tabbyapi.sh [tabby_dir] [models_dir]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TABBY=${1:-$REPO/../tabbyAPI}
MODELS=$(cd "${2:-$REPO/models}" && pwd)
TABBY_COMMIT=f07131cd8fe34e449fe87cdd3a066b52b96d3cac   # tested revision
[ -d "$TABBY" ] || git clone https://github.com/theroyallab/tabbyAPI "$TABBY"
cd "$TABBY"
git checkout -q "$TABBY_COMMIT"
git apply "$REPO/rocm/tabbyapi/0001-exllamav3-allow-rdna3.patch"
# TabbyAPI's own dependencies only: torch and exllamav3 come from this repository's environment
pip install "fastapi-slim>=0.115" "pydantic>=2.11,<3" ruamel.yaml rich "uvicorn>=0.28.1" "jinja2>=3.0.0" loguru \
            "sse-starlette>=2.2.0" packaging aiofiles aiohttp async_lru psutil "httptools>=0.5.0" requests uvloop setuptools
ln -sfn "$MODELS" models
cp -n "$REPO/rocm/tabbyapi/config.dflash2-192k.yml" config.yml
echo "TabbyAPI ready in $TABBY (config.yml = DFlash2 / 192K profile)"
