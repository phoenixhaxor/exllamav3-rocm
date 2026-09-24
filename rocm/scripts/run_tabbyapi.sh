#!/bin/bash
# Start TabbyAPI with the exllamav3 ROCm build. usage: rocm/scripts/run_tabbyapi.sh [config.yml] [tabby_dir]
# The environment from setup_env.sh must be active.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${1:-config.yml}
TABBY=${2:-$REPO/../tabbyAPI}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}   # skip an integrated GPU
cd "$TABBY"
exec python main.py --config "$CONFIG"
