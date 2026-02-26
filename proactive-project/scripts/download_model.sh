#!/bin/bash
# Download Qwen3-VL-8B-Instruct to local cache
# Requires: pip install huggingface_hub

set -e

MODEL_ID="${1:-Qwen/Qwen3-VL-8B-Instruct}"
CACHE_DIR="${2:-/home/v-tangxin/GUI/proactive-project/model_cache}"

echo "Downloading ${MODEL_ID} to ${CACHE_DIR}..."
echo "Estimated size: ~17GB (bf16)"
echo ""

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${MODEL_ID}',
    local_dir='${CACHE_DIR}/${MODEL_ID}',
    local_dir_use_symlinks=False,
)
print('Download complete!')
"
