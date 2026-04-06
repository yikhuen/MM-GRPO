#!/bin/bash
# Wait until GPUs have enough free memory, then launch the training.
#
# Usage:
#   bash scripts/wait_and_run.sh scripts/train_clevr_count.sh
#   bash scripts/wait_and_run.sh scripts/start_vllm_server.sh
#
# Optionally override defaults:
#   REQUIRED_FREE_MiB=20000 CHECK_INTERVAL=60 GPU_ID=6 bash scripts/wait_and_run.sh scripts/train_clevr_count.sh

SCRIPT=${1:?"Usage: $0 <script_to_run>"}
GPU_ID=${GPU_ID:-6}                  # Which GPU to monitor
REQUIRED_FREE_MiB=${REQUIRED_FREE_MiB:-20000}  # Minimum free memory in MiB
CHECK_INTERVAL=${CHECK_INTERVAL:-60}  # Seconds between checks

echo "Waiting for GPU $GPU_ID to have at least ${REQUIRED_FREE_MiB} MiB free..."
echo "Checking every ${CHECK_INTERVAL}s. Will run: $SCRIPT"

while true; do
    TOTAL=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
    USED=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    FREE=$((TOTAL - USED))

    echo "[$(date '+%H:%M:%S')] GPU $GPU_ID: ${FREE} MiB free / ${TOTAL} MiB total"

    if [ "$FREE" -ge "$REQUIRED_FREE_MiB" ]; then
        echo "GPU $GPU_ID has enough free memory. Launching: $SCRIPT"
        bash "$SCRIPT"
        exit $?
    fi

    sleep $CHECK_INTERVAL
done
