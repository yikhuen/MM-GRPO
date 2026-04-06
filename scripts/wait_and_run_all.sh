#!/bin/bash
# Wait until GPUs are free, then launch the vLLM server and training together.
#
# Usage:
#   bash scripts/wait_and_run_all.sh scripts/train_clevr_count.sh
#   bash scripts/wait_and_run_all.sh scripts/train_geoqa.sh
#   bash scripts/wait_and_run_all.sh scripts/train_open_r1_multimodal.sh
#
# Optionally override defaults:
#   REQUIRED_FREE_MiB=60000 CHECK_INTERVAL=60 bash scripts/wait_and_run_all.sh scripts/train_clevr_count.sh

TRAIN_SCRIPT=${1:?"Usage: $0 <training_script>"}
VLLM_GPU=${VLLM_GPU:-6}               # GPU for vLLM server
TRAIN_GPUS=${TRAIN_GPUS:-"0,1,2,3,4,5"}  # GPUs for training
REQUIRED_FREE_MiB=${REQUIRED_FREE_MiB:-40000}  # ~50% of 80GB H100, enough for training
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "========================================"
echo "Waiting for GPUs to free up..."
echo "  vLLM GPU:     $VLLM_GPU"
echo "  Training GPUs: $TRAIN_GPUS"
echo "  Required free: ${REQUIRED_FREE_MiB} MiB per GPU"
echo "  Training script: $TRAIN_SCRIPT"
echo "  Checking every ${CHECK_INTERVAL}s"
echo "========================================"

# Wait until all required GPUs have enough free memory
ALL_GPUS="$VLLM_GPU,${TRAIN_GPUS}"
IFS=',' read -ra GPU_LIST <<< "$ALL_GPUS"
# Deduplicate
GPU_LIST=($(printf '%s\n' "${GPU_LIST[@]}" | sort -u))

while true; do
    ALL_FREE=true
    STATUS=""

    for GPU_ID in "${GPU_LIST[@]}"; do
        TOTAL=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
        USED=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        FREE=$((TOTAL - USED))
        STATUS="$STATUS  GPU $GPU_ID: ${FREE}/${TOTAL} MiB free"

        if [ "$FREE" -lt "$REQUIRED_FREE_MiB" ]; then
            ALL_FREE=false
        fi
    done

    echo "[$(date '+%H:%M:%S')] $STATUS"

    if [ "$ALL_FREE" = true ]; then
        echo ""
        echo "All GPUs have enough free memory. Starting..."
        break
    fi

    sleep $CHECK_INTERVAL
done

# Step 1: Start vLLM server in background
echo "[$(date '+%H:%M:%S')] Starting vLLM server on GPU $VLLM_GPU..."
bash scripts/start_vllm_server.sh > "$LOG_DIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID (logs: $LOG_DIR/vllm_server.log)"

# Step 2: Wait for vLLM server to be ready
echo "[$(date '+%H:%M:%S')] Waiting for vLLM server to be ready on port 8000..."
until curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; do
    # Check the server process is still alive
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died. Check logs: $LOG_DIR/vllm_server.log"
        exit 1
    fi
    echo "[$(date '+%H:%M:%S')] Still waiting for vLLM server..."
    sleep 10
done
echo "[$(date '+%H:%M:%S')] vLLM server is ready."

# Step 3: Launch training
echo "[$(date '+%H:%M:%S')] Starting training: $TRAIN_SCRIPT"
bash "$TRAIN_SCRIPT" 2>&1 | tee "$LOG_DIR/training.log"
TRAIN_EXIT=$?

# Step 4: Shut down vLLM server when training finishes
echo "[$(date '+%H:%M:%S')] Training finished (exit code: $TRAIN_EXIT). Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "Done. Training exit code: $TRAIN_EXIT"
exit $TRAIN_EXIT
