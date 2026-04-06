#!/bin/bash
# Start the external vLLM rollout server for GRPO training (multi-GPU only).
# This should be launched BEFORE running any training script.
#
# NOT needed for single-GPU setups — those use colocated mode instead.
#
# Adjust CUDA_VISIBLE_DEVICES and vllm_data_parallel_size to match
# the GPUs you want to dedicate to the vLLM server.

VLLM_GPUS=${VLLM_GPUS:-"6,7"}
VLLM_DP_SIZE=${VLLM_DP_SIZE:-2}

CUDA_VISIBLE_DEVICES=$VLLM_GPUS \
swift rollout \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --vllm_data_parallel_size $VLLM_DP_SIZE
