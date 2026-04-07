#!/bin/bash
# Start the external vLLM rollout server for GRPO training.
# This should be launched BEFORE running any training script.
#
# Adjust CUDA_VISIBLE_DEVICES to match the GPUs you want to dedicate
# to the vLLM server (separate from the training GPUs).

CUDA_VISIBLE_DEVICES=6 \
swift rollout \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --vllm_gpu_memory_utilization 0.9 \
    --torch_dtype float16 \
    --max_model_len 4096
