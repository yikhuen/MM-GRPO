#!/bin/bash
# ============================================================================
# Task 1: ClevrCount - Count objects in CLEVR images
# ============================================================================
# Dataset: AI-ModelScope/clevr_cogen_a_train
# Model: Qwen2.5-VL-3B-Instruct
# Reward functions: external_r1v_acc (accuracy) + format (R1-style format)
#
# This is a simple counting task. The model learns to count objects in
# synthetic CLEVR images. Typically converges within ~500 steps.
#
# Prerequisites:
#   1. Start the vLLM server first: bash scripts/start_vllm_server.sh
#   2. Set your WANDB_API_KEY below (or remove --report_to wandb)
# ============================================================================

export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --tuner_type full \
    --torch_dtype float16 \
    --dataset 'AI-ModelScope/clevr_cogen_a_train' \
    --load_from_cache_file true \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir output/GRPO_CLEVR_COUNT \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001
