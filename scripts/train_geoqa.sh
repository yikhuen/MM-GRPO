#!/bin/bash
# ============================================================================
# Task 2: Geometric QA - Answer math questions about geometric figures
# ============================================================================
# Dataset: AI-ModelScope/GEOQA_R1V_Train_8K
# Model: Qwen2.5-VL-3B-Instruct
#
# Supports both single-GPU and multi-GPU setups:
#   Single GPU:  NUM_GPUS=1 bash scripts/train_geoqa.sh
#   Multi GPU:   bash scripts/train_geoqa.sh  (default: 6 GPUs)
#
# For multi-GPU, start the vLLM server first:
#   bash scripts/start_vllm_server.sh
# ============================================================================

export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}

NUM_GPUS=${NUM_GPUS:-6}

if [ "$NUM_GPUS" -eq 1 ]; then
    # ---- Single-GPU: colocated vLLM mode ----
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    MAX_PIXELS=262144 \
    NPROC_PER_NODE=1 \
    swift rlhf \
        --rlhf_type grpo \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --external_plugins examples/train/grpo/plugin/plugin.py \
        --reward_funcs external_r1v_acc format \
        --use_vllm true \
        --vllm_mode colocated \
        --tuner_type full \
        --torch_dtype bfloat16 \
        --dataset 'AI-ModelScope/GEOQA_R1V_Train_8K' \
        --load_from_cache_file true \
        --max_completion_length 1024 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --learning_rate 1e-6 \
        --gradient_accumulation_steps 8 \
        --save_strategy 'steps' \
        --eval_strategy 'steps' \
        --eval_steps 400 \
        --save_steps 400 \
        --save_total_limit 10 \
        --logging_steps 1 \
        --output_dir output/GRPO_GEOQA \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 4 \
        --num_generations 4 \
        --temperature 1.0 \
        --repetition_penalty 1.1 \
        --system 'examples/train/grpo/prompt.txt' \
        --deepspeed zero3 \
        --log_completions true \
        --report_to wandb \
        --num_iterations 2 \
        --async_generate false \
        --beta 0.001 \
        --max_grad_norm 0.5
else
    # ---- Multi-GPU: external vLLM server mode ----
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5} \
    MAX_PIXELS=401408 \
    NPROC_PER_NODE=$NUM_GPUS \
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
        --torch_dtype bfloat16 \
        --dataset 'AI-ModelScope/GEOQA_R1V_Train_8K' \
        --load_from_cache_file true \
        --max_completion_length 1024 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --learning_rate 1e-6 \
        --gradient_accumulation_steps 2 \
        --save_strategy 'steps' \
        --eval_strategy 'steps' \
        --eval_steps 400 \
        --save_steps 400 \
        --save_total_limit 10 \
        --logging_steps 1 \
        --output_dir output/GRPO_GEOQA \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 4 \
        --num_generations 8 \
        --temperature 1.0 \
        --repetition_penalty 1.1 \
        --system 'examples/train/grpo/prompt.txt' \
        --deepspeed zero3 \
        --log_completions true \
        --report_to wandb \
        --num_iterations 2 \
        --async_generate false \
        --beta 0.001 \
        --max_grad_norm 0.5
fi
