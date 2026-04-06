# Multimodal GRPO Training

Train multimodal vision-language models using GRPO (Group Relative Policy Optimization) with [ms-swift](https://github.com/modelscope/ms-swift). Based on the [SWIFT Multimodal GRPO Best Practices](https://swift.readthedocs.io/en/latest/BestPractices/GRPO-Multi-Modal-Training.html), referencing [R1-V](https://github.com/Deep-Agent/R1-V) and [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal).

## Setup

```bash
pip install -r requirements.txt
```

## Three Experiments

### 1. ClevrCount - Object Counting
- **Dataset:** `AI-ModelScope/clevr_cogen_a_train`
- **Task:** Count the number of objects in synthetic CLEVR images
- **Difficulty:** Easy - converges in ~500 steps
- **Expected:** Accuracy reward rises from ~0.4 to ~1.0

### 2. Geometric QA
- **Dataset:** `AI-ModelScope/GEOQA_R1V_Train_8K`
- **Task:** Answer math questions about geometric figures
- **Difficulty:** Medium - slower convergence, more oscillation
- **Expected:** Completion length converges to ~250 tokens

### 3. Multimodal Open R1
- **Dataset:** `lmms-lab/multimodal-open-r1-8k-verified`
- **Task:** Math reasoning on diverse multimodal data (Math360K + Geo170K)
- **Difficulty:** Hard - 8k samples, accuracy converges to ~0.5
- **Expected:** Completion length converges to ~200 tokens

## Usage

All experiments use **Qwen2.5-VL-3B-Instruct**. Scripts auto-configure for single or multi-GPU.

### Single GPU

No separate vLLM server needed — uses **colocated mode** (vLLM shares the GPU with training):

```bash
NUM_GPUS=1 bash scripts/train_clevr_count.sh
NUM_GPUS=1 bash scripts/train_geoqa.sh
NUM_GPUS=1 bash scripts/train_open_r1_multimodal.sh
```

Single-GPU uses reduced batch sizes (2), fewer generations (4-8), and higher gradient accumulation (8) to compensate.

### Multi-GPU (default: 8 GPUs)

**Step 1:** Start the vLLM rollout server (uses 2 GPUs):
```bash
bash scripts/start_vllm_server.sh
```

**Step 2:** Run a training script (uses 6 GPUs):
```bash
bash scripts/train_clevr_count.sh
bash scripts/train_geoqa.sh
bash scripts/train_open_r1_multimodal.sh
```

You can customize GPU count: `NUM_GPUS=4 bash scripts/train_clevr_count.sh`

### WandB Logging

Set your API key before running:
```bash
export WANDB_API_KEY=your_key_here
```

Or remove `--report_to wandb` from the training scripts to disable logging.

## Project Structure

```
.
├── examples/train/grpo/
│   ├── plugin/
│   │   └── plugin.py          # Reward functions (CountdownORM, MultiModalAccuracyORM)
│   └── prompt.txt             # System prompt for <think>/<answer> format
├── scripts/
│   ├── start_vllm_server.sh   # Launch vLLM rollout server
│   ├── train_clevr_count.sh   # Task 1: Object counting
│   ├── train_geoqa.sh         # Task 2: Geometry QA
│   └── train_open_r1_multimodal.sh  # Task 3: Open R1 multimodal
├── requirements.txt
└── README.md
```

## Reward Functions

- **`format`** (built-in): Checks for proper `<think>...</think><answer>...</answer>` format (from DeepSeek-R1)
- **`external_r1v_acc`** (custom): Accuracy reward using `math-verify` for symbolic verification with string-matching fallback

## Key Hyperparameters

| Parameter | ClevrCount | GeoQA | Open R1 |
|---|---|---|---|
| num_generations | 24 | 8 | 8 |
| num_iterations | 1 | 2 | 2 |
| max_grad_norm | default | 0.5 | 0.5 |
| MAX_PIXELS | default | 401408 | 262144 |
| warmup_ratio | 0.01 | 0.05 | 0.05 |
| repetition_penalty | default | 1.1 | 1.1 |

## GPU Configuration

| Setup | vLLM Mode | vLLM GPUs | Training GPUs |
|---|---|---|---|
| Single GPU | colocated (shared) | — | 1 |
| Multi-GPU (default) | external server | 2 (GPUs 6,7) | 6 (GPUs 0-5) |

Override GPU assignment with `CUDA_VISIBLE_DEVICES` and `NUM_GPUS` environment variables.

## Troubleshooting

- **vLLM errors with Qwen2.5-VL:** See [vllm#13285](https://github.com/vllm-project/vllm/issues/13285)
- **Training collapse (rewards drop, loss/grad_norm spike):** Lower `--max_grad_norm` (e.g., 0.5 or 0.3)
- **OOM:** Reduce `MAX_PIXELS`, `per_device_train_batch_size`, or `num_generations`

## Data Format

Custom datasets should follow this format:
```json
{
    "images": ["image_path1", "image_path2"],
    "messages": [
        {
            "role": "user",
            "content": "Your question here"
        }
    ],
    "solution": "<answer> ground_truth </answer>"
}
```
