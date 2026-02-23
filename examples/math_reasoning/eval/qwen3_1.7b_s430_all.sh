#!/bin/bash
#SBATCH --job-name=verl-ray
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgxh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=128G
#SBATCH --exclude=dgxh-1
#SBATCH --time=0-12:00:00

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
source ~/.bashrc && conda activate rllm

python -m examples.math_reasoning.evaluate_checkpoints \
    --eval-mode trained_checkpoint \
    --checkpoints-dir checkpoints/rllm-workflow-MARL-v2 \
    --dataset dapo_math \
    --experiment-filter orchestrator_workers-qwen3_1.7b_s430 \
    --base-model checkpoints/init_weight/qwen3_1.7b_s430 \
    --n-rollouts 1 \
    --n-parallel 512 \
    --port 8000