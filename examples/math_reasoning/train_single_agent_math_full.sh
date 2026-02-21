#!/bin/bash
#SBATCH --job-name=verl-ray
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgxh
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=128G
#SBATCH --exclude=dgxh-1
#SBATCH --time=0-12:00:00

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
source ~/.bashrc && conda activate rllm

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO

python3 -m examples.math_reasoning.train_single_agent_math \
    data.max_prompt_length=15360 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51200 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    trainer.project_name='rllm-workflow-MARL-v2' \
    trainer.experiment_name='qwen3_1.7b-fullp-math_single_agent' \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=false \
    trainer.agent_names=['generator'] \
    trainer.share_policy=true \
    trainer.total_training_steps=300

pkill -9 -f 'ray::WorkerDict'


# 1.7B
# actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \