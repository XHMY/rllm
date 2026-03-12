#!/bin/bash
#SBATCH --job-name=verl-ray
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
#SBATCH --constraint=l40s
#SBATCH --time=7-00:00:00
#SBATCH --requeue

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
source ~/.bashrc && conda activate rllm

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO

python3 -m examples.deepcoder.train_single_agent_deepcoder \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=23554 \
    trainer.project_name='rllm-workflow-MARL-v2' \
    trainer.experiment_name='single_agent-qwen3_1.7b-deepcoder' \
    trainer.n_gpus_per_node=4 \
    trainer.share_policy=False \
    trainer.agent_names=['generator'] \
    trainer.log_episodes=False \
    +rllm.workflow.enable_test_loop=False \
    rllm.workflow.code_executor_workers=64 \
    trainer.total_training_steps=301

pkill -9 -f 'ray::WorkerDict'


# 4B for H100: actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \