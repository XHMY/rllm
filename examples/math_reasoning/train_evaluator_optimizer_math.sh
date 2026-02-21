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

python3 -m examples.math_reasoning.train_evaluator_optimizer_math \
    data.max_prompt_length=30720 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=checkpoints/init_weight/qwen3_1.7b_s430 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    trainer.project_name='rllm-workflow-MARL-v2' \
    trainer.experiment_name='evaluator_optimizer-qwen3_1.7b_s430-math' \
    trainer.n_gpus_per_node=2 \
    trainer.agent_names=['generator','evaluator'] \
    rllm.workflow.use_final_outcome_reward=true \
    +rllm.workflow.max_iterations=3 \
    trainer.total_training_steps=400


# To warm-start generator from a single-agent checkpoint:
# +rllm.workflow.initial_lora_weights.generator=/path/to/checkpoints/.../actor/lora_adapter_generator

# To use final outcome reward for entire trajectory (experimental):
# This assigns the final episode outcome reward to ALL trajectories,
# testing whether multi-agent training can converge without fine-grained
# per-agent reward feedback.
# rllm.workflow.use_final_outcome_reward=true

# pkill -9 -f 'ray::WorkerDict'
