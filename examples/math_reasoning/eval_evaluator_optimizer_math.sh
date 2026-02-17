#!/bin/bash
# Training script for Evaluator-Optimizer Math Workflow (2-agent pattern)
#
# This script trains a 2-agent workflow:
# - Generator: Creates initial solution AND refines based on feedback
# - Evaluator: Reviews solutions and provides feedback
#
# Usage: bash examples/math_reasoning/train_evaluator_optimizer_math.sh

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
    data.max_prompt_length=15360 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name='rllm-workflow-MARL-v2' \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51200 \
    trainer.experiment_name='evaluator_optimizer-qwen3_0.6b-math-eval' \
    trainer.n_gpus_per_node=2 \
    trainer.agent_names=['generator','evaluator'] \
    trainer.val_before_train=true \
    trainer.total_training_steps=200 \
    trainer.log_episodes=true \
    rllm.workflow.use_final_outcome_reward=true \
    +rllm.workflow.max_iterations=3
    

# To use final outcome reward for entire trajectory (experimental):
# This assigns the final episode outcome reward to ALL trajectories,
# testing whether multi-agent training can converge without fine-grained
# per-agent reward feedback.
# rllm.workflow.use_final_outcome_reward=true

pkill -9 -f 'ray::WorkerDict'
