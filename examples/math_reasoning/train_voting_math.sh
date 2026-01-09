#!/bin/bash
# Training script for Voting Math Workflow (parallel generation + aggregation pattern)
#
# This script trains a 2-agent workflow:
# - Generator: Creates N solutions in parallel
# - Aggregator: Reviews all solutions and selects the best one
#
# Usage: bash examples/math_reasoning/train_voting_math.sh

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO


python3 -m examples.math_reasoning.train_voting_math \
    data.max_prompt_length=15360 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name='rllm-workflow-MARL' \
    trainer.experiment_name='voting-qwen3_0.6b-math' \
    trainer.n_gpus_per_node=2 \
    trainer.agent_names=['generator','aggregator'] \
    +rllm.workflow.n_votes=3

pkill -9 -f 'ray::WorkerDict'
