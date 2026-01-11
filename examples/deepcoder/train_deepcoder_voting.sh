#!/bin/bash
# Training script for Deepcoder Voting Workflow (2-agent parallel generation + aggregation)
#
# This script trains a 2-agent workflow for code generation:
# - Generator: Creates N code solutions in parallel
# - Aggregator: Reviews all solutions and selects the best one
#
# Pattern: Generate N parallel → Aggregate → Test selected solution
#
# Usage: bash examples/deepcoder/train_deepcoder_voting.sh

set -x

ulimit -n 1048576
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO

# Model - use Qwen3-1.7B for code generation
MODEL_PATH=Qwen/Qwen3-1.7B

python3 -m examples.deepcoder.train_deepcoder_voting \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    trainer.project_name='rllm-workflow-MARL' \
    trainer.experiment_name='voting-qwen3_0.6b-deepcoder' \
    trainer.n_gpus_per_node=4 \
    trainer.share_policy=False \
    trainer.agent_names=['generator','aggregator'] \
    +rllm.workflow.n_votes=3 \
    +rllm.workflow.enable_test_loop=False \
    +rllm.workflow.max_test_rounds=2 \
    +rllm.workflow.max_tests_to_show=3 \
    +rllm.workflow.public_test_only=False

pkill -9 -f 'ray::WorkerDict'
