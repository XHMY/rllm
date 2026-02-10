#!/bin/bash
# Training script for Single-Agent DeepCoder Workflow
#
# This script trains a single-agent workflow for code generation:
# - Generator: Creates code solution
# - Reward: Computed via test execution
#
# This is a simpler baseline compared to multi-agent workflows.
#
# Usage: bash examples/deepcoder/train_single_agent_deepcoder.sh

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
    data.max_prompt_length=30720 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51200 \
    trainer.project_name='rllm-workflow-MARL-v2-deepcoder' \
    trainer.experiment_name='single_agent-qwen3_1.7b-deepcoder' \
    trainer.n_gpus_per_node=2 \
    trainer.share_policy=False \
    trainer.agent_names=['generator'] \
    trainer.log_episodes=False \
    +rllm.workflow.enable_test_loop=False

pkill -9 -f 'ray::WorkerDict'


