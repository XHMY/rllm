#!/bin/bash
# Training script for Orchestrator-Workers Math Workflow (task decomposition pattern)
#
# This script trains a 2-agent workflow:
# - Orchestrator: Decomposes complex problems and synthesizes final answers
# - Worker: Solves individual subproblems in parallel
#
# Usage: bash examples/math_reasoning/train_orchestrator_workers_math.sh

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO


python3 -m examples.math_reasoning.train_orchestrator_workers_math \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name='rllm-workflow-MARL-v2' \
    trainer.experiment_name='orchestrator_workers-qwen3_0.6b-math-share_policy' \
    trainer.n_gpus_per_node=2 \
    trainer.agent_names=['orchestrator','worker'] \
    trainer.log_episodes=false \
    +rllm.workflow.max_subtasks=3 \
    +rllm.workflow.use_final_outcome_reward=true \
    trainer.share_policy=True

pkill -9 -f 'ray::WorkerDict'
