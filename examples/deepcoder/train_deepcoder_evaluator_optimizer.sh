#!/bin/bash
# Training script for Deepcoder Evaluator-Optimizer Workflow (2-agent pattern with test loops)
#
# This script trains a 2-agent workflow for code generation:
# - Generator: Creates code AND refines based on feedback
# - Evaluator: Reviews code logic and provides feedback
#
# Nested loops:
# - Outer loop: Test rounds (generate -> test -> refine with test feedback)
# - Inner loop: Eval-opt (generate -> evaluate logic -> refine with eval feedback)
#
# Usage: bash examples/deepcoder/train_deepcoder_evaluator_optimizer.sh

set -x

ulimit -n 10240
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO


python3 -m examples.deepcoder.train_deepcoder_evaluator_optimizer \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=checkpoints/init_weight/qwen3_4b_s300_deepcoder \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    trainer.project_name='rllm-workflow-MARL-v2' \
    trainer.experiment_name='evaluator_optimizer-qwen3_4b_s300-deepcoder' \
    trainer.n_gpus_per_node=2 \
    trainer.share_policy=False \
    trainer.agent_names=['generator','evaluator'] \
    rllm.workflow.use_final_outcome_reward=true \
    +rllm.workflow.max_iterations=2 \
    +rllm.workflow.enable_test_loop=False

# pkill -9 -f 'ray::WorkerDict'
