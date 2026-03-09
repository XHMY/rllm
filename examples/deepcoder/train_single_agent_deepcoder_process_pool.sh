#!/bin/bash
# ProcessPoolExecutor: code_executor_workers=64 (persistent worker processes)
#
# Uses a dedicated ProcessPoolExecutor with 64 workers for code reward evaluation.
# Each worker spawns a subprocess (multiprocessing.Process) per problem with kill-based
# timeout via p.kill(), preventing hangs from user code catching TimeoutException.
#
# Compare wall-clock time against train_single_agent_deepcoder_baseline.sh to measure speedup.
#
# Usage: bash examples/deepcoder/train_single_agent_deepcoder_process_pool.sh

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
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    trainer.project_name='rllm-workflow-MARL-v2-deepcoder' \
    trainer.experiment_name='single_agent-qwen3_4b-deepcoder-process-pool' \
    trainer.n_gpus_per_node=2 \
    trainer.share_policy=False \
    trainer.agent_names=['generator'] \
    trainer.log_episodes=False \
    +rllm.workflow.enable_test_loop=False \
    rllm.workflow.code_executor_workers=64 \
    trainer.total_training_steps=301

pkill -9 -f 'ray::WorkerDict'
