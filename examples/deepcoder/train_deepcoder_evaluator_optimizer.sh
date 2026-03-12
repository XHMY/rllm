#!/bin/bash
#SBATCH --job-name=verl-ray
#SBATCH --account=hw-grp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgxh
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=128G
#SBATCH --exclude=dgxh-1
#SBATCH --time=1-0:00:00

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
source ~/.bashrc && conda activate rllm

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
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    trainer.project_name='rllm-workflow-MARL-v2' \
    trainer.experiment_name='evaluator_optimizer-qwen3_4b-deepcoder' \
    trainer.n_gpus_per_node=2 \
    trainer.share_policy=False \
    trainer.agent_names=['generator','evaluator'] \
    rllm.workflow.use_final_outcome_reward=true \
    +rllm.workflow.max_iterations=2 \
    +rllm.workflow.enable_test_loop=False \
    rllm.workflow.code_executor_workers=48

# pkill -9 -f 'ray::WorkerDict'
