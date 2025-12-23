#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --exclude=dgxh-1
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=16
#SBATCH --mail-user=zengyif
#SBATCH --mail-type=ALL
#SBATCH --job-name=rllm

# Handle the conda env
source /nfs/stak/users/zengyif/.bashrc
conda activate rllm

cd /nfs/stak/users/zengyif/hpc-share/workspace/rllm_0.2.1

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO

python3 -m examples.math_reasoning.train_multi_agent_math \
    data.max_prompt_length=15360 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name='multi-agent-math-reasoning' \
    trainer.experiment_name='qwen3_1.7b-math_3agents-share_policy' \
    trainer.n_gpus_per_node=4 \
    trainer.agent_names=['generator','evaluator','refiner'] \
    trainer.share_policy=True \
    +rllm.workflow.max_refinement_iterations=3

pkill -9 -f 'ray::WorkerDict'
