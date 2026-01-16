#!/bin/bash
#SBATCH --job-name=verl-ray
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=preempt
#SBATCH --nodes=4
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1       # Important: 1 task per node to launch Ray daemon
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --constraint=a40
#SBATCH --time=4-00:00:00

source /nfs/stak/users/zengyif/.bashrc
conda activate rllm
cd /nfs/stak/users/zengyif/hpc-share/workspace/rllm_0.2.1

# Create logs directory if not exists
mkdir -p logs

# --- Environment Variables ---
set -x
# Note: ulimit may fail on some clusters, that's OK
ulimit -n 1048576 2>/dev/null || true
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO

# --- FIX: Unset conflicting SLURM memory variables for srun ---
# These cause "SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive" error
unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU
unset SLURM_MEM_PER_NODE
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1

# --- RAY CLUSTER LAUNCH ---

# 1. Get the Head Node IP (use hostname -i to get actual IP address)
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -i | head -n 1)
port=6379

echo "Starting Ray head on node $head_node (IP: $head_node_ip)"

# 2. Start Ray Head on the first node (Node 0)
# Calculate resources: cpus-per-gpu * gpus-per-node
num_cpus=$((${SLURM_CPUS_PER_GPU:-8} * 2))
num_gpus=2

srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus=$num_cpus --num-gpus=$num_gpus --block &

# Wait for head to start
sleep 15

# 3. Start Ray Workers on the remaining nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))

if [ $worker_num -gt 0 ]; then
    echo "Starting $worker_num Ray workers connecting to $head_node_ip:$port"
    srun --nodes=$worker_num --ntasks=$worker_num --exclude="$head_node" \
        ray start --address="$head_node_ip:$port" \
        --num-cpus=$num_cpus --num-gpus=$num_gpus --block &
fi

# Wait for workers to connect
sleep 30

# --- SET RAY_ADDRESS for Python to connect ---
export RAY_ADDRESS="$head_node_ip:$port"

# Verify Ray cluster is running
echo "Checking Ray cluster status..."
ray status || echo "Warning: Could not get ray status, but continuing..."

# --- RUN TRAINING ---

# Run the training script with multi-node configuration
python3 -m examples.deepcoder.train_single_agent_deepcoder \
    data.max_prompt_length=15360 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    trainer.project_name='rllm-workflow-MARL' \
    trainer.experiment_name='single_agent-qwen3_0.6b-deepcoder-multinode' \
    trainer.nnodes=$SLURM_NNODES \
    trainer.n_gpus_per_node=2 \
    trainer.share_policy=False \
    +rllm.workflow.enable_test_loop=False

pkill -9 -f 'ray::WorkerDict'

# Keep script running to prevent Ray from shutting down early
wait