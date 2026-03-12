#!/bin/bash
# Unified experiment launcher for math reasoning workflows.
# Generates and submits a single sbatch job from CLI arguments.
#
# Usage:
#   bash examples/math_reasoning/launch_experiment.sh \
#       --workflow voting --model 1.7B --share-policy true --node 2xH100
#
#   # Preview without submitting
#   bash examples/math_reasoning/launch_experiment.sh \
#       --workflow voting --model 1.7B --share-policy true --node 2xH100 --dry-run

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
WORKFLOW=""
MODEL=""
SHARE_POLICY=""
NODE=""
SLURM_CONFIG=""
SBATCH_DIRECTIVES=""
DRY_RUN=false
PROJECT_NAME="rllm-workflow-MARL-v2"
EXTRA_ARGS=""

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workflow)      WORKFLOW="$2";       shift 2 ;;
        --model)         MODEL="$2";          shift 2 ;;
        --share-policy)  SHARE_POLICY="$2";   shift 2 ;;
        --node)          NODE="$2";           shift 2 ;;
        --slurm-config)  SLURM_CONFIG="$2";   shift 2 ;;
        --dry-run)       DRY_RUN=true;        shift   ;;
        --project-name)  PROJECT_NAME="$2";   shift 2 ;;
        --extra-args)    EXTRA_ARGS="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ -z "$WORKFLOW" || -z "$MODEL" || -z "$SHARE_POLICY" ]]; then
    echo "ERROR: --workflow, --model, and --share-policy are required."
    echo "Usage: bash $0 --workflow <workflow> --model <model> --share-policy <true|false> [--slurm-config <path> | --node <node>]"
    echo ""
    echo "  --workflow      evaluator_optimizer | voting | orchestrator_workers"
    echo "  --model         0.6B | 1.7B | 4B"
    echo "  --share-policy  true | false"
    echo "  --slurm-config  Path to a .conf file with #SBATCH directives"
    echo "  --node          2xH100 | 2xL40s | 4xL40s  (legacy, use --slurm-config instead)"
    echo "  --dry-run       Print sbatch script without submitting"
    echo "  --project-name  Project name (default: rllm-workflow-MARL-v2)"
    echo "  --extra-args    Extra hydra overrides passed through verbatim"
    exit 1
fi
if [[ -z "$SLURM_CONFIG" && -z "$NODE" ]]; then
    echo "ERROR: Either --slurm-config or --node is required."
    exit 1
fi

# ── Lookup functions ─────────────────────────────────────────────────────────

get_entry_point() {
    case "$1" in
        evaluator_optimizer)  echo "examples.math_reasoning.train_evaluator_optimizer_math" ;;
        voting)               echo "examples.math_reasoning.train_voting_math" ;;
        orchestrator_workers) echo "examples.math_reasoning.train_orchestrator_workers_math" ;;
        *) echo "UNKNOWN"; return 1 ;;
    esac
}

get_agent_names() {
    case "$1" in
        evaluator_optimizer)  echo "['generator','evaluator']" ;;
        voting)               echo "['generator','aggregator']" ;;
        orchestrator_workers) echo "['orchestrator','worker','synthesizer']" ;;
    esac
}

get_workflow_params() {
    case "$1" in
        evaluator_optimizer)
            echo "+rllm.workflow.max_iterations=3 rllm.workflow.use_final_outcome_reward=true" ;;
        voting)
            echo "+rllm.workflow.n_votes=3 rllm.workflow.use_final_outcome_reward=true" ;;
        orchestrator_workers)
            echo "+rllm.workflow.max_subtasks=3 rllm.workflow.use_final_outcome_reward=true +rllm.workflow.share_main_task_with_workers=false" ;;
    esac
}

get_prompt_response_len() {
    case "$1" in
        evaluator_optimizer|voting) echo "30720 5120" ;;
        orchestrator_workers)       echo "20480 3072" ;;
    esac
}

get_model_path() {
    case "$1" in
        0.6B) echo "Qwen/Qwen3-0.6B" ;;
        1.7B) echo "Qwen/Qwen3-1.7B" ;;
        4B)   echo "Qwen/Qwen3-4B" ;;
        *) echo "UNKNOWN"; return 1 ;;
    esac
}

get_slurm_config() {
    case "$1" in
        2xH100)  echo "dgxh 2 none 8 128G 1-00:00:00 --exclude=dgxh-1" ;;
        2xL40s)  echo "preempt 2 l40s 4 48G 7-00:00:00 --requeue" ;;
        4xL40s)  echo "preempt 4 l40s 4 48G 7-00:00:00 --requeue" ;;
        *) echo "UNKNOWN"; return 1 ;;
    esac
}

get_gpu_type() {
    case "$1" in
        2xH100)        echo "H100" ;;
        2xL40s|4xL40s) echo "L40s" ;;
    esac
}

get_ppo_max_token_len() {
    local model="$1" gpu_type="$2"
    case "${model}_${gpu_type}" in
        0.6B_L40s) echo 30720 ;;
        0.6B_H100) echo 61440 ;;
        1.7B_L40s) echo 23554 ;;
        1.7B_H100) echo 51200 ;;
        4B_L40s)   echo 10240 ;;
        4B_H100)   echo 40960 ;;
        *) echo "UNKNOWN"; return 1 ;;
    esac
}

# ── Config-file parsing helpers ───────────────────────────────────────────────
parse_gpu_count() {
    grep -oP '#SBATCH\s+--gres=gpu:\K\d+' "$1"
}
parse_gpu_type() {
    grep -oP '^#\s*META:\s*GPU_TYPE=\K\S+' "$1"
}
read_sbatch_directives() {
    # Return #SBATCH lines, excluding job-name/output/error (those are set by the script)
    grep '^#SBATCH' "$1" | grep -v -E '(--job-name|--output|--error)'
}

# ── Read SLURM config ───────────────────────────────────────────────────────
if [[ -n "$SLURM_CONFIG" ]]; then
    if [[ ! -f "$SLURM_CONFIG" ]]; then
        echo "ERROR: Config not found: $SLURM_CONFIG"
        exit 1
    fi
    N_GPUS=$(parse_gpu_count "$SLURM_CONFIG")
    GPU_TYPE=$(parse_gpu_type "$SLURM_CONFIG")
    SBATCH_DIRECTIVES=$(read_sbatch_directives "$SLURM_CONFIG")
else
    # Legacy --node path
    read -r PARTITION N_GPUS CONSTRAINT CPUS_PER_GPU MEM_PER_GPU TIME_LIMIT SLURM_EXTRA \
        <<< "$(get_slurm_config "$NODE")"
    GPU_TYPE=$(get_gpu_type "$NODE")
fi

# ── Compatibility check ─────────────────────────────────────────────────────
if [[ "$WORKFLOW" == "orchestrator_workers" && "$N_GPUS" -lt 4 ]]; then
    echo "ERROR: orchestrator_workers requires >=4 GPUs (${NODE} has ${N_GPUS})"
    exit 1
fi

# ── Build experiment name ────────────────────────────────────────────────────
if [[ "$SHARE_POLICY" == "true" ]]; then
    policy_suffix="share_policy"
else
    policy_suffix="multi_lora"
fi
model_lower=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
exp_name="${WORKFLOW}-qwen3_${model_lower}-${policy_suffix}-math"

# ── Resolve parameters ──────────────────────────────────────────────────────
entry_point=$(get_entry_point "$WORKFLOW")
agent_names=$(get_agent_names "$WORKFLOW")
workflow_params=$(get_workflow_params "$WORKFLOW")
read -r max_prompt max_response <<< "$(get_prompt_response_len "$WORKFLOW")"
model_path=$(get_model_path "$MODEL")
ppo_max_token_len=$(get_ppo_max_token_len "$MODEL" "$GPU_TYPE")

# ── Build sbatch script ─────────────────────────────────────────────────────
sbatch_script="#!/bin/bash
#SBATCH --job-name=verl-ray
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err"

if [[ -n "$SBATCH_DIRECTIVES" ]]; then
    # New path: insert directives from config file
    sbatch_script+="
${SBATCH_DIRECTIVES}"
else
    # Legacy path: build from parsed variables
    sbatch_script+="
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --cpus-per-gpu=${CPUS_PER_GPU}
#SBATCH --mem-per-gpu=${MEM_PER_GPU}
#SBATCH --time=${TIME_LIMIT}"

    if [[ "$CONSTRAINT" != "none" ]]; then
        sbatch_script+="
#SBATCH --constraint=${CONSTRAINT}"
    fi
    if [[ "$SLURM_EXTRA" == *"--exclude="* ]]; then
        exclude_val="${SLURM_EXTRA#*--exclude=}"
        sbatch_script+="
#SBATCH --exclude=${exclude_val}"
    fi
    if [[ "$SLURM_EXTRA" == *"--requeue"* ]]; then
        sbatch_script+="
#SBATCH --requeue"
    fi
fi

sbatch_script+="

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
source ~/.bashrc && conda activate rllm
set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=\"expandable_segments:False\"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO

python3 -m ${entry_point} \\
    data.max_prompt_length=${max_prompt} \\
    data.max_response_length=${max_response} \\
    actor_rollout_ref.model.path=${model_path} \\
    trainer.project_name='${PROJECT_NAME}' \\
    trainer.experiment_name='${exp_name}' \\
    trainer.n_gpus_per_node=${N_GPUS} \\
    trainer.agent_names=${agent_names} \\
    trainer.share_policy=${SHARE_POLICY^} \\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len} \\
    ${workflow_params}"

if [[ -n "$EXTRA_ARGS" ]]; then
    sbatch_script+=" \\
    ${EXTRA_ARGS}"
fi

sbatch_script+="
"

# ── Submit or print ──────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "DRY RUN: ${exp_name}"
    echo "================================================================================"
    echo "$sbatch_script"
else
    tmpfile=$(mktemp /tmp/launch_exp_XXXXXX.sh)
    echo "$sbatch_script" > "$tmpfile"
    mkdir -p logs
    output=$(sbatch "$tmpfile" 2>&1)
    job_id=$(echo "$output" | grep -oP '\d+' | tail -1)
    rm -f "$tmpfile"
    echo "Submitted ${exp_name} → Job ${job_id}"
fi
