#!/bin/bash
# Unified experiment launcher for math and deepcoder workflows.
# Generates and submits a single sbatch job from CLI arguments.
#
# Usage:
#   bash dashboard/launch_experiment.sh \
#       --workflow voting --model 1.7B --share-policy true --slurm-config dashboard/slurm_configs/2xH100_dgxh.conf
#
#   # Deepcoder
#   bash dashboard/launch_experiment.sh \
#       --workflow voting --model 1.7B --share-policy true --slurm-config dashboard/slurm_configs/2xH100_dgxh_code.conf --task-type deepcoder
#
#   # Preview without submitting
#   bash dashboard/launch_experiment.sh \
#       --workflow voting --model 1.7B --share-policy true --slurm-config dashboard/slurm_configs/2xL40s.conf --dry-run

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
TASK_TYPE="math"
N_GPUS=""
CPUS_PER_GPU=""
MEM_PER_GPU=""
ENTRY_POINT=""
AGENT_NAMES_OVERRIDE=""
MODEL_PATH_OVERRIDE=""
MAX_PROMPT=""
MAX_RESPONSE=""
WORKFLOW_PARAMS_OVERRIDE=""

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
        --task-type)     TASK_TYPE="$2";      shift 2 ;;
        --n-gpus)        N_GPUS="$2";                shift 2 ;;
        --cpus-per-gpu)  CPUS_PER_GPU="$2";          shift 2 ;;
        --mem-per-gpu)   MEM_PER_GPU="$2";           shift 2 ;;
        --entry-point)       ENTRY_POINT="$2";              shift 2 ;;
        --agent-names)       AGENT_NAMES_OVERRIDE="$2";     shift 2 ;;
        --model-path)        MODEL_PATH_OVERRIDE="$2";      shift 2 ;;
        --max-prompt)        MAX_PROMPT="$2";               shift 2 ;;
        --max-response)      MAX_RESPONSE="$2";             shift 2 ;;
        --workflow-params)   WORKFLOW_PARAMS_OVERRIDE="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ -z "$WORKFLOW" || -z "$MODEL" || -z "$SHARE_POLICY" ]]; then
    echo "ERROR: --workflow, --model, and --share-policy are required."
    echo "Usage: bash $0 --workflow <workflow> --model <model> --share-policy <true|false> [--slurm-config <path> | --node <node>] [--task-type <math|deepcoder>]"
    echo ""
    echo "  --workflow      evaluator_optimizer | voting | orchestrator_workers | single_agent"
    echo "  --model         0.6B | 1.7B | 4B"
    echo "  --share-policy  true | false"
    echo "  --slurm-config  Path to a .conf file with #SBATCH directives"
    echo "  --node          2xH100 | 2xL40s | 4xL40s  (legacy, use --slurm-config instead)"
    echo "  --task-type     math (default) | deepcoder"
    echo "  --dry-run       Print sbatch script without submitting"
    echo "  --project-name  Project name (default: rllm-workflow-MARL-v2)"
    echo "  --extra-args    Extra hydra overrides passed through verbatim"
    exit 1
fi
if [[ -z "$SLURM_CONFIG" && -z "$NODE" ]]; then
    echo "ERROR: Either --slurm-config or --node is required."
    exit 1
fi

# ── Fallback lookup functions (used for direct CLI invocation; dashboard path passes values via CLI args) ──

get_entry_point() {
    local task="$1" wf="$2"
    case "${task}_${wf}" in
        math_evaluator_optimizer)      echo "examples.math_reasoning.train_evaluator_optimizer_math" ;;
        math_voting)                   echo "examples.math_reasoning.train_voting_math" ;;
        math_orchestrator_workers)     echo "examples.math_reasoning.train_orchestrator_workers_math" ;;
        math_single_agent)             echo "examples.math_reasoning.train_single_agent_math" ;;
        deepcoder_evaluator_optimizer) echo "examples.deepcoder.train_deepcoder_evaluator_optimizer" ;;
        deepcoder_voting)              echo "examples.deepcoder.train_deepcoder_voting" ;;
        deepcoder_orchestrator_workers) echo "examples.deepcoder.train_deepcoder_orchestrator_workers" ;;
        deepcoder_single_agent)        echo "examples.deepcoder.train_single_agent_deepcoder" ;;
        *) echo "UNKNOWN"; return 1 ;;
    esac
}

get_agent_names() {
    case "$1" in
        evaluator_optimizer)  echo "['generator','evaluator']" ;;
        voting)               echo "['generator','aggregator']" ;;
        orchestrator_workers) echo "['orchestrator','worker','synthesizer']" ;;
        single_agent)         echo "['generator']" ;;
    esac
}

get_workflow_params() {
    local task="$1" wf="$2"
    case "${task}_${wf}" in
        math_evaluator_optimizer)
            echo "+rllm.workflow.max_iterations=3 rllm.workflow.use_final_outcome_reward=true" ;;
        math_voting)
            echo "+rllm.workflow.n_votes=3 rllm.workflow.use_final_outcome_reward=true" ;;
        math_orchestrator_workers)
            echo "+rllm.workflow.max_subtasks=3 rllm.workflow.use_final_outcome_reward=true +rllm.workflow.share_main_task_with_workers=false" ;;
        math_single_agent)
            echo "" ;;
        deepcoder_evaluator_optimizer)
            echo "+rllm.workflow.max_iterations=2 rllm.workflow.use_final_outcome_reward=true +rllm.workflow.enable_test_loop=False" ;;
        deepcoder_voting)
            echo "+rllm.workflow.n_votes=3 rllm.workflow.use_final_outcome_reward=true +rllm.workflow.enable_test_loop=False" ;;
        deepcoder_orchestrator_workers)
            echo "+rllm.workflow.max_subtasks=3 rllm.workflow.use_final_outcome_reward=true +rllm.workflow.share_main_task_with_workers=false +rllm.workflow.enable_test_loop=False" ;;
        deepcoder_single_agent)
            echo "+rllm.workflow.enable_test_loop=False" ;;
        *)
            echo "" ;;
    esac
}

get_prompt_response_len() {
    local task="$1" wf="$2"
    case "${task}_${wf}" in
        math_evaluator_optimizer)                     echo "30720 5120" ;;
        math_voting)                                  echo "20480 5120" ;;
        math_orchestrator_workers)                    echo "20480 3072" ;;
        math_single_agent)                            echo "15360 5120" ;;
        deepcoder_evaluator_optimizer|deepcoder_voting|deepcoder_orchestrator_workers)
            echo "10240 2048" ;;
        deepcoder_single_agent)                       echo "4096 2048" ;;
        *)                                            echo "15360 5120" ;;
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
parse_gpu_type() {
    grep -oP '^#\s*META:\s*GPU_TYPE=\K\S+' "$1"
}
read_sbatch_directives() {
    # Return #SBATCH lines, excluding job-name/output/error and GPU/CPU/memory (injected from args)
    grep '^#SBATCH' "$1" | grep -v -E '(--job-name|--output|--error|--gres=gpu|--cpus-per-gpu|--mem-per-gpu)'
}

# ── Read SLURM config ───────────────────────────────────────────────────────
if [[ -n "$SLURM_CONFIG" ]]; then
    if [[ ! -f "$SLURM_CONFIG" ]]; then
        echo "ERROR: Config not found: $SLURM_CONFIG"
        exit 1
    fi
    if [[ -z "$N_GPUS" || -z "$CPUS_PER_GPU" || -z "$MEM_PER_GPU" ]]; then
        echo "ERROR: --n-gpus, --cpus-per-gpu, and --mem-per-gpu are required when using --slurm-config."
        exit 1
    fi
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
    echo "ERROR: orchestrator_workers requires >=4 GPUs (${NODE:-config} has ${N_GPUS})"
    exit 1
fi

# ── Build experiment name ────────────────────────────────────────────────────
if [[ "$SHARE_POLICY" == "true" ]]; then
    policy_suffix="share_policy"
else
    policy_suffix="multi_lora"
fi
model_lower=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
exp_name="${WORKFLOW}-qwen3_${model_lower}-${policy_suffix}-${TASK_TYPE}"

# ── Resolve parameters (prefer CLI args from dashboard, fall back to shell lookups) ──
entry_point="${ENTRY_POINT:-$(get_entry_point "$TASK_TYPE" "$WORKFLOW")}"
agent_names="${AGENT_NAMES_OVERRIDE:-$(get_agent_names "$WORKFLOW")}"
model_path="${MODEL_PATH_OVERRIDE:-$(get_model_path "$MODEL")}"
if [[ -n "$MAX_PROMPT" && -n "$MAX_RESPONSE" ]]; then
    max_prompt="$MAX_PROMPT"; max_response="$MAX_RESPONSE"
else
    read -r max_prompt max_response <<< "$(get_prompt_response_len "$TASK_TYPE" "$WORKFLOW")"
fi
workflow_params="${WORKFLOW_PARAMS_OVERRIDE:-$(get_workflow_params "$TASK_TYPE" "$WORKFLOW")}"
ppo_max_token_len=$(get_ppo_max_token_len "$MODEL" "$GPU_TYPE")

# ── Build sbatch script ─────────────────────────────────────────────────────
sbatch_script="#!/bin/bash
#SBATCH --job-name=${exp_name}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err"

if [[ -n "$SBATCH_DIRECTIVES" ]]; then
    # New path: inject GPU/CPU/memory from CLI args, plus directives from config file
    sbatch_script+="
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --cpus-per-gpu=${CPUS_PER_GPU}
#SBATCH --mem-per-gpu=${MEM_PER_GPU}
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
set -x"

# Add task-type specific sbatch commands
if [[ "$TASK_TYPE" == "deepcoder" ]]; then
    TOTAL_CPUS=$(( CPUS_PER_GPU * N_GPUS ))
    workflow_params+=" rllm.workflow.code_executor_workers=${TOTAL_CPUS}"
    sbatch_script+="

ulimit -n 1048576"
fi

sbatch_script+="

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

    # Write slurm_job_id into training_metadata.json
    meta_dir="checkpoints/${PROJECT_NAME}/${exp_name}"
    meta_file="${meta_dir}/training_metadata.json"
    mkdir -p "$meta_dir"
    if [[ -f "$meta_file" ]]; then
        # Update existing metadata — add/overwrite slurm_job_id
        tmp_meta=$(mktemp)
        python3 -c "
import json, sys
with open('$meta_file') as f:
    meta = json.load(f)
meta['slurm_job_id'] = '$job_id'
with open('$tmp_meta', 'w') as f:
    json.dump(meta, f, indent=2)
" && mv "$tmp_meta" "$meta_file"
    else
        # Create new metadata file
        echo "{\"slurm_job_id\": \"${job_id}\", \"experiment_name\": \"${exp_name}\", \"project_name\": \"${PROJECT_NAME}\"}" | python3 -m json.tool > "$meta_file"
    fi

    echo "Submitted ${exp_name} → Job ${job_id}"
fi
