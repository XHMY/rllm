#!/bin/bash
set -euo pipefail

# submit_eval_jobs.sh - Auto-submit sbatch evaluation jobs per experiment
#
# Usage:
#   bash dashboard/submit_eval_jobs.sh <model_filter> [options]
#
# Examples:
#   bash dashboard/submit_eval_jobs.sh qwen3_1.7b
#   bash dashboard/submit_eval_jobs.sh qwen3_1.7b --task-type deepcoder
#   bash dashboard/submit_eval_jobs.sh qwen3_0.6b --dataset aime2025 --dry-run

usage() {
    cat <<'EOF'
Usage: bash dashboard/submit_eval_jobs.sh <model_filter> [options]

Required:
  model_filter          Substring to match experiment directory names (e.g., qwen3_1.7b)

Options:
  --task-type TYPE      Task type: math (default) or deepcoder
  --base-model PATH     Base model path (default: omitted, auto-detected by evaluate_checkpoints.py)
  --checkpoints-dir DIR Checkpoint root (default: checkpoints/rllm-workflow-MARL-v2)
  --dataset NAME        Dataset name (default: dapo_math for math, deepcoder for deepcoder)
  --n-rollouts N        Number of rollouts (default: 1)
  --n-parallel N        Parallel instances (default: 512)
  --partition NAME      SLURM partition (default: preempt)
  --constraint NAME     GPU constraint (default: l40s)
  --mem-per-gpu SIZE    Memory per GPU (default: 80G)
  --time LIMIT          Time limit (default: 1-0:00:00)
  --dry-run             Print sbatch commands without submitting
  -h, --help            Show this help message
EOF
}

# --- Defaults ---
TASK_TYPE="math"
BASE_MODEL=""
CHECKPOINTS_DIR="checkpoints/rllm-workflow-MARL-v2"
DATASET=""
N_ROLLOUTS=1
N_PARALLEL=512
PARTITION="preempt"
CONSTRAINT="l40s"
MEM_PER_GPU="80G"
TIME_LIMIT="1-0:00:00"
DRY_RUN=false

# --- Parse arguments ---
if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

# Handle help as first argument
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

MODEL_FILTER="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-type)
            TASK_TYPE="$2"; shift 2 ;;
        --base-model)
            BASE_MODEL="$2"; shift 2 ;;
        --checkpoints-dir)
            CHECKPOINTS_DIR="$2"; shift 2 ;;
        --dataset)
            DATASET="$2"; shift 2 ;;
        --n-rollouts)
            N_ROLLOUTS="$2"; shift 2 ;;
        --n-parallel)
            N_PARALLEL="$2"; shift 2 ;;
        --partition)
            PARTITION="$2"; shift 2 ;;
        --constraint)
            CONSTRAINT="$2"; shift 2 ;;
        --mem-per-gpu)
            MEM_PER_GPU="$2"; shift 2 ;;
        --time)
            TIME_LIMIT="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Error: Unknown option: $1" >&2
            usage
            exit 1 ;;
    esac
done

# --- Resolve defaults per task type ---
if [[ -z "$DATASET" ]]; then
    case "$TASK_TYPE" in
        math)      DATASET="dapo_math" ;;
        deepcoder) DATASET="deepcoder" ;;
        *) echo "Error: Unknown task type: $TASK_TYPE" >&2; exit 1 ;;
    esac
fi

# Deepcoder needs more CPUs for code execution
CPUS_PER_GPU=4
if [[ "$TASK_TYPE" == "deepcoder" ]]; then
    CPUS_PER_GPU=8
fi

# --- Validate checkpoints directory ---
if [[ ! -d "$CHECKPOINTS_DIR" ]]; then
    echo "Error: Checkpoints directory not found: $CHECKPOINTS_DIR" >&2
    exit 1
fi

# --- Discover matching experiments ---
EXPERIMENTS=()
while IFS= read -r dir; do
    EXPERIMENTS+=("$dir")
done < <(ls -1 "$CHECKPOINTS_DIR" | grep -E "$MODEL_FILTER" || true)

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
    echo "No experiments matching '$MODEL_FILTER' found in $CHECKPOINTS_DIR" >&2
    exit 1
fi

echo "Found ${#EXPERIMENTS[@]} experiment(s) matching '$MODEL_FILTER':"
for exp in "${EXPERIMENTS[@]}"; do
    echo "  - $exp"
done
echo ""

# --- Ensure logs directory exists ---
mkdir -p logs

# --- Helper to generate sbatch script content ---
generate_sbatch_script() {
    local exp_name="$1"
    local port="$2"

    cat <<'HEADER'
#!/bin/bash
HEADER
    echo "#SBATCH --job-name=eval-${exp_name}"
    echo "#SBATCH --output=logs/eval_%x_%j.out"
    echo "#SBATCH --error=logs/eval_%x_%j.err"
    echo "#SBATCH --partition=${PARTITION}"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --gres=gpu:1"
    echo "#SBATCH --cpus-per-gpu=${CPUS_PER_GPU}"
    echo "#SBATCH --exclude=dgxh-1"
    echo "#SBATCH --mem-per-gpu=${MEM_PER_GPU}"
    echo "#SBATCH --constraint=${CONSTRAINT}"
    echo "#SBATCH --time=${TIME_LIMIT}"
    echo ""
    cat <<'SETUP'
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
source ~/.bashrc && conda activate rllm

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

SETUP

    # Add task-type specific commands
    if [[ "$TASK_TYPE" == "deepcoder" ]]; then
        echo "ulimit -n 1048576"
        echo ""
    fi

    # Build the python command with proper line continuations
    echo "python -m dashboard.evaluate_checkpoints \\"
    echo "    --task-type ${TASK_TYPE} \\"
    echo "    --eval-mode trained_checkpoint \\"
    echo "    --checkpoints-dir ${CHECKPOINTS_DIR} \\"
    echo "    --dataset ${DATASET} \\"
    echo "    --experiment-filter '^${exp_name}\$' \\"
    echo "    --n-rollouts ${N_ROLLOUTS} \\"
    echo "    --n-parallel ${N_PARALLEL} \\"
    if [[ -n "$BASE_MODEL" ]]; then
        echo "    --port ${port} \\"
        echo "    --base-model ${BASE_MODEL}"
    else
        echo "    --port ${port}"
    fi
}

# --- Submit one job per experiment ---
SUBMITTED=0
for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
    PORT=$(shuf -i 8000-9999 -n 1)

    TMPFILE=$(mktemp /tmp/eval_sbatch_XXXXXX.sh)
    generate_sbatch_script "$EXPERIMENT_NAME" "$PORT" > "$TMPFILE"

    if [[ "$DRY_RUN" == true ]]; then
        echo "=== [DRY RUN] Would submit job for: $EXPERIMENT_NAME (port $PORT) ==="
        cat "$TMPFILE"
        echo ""
        rm -f "$TMPFILE"
    else
        JOB_OUTPUT=$(sbatch "$TMPFILE" 2>&1)
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+' | tail -1)
        echo "Submitted: $EXPERIMENT_NAME | Job ID: ${JOB_ID:-N/A} | Port: $PORT"
        rm -f "$TMPFILE"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete. ${#EXPERIMENTS[@]} job(s) would be submitted."
else
    echo ""
    echo "Done. Submitted $SUBMITTED job(s). Check with: squeue -u \$USER"
fi
