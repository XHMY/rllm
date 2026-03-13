"""RLLM Experiment Dashboard — pure Python backend logic."""

import json
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from dashboard.task_configs import EVAL_DATASETS, infer_task_type

DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_PROJECT = "rllm-workflow-MARL-v2"
TOTAL_TRAINING_STEPS = 301
LAUNCHER_SCRIPT = "dashboard/launch_experiment.sh"
SLURM_CONFIGS_DIR = Path("dashboard/slurm_configs")

MODEL_TABS = ["0.6B", "1.7B", "4B"]
DATASET_CATEGORIES = {
    "Math": ["math"],
    "Code": ["deepcoder"],
}
STATUS_EMOJI = {"Finished": "\u2705", "Running": "\ud83d\udd04", "Unfinished": "\u23f8\ufe0f"}
POLICY_ABBREV = {
    "share_policy": "share",
    "multi_lora": "multi",
}


# ── SLURM config loading ───────────────────────────────────────────────────


def load_slurm_configs() -> list[str]:
    """Return sorted config names (filenames without .conf extension)."""
    if not SLURM_CONFIGS_DIR.is_dir():
        return []
    return sorted(p.stem for p in SLURM_CONFIGS_DIR.glob("*.conf"))


def parse_slurm_config(config_name: str) -> dict:
    """Extract gpu_type from a config file's META line."""
    config_path = SLURM_CONFIGS_DIR / f"{config_name}.conf"
    result = {"gpu_type": None}
    if not config_path.exists():
        return result
    text = config_path.read_text()
    m = re.search(r"^#\s*META:\s*GPU_TYPE=(\S+)", text, re.MULTILINE)
    if m:
        result["gpu_type"] = m.group(1)
    return result


# ── Eval results loading ──────────────────────────────────────────────────


def load_eval_results(
    experiment_name: str, checkpoint_dir: str, project: str = DEFAULT_PROJECT
) -> list[dict]:
    """Load eval results from per-experiment JSONL file, deduplicated.

    Source: {checkpoint_dir}/{project}/{experiment_name}/eval_results.jsonl

    Dedup key: (checkpoint_step, dataset, eval_mode, n_rollouts)
    """
    results: dict[tuple, dict] = {}

    per_exp_path = Path(checkpoint_dir) / project / experiment_name / "eval_results.jsonl"
    if per_exp_path.exists():
        try:
            for line in per_exp_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (
                        rec.get("checkpoint_step"),
                        rec.get("dataset"),
                        rec.get("eval_mode"),
                        rec.get("n_rollouts"),
                    )
                    results[key] = rec
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass

    return list(results.values())


# ── Checkpoint scanning ─────────────────────────────────────────────────────


def parse_experiment_name(name: str) -> dict:
    """Parse experiment directory name into components.

    Expected format: {workflow}-qwen3_{model}-{policy}-{dataset}
    Examples:
        evaluator_optimizer-qwen3_1.7b-multi_lora-math
        voting-qwen3_1.7b-share_policy-math
        single_agent-qwen3_1.7b-deepcoder
    """
    m = re.match(
        r"^(.+?)-qwen3_([\d.]+[bB])-(.+?)-(\w+)$", name
    )
    if m:
        return {
            "workflow": m.group(1),
            "model": m.group(2),
            "policy": m.group(3),
            "dataset": m.group(4),
        }
    m = re.match(r"^(.+?)-qwen3_([\d.]+[bB])-(\w+)$", name)
    if m:
        return {
            "workflow": m.group(1),
            "model": m.group(2),
            "policy": "\u2014",
            "dataset": m.group(3),
        }
    return {"workflow": name, "model": "\u2014", "policy": "\u2014", "dataset": "\u2014"}


def infer_gpu_count(exp_dir: Path) -> int | None:
    """Infer GPU count from checkpoint file naming: model_world_size_N_rank_0.pt."""
    for step_dir in exp_dir.glob("global_step_*"):
        actor_dir = step_dir / "actor"
        if not actor_dir.is_dir():
            continue
        for f in actor_dir.iterdir():
            m = re.match(r"model_world_size_(\d+)_rank_0\.pt", f.name)
            if m:
                return int(m.group(1))
    return None


def scan_checkpoints(checkpoint_dir: str, project: str = DEFAULT_PROJECT) -> list[dict]:
    """Scan checkpoint directories and return experiment metadata."""
    project_dir = Path(checkpoint_dir) / project
    if not project_dir.is_dir():
        return []

    experiments = []
    for entry in sorted(project_dir.iterdir()):
        if not entry.is_dir() or entry.is_symlink():
            continue

        name = entry.name
        parsed = parse_experiment_name(name)

        # Read current step
        iter_file = entry / "latest_checkpointed_iteration.txt"
        steps = 0
        if iter_file.exists():
            try:
                steps = int(iter_file.read_text().strip())
            except (ValueError, OSError):
                pass

        # Read metadata
        meta_file = entry / "training_metadata.json"
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        total = meta.get("total_training_steps", TOTAL_TRAINING_STEPS)

        # Infer GPU count from checkpoint files
        gpu_count = infer_gpu_count(entry)

        # Load eval results and compute best accuracy per dataset
        eval_results = load_eval_results(name, checkpoint_dir, project)
        eval_best: dict[str, float] = {}
        for rec in eval_results:
            ds = rec.get("dataset", "")
            acc = rec.get("accuracy")
            if ds and acc is not None:
                if ds not in eval_best or acc > eval_best[ds]:
                    eval_best[ds] = acc

        experiments.append(
            {
                "name": name,
                "workflow": parsed["workflow"],
                "model": parsed["model"],
                "policy": parsed["policy"],
                "dataset": parsed["dataset"],
                "steps": steps,
                "total_steps": total,
                "gpu_count": gpu_count,
                "eval_best": eval_best,
                "status": "",  # computed later in build_experiment_table
                "wandb_run": meta.get("wandb_run_id", "\u2014"),
                "slurm_job": meta.get("slurm_job_id") or "\u2014",
            }
        )
    return experiments


# ── SLURM helpers ────────────────────────────────────────────────────────────


def get_slurm_jobs() -> list[dict]:
    """Get current user's SLURM jobs as a list of dicts."""
    fmt = "%.18i %.9P %.80j %.8T %.10M %.6D %R"
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-o", fmt, "--noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        rows = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(None, 6)
            if len(parts) >= 6:
                rows.append(
                    {
                        "job_id": parts[0].strip(),
                        "partition": parts[1],
                        "name": parts[2],
                        "state": parts[3],
                        "time": parts[4],
                        "nodes": parts[5],
                        "nodelist": parts[6] if len(parts) > 6 else "\u2014",
                    }
                )
        return rows
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


# ── Build combined table ─────────────────────────────────────────────────────


def build_experiment_table(checkpoint_dir: str) -> tuple[list[dict], list[dict]]:
    """Return (experiments, slurm_jobs).

    experiments: list of dicts, one per experiment with SLURM status joined in.
    slurm_jobs: raw squeue output.
    """
    exps = scan_checkpoints(checkpoint_dir)
    slurm_jobs = get_slurm_jobs()

    # Build lookups
    slurm_lookup: dict[str, dict] = {}
    eval_slurm_lookup: dict[str, str] = {}
    for job in slurm_jobs:
        slurm_lookup[job["job_id"]] = {
            "slurm_state": job["state"],
            "time": job["time"],
            "node": job["nodelist"],
        }
        if job["name"].startswith("eval-"):
            eval_exp_name = job["name"][len("eval-"):]
            eval_slurm_lookup[eval_exp_name] = job["job_id"]

    # Enrich experiments
    for exp in exps:
        job_id = str(exp.get("slurm_job", "\u2014")).strip()
        info = slurm_lookup.get(job_id, {})
        slurm_state = info.get("slurm_state", "")
        exp["slurm_state"] = slurm_state or "\u2014"
        exp["time"] = info.get("time", "\u2014")
        exp["node"] = info.get("node", "\u2014")

        # Determine status
        steps = exp.get("steps", 0)
        total = exp.get("total_steps", TOTAL_TRAINING_STEPS)
        if steps >= total:
            exp["status"] = "Finished"
        elif slurm_state in ("RUNNING", "PENDING", "CONFIGURING"):
            exp["status"] = "Running"
        else:
            exp["status"] = "Unfinished"

        # Eval status
        exp_name = exp["name"]
        eval_best = exp.get("eval_best", {})
        has_eval = bool(eval_best)
        is_evaluating = exp_name in eval_slurm_lookup
        eval_job_id = eval_slurm_lookup.get(exp_name, "\u2014")
        exp["eval_job"] = eval_job_id

        if is_evaluating and has_eval:
            exp["eval_status"] = "Evaluating + Has Results"
        elif is_evaluating:
            exp["eval_status"] = "Evaluating"
        elif has_eval:
            exp["eval_status"] = "Has Results"
        else:
            exp["eval_status"] = ""

    return exps, slurm_jobs


# ── Eval results for API ────────────────────────────────────────────────────


def get_eval_results_table(experiment_name: str, checkpoint_dir: str) -> dict:
    """Return structured eval results for an experiment."""
    eval_results = load_eval_results(experiment_name, checkpoint_dir)
    if not eval_results:
        return {"datasets": [], "rows": [], "best": {}}

    datasets = sorted({r.get("dataset", "") for r in eval_results if r.get("dataset")})

    step_data: dict[tuple, dict[str, float]] = {}
    for rec in eval_results:
        step = rec.get("checkpoint_step", 0)
        mode = rec.get("eval_mode", "")
        n_roll = rec.get("n_rollouts", 1)
        ds = rec.get("dataset", "")
        acc = rec.get("accuracy")
        key = (step, mode, n_roll)
        if key not in step_data:
            step_data[key] = {}
        if ds and acc is not None:
            step_data[key][ds] = acc

    sorted_keys = sorted(step_data.keys())
    rows = []
    for step, mode, n_roll in sorted_keys:
        row = {"step": step, "mode": mode, "n_rollouts": n_roll, "scores": {}}
        for ds in datasets:
            acc = step_data[(step, mode, n_roll)].get(ds)
            row["scores"][ds] = acc
        rows.append(row)

    # Best per dataset
    best: dict[str, dict] = {}
    for ds in datasets:
        best_acc = None
        best_step = None
        for rec in eval_results:
            if rec.get("dataset") == ds and rec.get("accuracy") is not None:
                acc = rec["accuracy"]
                if best_acc is None or acc > best_acc:
                    best_acc = acc
                    best_step = rec.get("checkpoint_step", 0)
        if best_acc is not None:
            best[ds] = {"accuracy": best_acc, "step": best_step}

    return {"datasets": datasets, "rows": rows, "best": best}


# ── Assign SLURM Job ID ─────────────────────────────────────────────────────


def assign_job_id(experiment_name: str, job_id: str, checkpoint_dir: str) -> str:
    """Write slurm_job_id into an experiment's training_metadata.json."""
    if not experiment_name or not experiment_name.strip():
        return "Please select an experiment."
    if not job_id or not job_id.strip():
        return "Please enter a Job ID."
    job_id = job_id.strip()
    if not re.match(r"^\d+$", job_id):
        return f"Invalid Job ID: {job_id!r} (expected numeric)"

    meta_path = (
        Path(checkpoint_dir) / DEFAULT_PROJECT / experiment_name.strip() / "training_metadata.json"
    )
    if not meta_path.exists():
        return f"Metadata file not found: {meta_path}"

    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return f"Failed to read metadata: {e}"

    meta["slurm_job_id"] = job_id
    meta_path.write_text(json.dumps(meta, indent=2))
    return f"Assigned SLURM job {job_id} to {experiment_name}"


# ── Submit eval job ──────────────────────────────────────────────────────────


def submit_eval_job(
    experiment_name: str,
    dataset: str,
    n_rollouts: int,
    slurm_config: str,
    cpus_per_gpu: int,
    mem_per_gpu: str,
    checkpoint_dir: str,
    project: str = DEFAULT_PROJECT,
    dry_run: bool = False,
    task_type: str = "math",
    trajectory_analysis: bool = False,
) -> str:
    """Generate and submit an sbatch evaluation job."""
    if not experiment_name or not experiment_name.strip():
        return "Please select an experiment first."

    experiment_name = experiment_name.strip()
    checkpoints_dir = f"{checkpoint_dir}/{project}"
    output_json = f"{checkpoint_dir}/{project}/{experiment_name}/eval_results.jsonl"
    port = random.randint(8000, 9999)

    # Read SBATCH directives from config file (excluding gpu/cpu/mem lines)
    config_path = SLURM_CONFIGS_DIR / f"{slurm_config}.conf"
    if not config_path.exists():
        return f"SLURM config not found: {config_path}"
    sbatch_lines = []
    for line in config_path.read_text().splitlines():
        if line.startswith("#SBATCH") and not re.search(
            r"--(job-name|output|error|gres=gpu|cpus-per-gpu|mem-per-gpu)", line
        ):
            sbatch_lines.append(line)
    sbatch_directives = "\n".join(sbatch_lines)

    # Task-specific setup commands
    extra_setup = ""
    if task_type == "deepcoder":
        extra_setup = "\nulimit -n 1048576\n"

    # Trajectory analysis flags
    trajectory_flags = ""
    if trajectory_analysis:
        traj_dir = f"{checkpoint_dir}/{project}/{experiment_name}"
        trajectory_flags = f"""\\\n    --trajectory-output-dir {traj_dir} \\
    --max-samples 30"""

    script = f"""#!/bin/bash
#SBATCH --job-name=eval-{experiment_name}
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu={cpus_per_gpu}
#SBATCH --mem-per-gpu={mem_per_gpu}
{sbatch_directives}

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
source ~/.bashrc && conda activate rllm

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
{extra_setup}
python -m dashboard.evaluate_checkpoints \\
    --task-type {task_type} \\
    --eval-mode trained_checkpoint \\
    --checkpoints-dir {checkpoints_dir} \\
    --dataset {dataset} \\
    --experiment-filter '^{experiment_name}$' \\
    --n-rollouts {n_rollouts} \\
    --n-parallel 512 \\
    --port {port} \\
    --output-json {output_json} {trajectory_flags}
"""

    if dry_run:
        return f"=== DRY RUN \u2014 sbatch script for {experiment_name} (port {port}) ===\n{script}"

    os.makedirs("logs", exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", prefix="eval_sbatch_", delete=False
        ) as f:
            f.write(script)
            tmp_path = f.name

        result = subprocess.run(
            ["sbatch", tmp_path], capture_output=True, text=True, timeout=15
        )
        os.unlink(tmp_path)

        if result.returncode == 0:
            job_id_match = re.search(r"\d+", result.stdout)
            job_id = job_id_match.group() if job_id_match else "?"
            return (
                f"Submitted eval job for {experiment_name}\n"
                f"Job ID: {job_id} | Dataset: {dataset} | N-rollouts: {n_rollouts}\n"
                f"Output: {output_json}"
            )
        return f"sbatch failed (rc={result.returncode}):\n{result.stderr}"
    except FileNotFoundError:
        return "sbatch not found \u2014 is SLURM available?"
    except subprocess.TimeoutExpired:
        return "ERROR: sbatch timed out"
    except Exception as e:
        return f"ERROR: {e}"


# ── Launch / cancel ──────────────────────────────────────────────────────────


def launch_experiment(
    workflow: str,
    model: str,
    share_policy: str,
    node: str,
    extra_args: str,
    dry_run: bool,
    task_type: str = "math",
    n_gpus: int = 2,
    cpus_per_gpu: int = 4,
    mem_per_gpu: str = "48G",
) -> str:
    """Call launch_experiment.sh and return its output."""
    config_path = SLURM_CONFIGS_DIR / f"{node}.conf"
    cmd = [
        "bash",
        LAUNCHER_SCRIPT,
        "--workflow", workflow,
        "--model", model,
        "--share-policy", share_policy,
        "--slurm-config", str(config_path),
        "--task-type", task_type,
        "--n-gpus", str(n_gpus),
        "--cpus-per-gpu", str(cpus_per_gpu),
        "--mem-per-gpu", mem_per_gpu,
    ]
    if extra_args.strip():
        cmd += ["--extra-args", extra_args.strip()]
    if dry_run:
        cmd.append("--dry-run")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr
        return output if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: launch timed out after 30s"
    except Exception as e:
        return f"ERROR: {e}"


def cancel_job(job_id: str) -> str:
    """Cancel a SLURM job via scancel."""
    job_id = job_id.strip()
    if not job_id:
        return "Please enter a Job ID."
    if not re.match(r"^\d+$", job_id):
        return f"Invalid Job ID: {job_id!r} (expected numeric)"
    try:
        result = subprocess.run(
            ["scancel", job_id], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return f"Cancelled job {job_id}."
        return f"scancel failed (rc={result.returncode}): {result.stderr}"
    except FileNotFoundError:
        return "scancel not found \u2014 is SLURM available?"
    except Exception as e:
        return f"ERROR: {e}"
