"""RLLM Experiment Dashboard — monitor, launch, and cancel training experiments."""

import argparse
import json
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_PROJECT = "rllm-workflow-MARL-v2"
TOTAL_TRAINING_STEPS = 301
LAUNCHER_SCRIPT = "examples/math_reasoning/launch_experiment.sh"
SLURM_CONFIGS_DIR = Path("examples/math_reasoning/slurm_configs")

MODEL_TABS = ["0.6B", "1.7B", "4B"]
DATASET_CATEGORIES = {
    "Math": ["math"],
    "Code": ["deepcoder"],
}
STATUS_EMOJI = {"Finished": "✅", "Running": "🔄", "Unfinished": "⏸️"}
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
    """Extract gpu_count and gpu_type from a config file."""
    config_path = SLURM_CONFIGS_DIR / f"{config_name}.conf"
    result = {"gpu_count": None, "gpu_type": None}
    if not config_path.exists():
        return result
    text = config_path.read_text()
    m = re.search(r"#SBATCH\s+--gres=gpu:(\d+)", text)
    if m:
        result["gpu_count"] = int(m.group(1))
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

    # Per-experiment eval results
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
    # Try standard format: workflow-qwen3_MODEL-POLICY-DATASET
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
    # Fallback: workflow-qwen3_MODEL-DATASET (no policy field)
    m = re.match(r"^(.+?)-qwen3_([\d.]+[bB])-(\w+)$", name)
    if m:
        return {
            "workflow": m.group(1),
            "model": m.group(2),
            "policy": "—",
            "dataset": m.group(3),
        }
    return {"workflow": name, "model": "—", "policy": "—", "dataset": "—"}


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
                "Experiment": name,
                "Workflow": parsed["workflow"],
                "Model": parsed["model"],
                "Policy": parsed["policy"],
                "Dataset": parsed["dataset"],
                "Steps": f"{steps}/{total}",
                "_steps": steps,
                "_total": total,
                "_eval_best": eval_best,
                "Status": "",  # computed later in build_experiment_table
                "WandB Run": meta.get("wandb_run_id", "—"),
                "SLURM Job": meta.get("slurm_job_id") or "—",
            }
        )
    return experiments


# ── SLURM helpers ────────────────────────────────────────────────────────────


def get_slurm_jobs() -> pd.DataFrame:
    """Get current user's SLURM jobs."""
    fmt = "%.18i %.9P %.80j %.8T %.10M %.6D %R"
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-o", fmt, "--noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return pd.DataFrame(
                columns=["Job ID", "Partition", "Name", "State", "Time", "Nodes", "Nodelist"]
            )
        rows = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(None, 6)
            if len(parts) >= 6:
                rows.append(
                    {
                        "Job ID": parts[0],
                        "Partition": parts[1],
                        "Name": parts[2],
                        "State": parts[3],
                        "Time": parts[4],
                        "Nodes": parts[5],
                        "Nodelist": parts[6] if len(parts) > 6 else "—",
                    }
                )
        return pd.DataFrame(rows)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return pd.DataFrame(
            columns=["Job ID", "Partition", "Name", "State", "Time", "Nodes", "Nodelist"]
        )


# ── Build combined table ─────────────────────────────────────────────────────


MERGED_COLUMNS = [
    "Experiment", "Workflow", "Model", "Policy", "Dataset",
    "Steps", "Status", "Eval Status", "Eval Job",
    "WandB Run", "SLURM Job", "SLURM State", "Time", "Node",
]


def build_experiment_table(checkpoint_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (merged_df, slurm_df).

    merged_df: one row per experiment with SLURM status columns joined in.
    slurm_df: raw squeue output (kept for orphan-job visibility).
    """
    exps = scan_checkpoints(checkpoint_dir)
    exp_df = pd.DataFrame(exps) if exps else pd.DataFrame(
        columns=["Experiment", "Workflow", "Model", "Policy", "Dataset",
                 "Steps", "Status", "WandB Run", "SLURM Job"]
    )
    slurm_df = get_slurm_jobs()

    # Build a lookup: job_id -> {State, Time, Nodelist}
    slurm_lookup: dict[str, dict] = {}
    # Build eval SLURM lookup: experiment_name -> job_id for eval jobs
    eval_slurm_lookup: dict[str, str] = {}
    if not slurm_df.empty:
        for _, row in slurm_df.iterrows():
            slurm_lookup[str(row["Job ID"]).strip()] = {
                "SLURM State": row["State"],
                "Time": row["Time"],
                "Node": row["Nodelist"],
            }
            # Detect eval jobs by name prefix "eval-"
            job_name = str(row.get("Name", ""))
            if job_name.startswith("eval-"):
                eval_exp_name = job_name[len("eval-"):]
                eval_slurm_lookup[eval_exp_name] = str(row["Job ID"]).strip()

    # Enrich experiment rows with SLURM info and compute precise status
    slurm_states, times, nodes, statuses = [], [], [], []
    eval_statuses, eval_jobs = [], []
    for _, row in exp_df.iterrows():
        job_id = str(row.get("SLURM Job", "—")).strip()
        info = slurm_lookup.get(job_id, {})
        slurm_state = info.get("SLURM State", "")
        slurm_states.append(slurm_state or "—")
        times.append(info.get("Time", "—"))
        nodes.append(info.get("Node", "—"))

        # Determine status: finished / running / unfinished
        steps = row.get("_steps", 0)
        total = row.get("_total", TOTAL_TRAINING_STEPS)
        if steps >= total:
            statuses.append("Finished")
        elif slurm_state in ("RUNNING", "PENDING", "CONFIGURING"):
            statuses.append("Running")
        else:
            statuses.append("Unfinished")

        # Eval status
        exp_name = row.get("Experiment", "")
        eval_best = row.get("_eval_best", {})
        has_eval = bool(eval_best)
        is_evaluating = exp_name in eval_slurm_lookup
        eval_job_id = eval_slurm_lookup.get(exp_name, "—")
        eval_jobs.append(eval_job_id)

        if is_evaluating and has_eval:
            eval_statuses.append("Evaluating + Has Results")
        elif is_evaluating:
            eval_statuses.append("Evaluating")
        elif has_eval:
            eval_statuses.append("Has Results")
        else:
            eval_statuses.append("")

    exp_df["Status"] = statuses
    exp_df["SLURM State"] = slurm_states
    exp_df["Time"] = times
    exp_df["Node"] = nodes
    exp_df["Eval Status"] = eval_statuses
    exp_df["Eval Job"] = eval_jobs

    # Drop temporary columns but keep _eval_best for downstream use
    exp_df = exp_df.drop(columns=["_steps", "_total"], errors="ignore")
    # Ensure all MERGED_COLUMNS exist
    for col in MERGED_COLUMNS:
        if col not in exp_df.columns:
            exp_df[col] = "—"
    merged_df = exp_df

    return merged_df, slurm_df


# ── Pivot grid + detail helpers ──────────────────────────────────────────────

STATUS_BAR_COLORS = {"Finished": "#4caf50", "Running": "#2196f3", "Unfinished": "#ff9800"}


def _progress_bar_html(steps_str: str, status: str, label: str) -> str:
    """Return HTML for a single experiment line with emoji, label, and progress bar."""
    emoji = STATUS_EMOJI.get(status, "⏸️")
    color = STATUS_BAR_COLORS.get(status, "#ff9800")
    # Parse "150/301" into percentage
    parts = steps_str.split("/")
    try:
        current, total = int(parts[0]), int(parts[1])
        pct = min(100, round(current / total * 100)) if total > 0 else 0
    except (ValueError, IndexError):
        pct = 0
    return (
        f'<div style="margin:2px 0">'
        f'<span>{emoji} {label}</span><br>'
        f'<div style="display:flex;align-items:center;gap:4px">'
        f'<div style="background:#e0e0e0;border-radius:3px;height:12px;flex:1;min-width:60px">'
        f'<div style="background:{color};border-radius:3px;height:12px;width:{pct}%"></div>'
        f'</div>'
        f'<span style="font-size:0.85em;white-space:nowrap">{steps_str}</span>'
        f'</div></div>'
    )


def _eval_indicator_html(eval_status: str, eval_best: dict | None) -> str:
    """Generate compact HTML showing best eval accuracy per dataset."""
    if not eval_status:
        return ""
    parts = []
    if eval_best:
        for ds, acc in sorted(eval_best.items()):
            parts.append(f"{ds}: {acc:.1%}")
    indicator = " | ".join(parts) if parts else ""
    if "Evaluating" in eval_status:
        indicator = ("🔄 " + indicator) if indicator else "🔄 evaluating"
    if indicator:
        return f'<div style="font-size:0.8em;color:#666;margin-top:1px">📊 {indicator}</div>'
    return ""


def build_pivot_grid(
    merged_df: pd.DataFrame,
    model_filter: str,
    dataset_names: list[str] | None = None,
) -> tuple[pd.DataFrame, dict, int]:
    """Build a compact Dataset x Workflow pivot grid for one model size.

    Args:
        merged_df: Full experiment DataFrame.
        model_filter: Model size string (e.g. "1.7b").
        dataset_names: If provided, only include rows whose Dataset is in this list.

    Returns (pivot_df, lookup_dict, n_experiments).
    """
    if merged_df.empty:
        empty = pd.DataFrame({"Policy \\ Workflow": []})
        return empty, {}, 0

    mask = merged_df["Model"].str.lower() == model_filter.lower()
    if dataset_names:
        lower_names = [d.lower() for d in dataset_names]
        mask = mask & merged_df["Dataset"].str.lower().isin(lower_names)
    subset = merged_df[mask]
    if subset.empty:
        empty = pd.DataFrame({"Policy \\ Workflow": []})
        return empty, {}, 0
    n_experiments = len(subset)

    policies = sorted(subset["Policy"].unique())
    workflows = sorted(subset["Workflow"].unique())

    has_eval_best = "_eval_best" in subset.columns

    rows = []
    lookup: dict[str, list[str]] = {}

    for r_idx, policy in enumerate(policies):
        row_dict: dict[str, str] = {"Policy \\ Workflow": policy}
        for c_idx, wf in enumerate(workflows):
            cell_exps = subset[(subset["Policy"] == policy) & (subset["Workflow"] == wf)]
            if cell_exps.empty:
                row_dict[wf] = "—"
            else:
                html_parts = []
                exp_names_in_cell = []
                for _, exp_row in cell_exps.iterrows():
                    html_parts.append(_progress_bar_html(
                        exp_row["Steps"], exp_row["Status"], exp_row["Dataset"],
                    ))
                    # Append eval indicator
                    eval_status = exp_row.get("Eval Status", "")
                    eval_best = exp_row.get("_eval_best", {}) if has_eval_best else {}
                    if not isinstance(eval_best, dict):
                        eval_best = {}
                    html_parts.append(_eval_indicator_html(eval_status, eval_best))
                    exp_names_in_cell.append(exp_row["Experiment"])
                row_dict[wf] = "".join(html_parts)
                # col index in dataframe is c_idx+1 because col 0 is the row-label column
                lookup[f"{r_idx},{c_idx + 1}"] = exp_names_in_cell
        rows.append(row_dict)

    pivot_df = pd.DataFrame(rows, columns=["Policy \\ Workflow"] + workflows)
    return pivot_df, lookup, n_experiments


def format_detail_md(
    exp_name: str, merged_df: pd.DataFrame, checkpoint_dir: str = ""
) -> str:
    """Return a Markdown detail table for a single experiment."""
    if merged_df.empty:
        return ""
    match = merged_df[merged_df["Experiment"] == exp_name]
    if match.empty:
        return f"Experiment **{exp_name}** not found."
    row = match.iloc[0]
    emoji = STATUS_EMOJI.get(row["Status"], "⏸️")
    lines = [
        f"### {exp_name}",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Status | {emoji} {row['Status']} |",
        f"| Steps | {row['Steps']} |",
        f"| Workflow | {row['Workflow']} |",
        f"| Policy | {row['Policy']} |",
        f"| Dataset | {row['Dataset']} |",
        f"| Eval Status | {row.get('Eval Status', '')} |",
        f"| Eval Job | {row.get('Eval Job', '—')} |",
        f"| WandB Run | {row['WandB Run']} |",
        f"| SLURM Job | {row['SLURM Job']} |",
        f"| SLURM State | {row['SLURM State']} |",
        f"| Time | {row['Time']} |",
        f"| Node | {row['Node']} |",
    ]
    return "\n".join(lines)


def format_eval_results_md(exp_name: str, checkpoint_dir: str) -> str:
    """Return a Markdown table of evaluation results for an experiment."""
    if not checkpoint_dir:
        return ""
    eval_results = load_eval_results(exp_name, checkpoint_dir)
    if not eval_results:
        return "*No evaluation results found.*"

    # Collect all datasets and steps
    datasets = sorted({r.get("dataset", "") for r in eval_results if r.get("dataset")})
    # Build (step, eval_mode, n_rollouts) -> {dataset: accuracy}
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

    # Sort by step
    sorted_keys = sorted(step_data.keys())

    # Build markdown table
    header = "| Step | Mode | N | " + " | ".join(datasets) + " |"
    sep = "|------|------|---|" + "|".join("---" for _ in datasets) + "|"
    rows = []
    for step, mode, n_roll in sorted_keys:
        cells = []
        for ds in datasets:
            acc = step_data[(step, mode, n_roll)].get(ds)
            cells.append(f"{acc:.1%}" if acc is not None else "—")
        rows.append(f"| {step} | {mode} | {n_roll} | " + " | ".join(cells) + " |")

    # Best per dataset
    best_parts = []
    for ds in datasets:
        best_acc = None
        best_step = None
        for rec in eval_results:
            if rec.get("dataset") == ds and rec.get("accuracy") is not None:
                acc = rec["accuracy"]
                if best_acc is None or acc > best_acc:
                    best_acc = acc
                    best_step = rec.get("checkpoint_step", "?")
        if best_acc is not None:
            best_parts.append(f"{ds}={best_acc:.1%} (step {best_step})")

    lines = [
        "### Evaluation Results",
        "",
        header,
        sep,
        *rows,
    ]
    if best_parts:
        lines.append("")
        lines.append("**Best:** " + ", ".join(best_parts))

    return "\n".join(lines)


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
    partition: str,
    constraint: str,
    checkpoint_dir: str,
    project: str = DEFAULT_PROJECT,
    dry_run: bool = False,
) -> str:
    """Generate and submit an sbatch evaluation job."""
    if not experiment_name or not experiment_name.strip():
        return "Please select an experiment first."

    experiment_name = experiment_name.strip()
    checkpoints_dir = f"{checkpoint_dir}/{project}"
    output_json = f"{checkpoint_dir}/{project}/{experiment_name}/eval_results.jsonl"
    port = random.randint(8000, 9999)

    script = f"""#!/bin/bash
#SBATCH --job-name=eval-{experiment_name}
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --exclude=dgxh-1
#SBATCH --mem-per-gpu=80G
#SBATCH --constraint={constraint}
#SBATCH --time=1-0:00:00

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
source ~/.bashrc && conda activate rllm

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

python -m examples.math_reasoning.evaluate_checkpoints \\
    --eval-mode trained_checkpoint \\
    --checkpoints-dir {checkpoints_dir} \\
    --dataset {dataset} \\
    --experiment-filter '^{experiment_name}$' \\
    --n-rollouts {n_rollouts} \\
    --n-parallel 512 \\
    --port {port} \\
    --output-json {output_json}
"""

    if dry_run:
        return f"=== DRY RUN — sbatch script for {experiment_name} (port {port}) ===\n{script}"

    # Ensure logs directory exists
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
        return "sbatch not found — is SLURM available?"
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
        return "scancel not found — is SLURM available?"
    except Exception as e:
        return f"ERROR: {e}"


# ── Gradio app ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="RLLM Experiment Dashboard")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
        help="Root checkpoint directory"
    )
    args = parser.parse_args()

    def refresh():
        merged_df, slurm_df = build_experiment_table(args.checkpoint_dir)
        results = []
        for ds_key, ds_names in DATASET_CATEGORIES.items():
            for model in MODEL_TABS:
                pivot, lookup, n_exp = build_pivot_grid(merged_df, model.lower(), ds_names)
                label = f"{ds_key} — {model} ({n_exp} experiment{'s' if n_exp != 1 else ''})"
                results.append(gr.update(value=pivot, label=label))
                results.append(lookup)
        return (
            *results,
            slurm_df, merged_df,
            gr.update(visible=False), "", "",  # hide detail panel on refresh
            "", "",  # clear eval_results_md, eval_output
        )

    def on_cell_select(select_data: gr.SelectData, lookup, full_data):
        row, col = select_data.index
        if col == 0:
            return gr.update(visible=False), "", "", "", "", ""
        key = f"{row},{col}"
        exp_names = lookup.get(key, [])
        if not exp_names:
            return gr.update(visible=False), "", "", "", "", ""
        exp_name = exp_names[0]
        detail = format_detail_md(exp_name, full_data, args.checkpoint_dir)
        eval_md = format_eval_results_md(exp_name, args.checkpoint_dir)
        # Pre-fill SLURM Job ID from the experiment's existing value
        match = full_data[full_data["Experiment"] == exp_name]
        existing_job = ""
        if not match.empty:
            val = match.iloc[0].get("SLURM Job", "—")
            if val and val != "—":
                existing_job = str(val)
        return gr.update(visible=True), detail, exp_name, existing_job, eval_md, ""

    def do_assign(exp_name, job_id):
        result = assign_job_id(exp_name, job_id, args.checkpoint_dir)
        return result

    def do_cancel(exp_name, full_data):
        if not exp_name or full_data.empty:
            return "No experiment selected."
        match = full_data[full_data["Experiment"] == exp_name]
        if match.empty:
            return "Experiment not found."
        job_id = str(match.iloc[0].get("SLURM Job", ""))
        if not job_id or job_id == "—":
            return "No SLURM Job ID assigned to this experiment."
        return cancel_job(job_id)

    def do_eval_submit(exp_name, dataset, n_rollouts, partition, constraint, dry_run):
        return submit_eval_job(
            experiment_name=exp_name,
            dataset=dataset,
            n_rollouts=int(n_rollouts),
            partition=partition,
            constraint=constraint,
            checkpoint_dir=args.checkpoint_dir,
            dry_run=dry_run,
        )

    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="RLLM Experiment Dashboard",
    ) as app:
        # ── State ────────────────────────────────────────────────────────
        full_data_state = gr.State(pd.DataFrame())
        selected_exp_state = gr.State("")

        # Per (dataset_category, model) lookup states and grid references
        grids: dict[str, gr.Dataframe] = {}
        lookup_states: dict[str, gr.State] = {}
        for ds_key in DATASET_CATEGORIES:
            for model in MODEL_TABS:
                key = f"{ds_key}_{model}"
                lookup_states[key] = gr.State({})

        # ── Header ───────────────────────────────────────────────────────
        with gr.Row():
            gr.Markdown("# RLLM Experiment Dashboard")
            refresh_btn = gr.Button("Refresh", scale=0)

        # ── Nested tabs: Dataset (outer) → Model (inner) ────────────────
        with gr.Tabs():
            for ds_key in DATASET_CATEGORIES:
                with gr.TabItem(ds_key):
                    with gr.Tabs():
                        for model in MODEL_TABS:
                            with gr.TabItem(model):
                                key = f"{ds_key}_{model}"
                                grids[key] = gr.Dataframe(
                                    label=f"{ds_key} — {model} Experiments",
                                    interactive=False,
                                    wrap=True,
                                    datatype="html",
                                )

        # ── Detail panel (hidden until cell click) ───────────────────────
        with gr.Column(visible=False) as detail_panel:
            gr.Markdown("## Experiment Detail")
            detail_md = gr.Markdown("")
            with gr.Row():
                assign_job_box = gr.Textbox(
                    label="SLURM Job ID", placeholder="e.g. 123456", scale=2,
                )
                assign_btn = gr.Button("Assign", scale=0)
            with gr.Row():
                cancel_btn = gr.Button("Cancel Job", scale=0)
            action_output = gr.Textbox(label="Result", lines=1, interactive=False)

            # ── Evaluation accordion ─────────────────────────────────────
            with gr.Accordion("Evaluation", open=False):
                eval_results_md = gr.Markdown("")
                with gr.Row():
                    eval_dataset_dd = gr.Dropdown(
                        choices=["dapo_math", "aime2025"],
                        label="Dataset",
                        value="dapo_math",
                        scale=1,
                    )
                    eval_n_rollouts_dd = gr.Dropdown(
                        choices=["1", "3", "5"],
                        label="N-Rollouts",
                        value="1",
                        scale=1,
                    )
                    eval_partition_dd = gr.Dropdown(
                        choices=["preempt", "dgxh"],
                        label="Partition",
                        value="preempt",
                        scale=1,
                    )
                    eval_constraint_dd = gr.Dropdown(
                        choices=["l40s", "a40"],
                        label="GPU Constraint",
                        value="l40s",
                        scale=1,
                    )
                with gr.Row():
                    eval_submit_btn = gr.Button("Submit Eval Job")
                    eval_dry_run_btn = gr.Button("Eval Dry Run")
                eval_output = gr.Textbox(
                    label="Eval Output", lines=8, interactive=False,
                )

        # ── Raw SLURM Queue ──────────────────────────────────────────────
        with gr.Accordion("Raw SLURM Queue", open=False):
            slurm_table = gr.Dataframe(
                label="SLURM Jobs (squeue)",
                interactive=False,
                wrap=True,
            )

        # ── Launch section ───────────────────────────────────────────────
        gr.Markdown("## Launch Experiment")
        with gr.Row():
            workflow_dd = gr.Dropdown(
                choices=["evaluator_optimizer", "voting", "orchestrator_workers"],
                label="Workflow",
                value="evaluator_optimizer",
            )
            model_dd = gr.Dropdown(
                choices=["0.6B", "1.7B", "4B"],
                label="Model",
                value="1.7B",
            )
            policy_dd = gr.Dropdown(
                choices=["true", "false"],
                label="Share Policy",
                value="false",
            )
            slurm_choices = load_slurm_configs()
            node_dd = gr.Dropdown(
                choices=slurm_choices or ["(no configs found)"],
                label="SLURM Config",
                value=slurm_choices[0] if slurm_choices else None,
            )
        gpu_info_md = gr.Markdown("")
        extra_args_box = gr.Textbox(
            label="Extra Args (hydra overrides)",
            placeholder="e.g. trainer.total_training_steps=100",
        )
        with gr.Row():
            launch_btn = gr.Button("Launch")
            dry_run_btn = gr.Button("Dry Run")
        launch_output = gr.Textbox(label="Launch Output", lines=15, interactive=False)

        # ── GPU info display ──────────────────────────────────────────────
        def update_gpu_info(config_name):
            if not config_name or config_name == "(no configs found)":
                return ""
            info = parse_slurm_config(config_name)
            parts = []
            if info["gpu_count"]:
                parts.append(f"**GPUs:** {info['gpu_count']}")
            if info["gpu_type"]:
                parts.append(f"**Type:** {info['gpu_type']}")
            return " | ".join(parts) if parts else "Could not parse config"

        node_dd.change(fn=update_gpu_info, inputs=[node_dd], outputs=[gpu_info_md])

        # ── Wiring ───────────────────────────────────────────────────────
        # Build refresh outputs in same order as refresh() returns:
        # for each (ds, model): pivot_df, lookup_dict; then slurm, full_data, detail hide
        refresh_outputs = []
        for ds_key in DATASET_CATEGORIES:
            for model in MODEL_TABS:
                key = f"{ds_key}_{model}"
                refresh_outputs.append(grids[key])
                refresh_outputs.append(lookup_states[key])
        refresh_outputs += [
            slurm_table, full_data_state,
            detail_panel, detail_md, selected_exp_state,
            eval_results_md, eval_output,
        ]
        refresh_btn.click(fn=refresh, outputs=refresh_outputs)
        app.load(fn=refresh, outputs=refresh_outputs)

        # Cell click handlers — one per grid
        cell_select_outputs = [
            detail_panel, detail_md, selected_exp_state, assign_job_box,
            eval_results_md, eval_output,
        ]
        for ds_key in DATASET_CATEGORIES:
            for model in MODEL_TABS:
                key = f"{ds_key}_{model}"
                grids[key].select(
                    fn=on_cell_select,
                    inputs=[lookup_states[key], full_data_state],
                    outputs=cell_select_outputs,
                )

        # Assign job ID → refresh
        assign_btn.click(
            fn=do_assign,
            inputs=[selected_exp_state, assign_job_box],
            outputs=action_output,
        ).then(fn=refresh, outputs=refresh_outputs)

        # Cancel job from detail panel
        cancel_btn.click(
            fn=do_cancel,
            inputs=[selected_exp_state, full_data_state],
            outputs=action_output,
        ).then(fn=refresh, outputs=refresh_outputs)

        # Eval submit / dry-run
        eval_submit_btn.click(
            fn=lambda e, d, n, p, c: do_eval_submit(e, d, n, p, c, dry_run=False),
            inputs=[
                selected_exp_state, eval_dataset_dd, eval_n_rollouts_dd,
                eval_partition_dd, eval_constraint_dd,
            ],
            outputs=eval_output,
        ).then(fn=refresh, outputs=refresh_outputs)

        eval_dry_run_btn.click(
            fn=lambda e, d, n, p, c: do_eval_submit(e, d, n, p, c, dry_run=True),
            inputs=[
                selected_exp_state, eval_dataset_dd, eval_n_rollouts_dd,
                eval_partition_dd, eval_constraint_dd,
            ],
            outputs=eval_output,
        )

        # Launch / dry-run
        launch_btn.click(
            fn=lambda w, m, sp, n, ea: launch_experiment(w, m, sp, n, ea, dry_run=False),
            inputs=[workflow_dd, model_dd, policy_dd, node_dd, extra_args_box],
            outputs=launch_output,
        )
        dry_run_btn.click(
            fn=lambda w, m, sp, n, ea: launch_experiment(w, m, sp, n, ea, dry_run=True),
            inputs=[workflow_dd, model_dd, policy_dd, node_dd, extra_args_box],
            outputs=launch_output,
        )

    app.launch(server_port=args.port, share=True)


if __name__ == "__main__":
    main()
