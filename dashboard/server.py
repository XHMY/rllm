"""RLLM Experiment Dashboard — FastAPI server."""

import argparse
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dashboard.backend import (
    DEFAULT_CHECKPOINT_DIR,
    DATASET_CATEGORIES,
    EVAL_DATASETS,
    MODEL_TABS,
    assign_job_id,
    build_experiment_table,
    cancel_job,
    get_analysis_markdown,
    get_analysis_status,
    get_eval_results_table,
    has_trajectory_dirs,
    launch_experiment,
    launch_trajectory_analysis,
    load_slurm_configs,
    parse_slurm_config,
    submit_eval_job,
)

app = FastAPI(title="RLLM Dashboard")

# Global config — set by CLI args before server starts
_checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR

STATIC_DIR = Path(__file__).parent / "static"


# ── Request models ───────────────────────────────────────────────────────────


class AssignJobRequest(BaseModel):
    experiment_name: str
    job_id: str


class CancelJobRequest(BaseModel):
    job_id: str


class LaunchRequest(BaseModel):
    workflow: str
    model: str
    share_policy: str
    node: str
    extra_args: str = ""
    dry_run: bool = False
    task_type: str = "math"
    n_gpus: int = 2
    cpus_per_gpu: int = 4
    mem_per_gpu: str = "48G"


class EvalSubmitRequest(BaseModel):
    experiment_name: str
    dataset: str
    n_rollouts: int = 1
    slurm_config: str = "preempt_l40s"
    cpus_per_gpu: int = 4
    mem_per_gpu: str = "80G"
    dry_run: bool = False
    task_type: str = "math"
    trajectory_analysis: bool = False


class AnalyzeRequest(BaseModel):
    experiment_name: str


# ── API endpoints ────────────────────────────────────────────────────────────


@app.get("/api/experiments")
def api_experiments():
    """All experiments + SLURM status + metadata."""
    experiments, slurm_jobs = build_experiment_table(_checkpoint_dir)
    return {
        "experiments": experiments,
        "slurm_jobs": slurm_jobs,
        "metadata": {
            "dataset_categories": DATASET_CATEGORIES,
            "model_tabs": MODEL_TABS,
            "eval_datasets": EVAL_DATASETS,
        },
    }


@app.get("/api/experiment/{name}")
def api_experiment_detail(name: str):
    """Single experiment detail + eval results + trajectory analysis info."""
    experiments, _ = build_experiment_table(_checkpoint_dir)
    exp = next((e for e in experiments if e["name"] == name), None)
    if not exp:
        return {"error": f"Experiment '{name}' not found"}
    eval_data = get_eval_results_table(name, _checkpoint_dir)
    traj_dirs = has_trajectory_dirs(name, _checkpoint_dir)
    analysis_md = get_analysis_markdown(name, _checkpoint_dir)
    analysis_status = get_analysis_status(name)
    return {
        "experiment": exp,
        "eval": eval_data,
        "trajectory_dirs": traj_dirs,
        "analysis_markdown": analysis_md,
        "analysis_status": analysis_status,
    }


@app.get("/api/slurm-configs")
def api_slurm_configs():
    """Available SLURM configs with GPU info."""
    configs = load_slurm_configs()
    result = []
    for name in configs:
        info = parse_slurm_config(name)
        result.append({"name": name, **info})
    return {"configs": result}


@app.post("/api/assign-job")
def api_assign_job(req: AssignJobRequest):
    msg = assign_job_id(req.experiment_name, req.job_id, _checkpoint_dir)
    return {"message": msg}


@app.post("/api/cancel-job")
def api_cancel_job(req: CancelJobRequest):
    msg = cancel_job(req.job_id)
    return {"message": msg}


@app.post("/api/launch")
def api_launch(req: LaunchRequest):
    output = launch_experiment(
        req.workflow, req.model, req.share_policy,
        req.node, req.extra_args, req.dry_run,
        task_type=req.task_type,
        n_gpus=req.n_gpus,
        cpus_per_gpu=req.cpus_per_gpu,
        mem_per_gpu=req.mem_per_gpu,
    )
    return {"output": output}


@app.post("/api/eval/submit")
def api_eval_submit(req: EvalSubmitRequest):
    output = submit_eval_job(
        experiment_name=req.experiment_name,
        dataset=req.dataset,
        n_rollouts=req.n_rollouts,
        slurm_config=req.slurm_config,
        cpus_per_gpu=req.cpus_per_gpu,
        mem_per_gpu=req.mem_per_gpu,
        checkpoint_dir=_checkpoint_dir,
        dry_run=req.dry_run,
        task_type=req.task_type,
        trajectory_analysis=req.trajectory_analysis,
    )
    return {"output": output}


@app.post("/api/analyze-trajectories")
def api_analyze_trajectories(req: AnalyzeRequest):
    msg = launch_trajectory_analysis(req.experiment_name, _checkpoint_dir)
    return {"message": msg}


@app.get("/api/analysis-status/{name}")
def api_analysis_status(name: str):
    status = get_analysis_status(name)
    md = get_analysis_markdown(name, _checkpoint_dir) if status != "running" else None
    return {"status": status, "markdown": md}


# ── Static file serving ─────────────────────────────────────────────────────


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── CLI entry point ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="RLLM Dashboard Server")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
        help="Root checkpoint directory",
    )
    args = parser.parse_args()

    global _checkpoint_dir
    _checkpoint_dir = args.checkpoint_dir

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
