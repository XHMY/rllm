"""RLLM Experiment Dashboard — monitor, launch, and cancel training experiments.

Backend logic is shared with the FastAPI dashboard in dashboard/backend.py.
This file imports from there and adds the Gradio UI layer.
"""

import argparse
import re

import gradio as gr
import pandas as pd

from dashboard.backend import (
    DEFAULT_CHECKPOINT_DIR,
    DATASET_CATEGORIES,
    MODEL_TABS,
    POLICY_ABBREV,
    STATUS_EMOJI,
    TOTAL_TRAINING_STEPS,
    assign_job_id,
    cancel_job,
    launch_experiment,
    load_eval_results,
    load_slurm_configs,
    parse_slurm_config,
    submit_eval_job,
)

# ── Gradio-specific helpers that need pandas DataFrames ──────────────────────


def _get_slurm_jobs_df() -> pd.DataFrame:
    """Get SLURM jobs as a pandas DataFrame (Gradio needs this format)."""
    from dashboard.backend import get_slurm_jobs
    jobs = get_slurm_jobs()
    if not jobs:
        return pd.DataFrame(
            columns=["Job ID", "Partition", "Name", "State", "Time", "Nodes", "Nodelist"]
        )
    return pd.DataFrame([{
        "Job ID": j["job_id"],
        "Partition": j["partition"],
        "Name": j["name"],
        "State": j["state"],
        "Time": j["time"],
        "Nodes": j["nodes"],
        "Nodelist": j["nodelist"],
    } for j in jobs])


def _scan_checkpoints_df(checkpoint_dir: str) -> pd.DataFrame:
    """Scan checkpoints and return as a pandas DataFrame (Gradio format)."""
    from dashboard.backend import scan_checkpoints
    exps = scan_checkpoints(checkpoint_dir)
    if not exps:
        return pd.DataFrame(
            columns=["Experiment", "Workflow", "Model", "Policy", "Dataset",
                     "Steps", "Status", "WandB Run", "SLURM Job"]
        )
    rows = []
    for e in exps:
        rows.append({
            "Experiment": e["name"],
            "Workflow": e["workflow"],
            "Model": e["model"],
            "Policy": e["policy"],
            "Dataset": e["dataset"],
            "Steps": f"{e['steps']}/{e['total_steps']}",
            "_steps": e["steps"],
            "_total": e["total_steps"],
            "_eval_best": e.get("eval_best", {}),
            "Status": "",
            "WandB Run": e.get("wandb_run", "\u2014"),
            "SLURM Job": e.get("slurm_job", "\u2014"),
        })
    return pd.DataFrame(rows)


MERGED_COLUMNS = [
    "Experiment", "Workflow", "Model", "Policy", "Dataset",
    "Steps", "Status", "Eval Status", "Eval Job",
    "WandB Run", "SLURM Job", "SLURM State", "Time", "Node",
]


def build_experiment_table(checkpoint_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (merged_df, slurm_df) for the Gradio UI."""
    exp_df = _scan_checkpoints_df(checkpoint_dir)
    slurm_df = _get_slurm_jobs_df()

    slurm_lookup: dict[str, dict] = {}
    eval_slurm_lookup: dict[str, str] = {}
    if not slurm_df.empty:
        for _, row in slurm_df.iterrows():
            slurm_lookup[str(row["Job ID"]).strip()] = {
                "SLURM State": row["State"],
                "Time": row["Time"],
                "Node": row["Nodelist"],
            }
            job_name = str(row.get("Name", ""))
            if job_name.startswith("eval-"):
                eval_exp_name = job_name[len("eval-"):]
                eval_slurm_lookup[eval_exp_name] = str(row["Job ID"]).strip()

    slurm_states, times, nodes, statuses = [], [], [], []
    eval_statuses, eval_jobs = [], []
    for _, row in exp_df.iterrows():
        job_id = str(row.get("SLURM Job", "\u2014")).strip()
        info = slurm_lookup.get(job_id, {})
        slurm_state = info.get("SLURM State", "")
        slurm_states.append(slurm_state or "\u2014")
        times.append(info.get("Time", "\u2014"))
        nodes.append(info.get("Node", "\u2014"))

        steps = row.get("_steps", 0)
        total = row.get("_total", TOTAL_TRAINING_STEPS)
        if steps >= total:
            statuses.append("Finished")
        elif slurm_state in ("RUNNING", "PENDING", "CONFIGURING"):
            statuses.append("Running")
        else:
            statuses.append("Unfinished")

        exp_name = row.get("Experiment", "")
        eval_best = row.get("_eval_best", {})
        has_eval = bool(eval_best)
        is_evaluating = exp_name in eval_slurm_lookup
        eval_job_id = eval_slurm_lookup.get(exp_name, "\u2014")
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

    exp_df = exp_df.drop(columns=["_steps", "_total"], errors="ignore")
    for col in MERGED_COLUMNS:
        if col not in exp_df.columns:
            exp_df[col] = "\u2014"
    merged_df = exp_df

    return merged_df, slurm_df


# ── Pivot grid + detail helpers ──────────────────────────────────────────────
STATUS_BAR_COLORS = {"Finished": "#4caf50", "Running": "#2196f3", "Unfinished": "#ff9800"}


def _progress_bar_html(steps_str: str, status: str, label: str) -> str:
    """Return HTML for a single experiment line with emoji, label, and progress bar."""
    emoji = STATUS_EMOJI.get(status, "\u23f8\ufe0f")
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
        indicator = ("\ud83d\udd04 " + indicator) if indicator else "\ud83d\udd04 evaluating"
    if indicator:
        return f'<div style="font-size:0.8em;color:#666;margin-top:1px">\ud83d\udcca {indicator}</div>'
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
                row_dict[wf] = "\u2014"
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
    emoji = STATUS_EMOJI.get(row["Status"], "\u23f8\ufe0f")
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
        f"| Eval Job | {row.get('Eval Job', chr(8212))} |",
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
            cells.append(f"{acc:.1%}" if acc is not None else "\u2014")
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
                label = f"{ds_key} \u2014 {model} ({n_exp} experiment{'s' if n_exp != 1 else ''})"
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
            val = match.iloc[0].get("SLURM Job", "\u2014")
            if val and val != "\u2014":
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
        if not job_id or job_id == "\u2014":
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

        # ── Nested tabs: Dataset (outer) > Model (inner) ────────────────
        with gr.Tabs():
            for ds_key in DATASET_CATEGORIES:
                with gr.TabItem(ds_key):
                    with gr.Tabs():
                        for model in MODEL_TABS:
                            with gr.TabItem(model):
                                key = f"{ds_key}_{model}"
                                grids[key] = gr.Dataframe(
                                    label=f"{ds_key} \u2014 {model} Experiments",
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

        # Assign job ID -> refresh
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
