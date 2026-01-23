"""Analyze evaluation results and create comparison tables.

This script reads the eval_results.jsonl file and creates comparison tables
showing the performance of different workflows before and after training.

Usage:
    python -m examples.math_reasoning.analyze_eval_results --input eval_results.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_results(input_path: str) -> pd.DataFrame:
    """Load evaluation results from JSON Lines file."""
    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def get_last_checkpoint(df: pd.DataFrame, workflow_type: str, eval_mode: str) -> pd.Series | None:
    """Get the last (highest step) checkpoint for a workflow type and eval mode."""
    subset = df[(df["workflow_type"] == workflow_type) & (df["eval_mode"] == eval_mode)]
    if subset.empty:
        return None
    # For trained_checkpoint, get the highest step; for others, just get the first
    if eval_mode == "trained_checkpoint":
        return subset.loc[subset["checkpoint_step"].idxmax()]
    return subset.iloc[0]


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a comparison table showing before/after training performance.

    Columns:
    - Workflow Type
    - Base Model (no training)
    - Single-Agent Trained (for reference)
    - Single-Agent Transfer (single-agent LoRA applied to multi-agent workflow)
    - Multi-Agent Trained (workflow-specific training)
    - Improvement (trained vs base)
    """
    workflows = ["single_agent", "voting", "evaluator_optimizer", "orchestrator_workers"]

    rows = []
    for wf in workflows:
        row = {"Workflow": wf}

        # Base model performance
        base = get_last_checkpoint(df, wf, "base_model")
        row["Base Model"] = f"{base['accuracy']:.2%}" if base is not None else "-"
        base_acc = base["accuracy"] if base is not None else None

        # Single-agent trained (only for single_agent workflow)
        if wf == "single_agent":
            trained = get_last_checkpoint(df, wf, "trained_checkpoint")
            if trained is not None:
                row["Single-Agent Trained"] = f"{trained['accuracy']:.2%} (step {trained['checkpoint_step']})"
                single_agent_acc = trained["accuracy"]
            else:
                row["Single-Agent Trained"] = "-"
                single_agent_acc = None
        else:
            row["Single-Agent Trained"] = "-"
            single_agent_acc = None

        # Single-agent transfer (using single-agent LoRA for multi-agent workflow)
        transfer = get_last_checkpoint(df, wf, "single_agent_transfer")
        if transfer is not None:
            row["Single-Agent Transfer"] = f"{transfer['accuracy']:.2%}"
            transfer_acc = transfer["accuracy"]
        else:
            row["Single-Agent Transfer"] = "-"
            transfer_acc = None

        # Multi-agent trained (workflow-specific training)
        if wf != "single_agent":
            trained = get_last_checkpoint(df, wf, "trained_checkpoint")
            if trained is not None:
                row["Multi-Agent Trained"] = f"{trained['accuracy']:.2%} (step {trained['checkpoint_step']})"
                trained_acc = trained["accuracy"]
            else:
                row["Multi-Agent Trained"] = "-"
                trained_acc = None
        else:
            row["Multi-Agent Trained"] = "-"
            trained_acc = None

        # Calculate improvements
        if wf == "single_agent":
            # For single-agent, compare trained vs base
            if base_acc is not None and single_agent_acc is not None:
                improvement = single_agent_acc - base_acc
                row["Improvement"] = f"+{improvement:.2%}" if improvement > 0 else f"{improvement:.2%}"
            else:
                row["Improvement"] = "-"
        else:
            # For multi-agent, show best improvement
            best_acc = None
            best_source = None
            if transfer_acc is not None:
                best_acc = transfer_acc
                best_source = "transfer"
            if trained_acc is not None and (best_acc is None or trained_acc > best_acc):
                best_acc = trained_acc
                best_source = "trained"

            if base_acc is not None and best_acc is not None:
                improvement = best_acc - base_acc
                row["Improvement"] = f"+{improvement:.2%} ({best_source})" if improvement > 0 else f"{improvement:.2%}"
            else:
                row["Improvement"] = "-"

        rows.append(row)

    return pd.DataFrame(rows)


def create_detailed_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a detailed table with all results."""
    # Select and order columns
    columns = [
        "workflow_type",
        "eval_mode",
        "checkpoint_step",
        "accuracy",
        "num_correct",
        "num_total",
        "model_size",
        "share_policy",
        "hostname",
        "timestamp",
    ]

    # Filter to available columns
    available_cols = [c for c in columns if c in df.columns]
    result = df[available_cols].copy()

    # Sort by workflow_type, eval_mode, and checkpoint_step
    result = result.sort_values(
        ["workflow_type", "eval_mode", "checkpoint_step"],
        ascending=[True, True, False]
    )

    # Format accuracy as percentage
    result["accuracy"] = result["accuracy"].apply(lambda x: f"{x:.2%}")

    return result


def create_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table for easy comparison.

    Rows: Workflow types
    Columns: Evaluation modes
    Values: Accuracy (last checkpoint)
    """
    workflows = ["single_agent", "voting", "evaluator_optimizer", "orchestrator_workers"]
    eval_modes = ["base_model", "single_agent_transfer", "trained_checkpoint"]

    data = []
    for wf in workflows:
        row = {"Workflow": wf}
        for mode in eval_modes:
            result = get_last_checkpoint(df, wf, mode)
            if result is not None:
                acc = result["accuracy"]
                step = result["checkpoint_step"]
                if mode == "trained_checkpoint" and step > 0:
                    row[mode] = f"{acc:.2%} (step {step})"
                else:
                    row[mode] = f"{acc:.2%}"
            else:
                row[mode] = "-"
        data.append(row)

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument(
        "--input",
        type=str,
        default="eval_results.jsonl",
        help="Input JSON Lines file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (optional)",
    )
    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.input}")
    df = load_results(args.input)
    print(f"Loaded {len(df)} evaluation records\n")

    # Print summary statistics
    print("=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nTotal records: {len(df)}")
    print(f"Workflow types: {df['workflow_type'].unique().tolist()}")
    print(f"Eval modes: {df['eval_mode'].unique().tolist()}")
    print(f"Model sizes: {df['model_size'].unique().tolist()}")

    # Create and print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE: Before vs After Training")
    print("=" * 80)
    comparison_df = create_comparison_table(df)
    print(comparison_df.to_string(index=False))

    # Create and print pivot table
    print("\n" + "=" * 80)
    print("PIVOT TABLE: Accuracy by Workflow and Eval Mode")
    print("=" * 80)
    pivot_df = create_pivot_table(df)
    print(pivot_df.to_string(index=False))

    # Create and print detailed table
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    detailed_df = create_detailed_table(df)
    print(detailed_df.to_string(index=False))

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Single-agent improvement
    single_base = get_last_checkpoint(df, "single_agent", "base_model")
    single_trained = get_last_checkpoint(df, "single_agent", "trained_checkpoint")
    if single_base is not None and single_trained is not None:
        improvement = single_trained["accuracy"] - single_base["accuracy"]
        print(f"\n1. Single-Agent Training Effect:")
        print(f"   Base model: {single_base['accuracy']:.2%}")
        print(f"   After training (step {single_trained['checkpoint_step']}): {single_trained['accuracy']:.2%}")
        print(f"   Improvement: +{improvement:.2%} ({improvement/single_base['accuracy']*100:.1f}% relative)")

    # Single-agent transfer effect on multi-agent workflows
    print(f"\n2. Single-Agent Transfer to Multi-Agent Workflows:")
    for wf in ["voting", "evaluator_optimizer", "orchestrator_workers"]:
        base = get_last_checkpoint(df, wf, "base_model")
        transfer = get_last_checkpoint(df, wf, "single_agent_transfer")
        if base is not None and transfer is not None:
            improvement = transfer["accuracy"] - base["accuracy"]
            print(f"   {wf}: {base['accuracy']:.2%} -> {transfer['accuracy']:.2%} (+{improvement:.2%})")

    # Multi-agent training effect
    print(f"\n3. Multi-Agent Workflow-Specific Training:")
    for wf in ["voting", "evaluator_optimizer", "orchestrator_workers"]:
        base = get_last_checkpoint(df, wf, "base_model")
        trained = get_last_checkpoint(df, wf, "trained_checkpoint")
        if base is not None and trained is not None:
            improvement = trained["accuracy"] - base["accuracy"]
            print(f"   {wf}: {base['accuracy']:.2%} -> {trained['accuracy']:.2%} (step {trained['checkpoint_step']}, +{improvement:.2%})")
        elif base is not None:
            print(f"   {wf}: {base['accuracy']:.2%} -> not yet trained")

    # Comparison: single-agent transfer vs multi-agent training
    print(f"\n4. Single-Agent Transfer vs Multi-Agent Training:")
    for wf in ["voting", "evaluator_optimizer", "orchestrator_workers"]:
        transfer = get_last_checkpoint(df, wf, "single_agent_transfer")
        trained = get_last_checkpoint(df, wf, "trained_checkpoint")
        if transfer is not None and trained is not None:
            diff = trained["accuracy"] - transfer["accuracy"]
            winner = "Multi-Agent" if diff > 0 else "Transfer"
            print(f"   {wf}: Transfer={transfer['accuracy']:.2%}, Multi-Agent={trained['accuracy']:.2%} ({winner} better by {abs(diff):.2%})")
        elif transfer is not None:
            print(f"   {wf}: Transfer={transfer['accuracy']:.2%}, Multi-Agent=not yet trained")
        elif trained is not None:
            print(f"   {wf}: Transfer=not evaluated, Multi-Agent={trained['accuracy']:.2%}")

    # Save to CSV if requested
    if args.output:
        comparison_df.to_csv(args.output, index=False)
        print(f"\nComparison table saved to: {args.output}")


if __name__ == "__main__":
    main()
