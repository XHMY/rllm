#!/usr/bin/env python3
"""Migrate records from global eval_results.jsonl into per-experiment files.

For each record whose experiment_name matches an existing checkpoint directory,
appends the record to {checkpoint_dir}/{project}/{experiment_name}/eval_results.jsonl.

Deduplicates by (checkpoint_step, dataset, eval_mode, n_rollouts) so the script
is idempotent — running it twice produces no duplicates.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_PROJECT = "rllm-workflow-MARL-v2"


def load_existing_keys(path: Path) -> set[tuple]:
    """Load dedup keys from an existing per-experiment eval_results.jsonl."""
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            keys.add((
                rec.get("checkpoint_step"),
                rec.get("dataset"),
                rec.get("eval_mode"),
                rec.get("n_rollouts"),
            ))
        except json.JSONDecodeError:
            continue
    return keys


def main():
    parser = argparse.ArgumentParser(description="Migrate global eval_results.jsonl to per-experiment files")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Root checkpoint directory")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Project name")
    parser.add_argument("--global-file", default="eval_results.jsonl", help="Path to global eval_results.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing")
    args = parser.parse_args()

    global_path = Path(args.global_file)
    if not global_path.exists():
        print(f"Global file not found: {global_path}")
        return

    project_dir = Path(args.checkpoint_dir) / args.project

    # Group records by experiment_name
    by_experiment: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for line in global_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            total += 1
            exp_name = rec.get("experiment_name", "")
            if exp_name:
                by_experiment[exp_name].append(rec)
        except json.JSONDecodeError:
            continue

    print(f"Read {total} records from {global_path} ({len(by_experiment)} experiments)")

    migrated_total = 0
    migrated_experiments = 0
    orphaned_total = 0
    orphaned_experiments = []

    for exp_name in sorted(by_experiment):
        records = by_experiment[exp_name]
        exp_dir = project_dir / exp_name

        if not exp_dir.is_dir():
            orphaned_total += len(records)
            orphaned_experiments.append(f"  {len(records):3d}  {exp_name} (no checkpoint dir)")
            continue

        # Deduplicate against existing per-experiment file
        per_exp_path = exp_dir / "eval_results.jsonl"
        existing_keys = load_existing_keys(per_exp_path)

        new_records = []
        for rec in records:
            key = (
                rec.get("checkpoint_step"),
                rec.get("dataset"),
                rec.get("eval_mode"),
                rec.get("n_rollouts"),
            )
            if key not in existing_keys:
                new_records.append(rec)
                existing_keys.add(key)  # prevent duplicates within batch

        if not new_records:
            print(f"  {exp_name}: {len(records)} records, all already present — skipped")
            continue

        if args.dry_run:
            print(f"  {exp_name}: would write {len(new_records)} new records (of {len(records)} total)")
        else:
            with open(per_exp_path, "a") as f:
                for rec in new_records:
                    f.write(json.dumps(rec) + "\n")
            print(f"  {exp_name}: wrote {len(new_records)} new records to {per_exp_path}")

        migrated_total += len(new_records)
        migrated_experiments += 1

    print(f"\nSummary:")
    print(f"  Migrated: {migrated_total} records across {migrated_experiments} experiments")
    print(f"  Orphaned: {orphaned_total} records across {len(orphaned_experiments)} experiments (no checkpoint dir)")
    if orphaned_experiments:
        print(f"\nOrphaned experiments:")
        for line in orphaned_experiments:
            print(line)


if __name__ == "__main__":
    main()
