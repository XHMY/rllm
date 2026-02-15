#!/usr/bin/env python3
"""
Unified trajectory analysis script for all multi-agent workflow types.

Auto-detects workflow type from trajectory agent names and runs appropriate analysis:
  - single-agent: baseline accuracy and token usage
  - evaluator-optimizer: iteration benefit, evaluator calibration, recovery analysis
  - voting: generator agreement, aggregator selection quality
  - orchestrator-workers: decomposition quality, worker success, synthesis quality

Usage:
    # Single directory (original behavior)
    python scripts/analyze_trajectories.py evaluation_trajectories/voting-qwen3_1.7b-math_step290/

    # Parent directory — auto-discovers subdirs, shows comparison table + full analysis
    python scripts/analyze_trajectories.py evaluation_trajectories/

    # Compare steps only (no full per-dir analysis)
    python scripts/analyze_trajectories.py evaluation_trajectories/ --compare-steps

    # Multiple explicit directories
    python scripts/analyze_trajectories.py eval_traj/voting_step270/ eval_traj/voting_step540/

    # With run filter
    python scripts/analyze_trajectories.py evaluation_trajectories/ --run-filter run_0
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict


# ============================================================================
# Parsing
# ============================================================================

def load_trajectory_files(traj_dir, run_filter=None):
    """Load all JSON trajectory files from a directory, optionally filtering by run."""
    files = sorted(f for f in os.listdir(traj_dir) if f.endswith(".json"))
    if run_filter:
        files = [f for f in files if run_filter in f]
    all_data = []
    for fname in files:
        filepath = os.path.join(traj_dir, fname)
        with open(filepath) as f:
            data = json.load(f)
        data["_filename"] = fname
        all_data.append(data)
    return all_data


def detect_workflow_type(all_data):
    """Detect workflow type based on trajectory agent names."""
    agent_names = set()
    for d in all_data:
        for t in d.get("trajectories", []):
            agent_names.add(t["name"])

    if "orchestrator" in agent_names and "worker" in agent_names:
        return "orchestrator-workers"
    if "aggregator" in agent_names:
        return "voting"
    if "evaluator" in agent_names:
        return "evaluator-optimizer"
    return "single-agent"


def get_agent_names(all_data):
    """Get sorted list of unique agent names across all trajectories."""
    names = set()
    for d in all_data:
        for t in d.get("trajectories", []):
            names.add(t["name"])
    return sorted(names)


# ============================================================================
# Common analysis (all workflows)
# ============================================================================

def print_common_analysis(all_data, traj_dir, workflow_type):
    """Print analysis common to all workflow types."""
    total = len(all_data)
    correct = sum(1 for d in all_data if d["is_correct"])

    print("=" * 80)
    print(f"TRAJECTORY ANALYSIS — {workflow_type.upper()}")
    print(f"Directory: {traj_dir}")
    print(f"Total files: {total}")
    print(f"Agent names: {get_agent_names(all_data)}")
    print("=" * 80)

    # Overall accuracy
    print(f"\n{'='*80}")
    print("OVERALL ACCURACY")
    print(f"{'='*80}")
    print(f"  Correct: {correct}/{total} ({correct/total*100:.1f}%)")

    # Per-problem accuracy (group by problem_idx)
    problems = defaultdict(list)
    for d in all_data:
        idx = d["task"].get("problem_idx", d["_filename"])
        problems[idx].append(d)

    print(f"  Unique problems: {len(problems)}")
    if len(problems) < total:
        # Multiple runs per problem
        n_runs = total // len(problems) if len(problems) > 0 else 0
        print(f"  Runs per problem: ~{n_runs}")

        per_problem_acc = {}
        for idx, runs in sorted(problems.items()):
            c = sum(1 for r in runs if r["is_correct"])
            per_problem_acc[idx] = (c, len(runs))

        always_correct = sum(1 for c, n in per_problem_acc.values() if c == n)
        always_wrong = sum(1 for c, n in per_problem_acc.values() if c == 0)
        sometimes = len(per_problem_acc) - always_correct - always_wrong
        print(f"  Always correct (all runs): {always_correct}")
        print(f"  Always wrong (all runs):   {always_wrong}")
        print(f"  Mixed (some runs correct):  {sometimes}")

    # Token usage (character-based approximation)
    print(f"\n{'='*80}")
    print("TOKEN USAGE (character counts as proxy)")
    print(f"{'='*80}")

    agent_prompt_chars = defaultdict(list)
    agent_response_chars = defaultdict(list)

    for d in all_data:
        for t in d["trajectories"]:
            name = t["name"]
            for step in t["steps"]:
                cc = step.get("chat_completions", [])
                if not isinstance(cc, list):
                    continue
                prompt_c = sum(len(str(m.get("content", ""))) for m in cc if m.get("role") != "assistant")
                response_c = sum(
                    len(str(m.get("content", ""))) + len(str(m.get("reasoning", "")))
                    for m in cc if m.get("role") == "assistant"
                )
                agent_prompt_chars[name].append(prompt_c)
                agent_response_chars[name].append(response_c)

    for name in sorted(agent_prompt_chars.keys()):
        p = agent_prompt_chars[name]
        r = agent_response_chars[name]
        print(f"\n  {name}:")
        print(f"    Samples:  {len(p)}")
        print(f"    Prompt:   avg={sum(p)/len(p):.0f}  min={min(p)}  max={max(p)}")
        print(f"    Response: avg={sum(r)/len(r):.0f}  min={min(r)}  max={max(r)}")


# ============================================================================
# Single-Agent analysis
# ============================================================================

def analyze_single_agent(all_data):
    """Baseline accuracy and per-problem breakdown."""
    total = len(all_data)
    correct = sum(1 for d in all_data if d["is_correct"])

    print(f"\n{'='*80}")
    print("SINGLE-AGENT BASELINE")
    print(f"{'='*80}")
    print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  This serves as the baseline for comparing multi-agent workflows.")


# ============================================================================
# Evaluator-Optimizer analysis
# ============================================================================

def analyze_evaluator_optimizer(all_data):
    """Full evaluator-optimizer analysis: iteration benefit, calibration, recovery, failure modes."""
    total = len(all_data)

    # Extract generator/evaluator steps per file
    parsed = []
    for d in all_data:
        entry = {
            "filename": d["_filename"],
            "is_correct": d["is_correct"],
            "ground_truth": d["task"]["ground_truth"],
            "total_iterations": d["metrics"]["total_iterations"],
            "generator_attempts": d["metrics"]["generator_attempts"],
            "generator_steps": [],
            "evaluator_steps": [],
        }
        for t in d["trajectories"]:
            if t["name"] == "generator":
                step = t["steps"][0]
                entry["generator_steps"].append({"reward": step["reward"]})
            elif t["name"] == "evaluator":
                step = t["steps"][0]
                action = step["action"]
                verdict = None
                if isinstance(action, dict):
                    verdict = action.get("verdict")
                entry["evaluator_steps"].append({"verdict": verdict, "reward": step["reward"]})
        parsed.append(entry)

    # ---- 1. Iteration Benefit ----
    print(f"\n{'='*80}")
    print("1. ITERATION BENEFIT ANALYSIS")
    print(f"{'='*80}")

    initially_correct = [p for p in parsed if p["generator_steps"][0]["reward"] == 1.0]
    initially_wrong_recovered = [p for p in parsed if p["generator_steps"][0]["reward"] != 1.0 and p["is_correct"]]
    initially_wrong_still_wrong = [p for p in parsed if p["generator_steps"][0]["reward"] != 1.0 and not p["is_correct"]]
    stayed_wrong_all = [p for p in initially_wrong_still_wrong if all(s["reward"] == 0.0 for s in p["generator_steps"])]

    print(f"\n  Total problems: {total}")
    print(f"  Initially correct (attempt 1 correct):           {len(initially_correct):4d} ({len(initially_correct)/total*100:.1f}%)")
    print(f"  Initially wrong, eventually correct (recovery):  {len(initially_wrong_recovered):4d} ({len(initially_wrong_recovered)/total*100:.1f}%)")
    print(f"  Initially wrong, still wrong (no recovery):      {len(initially_wrong_still_wrong):4d} ({len(initially_wrong_still_wrong)/total*100:.1f}%)")
    print(f"    - Of which ALL attempts wrong:                 {len(stayed_wrong_all):4d}")

    initially_correct_but_final_wrong = [p for p in initially_correct if not p["is_correct"]]
    if initially_correct_but_final_wrong:
        print(f"\n  WARNING: Initially correct but final answer wrong: {len(initially_correct_but_final_wrong)}")
        for p in initially_correct_but_final_wrong:
            print(f"    {p['filename']}: iterations={p['total_iterations']}")

    # ---- 2. Per-Iteration Accuracy ----
    print(f"\n{'='*80}")
    print("2. PER-ITERATION ACCURACY")
    print(f"{'='*80}")

    max_attempts = max(p["generator_attempts"] for p in parsed)
    print(f"\n  Max generator attempts: {max_attempts}")

    for idx in range(max_attempts):
        with_attempt = [p for p in parsed if len(p["generator_steps"]) > idx]
        correct_at = [p for p in with_attempt if p["generator_steps"][idx]["reward"] == 1.0]
        n, c = len(with_attempt), len(correct_at)
        if n > 0:
            print(f"\n  Attempt {idx+1}:")
            print(f"    Files reaching this attempt: {n}")
            print(f"    Correct at this attempt:     {c} ({c/n*100:.1f}%)")

    print(f"\n  Cumulative accuracy (correct by attempt N, across all {total} files):")
    for idx in range(max_attempts):
        correct_by = 0
        for p in parsed:
            for j in range(min(idx + 1, len(p["generator_steps"]))):
                if p["generator_steps"][j]["reward"] == 1.0:
                    correct_by += 1
                    break
        print(f"    By attempt {idx+1}: {correct_by}/{total} ({correct_by/total*100:.1f}%)")

    # ---- 3. Evaluator Calibration ----
    print(f"\n{'='*80}")
    print("3. EVALUATOR CALIBRATION")
    print(f"{'='*80}")

    tp = fp = fn = tn = unknown = 0
    for p in parsed:
        n_pairs = min(len(p["generator_steps"]), len(p["evaluator_steps"]))
        for i in range(n_pairs):
            gen_correct = p["generator_steps"][i]["reward"] == 1.0
            verdict = p["evaluator_steps"][i]["verdict"]
            if verdict is None:
                unknown += 1
                continue
            v = verdict.strip().lower()
            if v == "correct":
                if gen_correct:
                    tp += 1
                else:
                    fp += 1
            elif v == "incorrect":
                if gen_correct:
                    fn += 1
                else:
                    tn += 1
            else:
                unknown += 1

    total_pairs = tp + fp + fn + tn
    print(f"\n  Total (generator, evaluator) pairs: {total_pairs}")
    if unknown > 0:
        print(f"  Unknown/missing verdict: {unknown}")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':30s} Generator CORRECT  Generator WRONG")
    print(f"  Evaluator says 'correct':    {tp:10d}         {fp:10d}")
    print(f"  Evaluator says 'incorrect':  {fn:10d}         {tn:10d}")

    says_correct = tp + fp
    says_incorrect = fn + tn
    actually_correct = tp + fn
    actually_wrong = fp + tn

    if says_correct > 0:
        print(f"\n  Precision (eval says correct, gen is correct): {tp}/{says_correct} = {tp/says_correct*100:.1f}%")
    if says_incorrect > 0:
        print(f"  Precision-neg (eval says incorrect, gen is wrong): {tn}/{says_incorrect} = {tn/says_incorrect*100:.1f}%")
    if actually_correct > 0:
        print(f"  Recall (gen correct, eval says correct): {tp}/{actually_correct} = {tp/actually_correct*100:.1f}%")
    if actually_wrong > 0:
        print(f"  Recall-neg (gen wrong, eval says incorrect): {tn}/{actually_wrong} = {tn/actually_wrong*100:.1f}%")
    if total_pairs > 0:
        acc = (tp + tn) / total_pairs
        print(f"  Overall evaluator accuracy: {tp+tn}/{total_pairs} = {acc*100:.1f}%")

    # ---- 4. Recovery Analysis ----
    print(f"\n{'='*80}")
    print("4. RECOVERY ANALYSIS")
    print(f"{'='*80}")

    attempt1_correct = [p for p in parsed if p["generator_steps"][0]["reward"] == 1.0]
    print(f"\n  Attempt 1 correct: {len(attempt1_correct)} ({len(attempt1_correct)/total*100:.1f}%)")

    one_step = [p for p in parsed if len(p["generator_steps"]) >= 2 and p["generator_steps"][0]["reward"] == 0.0 and p["generator_steps"][1]["reward"] == 1.0]
    two_step = [p for p in parsed if len(p["generator_steps"]) >= 3 and p["generator_steps"][0]["reward"] == 0.0 and p["generator_steps"][1]["reward"] == 0.0 and p["generator_steps"][2]["reward"] == 1.0]
    no_recovery = [p for p in parsed if all(s["reward"] == 0.0 for s in p["generator_steps"])]

    print(f"  1-step recovery (attempt 1 wrong, attempt 2 correct): {len(one_step)} ({len(one_step)/total*100:.1f}%)")
    print(f"  2-step recovery (attempts 1-2 wrong, attempt 3 correct): {len(two_step)} ({len(two_step)/total*100:.1f}%)")
    print(f"  No recovery (all attempts wrong): {len(no_recovery)} ({len(no_recovery)/total*100:.1f}%)")

    total_final_correct = sum(1 for p in parsed if p["is_correct"])
    total_recovery = len(one_step) + len(two_step)
    print(f"\n  Final overall accuracy: {total_final_correct}/{total} ({total_final_correct/total*100:.1f}%)")
    print(f"  Total recovered through refinement: {total_recovery} ({total_recovery/total*100:.1f}%)")

    if one_step:
        print(f"\n  1-step recovery files:")
        for p in one_step:
            print(f"    {p['filename']} (ground_truth={p['ground_truth']})")
    if two_step:
        print(f"\n  2-step recovery files:")
        for p in two_step:
            print(f"    {p['filename']} (ground_truth={p['ground_truth']})")

    # ---- 5. Evaluator Failure Modes ----
    print(f"\n{'='*80}")
    print("5. EVALUATOR FAILURE MODES")
    print(f"{'='*80}")

    false_positives = []
    false_negatives = []
    for p in parsed:
        n_pairs = min(len(p["generator_steps"]), len(p["evaluator_steps"]))
        for i in range(n_pairs):
            gen_correct = p["generator_steps"][i]["reward"] == 1.0
            verdict = p["evaluator_steps"][i]["verdict"]
            if verdict is None:
                continue
            v = verdict.strip().lower()
            if v == "correct" and not gen_correct:
                false_positives.append({"filename": p["filename"], "attempt": i + 1, "ground_truth": p["ground_truth"], "final_correct": p["is_correct"]})
            elif v == "incorrect" and gen_correct:
                false_negatives.append({"filename": p["filename"], "attempt": i + 1, "ground_truth": p["ground_truth"], "final_correct": p["is_correct"]})

    print(f"\n  FALSE POSITIVES (eval says correct, gen was WRONG): {len(false_positives)}")
    print(f"  These cause the loop to STOP prematurely with a wrong answer.")
    for fp in false_positives:
        print(f"    {fp['filename']}, attempt {fp['attempt']}, gt={fp['ground_truth']}, final_correct={fp['final_correct']}")

    print(f"\n  FALSE NEGATIVES (eval says incorrect, gen was CORRECT): {len(false_negatives)}")
    print(f"  These cause unnecessary refinement (wasted iterations).")
    for fn_ in false_negatives:
        print(f"    {fn_['filename']}, attempt {fn_['attempt']}, gt={fn_['ground_truth']}, final_correct={fn_['final_correct']}")

    fp_final_wrong = sum(1 for fp in false_positives if not fp["final_correct"])
    fn_final_correct = sum(1 for fn_ in false_negatives if fn_["final_correct"])
    fn_final_wrong = sum(1 for fn_ in false_negatives if not fn_["final_correct"])
    print(f"\n  IMPACT:")
    print(f"    False positives leading to final wrong answer: {fp_final_wrong}")
    print(f"    False negatives where final still correct: {fn_final_correct}")
    print(f"    False negatives where final ended up wrong: {fn_final_wrong}")

    # ---- Distribution ----
    print(f"\n{'='*80}")
    print("DISTRIBUTION OF ITERATIONS")
    print(f"{'='*80}")
    iter_counts = defaultdict(int)
    for p in parsed:
        iter_counts[p["total_iterations"]] += 1
    for k in sorted(iter_counts):
        print(f"  {k} iteration(s): {iter_counts[k]} files ({iter_counts[k]/total*100:.1f}%)")

    attempt_counts = defaultdict(int)
    for p in parsed:
        attempt_counts[p["generator_attempts"]] += 1
    print(f"\n  Generator attempt distribution:")
    for k in sorted(attempt_counts):
        print(f"    {k} attempt(s): {attempt_counts[k]} files ({attempt_counts[k]/total*100:.1f}%)")


# ============================================================================
# Voting analysis
# ============================================================================

def analyze_voting(all_data):
    """Voting workflow analysis: generator agreement, aggregator quality."""
    total = len(all_data)

    # Extract per-file generator rewards and aggregator reward
    parsed = []
    for d in all_data:
        gen_trajs = sorted(
            [t for t in d["trajectories"] if t["name"].startswith("generator")],
            key=lambda t: t["name"],
        )
        agg_trajs = [t for t in d["trajectories"] if t["name"] == "aggregator"]
        gen_rewards = [t["steps"][0]["reward"] for t in gen_trajs]
        agg_reward = agg_trajs[0]["steps"][0]["reward"] if agg_trajs else None
        n_votes = d["metrics"].get("n_votes", len(gen_trajs))
        any_correct = d["metrics"].get("any_correct", int(any(r == 1.0 for r in gen_rewards)))
        parsed.append({
            "filename": d["_filename"],
            "is_correct": d["is_correct"],
            "ground_truth": d["task"]["ground_truth"],
            "gen_rewards": gen_rewards,
            "gen_names": [t["name"] for t in gen_trajs],
            "agg_reward": agg_reward,
            "n_votes": n_votes,
            "any_correct": any_correct,
        })

    n_votes = parsed[0]["n_votes"] if parsed else 0

    # ---- 1. Generator Agreement ----
    print(f"\n{'='*80}")
    print("1. GENERATOR AGREEMENT ANALYSIS")
    print(f"{'='*80}")

    all_agree_correct = [p for p in parsed if all(r == 1.0 for r in p["gen_rewards"])]
    all_agree_wrong = [p for p in parsed if all(r == 0.0 for r in p["gen_rewards"])]
    disagreement = [p for p in parsed if len(set(p["gen_rewards"])) > 1]

    print(f"\n  N votes per problem: {n_votes}")
    print(f"  All generators agree correct: {len(all_agree_correct):4d} ({len(all_agree_correct)/total*100:.1f}%)")
    print(f"  All generators agree wrong:   {len(all_agree_wrong):4d} ({len(all_agree_wrong)/total*100:.1f}%)")
    print(f"  Generators disagree:          {len(disagreement):4d} ({len(disagreement)/total*100:.1f}%)")

    # Per-generator accuracy
    print(f"\n{'='*80}")
    print("2. PER-GENERATOR ACCURACY")
    print(f"{'='*80}")

    gen_names = sorted(set(name for p in parsed for name in p["gen_names"]))
    for i, name in enumerate(gen_names):
        correct_count = sum(1 for p in parsed if len(p["gen_rewards"]) > i and p["gen_rewards"][i] == 1.0)
        print(f"  {name}: {correct_count}/{total} ({correct_count/total*100:.1f}%)")

    any_gen_correct = sum(1 for p in parsed if any(r == 1.0 for r in p["gen_rewards"]))
    print(f"\n  Any generator correct (oracle upper bound): {any_gen_correct}/{total} ({any_gen_correct/total*100:.1f}%)")

    # ---- 3. Aggregator Selection Quality ----
    print(f"\n{'='*80}")
    print("3. AGGREGATOR SELECTION QUALITY")
    print(f"{'='*80}")

    final_correct = sum(1 for p in parsed if p["is_correct"])
    print(f"\n  Final accuracy (after aggregation): {final_correct}/{total} ({final_correct/total*100:.1f}%)")

    # Scenario A: some gen correct, aggregator picked wrong
    missed_correct = [p for p in parsed if any(r == 1.0 for r in p["gen_rewards"]) and not p["is_correct"]]
    print(f"\n  Scenario A — Missed correct answer:")
    print(f"    Cases where >= 1 generator was correct but aggregator chose wrong: {len(missed_correct)}")
    for p in missed_correct[:10]:
        print(f"      {p['filename']}: gen_rewards={p['gen_rewards']}, gt={p['ground_truth']}")

    # Scenario B: disagreement, aggregator picked correctly
    contested_correct = [p for p in disagreement if p["is_correct"]]
    print(f"\n  Scenario B — Correct selection from disagreement:")
    print(f"    Cases with disagreement where aggregator chose correctly: {len(contested_correct)}")
    for p in contested_correct[:10]:
        print(f"      {p['filename']}: gen_rewards={p['gen_rewards']}, gt={p['ground_truth']}")

    # Aggregator recovery rate on contested cases
    if disagreement:
        recovery_rate = len(contested_correct) / len(disagreement)
        print(f"\n  Aggregator recovery rate on contested cases: {len(contested_correct)}/{len(disagreement)} ({recovery_rate*100:.1f}%)")


# ============================================================================
# Orchestrator-Workers analysis
# ============================================================================

def analyze_orchestrator_workers(all_data):
    """Orchestrator-workers analysis: decomposition, worker success, synthesis."""
    total = len(all_data)

    parsed = []
    for d in all_data:
        m = d["metrics"]
        orch_trajs = [t for t in d["trajectories"] if t["name"] == "orchestrator"]
        worker_trajs = [t for t in d["trajectories"] if t["name"] == "worker"]

        # Extract decomposition info from first orchestrator call
        subtasks = []
        if orch_trajs:
            action = orch_trajs[0]["steps"][0].get("action", {})
            if isinstance(action, dict):
                subtasks = action.get("subtasks", [])

        worker_rewards = [t["steps"][0]["reward"] for t in worker_trajs]

        parsed.append({
            "filename": d["_filename"],
            "is_correct": d["is_correct"],
            "ground_truth": d["task"]["ground_truth"],
            "n_subtasks": m.get("n_subtasks", len(subtasks)),
            "n_workers": m.get("n_workers", len(worker_trajs)),
            "successful_workers": m.get("successful_workers", sum(1 for r in worker_rewards if r == 1.0)),
            "worker_success_rate": m.get("worker_success_rate", 0),
            "orchestrator_calls": m.get("orchestrator_calls", len(orch_trajs)),
            "worker_rewards": worker_rewards,
            "subtasks": subtasks,
        })

    # ---- 1. Decomposition Quality ----
    print(f"\n{'='*80}")
    print("1. DECOMPOSITION QUALITY")
    print(f"{'='*80}")

    subtask_counts = defaultdict(int)
    for p in parsed:
        subtask_counts[p["n_subtasks"]] += 1
    print(f"\n  Subtask count distribution:")
    for k in sorted(subtask_counts):
        print(f"    {k} subtasks: {subtask_counts[k]} files ({subtask_counts[k]/total*100:.1f}%)")

    avg_subtasks = sum(p["n_subtasks"] for p in parsed) / total if total else 0
    print(f"  Average subtasks per problem: {avg_subtasks:.1f}")

    # ---- 2. Worker Success ----
    print(f"\n{'='*80}")
    print("2. WORKER SUCCESS ANALYSIS")
    print(f"{'='*80}")

    all_workers_succeed = sum(1 for p in parsed if p["worker_success_rate"] == 1.0)
    some_workers_fail = sum(1 for p in parsed if p["worker_success_rate"] < 1.0)
    print(f"\n  All workers succeed: {all_workers_succeed}/{total} ({all_workers_succeed/total*100:.1f}%)")
    print(f"  Some workers fail:   {some_workers_fail}/{total} ({some_workers_fail/total*100:.1f}%)")

    avg_worker_success = sum(p["worker_success_rate"] for p in parsed) / total if total else 0
    print(f"  Average worker success rate: {avg_worker_success*100:.1f}%")

    # ---- 3. Worker Redundancy ----
    print(f"\n{'='*80}")
    print("3. WORKER REDUNDANCY ANALYSIS")
    print(f"{'='*80}")

    # Check: do workers individually solve the problem (reward=1) even though orchestrator breaks it into subtasks?
    workers_all_correct_but_final_wrong = [p for p in parsed if all(r == 1.0 for r in p["worker_rewards"]) and not p["is_correct"]]
    workers_some_wrong_but_final_correct = [p for p in parsed if any(r == 0.0 for r in p["worker_rewards"]) and p["is_correct"]]

    print(f"\n  All workers correct but final wrong (synthesis failure): {len(workers_all_correct_but_final_wrong)}")
    print(f"  Some workers wrong but final correct (synthesis recovery): {len(workers_some_wrong_but_final_correct)}")

    # ---- 4. Synthesis Quality ----
    print(f"\n{'='*80}")
    print("4. SYNTHESIS QUALITY")
    print(f"{'='*80}")

    final_correct = sum(1 for p in parsed if p["is_correct"])
    print(f"\n  Final accuracy: {final_correct}/{total} ({final_correct/total*100:.1f}%)")

    # Conditional on worker success
    all_workers_ok = [p for p in parsed if p["worker_success_rate"] == 1.0]
    if all_workers_ok:
        c = sum(1 for p in all_workers_ok if p["is_correct"])
        print(f"  Accuracy when all workers succeed: {c}/{len(all_workers_ok)} ({c/len(all_workers_ok)*100:.1f}%)")

    some_workers_bad = [p for p in parsed if p["worker_success_rate"] < 1.0]
    if some_workers_bad:
        c = sum(1 for p in some_workers_bad if p["is_correct"])
        print(f"  Accuracy when some workers fail:   {c}/{len(some_workers_bad)} ({c/len(some_workers_bad)*100:.1f}%)")


# ============================================================================
# Multi-directory / step comparison
# ============================================================================

def parse_experiment_key(dirname):
    """Strip _step\\d+ suffix from directory name to get (experiment_key, step_number).

    Examples:
        'voting-qwen3_1.7b-math_step290' -> ('voting-qwen3_1.7b-math', 290)
        'qwen3_1.7b-math_single_agent-length5120_step430' -> ('qwen3_1.7b-math_single_agent-length5120', 430)
        'some_dir_without_step' -> ('some_dir_without_step', None)
    """
    m = re.search(r'^(.+?)_step(\d+)$', dirname)
    if m:
        return m.group(1), int(m.group(2))
    return dirname, None


def resolve_directories(paths):
    """Resolve input paths into a list of trajectory directories.

    If a path contains .json files, treat it as a trajectory directory.
    Otherwise, scan its immediate subdirectories for trajectory directories.
    """
    traj_dirs = []
    for p in paths:
        p = os.path.abspath(p)
        if not os.path.isdir(p):
            print(f"Warning: {p} is not a directory, skipping", file=sys.stderr)
            continue
        # Check if this directory itself contains JSON files
        json_files = [f for f in os.listdir(p) if f.endswith(".json")]
        if json_files:
            traj_dirs.append(p)
        else:
            # Scan subdirectories
            for sub in sorted(os.listdir(p)):
                subpath = os.path.join(p, sub)
                if os.path.isdir(subpath):
                    sub_jsons = [f for f in os.listdir(subpath) if f.endswith(".json")]
                    if sub_jsons:
                        traj_dirs.append(subpath)
    return traj_dirs


def group_by_experiment(traj_dirs):
    """Group trajectory directories by experiment key, sorted by step within each group.

    Returns: dict mapping experiment_key -> [(step_number, dir_path), ...]
    """
    groups = defaultdict(list)
    for d in traj_dirs:
        dirname = os.path.basename(d)
        key, step = parse_experiment_key(dirname)
        groups[key].append((step, d))
    # Sort each group by step number (None sorts first)
    for key in groups:
        groups[key].sort(key=lambda x: (x[0] is None, x[0] or 0))
    return dict(groups)


def compute_accuracy(traj_dir, run_filter=None):
    """Lightweight accuracy computation. Returns (correct, total, pct, workflow_type) or None on error."""
    try:
        all_data = load_trajectory_files(traj_dir, run_filter=run_filter)
    except Exception as e:
        print(f"  Warning: could not load {traj_dir}: {e}", file=sys.stderr)
        return None
    if not all_data:
        return None
    workflow_type = detect_workflow_type(all_data)
    correct = sum(1 for d in all_data if d["is_correct"])
    total = len(all_data)
    pct = correct / total * 100 if total > 0 else 0.0
    return correct, total, pct, workflow_type


def print_step_comparison(groups, run_filter=None):
    """Print a comparison table showing accuracy across training steps for each experiment."""
    print("=" * 80)
    print("STEP COMPARISON ACROSS TRAINING CHECKPOINTS")
    print("=" * 80)

    if run_filter:
        print(f"  (filtered to: {run_filter})")

    for key in sorted(groups.keys()):
        entries = groups[key]
        print(f"\nExperiment: {key}")
        print(f"  {'Step':>10s}   {'Correct':>7s}   {'Total':>5s}   {'Accuracy':>8s}   {'Workflow'}")

        results = []
        for step, dirpath in entries:
            acc = compute_accuracy(dirpath, run_filter=run_filter)
            if acc is None:
                step_label = f"step{step}" if step is not None else os.path.basename(dirpath)
                print(f"  {step_label:>10s}   {'?':>7s}   {'?':>5s}   {'N/A':>8s}   (no data)")
                continue
            correct, total, pct, wf_type = acc
            step_label = f"step{step}" if step is not None else os.path.basename(dirpath)
            print(f"  {step_label:>10s}   {correct:>7d}   {total:>5d}   {pct:>7.1f}%   {wf_type}")
            results.append((step, pct, step_label))

        # Show trend if multiple results with step numbers
        if len(results) >= 2 and all(r[0] is not None for r in results):
            first_pct = results[0][1]
            last_pct = results[-1][1]
            diff = last_pct - first_pct
            if diff > 0.5:
                trend = f"improving (+{diff:.1f}% from {results[0][2]} to {results[-1][2]})"
            elif diff < -0.5:
                trend = f"regressing ({diff:.1f}% from {results[0][2]} to {results[-1][2]})"
            else:
                trend = f"stable ({diff:+.1f}% from {results[0][2]} to {results[-1][2]})"
            print(f"\n  Trend: {trend}")

    print()


def analyze_single_directory(traj_dir, run_filter=None):
    """Run full analysis on a single trajectory directory."""
    all_data = load_trajectory_files(traj_dir, run_filter=run_filter)
    if not all_data:
        print(f"Error: No JSON files found in {traj_dir}", file=sys.stderr)
        return False

    workflow_type = detect_workflow_type(all_data)
    print_common_analysis(all_data, traj_dir, workflow_type)

    if workflow_type == "single-agent":
        analyze_single_agent(all_data)
    elif workflow_type == "evaluator-optimizer":
        analyze_evaluator_optimizer(all_data)
    elif workflow_type == "voting":
        analyze_voting(all_data)
    elif workflow_type == "orchestrator-workers":
        analyze_orchestrator_workers(all_data)

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified trajectory analysis for multi-agent workflows")
    parser.add_argument("paths", nargs="+", help="Path(s) to trajectory directory or parent directory containing subdirs")
    parser.add_argument("--run-filter", help="Filter files by run (e.g. 'run_0')", default=None)
    parser.add_argument("--compare-steps", action="store_true", help="Show step comparison table only (no full per-dir analysis)")
    args = parser.parse_args()

    # Resolve all paths into trajectory directories
    traj_dirs = resolve_directories(args.paths)
    if not traj_dirs:
        print("Error: No trajectory directories found in the given path(s)", file=sys.stderr)
        sys.exit(1)

    # Single directory, no compare flag -> original behavior
    if len(traj_dirs) == 1 and not args.compare_steps:
        analyze_single_directory(traj_dirs[0], run_filter=args.run_filter)
        return

    # Multiple directories: show comparison table
    groups = group_by_experiment(traj_dirs)
    print_step_comparison(groups, run_filter=args.run_filter)

    # If --compare-steps, stop after the table
    if args.compare_steps:
        return

    # Otherwise, also run full analysis on each directory
    for traj_dir in traj_dirs:
        print(f"\n\n{'#'*80}")
        print(f"# FULL ANALYSIS: {os.path.basename(traj_dir)}")
        print(f"{'#'*80}\n")
        analyze_single_directory(traj_dir, run_filter=args.run_filter)


if __name__ == "__main__":
    main()
