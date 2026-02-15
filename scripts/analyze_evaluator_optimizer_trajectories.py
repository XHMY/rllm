#!/usr/bin/env python3
"""
Analyze evaluator-optimizer trajectory files for iteration benefit,
per-iteration accuracy, evaluator calibration, recovery analysis,
and evaluator failure modes.
"""

import json
import os
from collections import defaultdict

TRAJ_DIR = "/nfs/hpc/share/zengyif/workspace/rllm_0.2.1/evaluation_trajectories/evaluator_optimizer-qwen3_1.7b-math_step270/"


def parse_trajectory_file(filepath):
    """Parse a single trajectory file and extract structured info."""
    with open(filepath) as f:
        data = json.load(f)

    result = {
        "filename": os.path.basename(filepath),
        "is_correct": data["is_correct"],
        "ground_truth": data["task"]["ground_truth"],
        "total_iterations": data["metrics"]["total_iterations"],
        "generator_attempts": data["metrics"]["generator_attempts"],
        "generator_steps": [],
        "evaluator_steps": [],
    }

    for traj in data["trajectories"]:
        if traj["name"] == "generator":
            step = traj["steps"][0]
            result["generator_steps"].append({
                "reward": step["reward"],
            })
        elif traj["name"] == "evaluator":
            step = traj["steps"][0]
            action = step["action"]
            verdict = None
            if isinstance(action, dict):
                verdict = action.get("verdict", None)
            result["evaluator_steps"].append({
                "verdict": verdict,
                "reward": step["reward"],
            })

    return result


def main():
    # Parse all files
    files = sorted(os.listdir(TRAJ_DIR))
    all_data = []
    for fname in files:
        if not fname.endswith(".json"):
            continue
        filepath = os.path.join(TRAJ_DIR, fname)
        all_data.append(parse_trajectory_file(filepath))

    total_files = len(all_data)
    print("=" * 80)
    print(f"EVALUATOR-OPTIMIZER TRAJECTORY ANALYSIS")
    print(f"Directory: {TRAJ_DIR}")
    print(f"Total files analyzed: {total_files}")
    print("=" * 80)

    # =========================================================================
    # 1. ITERATION BENEFIT ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. ITERATION BENEFIT ANALYSIS")
    print("=" * 80)

    initially_wrong_eventually_correct = []
    initially_correct = []
    stayed_wrong_all_iterations = []
    initially_wrong_still_wrong = []

    for d in all_data:
        gen_steps = d["generator_steps"]
        first_correct = gen_steps[0]["reward"] == 1.0

        if first_correct:
            initially_correct.append(d)
        else:
            if d["is_correct"]:
                initially_wrong_eventually_correct.append(d)
            else:
                initially_wrong_still_wrong.append(d)
                all_wrong = all(s["reward"] == 0.0 for s in gen_steps)
                if all_wrong:
                    stayed_wrong_all_iterations.append(d)

    print(f"\nTotal problems: {total_files}")
    print(f"  Initially correct (attempt 1 correct):           {len(initially_correct):4d} ({len(initially_correct)/total_files*100:.1f}%)")
    print(f"  Initially wrong, eventually correct (recovery):  {len(initially_wrong_eventually_correct):4d} ({len(initially_wrong_eventually_correct)/total_files*100:.1f}%)")
    print(f"  Initially wrong, still wrong (no recovery):      {len(initially_wrong_still_wrong):4d} ({len(initially_wrong_still_wrong)/total_files*100:.1f}%)")
    print(f"    - Of which ALL attempts wrong:                 {len(stayed_wrong_all_iterations):4d}")

    initially_correct_but_final_wrong = [d for d in initially_correct if not d["is_correct"]]
    print(f"\n  Initially correct but final answer wrong:         {len(initially_correct_but_final_wrong):4d}")
    if initially_correct_but_final_wrong:
        print("    WARNING: These cases should not happen if loop stops on correct!")
        for d in initially_correct_but_final_wrong:
            print(f"      {d['filename']}: iterations={d['total_iterations']}")

    # =========================================================================
    # 2. PER-ITERATION ACCURACY
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. PER-ITERATION ACCURACY")
    print("=" * 80)

    max_attempts = max(d["generator_attempts"] for d in all_data)
    print(f"\nMax generator attempts across all files: {max_attempts}")

    for attempt_idx in range(max_attempts):
        files_with_attempt = [d for d in all_data if len(d["generator_steps"]) > attempt_idx]
        correct_at_attempt = [d for d in files_with_attempt if d["generator_steps"][attempt_idx]["reward"] == 1.0]
        n = len(files_with_attempt)
        c = len(correct_at_attempt)
        if n > 0:
            print(f"\n  Attempt {attempt_idx + 1}:")
            print(f"    Files reaching this attempt: {n}")
            print(f"    Correct at this attempt:     {c} ({c/n*100:.1f}%)")

    print(f"\n  Cumulative accuracy (correct by attempt N, across all {total_files} files):")
    for attempt_idx in range(max_attempts):
        correct_by_attempt = 0
        for d in all_data:
            for j in range(min(attempt_idx + 1, len(d["generator_steps"]))):
                if d["generator_steps"][j]["reward"] == 1.0:
                    correct_by_attempt += 1
                    break
        print(f"    By attempt {attempt_idx + 1}: {correct_by_attempt}/{total_files} ({correct_by_attempt/total_files*100:.1f}%)")

    # =========================================================================
    # 3. EVALUATOR CALIBRATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. EVALUATOR CALIBRATION")
    print("=" * 80)

    eval_says_correct_gen_correct = 0   # TP
    eval_says_correct_gen_wrong = 0     # FP
    eval_says_incorrect_gen_correct = 0 # FN
    eval_says_incorrect_gen_wrong = 0   # TN
    eval_unknown_verdict = 0

    for d in all_data:
        gen_steps = d["generator_steps"]
        eval_steps = d["evaluator_steps"]
        n_pairs = min(len(gen_steps), len(eval_steps))

        for i in range(n_pairs):
            gen_correct = gen_steps[i]["reward"] == 1.0
            verdict = eval_steps[i]["verdict"]

            if verdict is None:
                eval_unknown_verdict += 1
                continue

            verdict_lower = verdict.strip().lower()

            if verdict_lower == "correct":
                if gen_correct:
                    eval_says_correct_gen_correct += 1
                else:
                    eval_says_correct_gen_wrong += 1
            elif verdict_lower == "incorrect":
                if gen_correct:
                    eval_says_incorrect_gen_correct += 1
                else:
                    eval_says_incorrect_gen_wrong += 1
            else:
                eval_unknown_verdict += 1

    total_eval_pairs = eval_says_correct_gen_correct + eval_says_correct_gen_wrong + \
                       eval_says_incorrect_gen_correct + eval_says_incorrect_gen_wrong

    print(f"\n  Total (generator, evaluator) pairs analyzed: {total_eval_pairs}")
    if eval_unknown_verdict > 0:
        print(f"  Pairs with unknown/missing verdict: {eval_unknown_verdict}")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':30s} Generator CORRECT  Generator WRONG")
    print(f"  Evaluator says 'correct':    {eval_says_correct_gen_correct:10d}         {eval_says_correct_gen_wrong:10d}")
    print(f"  Evaluator says 'incorrect':  {eval_says_incorrect_gen_correct:10d}         {eval_says_incorrect_gen_wrong:10d}")

    eval_says_correct_total = eval_says_correct_gen_correct + eval_says_correct_gen_wrong
    eval_says_incorrect_total = eval_says_incorrect_gen_correct + eval_says_incorrect_gen_wrong
    gen_actually_correct_total = eval_says_correct_gen_correct + eval_says_incorrect_gen_correct
    gen_actually_wrong_total = eval_says_correct_gen_wrong + eval_says_incorrect_gen_wrong

    print(f"\n  Precision (when evaluator says 'correct', how often is gen actually correct):")
    if eval_says_correct_total > 0:
        prec_pos = eval_says_correct_gen_correct / eval_says_correct_total
        print(f"    {eval_says_correct_gen_correct}/{eval_says_correct_total} = {prec_pos*100:.1f}%")
    else:
        print(f"    N/A (evaluator never said 'correct')")

    print(f"\n  Precision for negative (when evaluator says 'incorrect', how often is gen actually wrong):")
    if eval_says_incorrect_total > 0:
        prec_neg = eval_says_incorrect_gen_wrong / eval_says_incorrect_total
        print(f"    {eval_says_incorrect_gen_wrong}/{eval_says_incorrect_total} = {prec_neg*100:.1f}%")
    else:
        print(f"    N/A (evaluator never said 'incorrect')")

    print(f"\n  Recall (when gen is correct, how often does evaluator say 'correct'):")
    if gen_actually_correct_total > 0:
        rec_pos = eval_says_correct_gen_correct / gen_actually_correct_total
        print(f"    {eval_says_correct_gen_correct}/{gen_actually_correct_total} = {rec_pos*100:.1f}%")
    else:
        print(f"    N/A (gen never correct)")

    print(f"\n  Recall for negative (when gen is wrong, how often does evaluator say 'incorrect'):")
    if gen_actually_wrong_total > 0:
        rec_neg = eval_says_incorrect_gen_wrong / gen_actually_wrong_total
        print(f"    {eval_says_incorrect_gen_wrong}/{gen_actually_wrong_total} = {rec_neg*100:.1f}%")
    else:
        print(f"    N/A (gen never wrong)")

    eval_correct_decisions = eval_says_correct_gen_correct + eval_says_incorrect_gen_wrong
    if total_eval_pairs > 0:
        print(f"\n  Overall evaluator accuracy: {eval_correct_decisions}/{total_eval_pairs} = {eval_correct_decisions/total_eval_pairs*100:.1f}%")

    # =========================================================================
    # 4. RECOVERY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. RECOVERY ANALYSIS")
    print("=" * 80)

    attempt1_correct = [d for d in all_data if d["generator_steps"][0]["reward"] == 1.0]
    print(f"\n  Attempt 1 correct (no refinement needed): {len(attempt1_correct)} ({len(attempt1_correct)/total_files*100:.1f}%)")

    one_step_recovery = []
    for d in all_data:
        gs = d["generator_steps"]
        if len(gs) >= 2 and gs[0]["reward"] == 0.0 and gs[1]["reward"] == 1.0:
            one_step_recovery.append(d)
    print(f"  1-step recovery (attempt 1 wrong, attempt 2 correct): {len(one_step_recovery)} ({len(one_step_recovery)/total_files*100:.1f}%)")

    two_step_recovery = []
    for d in all_data:
        gs = d["generator_steps"]
        if len(gs) >= 3 and gs[0]["reward"] == 0.0 and gs[1]["reward"] == 0.0 and gs[2]["reward"] == 1.0:
            two_step_recovery.append(d)
    print(f"  2-step recovery (attempts 1-2 wrong, attempt 3 correct): {len(two_step_recovery)} ({len(two_step_recovery)/total_files*100:.1f}%)")

    no_recovery = [d for d in all_data if all(s["reward"] == 0.0 for s in d["generator_steps"])]
    print(f"  No recovery (all attempts wrong): {len(no_recovery)} ({len(no_recovery)/total_files*100:.1f}%)")

    total_final_correct = sum(1 for d in all_data if d["is_correct"])
    print(f"\n  Final overall accuracy: {total_final_correct}/{total_files} ({total_final_correct/total_files*100:.1f}%)")
    total_recovery = len(one_step_recovery) + len(two_step_recovery)
    print(f"  Total recovered through refinement: {total_recovery} ({total_recovery/total_files*100:.1f}%)")

    if one_step_recovery:
        print(f"\n  1-step recovery files:")
        for d in one_step_recovery:
            print(f"    {d['filename']} (ground_truth={d['ground_truth']})")

    if two_step_recovery:
        print(f"\n  2-step recovery files:")
        for d in two_step_recovery:
            print(f"    {d['filename']} (ground_truth={d['ground_truth']})")

    # =========================================================================
    # 5. EVALUATOR FAILURE MODES
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. EVALUATOR FAILURE MODES")
    print("=" * 80)

    false_positives = []
    false_negatives = []

    for d in all_data:
        gen_steps = d["generator_steps"]
        eval_steps = d["evaluator_steps"]
        n_pairs = min(len(gen_steps), len(eval_steps))

        for i in range(n_pairs):
            gen_correct = gen_steps[i]["reward"] == 1.0
            verdict = eval_steps[i]["verdict"]
            if verdict is None:
                continue
            verdict_lower = verdict.strip().lower()

            if verdict_lower == "correct" and not gen_correct:
                false_positives.append({
                    "filename": d["filename"],
                    "attempt": i + 1,
                    "ground_truth": d["ground_truth"],
                    "final_correct": d["is_correct"],
                    "total_iterations": d["total_iterations"],
                })
            elif verdict_lower == "incorrect" and gen_correct:
                false_negatives.append({
                    "filename": d["filename"],
                    "attempt": i + 1,
                    "ground_truth": d["ground_truth"],
                    "final_correct": d["is_correct"],
                    "total_iterations": d["total_iterations"],
                })

    print(f"\n  FALSE POSITIVES (evaluator says 'correct' but generator was WRONG): {len(false_positives)}")
    print(f"  These cause the loop to STOP prematurely with a wrong answer.")
    if false_positives:
        for fp in false_positives:
            print(f"    {fp['filename']}, attempt {fp['attempt']}, ground_truth={fp['ground_truth']}, "
                  f"final_correct={fp['final_correct']}, total_iter={fp['total_iterations']}")

    print(f"\n  FALSE NEGATIVES (evaluator says 'incorrect' but generator was CORRECT): {len(false_negatives)}")
    print(f"  These cause unnecessary refinement (wasted iterations).")
    if false_negatives:
        for fn in false_negatives:
            print(f"    {fn['filename']}, attempt {fn['attempt']}, ground_truth={fn['ground_truth']}, "
                  f"final_correct={fn['final_correct']}, total_iter={fn['total_iterations']}")

    print(f"\n  IMPACT ANALYSIS:")
    fp_final_wrong = [fp for fp in false_positives if not fp["final_correct"]]
    print(f"    False positives that led to final wrong answer: {len(fp_final_wrong)}")

    fn_final_correct = [fn for fn in false_negatives if fn["final_correct"]]
    fn_final_wrong = [fn for fn in false_negatives if not fn["final_correct"]]
    print(f"    False negatives where final answer was still correct: {len(fn_final_correct)}")
    print(f"    False negatives where final answer ended up wrong: {len(fn_final_wrong)}")

    # =========================================================================
    # DISTRIBUTION SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("DISTRIBUTION OF ITERATIONS")
    print("=" * 80)
    iter_counts = defaultdict(int)
    for d in all_data:
        iter_counts[d["total_iterations"]] += 1
    for k in sorted(iter_counts):
        print(f"  {k} iteration(s): {iter_counts[k]} files ({iter_counts[k]/total_files*100:.1f}%)")

    attempt_counts = defaultdict(int)
    for d in all_data:
        attempt_counts[d["generator_attempts"]] += 1
    print(f"\n  Generator attempt distribution:")
    for k in sorted(attempt_counts):
        print(f"    {k} attempt(s): {attempt_counts[k]} files ({attempt_counts[k]/total_files*100:.1f}%)")

    # =========================================================================
    # PER-FILE SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("PER-FILE SUMMARY (sorted by filename)")
    print("=" * 80)
    print(f"  {'Filename':<30s} {'Correct':>7s} {'Iters':>5s} {'Attempts':>8s} {'GenRewards':>30s} {'EvalVerdicts':>35s}")
    print("  " + "-" * 120)
    for d in sorted(all_data, key=lambda x: x["filename"]):
        gen_rewards = [f"{s['reward']:.0f}" for s in d["generator_steps"]]
        eval_verdicts = [s["verdict"] if s["verdict"] else "?" for s in d["evaluator_steps"]]
        print(f"  {d['filename']:<30s} {str(d['is_correct']):>7s} {d['total_iterations']:>5d} "
              f"{d['generator_attempts']:>8d} {','.join(gen_rewards):>30s} {','.join(eval_verdicts):>35s}")


if __name__ == "__main__":
    main()
