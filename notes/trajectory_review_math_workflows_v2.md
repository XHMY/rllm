# Trajectory Review v2: Multi-Agent Workflow Comparison (s430 Checkpoints)

**Model:** Qwen3-1.7B (s430 training run)
**Dataset:** AIME 2025 (30 problems, single run)
**Trajectory directory:** `trajectory_outputs/`
**Date:** 2026-03-05

This review compares 8 workflow variants across 3 dimensions:
- **Workflow type:** Evaluator-Optimizer, Orchestrator-Workers, Voting
- **Policy sharing:** Multi-policy (independent per agent) vs Share-policy (single shared policy)
- **Reward structure:** Team reward (default) vs Per-agent reward

---

## Accuracy Summary

| Workflow | Variant | Step | Correct | Accuracy |
|----------|---------|------|---------|----------|
| Evaluator-Optimizer | Multi-policy | 300 | 17/30 | **56.7%** |
| Evaluator-Optimizer | Per-agent-reward | 290 | 14/30 | 46.7% |
| Evaluator-Optimizer | Share-policy | 300 | 12/30 | 40.0% |
| Voting | Per-agent-reward | 290 | 14/30 | 46.7% |
| Orchestrator-Workers | Multi-policy | 290 | 12/30 | 40.0% |
| Voting | Share-policy | 290 | 10/30 | 33.3% |
| Orchestrator-Workers | Share-policy | 290 | 10/30 | 33.3% |
| Voting | Multi-policy | 220 | 8/30 | 26.7% |

### Key Rankings
1. **Eval-Opt multi-policy (56.7%)** is the clear winner, 10+ points ahead of all others
2. **Multi-policy > Share-policy** in every workflow comparison (eval-opt: +16.7pp, orch: +6.7pp)
3. **Per-agent reward dramatically helps Voting** (+20pp over multi-policy at step220, though different steps)
4. **Per-agent reward hurts Eval-Opt** (-10pp vs multi-policy, though comparing step290 vs step300)

---

## 1. Evaluator-Optimizer Workflows

### 1.1 Multi-policy (56.7% accuracy, step300) -- BEST OVERALL

**Directory:** `evaluator_optimizer-qwen3_1.7b_s430-math/`

#### Iteration Behavior
- 1 iteration: 22 files (73.3%) -- evaluator terminates early
- 2 iterations: 3 files (10.0%)
- 3 iterations: 5 files (16.7%)

#### Iteration Benefit
- Initially correct (attempt 1): 12/30 (40.0%)
- Recovered via refinement: +5 (16.7%) -- all 1-step recoveries
- Final accuracy: 17/30 (56.7%)
- **Refinement adds +16.7pp over first-attempt accuracy**

Recovery files: eval_3 (gt=6900), eval_7 (gt=14), eval_17 (gt=20), eval_28 (gt=1), eval_29 (gt=561)

#### Evaluator Calibration
Total (generator, evaluator) pairs: 33 (10 unknown/missing verdict)

|  | Generator CORRECT | Generator WRONG |
|--|-------------------|-----------------|
| Evaluator says "correct" | 15 (TP) | 10 (FP) |
| Evaluator says "incorrect" | 0 (FN) | 8 (TN) |

- Precision: 60.0% (15/25)
- Recall: 100% (15/15) -- never rejects a correct answer
- False positive rate: 55.6% (10/18) -- approves wrong answers too often
- Overall evaluator accuracy: 69.7%

#### Critical Issue: False Positives
10 false positives, all at attempt 1, all leading to final wrong answers. The evaluator approves wrong solutions prematurely, preventing the refinement loop from activating. Problems affected: eval_1 (gt=770), eval_2 (gt=24), eval_5 (gt=144), eval_8 (gt=91), eval_12 (gt=13), eval_14 (gt=701), eval_16 (gt=22), eval_19 (gt=45), eval_26 (gt=2032), eval_27 (gt=280).

**If false positives were eliminated (evaluator always caught wrong answers):** The generator could attempt refinement on all 18 initially-wrong problems. Given the 62.5% success rate at attempt 2 (5/8), this could potentially push accuracy well above 60%.

---

### 1.2 Per-Agent Reward (46.7% accuracy, step290)

**Directory:** `evaluator_optimizer-qwen3_1.7b_s430-math-per_agent_reward/`

#### Iteration Behavior
- 1 iteration: 12 files (40.0%)
- 2 iterations: 1 file (3.3%)
- 3 iterations: 17 files (56.7%) -- much more aggressive iteration vs multi-policy

#### Iteration Benefit
- Initially correct: 11/30 (36.7%)
- Recovered: +3 (10.0%)
- Final accuracy: 14/30 (46.7%)

#### Evaluator Calibration -- VERY DIFFERENT from multi-policy

|  | Generator CORRECT | Generator WRONG |
|--|-------------------|-----------------|
| Evaluator says "correct" | 9 (TP) | 4 (FP) |
| Evaluator says "incorrect" | 13 (FN) | 39 (TN) |

- Precision: 69.2% (9/13) -- better than multi-policy (60%)
- Recall: 40.9% (9/22) -- much worse, rejects many correct answers
- FP rate: 9.3% (4/43) -- dramatically lower than multi-policy (55.6%)
- FN rate: 59.1% (13/22) -- the evaluator is now over-conservative
- Overall evaluator accuracy: 73.8%

#### Key Insight: Per-Agent Reward Makes the Evaluator Conservative
Per-agent reward incentivizes the evaluator to maximize its own accuracy. The result: the evaluator becomes cautious -- it says "incorrect" more often to avoid being wrong about approving bad answers. This reduces false positives (good) but creates excessive false negatives (bad). The generator correct answers get unnecessarily pushed into refinement, wasting iterations and sometimes degrading the answer.

**13 false negatives** where the evaluator rejected correct generator answers:
- eval_5 (gt=144): Correctly solved 3 times, evaluator rejected all 3
- eval_19 (gt=45): Same pattern -- correct every time, rejected every time
- eval_17, eval_20, eval_7: Similar patterns

The evaluator is pessimistic but the generator maintains correct answers through refinement in most cases (all 13 FN cases still ended up correct).

#### 4 false positives (much better than multi-policy's 10):
eval_3 (gt=6900), eval_14 (gt=701), eval_24 (gt=57), eval_26 (gt=2032)

---

### 1.3 Share-Policy (40.0% accuracy, step300)

**Directory:** `evaluator_optimizer-qwen3_1.7b_s430-share_policy-math/`

#### Iteration Behavior
- 1 iteration: 22 files (73.3%)
- 2 iterations: 1 file (3.3%)
- 3 iterations: 7 files (23.3%)

#### Iteration Benefit
- Initially correct: 12/30 (40.0%)
- Recovered: 0/30 (0.0%) -- **NO RECOVERY AT ALL**
- Final accuracy: 12/30 (40.0%)
- **The refinement loop is completely broken under shared policy**

#### Evaluator Calibration

|  | Generator CORRECT | Generator WRONG |
|--|-------------------|-----------------|
| Evaluator says "correct" | 12 (TP) | 11 (FP) |
| Evaluator says "incorrect" | 0 (FN) | 22 (TN) |

- Precision: 52.2% (12/23) -- worst of all three variants
- Recall: 100% -- never rejects correct answers (like multi-policy)
- FP rate: 33.3% (11/33)
- Overall: 75.6%

#### Critical Issue: Zero Recovery
Despite entering the refinement loop 8 times (8 multi-iteration files), the generator NEVER corrects its answer. The shared policy appears to prevent the kind of specialization needed for the generator to improve its response based on evaluator feedback.

**11 false positives** leading to premature termination with wrong answers, even more than multi-policy (10).

#### Why Share-Policy Fails for Eval-Opt
The evaluator-optimizer workflow fundamentally relies on role differentiation:
- The evaluator must learn to critique (different skill from generating)
- The generator must learn to refine based on feedback (different from first-attempt generation)
When both roles share a policy, neither specialization emerges effectively.

---

### Evaluator-Optimizer Comparison Summary

| Metric | Multi-policy | Per-agent-reward | Share-policy |
|--------|-------------|------------------|-------------|
| Accuracy | **56.7%** | 46.7% | 40.0% |
| 1st-attempt accuracy | 40.0% | 36.7% | 40.0% |
| Recovery rate | **+16.7pp** | +10.0pp | +0.0pp |
| Evaluator FP count | 10 | **4** | 11 |
| Evaluator FN count | **0** | 13 | **0** |
| Evaluator precision | 60.0% | **69.2%** | 52.2% |
| Evaluator recall | **100%** | 40.9% | **100%** |
| Multi-iteration files | 8 | 18 | 8 |

**Best variant:** Multi-policy -- best accuracy due to highest recovery rate. The evaluator has more false positives than per-agent-reward, but the generator's ability to actually recover from feedback more than compensates.

---

## 2. Orchestrator-Workers Workflows

> **Metric correction:** Previous analysis reported "100% worker success rate" as a key finding. This metric is **meaningless** — it only measures whether workers executed without crashing, NOT whether their responses are correct. There is no ground truth for individual subtasks. Furthermore, worker `reward` values are overwritten with the final outcome reward (`use_final_outcome_reward=True`), so a worker reward of 1.0 just means the final synthesis was correct, not that the worker's subtask answer was correct. The analysis below replaces these bogus metrics with qualitative examination of actual worker behavior.

### 2.1 Multi-policy (40.0% accuracy, step290)

**Directory:** `orchestrator_workers-qwen3_1.7b_s430-math/`

#### Structure
Every problem decomposes into exactly 3 subtasks (100%). 5 trajectories per problem: orchestrator (decompose) + 3 workers + orchestrator (synthesize).

#### Decomposition Quality: Sequential, Not Independent

The orchestrator consistently produces **sequential decompositions** (step 1 → step 2 → step 3) rather than truly independent subproblems. Typical pattern:
- Subtask 1: "Identify/set up the problem constraints"
- Subtask 2: "Calculate the key quantity"
- Subtask 3: "Compute the final answer"

These are not parallelizable — subtask 3 logically depends on subtask 2, which depends on subtask 1. Examples:
- eval_0 (gt=100): (1) Find M using angle bisector theorem → (2) Calculate AM and CM → (3) Compute AM/CM and multiply by 100
- eval_1 (gt=770): (1) Identify periodic ant behavior → (2) Calculate number of enclosed squares → (3) Determine total area
- eval_4 (gt=961): (1) Identify geometric constraints → (2) Determine area of union → (3) Compute final fraction

#### Workers Solve the Entire Problem, Not Their Subtask

Workers receive the **full original problem** in their prompt along with their assigned subtask. Instead of solving only their part, workers routinely solve the entire problem end-to-end. Evidence:
- eval_0 (correct): All 3 workers produced the same final answer (100). Worker 1 was assigned "find point M" but computed AM/CM=1 and reported 100×1=100.
- eval_13 (correct): All 3 workers answered 6, regardless of whether their subtask was "find cube volume," "determine how many balls fit," or "account for reshaping."
- Of 90 total worker responses across 30 files, 39 (43%) contain the ground truth answer in the response text — workers who "should" only be solving a partial step frequently arrive at the full answer.

This means the decompose-solve-synthesize structure effectively functions as a **3-attempt majority-vote system** rather than genuine task decomposition.

#### Root Cause of Failures: Workers Get Wrong Answers

The dominant failure mode is NOT synthesis quality — it's workers producing wrong answers:

| Failure Mode | Count | Examples |
|-------------|-------|---------|
| All workers wrong (problem too hard) | 17/18 | eval_1 (gt=770, all workers said 10), eval_14 (gt=701, all said 101), eval_8 (gt=91, workers said 111) |
| Synthesis picked wrong worker (one had correct answer) | 1/18 | eval_16 (gt=22, workers=[22, 37, 37], synth=37) |

In the one synthesis failure (eval_16), worker 1 had the correct answer (22) but workers 2 and 3 agreed on 37, and the synthesis step followed the majority — mimicking a naive majority vote and failing.

For the 17 "all workers wrong" cases, the problem was simply too difficult for the model. No amount of decomposition or synthesis improvement could help — the model cannot solve these problems in any framing.

#### Why "Worker Success Rate" Was Misleading

The previously reported "100% worker success, but only 40% accuracy" framing implied synthesis was the bottleneck. In reality:
- Workers produce responses 100% of the time (trivially true — they always generate text)
- Workers produce **correct** final answers in only ~43% of responses
- When workers do produce correct answers, synthesis almost always passes them through correctly (11/12 correct cases had workers with correct answers → correct synthesis)
- The one exception (eval_16) is the ONLY genuine synthesis failure

---

### 2.2 Share-Policy (33.3% accuracy, step290)

**Directory:** `orchestrator_workers-qwen3_1.7b_s430-share_policy-math/`

#### Structure: More Varied Decomposition
Unlike multi-policy's rigid 3-subtask pattern, share-policy produces varied decomposition:
- 1 subtask: 2 files (6.7%)
- 2 subtasks: 9 files (30.0%)
- 3 subtasks: 19 files (63.3%)

Average 2.6 subtasks.

#### Critical Behavior: Orchestrator Pre-Solves During Decomposition

Under shared policy, the orchestrator and workers share the same model weights. The orchestrator therefore behaves like a solver: it computes the answer while writing subtask descriptions, embedding the complete solution into the decomposition itself. Workers then receive subtask descriptions containing pre-computed answers and simply echo them back in ~15 tokens.

Evidence from detailed trajectory analysis:
- **eval_13** (gt=6, correct): Orchestrator decomposition (464 tokens) already computed 81/(4π) ≈ 6.47, floored to 6, and wrote `\boxed{6}` inside the subtask description. Worker 3 responded with only 13 tokens echoing the answer.
- **eval_18** (gt=697, correct): Orchestrator decomposition (765 tokens) solved x = -656, n = 697 completely. All three workers responded with just 15 tokens each: `\boxed{697}`.
- **Contrast with multi-policy**: For the same problems, multi-policy orchestrators produce 73-129 token decompositions (bare subtask titles only), and workers do substantive computation (1000-4600 tokens each).

This means **share-policy orchestrator-workers collapses to single-agent behavior**: the orchestrator solves the problem in its decomposition turn, and workers/synthesis are mechanical confirmation steps. The "multi-agent" framework provides no actual parallelism or collaboration.

#### Decomposition Quality: Blurred Boundaries

Share-policy decomposition is qualitatively different from multi-policy:
- Subtask descriptions contain partial or complete solutions embedded in them
- eval_14 (gt=701): Only 1 subtask — the orchestrator gave up on decomposition entirely
- eval_6 (gt=977): Only 1 subtask with the problem restated verbatim
- When 2 subtasks are used, they tend to be "analyze the problem" + "compute the answer" — a weaker split than multi-policy's 3-step pattern

#### Token Budget Exhaustion as Failure Mechanism

A failure mode specific to share-policy: the orchestrator spends its token budget solving the problem during decomposition instead of delegating. On hard problems, both orchestrator and workers hit the 5120-token limit and are truncated:
- **eval_7** (gt=14): All 3 workers AND synthesis truncated at 5120 tokens — no complete answer produced
- **eval_5** (gt=144): Orchestrator AND all 3 workers truncated
- **eval_12** (gt=13): Orchestrator truncated during exhaustive search; workers received malformed subtask descriptions

#### Root Cause of Failures

| Failure Mode | Count | Examples |
|-------------|-------|---------|
| All workers wrong | 17/20 | eval_1 (gt=770, workers said 10), eval_4 (gt=961, workers said 81), eval_8 (gt=91, workers said 111) |
| Synthesis picked wrong worker | 3/20 | eval_16 (gt=22, [22,37]→37), eval_17 (gt=20, [20,25]→25), eval_22 (gt=2027025, [2027025,126]→126) |

Share-policy has **3x more synthesis failures** than multi-policy (3 vs 1). Key examples:
- **eval_22** (gt=2027025): Worker 1 correctly computed 2027025, Worker 2 got 126. Synthesis chose 126 — the dramatically wrong answer. This suggests the synthesis step under shared policy lacks the evaluation skill to distinguish correct from incorrect worker outputs.
- **eval_17** (gt=20): Worker 1 had 20 (correct), Worker 2 had 25. Synthesis chose 25.
- **eval_16** (gt=22): Same pattern as multi-policy — majority/wrong answer wins.

#### Recurring Error: Intermediate-vs-Final Confusion

A pattern across multiple incorrect share-policy cases — synthesis (or workers) report a value that appears in the problem statement or is an intermediate result, rather than the actual answer:
- **eval_1** (gt=770): Workers returned 10 — the number of squares stated in the problem, not their total area (which is what the question asks)
- **eval_27** (gt=280): Workers found one valid N value (81) rather than computing the sum of all valid N values
- **eval_20** (gt=75): Synthesis reported 45° — the given angle (ABC), not the target angle (ACB) that the problem asks for

#### Share-Policy vs Multi-Policy Behavioral Differences

The shared policy produces fundamentally different workflow dynamics:
- **Orchestrator pre-solves** — share-policy orchestrator embeds complete solutions in decomposition (464-4393 tokens); multi-policy orchestrator produces bare subtask titles (73-129 tokens)
- **Workers are vestigial under share-policy** — 15-token echo responses vs 1000-4600 token genuine computation under multi-policy
- **Fewer subtasks on average** (2.6 vs 3.0) — less rigid, but also less structured
- **More synthesis failures** (3 vs 1) — the shared policy makes the orchestrator worse at evaluating conflicting worker outputs
- **Token budget concentrated in orchestrator** — under share-policy, the orchestrator spends the token budget solving; under multi-policy, computation is distributed across workers. This makes share-policy more vulnerable to token truncation on hard problems

---

### Orchestrator-Workers Comparison

| Metric | Multi-policy | Share-policy |
|--------|-------------|-------------|
| Accuracy | **40.0%** | 33.3% |
| Avg subtasks | 3.0 (rigid) | 2.6 (varied) |
| Orchestrator decomposition tokens | 73-129 (bare titles) | 464-4393 (pre-solves) |
| Worker response tokens | 1000-4600 (genuine computation) | 15-3789 (often vestigial) |
| Failures: all workers wrong | 17/18 (94%) | 17/20 (85%) |
| Failures: synthesis chose wrong worker | 1/18 (6%) | 3/20 (15%) |
| Effective behavior | 3-attempt majority vote + synthesis | Single-agent with echo confirmation |

**Core problems across both variants:**

1. **Decomposition is cosmetic, not functional.** Subtasks are sequential and interdependent. Workers receive the full problem and overwhelmingly solve (or fail) the entire problem regardless of their assigned subtask. The workflow effectively functions as a multi-attempt system with an extra synthesis overhead.

2. **The real bottleneck is model capability, not synthesis.** In 34/38 incorrect cases across both variants, no worker produced the correct answer. The decomposition pattern cannot create correct answers from a model that cannot solve the problem.

3. **When workers disagree, synthesis frequently picks wrong.** In the 4 cases where at least one worker had the right answer, synthesis only succeeded once (25%). This is worse than random selection and suggests the synthesis step has no genuine evaluation ability.

4. **5x computational cost for no benefit.** The workflow makes 5 model calls (decompose + 3 workers + synthesize) versus 1 for single-agent, without proportional accuracy gain. Since workers solve the whole problem anyway, this is an expensive way to get 3 attempts with a bad selection mechanism.

---

## 3. Voting Workflows

### 3.1 Multi-policy (26.7% accuracy, step220)

**Directory:** `voting-qwen3_1.7b_s430-math/`

NOTE: This checkpoint is at step220 (earlier than others at step290), which may partly explain lower accuracy.

#### Generator Agreement
- All correct: 3 (10.0%)
- All wrong: 19 (63.3%)
- Disagree: 8 (26.7%)

#### Per-Generator Accuracy
- generator0: 26.7%, generator1: 23.3%, generator2: 20.0%
- Oracle (any correct): 36.7%

#### Aggregator Selection Quality
- Final accuracy: 26.7% (8/30)
- Missed correct answers (selection failures): 3 cases (eval_13, eval_16, eval_17)
- Correct selections from disagreement: 5/8 (62.5%)
- **Recovery rate on contested cases: 62.5%**

The 3 selection failures all involve minority-correct scenarios (1/3 generators correct). The aggregator is biased toward majority consensus.

---

### 3.2 Per-Agent Reward (46.7% accuracy, step290) -- BEST VOTING VARIANT

**Directory:** `voting-qwen3_1.7b_s430-math-per_agent_reward/`

#### Generator Agreement
- All correct: 7 (23.3%)
- All wrong: 16 (53.3%)
- Disagree: 7 (23.3%)

#### Per-Generator Accuracy
- generator0: 30.0%, generator1: **40.0%**, generator2: 36.7%
- Oracle: 46.7%

#### Aggregator Selection Quality -- PERFECT
- Final accuracy: 46.7% (14/30)
- **Selection failures: 0** -- the aggregator NEVER missed a correct answer
- Correct selections from disagreement: 7/7 (100%)
- **Recovery rate on contested cases: 100%**

This is the standout finding: per-agent reward produces an aggregator that perfectly identifies and selects correct answers whenever at least one generator produces one. The accuracy ceiling is entirely determined by generator quality (oracle = final accuracy).

#### Why Per-Agent Reward Works So Well for Voting
Each agent is rewarded based on its own correctness:
- Generators get reward for producing correct answers (incentivizes diverse, high-quality solutions)
- Aggregator gets reward for selecting the correct final answer (incentivizes evaluation skill)
This separation of incentives produces genuine specialization, unlike team reward which can create free-rider effects.

---

### 3.3 Share-Policy (33.3% accuracy, step290)

**Directory:** `voting-qwen3_1.7b_s430-share_policy-math/`

#### Generator Agreement
- All correct: 8 (26.7%)
- All wrong: 15 (50.0%)
- Disagree: 7 (23.3%)

#### Per-Generator Accuracy
- generator0: 30.0%, generator1: **43.3%**, generator2: 33.3%
- Oracle: **50.0%** -- highest oracle among all voting variants

#### Aggregator Selection Quality -- POOR
- Final accuracy: 33.3% (10/30)
- **Selection failures: 5** (eval_2, eval_3, eval_16, eval_19, eval_28)
- Correct selections from disagreement: 2/7 (28.6%)
- **Recovery rate on contested cases: 28.6%** -- worst of all voting variants

#### Paradox: Best Generators, Worst Aggregator
Share-policy produces the best individual generators (oracle=50%) but the worst aggregator (only 28.6% contested recovery). The shared policy means the aggregator can't develop distinct evaluation skills. It defaults to majority-vote behavior, which fails when the minority answer is correct.

The 5 selection failures all involve cases where generator1 alone was correct but generators 0 and 2 were wrong -- the aggregator picks from the majority.

---

### Voting Comparison

| Metric | Multi-policy (s220) | Per-agent-reward (s290) | Share-policy (s290) |
|--------|-------------------|----------------------|-------------------|
| Accuracy | 26.7% | **46.7%** | 33.3% |
| Oracle (any correct) | 36.7% | 46.7% | **50.0%** |
| Selection failures | 3 | **0** | 5 |
| Contested recovery | 62.5% | **100%** | 28.6% |
| Generator diversity | 26.7% disagree | 23.3% disagree | 23.3% disagree |

**Best variant:** Per-agent-reward -- perfect aggregator selection makes full use of generator capability. Share-policy has the best generators but wastes them with a poor aggregator.

---

## Cross-Workflow Analysis

### Multi-policy vs Share-policy (Consistent Finding)

| Workflow | Multi-policy | Share-policy | Delta |
|----------|-------------|-------------|-------|
| Evaluator-Optimizer | **56.7%** | 40.0% | +16.7pp |
| Orchestrator-Workers | **40.0%** | 33.3% | +6.7pp |
| Voting | 26.7%* | 33.3% | -6.7pp* |

*Voting multi-policy is at step220 vs step290, so not directly comparable.

**Conclusion:** Multi-policy training enables agent specialization that shared policy cannot achieve. The benefit is largest for eval-opt, where role differentiation (generator vs evaluator) is most critical.

### Per-Agent Reward Impact

| Workflow | Team Reward | Per-Agent | Delta |
|----------|-----------|-----------|-------|
| Evaluator-Optimizer | 56.7% (s300) | 46.7% (s290) | -10.0pp* |
| Voting | 26.7% (s220) | **46.7%** (s290) | +20.0pp* |

*Different steps, not strictly comparable.

Per-agent reward dramatically improves voting (perfect aggregator selection) but appears to hurt eval-opt (over-conservative evaluator). The mechanism:
- **Voting:** Per-agent reward aligns incentives correctly -- each agent maximizes its own contribution
- **Eval-opt:** Per-agent reward creates an over-cautious evaluator that rejects correct answers (59.1% FN rate) to avoid being wrong

### Workflow-Specific Bottlenecks

| Workflow | Bottleneck | Evidence |
|----------|-----------|----------|
| Evaluator-Optimizer | Evaluator false positives | 10 FP cases stop refinement prematurely |
| Orchestrator-Workers | Model capability (workers can't solve hard problems) + poor synthesis selection | 94% of failures are all-workers-wrong; when workers disagree, synthesis picks wrong 75% of the time |
| Voting | Generator coverage | Accuracy = oracle when aggregator is good (per-agent reward) |

### Token Efficiency

| Workflow | Avg Agent Calls | Avg Response Chars | Relative Cost |
|----------|----------------|-------------------|---------------|
| Evaluator-Optimizer | 2-6 (1-3 iterations) | ~13-17K per agent | Medium |
| Orchestrator-Workers | 5 (fixed) | ~12K workers + synth | High |
| Voting | 4 (3 gen + 1 agg) | ~9-11K generators | Medium |

Orchestrator-Workers is the least efficient: highest computational cost with no accuracy advantage. Workers solve the whole problem regardless of subtask assignment, making the decomposition overhead pure waste.

---

## Recommendations

### 1. Continue Investing in Evaluator-Optimizer Multi-policy
At 56.7%, it leads all variants by a wide margin. Focus on:
- **Reducing evaluator false positives:** Better negative examples in evaluator prompt, or ensemble evaluation
- **Potential ceiling:** If FP were eliminated, recovery could push accuracy above 65-70%

### 2. Voting Per-Agent Reward is Promising
Perfect aggregator selection at 46.7% means the only ceiling is generator quality. To improve:
- **Increase generator diversity:** Different temperatures, prompt variations, or more voters
- **Scale n_votes:** With perfect selection, adding more generators directly increases oracle probability

### 3. Deprioritize Orchestrator-Workers for Math
The workflow is structurally mismatched with mathematical reasoning: decomposition is cosmetic (workers solve the whole problem anyway), the real bottleneck is model capability (not synthesis), and when workers disagree the synthesis step picks the wrong answer 75% of the time. Consider:
- Only using it for genuinely decomposable tasks (multi-part questions with independent calculations)
- If kept, improving synthesis to evaluate conflicting worker answers rather than defaulting to majority/last answer

### 4. Avoid Share-Policy for Role-Differentiated Workflows
Share-policy consistently underperforms, especially for eval-opt (zero recovery) and voting (poor aggregator). The only scenario where it might work is when agents play identical roles (not the case here).

### 5. Per-Agent Reward Should Be Workflow-Specific
- Use per-agent reward for voting (proven benefit: perfect aggregator)
- Avoid per-agent reward for eval-opt (creates over-conservative evaluator)
- Not tested for orchestrator-workers

---

## Appendix: Raw Script Outputs

All analysis performed using `scripts/analyze_trajectories.py` on each `trajectory_outputs/<workflow>/` directory. Key metrics reproduced in sections above. For full script output, run:

```bash
python3 scripts/analyze_trajectories.py trajectory_outputs/<workflow_dir>/
```

### Problem-Level Correctness Overlap

Problems correct across ALL 8 variants (if any) vs problems wrong across all would indicate problem difficulty distribution. This analysis could be extended with per-problem breakdowns.
