# Trajectory Review: Multi-Agent Workflow Behavior on AIME 2025

**Model:** Qwen3-1.7B
**Dataset:** AIME 2025 (30 problems, 5 runs each, analysis on run_0 only)
**Evaluation directory:** `evaluation_trajectories/`

---

## Accuracy Summary

| Workflow | Checkpoint | Correct | Total | Accuracy |
|----------|-----------|---------|-------|----------|
| Single Agent (baseline) | step 430 | 7 | 30 | 23.3% |
| Evaluator-Optimizer | step 270 | 9 | 30 | **30.0%** |
| Orchestrator-Workers | step 430 | 7 | 30 | 23.3% |
| Voting | step 290 | 5 | 30 | 16.7% |

---

## 1. Single Agent (Baseline)

**Directory:** `qwen3_1.7b-math_single_agent-length5120_step430/`

### Structure
- 1 trajectory per problem: `generator`
- 1 step per trajectory

### Behavior
The generator receives the problem and produces a solution in `\boxed{}` format. Simple, clean, single-turn. Functions as expected.

**Verdict: Working as designed.**

---

## 2. Evaluator-Optimizer Workflow

**Directory:** `evaluator_optimizer-qwen3_1.7b-math_step270/`
**Workflow file:** `examples/math_reasoning/evaluator_optimizer_math_workflow.py`
**Agents:** `generator`, `evaluator`

### Design
Generator solves -> Evaluator checks (`\boxed{Correct}` or `\boxed{Incorrect}`) -> if incorrect, Generator refines with feedback -> loop up to `max_iterations=3`.

### Observed Behavior

**Correct case (eval_0, answer=70):**
- 2 trajectories: `generator` (correct, reward=1.0) + `evaluator` (says Correct, reward=1.0)
- 1 iteration, early termination on success
- Metrics: `generator_acc: 1.0, evaluator_acc: 1.0, total_iterations: 1`

**Incorrect case (eval_10, answer=259):**
- 6 trajectories: 3x (`generator` -> `evaluator`) pairs
- All 3 generator attempts wrong (reward=0.0), all 3 evaluator verdicts "incorrect" (reward=1.0)
- Full loop exhausted: `total_iterations: 3, generator_attempts: 3, evaluator_predictions: 3`

### Conversation History Sharing
Verified that `share_conversation_history=True` works correctly:
- 2nd generator (1st refinement): sees 5 messages (original prompt + 1st attempt + evaluator feedback + refinement prompt + new response)
- 3rd generator (2nd refinement): sees 7 messages (full history of all prior attempts and feedback)

### Note on Role Assignment
The evaluator's feedback is injected with `role=assistant` instead of `role=user`. This means the generator sees the evaluator critique as if it said it itself rather than receiving it from an external critic. This works because the refinement prompt (`role=user`) provides context that "the teacher's feedback [is] above," but it's a somewhat unnatural chat format. Worth monitoring whether `role=user` would give clearer agent separation.

### Reward Structure
- Generator: reward from ground-truth math evaluation (0.0 or 1.0)
- Evaluator: reward=1.0 if verdict matches ground truth correctness, 0.0 otherwise
- Evaluator shows strong calibration (`evaluator_acc: 1.0` on both correct and incorrect samples)

**Verdict: Working as designed. Best accuracy among all workflows (+6.7% over baseline).**

---

## 3. Orchestrator-Workers Workflow

**Directory:** `orchestrator_workers-qwen3_1.7b-math_step430/`
**Workflow file:** `examples/math_reasoning/orchestrator_workers_math_workflow.py`
**Agents:** `orchestrator`, `worker`

### Design
Orchestrator decomposes into subtasks (max 3) -> Workers solve subtasks in parallel -> Orchestrator synthesizes final answer.

### Observed Behavior

**Correct case (eval_0, answer=70):**
- 5 trajectories: `orchestrator` (decompose) + 3x `worker` + `orchestrator` (synthesize)
- Decomposition: "Understanding the Problem", "Converting Numbers", "Solving the Equation"
- All workers independently solved the **entire problem** and each got the right answer
- Synthesis correctly produced `\boxed{70}`
- Metrics: `n_subtasks: 3, n_workers: 3, successful_workers: 3`

**Incorrect case (eval_10, answer=259):**
- Same 5-trajectory structure
- Decomposition: "Understanding the Function", "Finding Intersections", "Summing y-coords"
- Workers produced lengthy responses (~13K chars each)
- Synthesis prompt was ~44K chars, final answer wrong (`\boxed{70}` instead of 259)
- Metrics: `success: 0`

### Key Issue: Decomposition Quality
The model decomposes problems into **sequential reasoning steps** rather than truly independent subproblems:
- "Step 1: Understand" -> "Step 2: Set up equations" -> "Step 3: Solve"
- These are NOT parallelizable - each depends on the previous
- Each worker ends up solving the entire problem from scratch (massive redundancy)
- The synthesis step processes extremely long prompts with redundant information

This explains why accuracy matches the baseline exactly (23.3%) - the workflow adds overhead without real benefit.

### Possible Improvements
- Add few-shot examples of good decompositions in the orchestrator prompt
- For math, consider whether this pattern is inherently a poor fit (most problems are sequential)
- The `share_context_with_workers=True` setting means workers get the original problem, which enables them to solve the whole thing. Consider whether limiting context would force better use of decomposition.

**Verdict: Structurally working as designed, but decomposition quality is poor. No accuracy gain over baseline.**

---

## 4. Voting Workflow

**Directory:** `voting-qwen3_1.7b-math_step290/`
**Workflow file:** `examples/math_reasoning/voting_math_workflow.py`
**Agents:** `generator0`, `generator1`, `generator2`, `aggregator`

### Design
N generators (n_votes=3) solve independently in parallel -> Aggregator reviews all solutions and selects the best via `\boxed{N}` where N is the solution number.

### Observed Behavior

**Correct case (eval_0, answer=70):**
- 4 trajectories: `generator0`, `generator1`, `generator2`, `aggregator`
- All 3 generators correct (reward=1.0 each)
- Aggregator selected Solution 2 (`\boxed{2}`), reward=1.0
- Metrics: `generator_acc: 1.0, aggregator_acc: 1.0, n_votes: 3, any_correct: 1`

**Incorrect case (eval_10, answer=259):**
- 4 trajectories: same structure
- All 3 generators wrong (reward=0.0), producing answers like incomplete, `\boxed{2}`, `\boxed{39}`
- Aggregator selected generator2's answer, output `\boxed{39}` instead of `\boxed{2}` (format confusion)
- Metrics: `generator_acc: 0.0, aggregator_acc: 0.0, any_correct: 0`

### Key Issues

1. **Aggregator format confusion:** On the incorrect case, the aggregator output `\boxed{39}` (the answer value) instead of `\boxed{2}` (the solution number it intended to select). The `parse_aggregator_response` method handles this via fallback digit parsing, but the confusion indicates the aggregator doesn't always follow the selection format.

2. **No generator diversity:** All 3 generators receive the identical prompt. This limits the benefit of multiple attempts - they tend to make similar mistakes.

3. **Lowest accuracy (16.7%):** Worse than baseline. Possible causes:
   - When no generator gets it right, the workflow cannot recover (unlike evaluator-optimizer which can refine)
   - The aggregator may actively select worse answers even when a correct one exists among the candidates

### Possible Improvements
- Introduce diversity mechanisms (different temperature, different prompt phrasing per generator)
- Strengthen the aggregator's format adherence (always select by number, not by answer value)
- Investigate cases where `any_correct=1` but `aggregator_acc=0` to understand selection failures

**Verdict: Structurally working as designed, but underperforming the baseline.**

---

## Cross-Workflow Comparison

### What Works Well
- **Evaluator-Optimizer:** The iterative refinement loop is the most effective pattern. The evaluator is well-calibrated and the conversation history enables meaningful refinement.
- **All workflows:** The trajectory format correctly captures agent names, conversation histories, rewards, and metrics. The multi-agent LoRA training infrastructure appears functional.

### What Needs Improvement
| Issue | Workflow | Impact |
|-------|----------|--------|
| Sequential decomposition (not parallel) | Orchestrator-Workers | No accuracy gain, wasted tokens |
| No generator diversity | Voting | Similar mistakes across all generators |
| Aggregator format confusion | Voting | May select wrong answer |
| Evaluator feedback as `assistant` role | Evaluator-Optimizer | Minor - works but unnatural |

### Token Efficiency
- **Single Agent:** ~1K tokens per problem
- **Evaluator-Optimizer:** ~1K-6K tokens (1-3 iterations)
- **Voting:** ~3K-4K tokens (3 generators + aggregator)
- **Orchestrator-Workers:** ~5K-15K tokens (very expensive due to worker redundancy and long synthesis prompts)

---

## Raw Data Reference

Each trajectory file has the following structure:
```json
{
  "id": "eval_{problem_idx}_run_{run_idx}",
  "task": { "question": "...", "ground_truth": ..., ... },
  "termination_reason": null,
  "is_correct": true/false,
  "trajectories": [
    {
      "uid": "uuid",
      "name": "agent_name",  // e.g., "generator", "evaluator", "orchestrator", "worker", "aggregator"
      "steps": [
        {
          "chat_completions": [ { "role": "...", "content": "..." }, ... ],
          "reward": 0.0/1.0,
          "model_output": { "prompt_length": ..., "completion_length": ..., "finish_reason": "stop" }
        }
      ],
      "reward": 0.0/1.0
    }
  ],
  "metrics": { ... }  // workflow-specific metrics
}
```
