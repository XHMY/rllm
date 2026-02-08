# Ray Training Failure Investigation Report

## Summary

The training job was an RLLM (Reinforcement Learning from LLM) multi-agent workflow training a Qwen3-1.7B model using PPO with LoRA adapters (generator + evaluator) on the DeepCoder evaluator-optimizer workflow. The training completed all 10 scheduled steps (step 0 through step 9) and shut down intentionally via `ray.shutdown()`. However, there were significant data quality errors and performance degradation throughout the run.

## Fundamental Root Cause: Malformed `tests` field in the dataset

In `rllm/rewards/code_reward.py:455`, the code calls `json.loads(tests)` on the `tests` field of DeepCoder task entries. For **242 tasks**, this field was empty or null, causing:

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

This cascades into:

1. `code_reward_fn` throws an exception
2. The rollout is marked as `TerminationReason.ERROR` (37 total error/failure events)
3. Retries are attempted (up to 3), but fail again since the data itself is bad
4. These tasks get zero rewards, corrupting the PPO training signal

## Secondary Issue: vLLM KV Cache Exhaustion

Both vLLM engines repeatedly hit **99-100% GPU KV cache usage**, causing:

- Waiting queues growing to 218+ pending requests
- Throughput collapsing from ~650 tokens/s down to ~80 tokens/s
- Each training step taking 2600-4260 seconds (~45-70 min)

## Timeline

| Time | Event |
|------|-------|
| 11:59:08 | Ray session started (PID 1782451) |
| ~12:01 | Workers initialized: vLLM engines, WorkerDict actors, TaskRunner |
| ~12:03 | Model loaded (Qwen3-1.7B), LoRA adapters initialized, CUDA graphs captured |
| ~12:03 | Initial validation: pass@1 = 7.28% |
| ~12:03 | Step 0 (initial eval) + Step 1 trajectory generation begins |
| 12:03-22:32 | Training steps 0-9 executed (~1 hour per step) |
| Throughout | Persistent "invalid 'tests' field" JSON errors (242 tasks affected) |
| Throughout | Periodic "timeout occured: alarm went off" during code execution (10 occurrences) |
| Throughout | vLLM KV cache repeatedly hitting 95-100%, causing throughput degradation |
| 22:32:41 | Driver initiates `ray.shutdown()` with `exit_type=INTENDED_USER_EXIT` |

## Key Log Files

| File | What it shows |
|------|--------------|
| `worker-d241a21e...-1787568.out` | 19 tracebacks, 37 error events (main trainer) |
| `worker-d241a21e...-1787568.err` | 242 "invalid 'tests' field" errors |
| `worker-49fb9a00...-1790684.out` | vLLM engine 0: KV cache saturation |
| `worker-d9ecfe7c...-1790682.out` | vLLM engine 1: KV cache saturation |
| `raylet.err`, `gcs_server.err` | Empty (no system-level errors) |

## Recommendations

1. **Fix the data pipeline** -- Add a guard in `code_reward.py` before line 455: `if not tests: return 0.0`, or filter out tasks with empty `tests` fields during data loading.
2. **Address vLLM KV cache pressure** -- Increase `gpu_memory_utilization`, reduce `max_seq_len` (currently 22528), or reduce batch concurrency to prevent throughput collapse.
3. **Don't retry on bad data** -- Rollouts failing due to empty `tests` should not be retried since the data issue persists across retries, wasting compute.
