# Training Hang/Waiting Issue Report

**Date**: 2026-02-07
**Job**: `evaluator_optimizer-qwen3_1.7b-deepcoder` (verl RLHF/PPO with FSDP + vLLM hybrid engine)
**Symptom**: GPU utilization drops to zero for increasingly long periods during the "Generating trajectories" phase. The gap grows with each training step until the job appears hung.

## Key Finding

vLLM finishes ALL generation requests long before the trajectory progress bar completes. The remaining time is spent waiting for something outside vLLM (likely code execution or agent loop orchestration).

### Evidence: vLLM idle vs. trajectory completion

| Step | vLLM reaches 0 running / 0 waiting | Trajectories complete | GPU-idle gap |
|------|-------------------------------------|----------------------|-------------|
| Step 1 | 13:38 | ~13:57 | ~20 min |
| Step 2 | 14:29 | ~14:50 | ~21 min |
| Step 3 | 15:32 | ~15:58 | ~26 min |
| Step 4 | **16:41** | **never (killed ~18:20)** | **>99 min** |

Source: vLLM engine log `worker-b4dba5dc...-2204670.out` and tqdm progress bars in `worker-29e374e9...-2194715.err`.

### vLLM engine state at step 4 tail

```
16:41:29  Running: 1, Waiting: 139, KV cache: 5.9%    # last generation request
16:41:39  Running: 98, Waiting: 0                       # final flush
16:41:49  Running: 0, Waiting: 0, KV cache: 0.0%       # vLLM completely done
16:41:59  throughput: 0.0 tokens/s                      # idle, no more log entries
```

After 16:41, vLLM had zero requests for ~99 minutes until CTRL+C at ~18:20. The progress bar was stuck at 511/512.

### Straggler tail in progress bar (step 4, line 35 of .err)

```
96% | 490/512 [54:27]     # still moving
96% | 491/512 [55:32]     # ~1 min per trajectory
98% | 501/512 [1:03:23]   # ~88s per trajectory
99% | 507/512 [1:04:31]
100%| 510/512 [1:09:21]   # ~97s per trajectory
100%| 511/512 [1:10:01]   # last entry before CTRL+C, 512th never completed
```

## Relevant Config

```python
sandbox_fusion.max_concurrent: 64
sandbox_fusion.url: None
trajectory_timeout: None          # no per-trajectory timeout
workflow.timeout: 1000000.0       # ~11.5 days, effectively infinite
workflow.max_iterations: 2        # up to 2 rounds of generate -> execute
agent.max_steps: 20
```

## Suspected Root Cause

After vLLM completes all generation, the remaining trajectories are waiting on something in the agent loop -- most likely code execution in the sandbox, or possibly the orchestration logic between generation rounds. The wait time grows each step (20 min -> 21 min -> 26 min -> 99+ min), suggesting the problem compounds as the model improves or as state accumulates.

**Open question**: Code execution alone should not take this long. Possible explanations to investigate:
- Sandbox processes hanging or deadlocking (especially with `sandbox_fusion.url: None` -- is a local sandbox being used?)
- `max_concurrent: 64` causing queuing, but this alone doesn't explain 99 min for the last few trajectories
- Agent loop orchestration (e.g., `AgentLoopManager`) blocking on something after generation completes
- Resource contention or memory pressure during the code execution phase
- Async callback or future not resolving for the last trajectory
- `timeout occured: alarm went off` messages appear 14 times in the driver log -- possibly related to sandbox timeouts but not killing the trajectory

## Files Referenced

- **Driver output**: `worker-29e374e9...-2194715.out` (rollout completion logs, config dump)
- **Driver stderr**: `worker-29e374e9...-2194715.err` (tqdm progress bars with timing)
- **vLLM engine**: `worker-b4dba5dc...-2204670.out` (engine throughput, request counts, KV cache)
- **Events**: `events/event_RAYLET.log` (driver death at 18:20)

## Suggested Next Steps

1. Add logging/timing inside the agent loop to pinpoint where time is spent after vLLM generation completes (code execution? callback? orchestration?)
2. Set `trajectory_timeout` to a reasonable value (e.g., 300-600s) to prevent infinite waits
3. Investigate the sandbox setup (`sandbox_fusion.url: None`) -- is a local sandbox hanging?
4. Check if the `AgentLoopManager` has a bug where it fails to resolve the last few trajectories
5. Add per-trajectory timing breakdowns (vLLM time vs. code execution time vs. wait time) to wandb logging
