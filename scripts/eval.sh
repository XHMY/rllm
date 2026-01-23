export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Case 1: Evaluate trained checkpoints (last only)
CUDA_VISIBLE_DEVICES=0 \
python -m examples.math_reasoning.evaluate_checkpoints \
    --eval-mode trained_checkpoint \
    --checkpoints-dir checkpoints/rllm-workflow-MARL-v2 \
    --experiment-filter orchestrator_workers-qwen3_0.6b-math$ \
    --last-checkpoint-only \
    --n-parallel 256 \
    --port 8000

CUDA_VISIBLE_DEVICES=1 \
python -m examples.math_reasoning.evaluate_checkpoints \
    --eval-mode trained_checkpoint \
    --checkpoints-dir checkpoints/rllm-workflow-MARL-v2 \
    --experiment-filter evaluator_optimizer-qwen3_0.6b-math$ \
    --last-checkpoint-only \
    --n-parallel 512 \
    --port 8001
    

# Case 2: Base model baseline
python -m examples.math_reasoning.evaluate_checkpoints \
    --eval-mode base_model \
    --base-model Qwen/Qwen3-0.6B \
    --workflow-types single_agent voting evaluator_optimizer orchestrator_workers \
    --port 8000 \
    --n-parallel 512 \
    --data-parallel 2

# Case 3: Single-agent transfer to multi-agent workflows
CUDA_VISIBLE_DEVICES=0 \
python -m examples.math_reasoning.evaluate_checkpoints \
    --eval-mode single_agent_transfer \
    --base-model Qwen/Qwen3-0.6B \
    --single-agent-lora-path checkpoints/rllm-workflow-MARL-v2/qwen3_0.6b-math_single_agent/global_step_180/actor/lora_adapter \
    --workflow-types voting evaluator_optimizer \
    --n-parallel 512 \
    --port 8000

CUDA_VISIBLE_DEVICES=1 \
python -m examples.math_reasoning.evaluate_checkpoints \
    --eval-mode single_agent_transfer \
    --base-model Qwen/Qwen3-0.6B \
    --single-agent-lora-path checkpoints/rllm-workflow-MARL-v2/qwen3_0.6b-math_single_agent/global_step_180/actor/lora_adapter \
    --workflow-types orchestrator_workers \
    --n-parallel 512 \
    --port 8001