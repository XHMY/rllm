#!/bin/bash

# Training script for multi-agent collaborative math problem solving
# This script launches training with 3 agents (generator_initial, evaluator_critique, generator_refinement)
# Each agent has its own LoRA adapter for independent policy learning

# Set environment variables for optimal performance
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# Find the directory where rllm package is located
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# Launch training with Hydra config overrides
python3 -m examples.math_reasoning.train_multi_agent_math \
    multi_agent.enabled=true \
    multi_agent.num_agents=3 \
    multi_agent.agent_roles=['generator_initial','evaluator_critique','generator_refinement'] \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-1.5B-Instruct \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=10 \
    trainer.project_name=multi_agent_math \
    trainer.experiment_name=deepmath_3agents_qwen \
    trainer.logger=['console','wandb'] \
    agent.max_steps=6 \
    agent.trajectory_timeout=300 \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0
