set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

rm -rf /tmp/rllm_tmp_lora/

python3 -m examples.math_reasoning.train_multi_agent_math \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.lora_rank=256 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules=['q_proj','k_proj','v_proj'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    trainer.project_name='multi-agent-math-reasoning' \
    trainer.experiment_name='qwen3_4b_math_3agents' \
    trainer.n_gpus_per_node=2 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=4 \
    +trainer.agent_names=['generator','evaluator','refiner'] \
    +rllm.workflow.max_refinement_iterations=3

pkill -9 -f 'ray::WorkerDict'
