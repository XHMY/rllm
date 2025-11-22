set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

python3 -m examples.solver_judge.train_solver_judge_flow \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.lora_rank=128 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=['q_proj','k_proj','v_proj'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.rollout.n=4 \
    trainer.project_name='solver-judge-workflow' \
    trainer.experiment_name='qwen3_4b' \
    trainer.n_gpus_per_node=2 \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    +trainer.agent_names=['solver','judge']

pkill -9 -f 'ray::WorkerDict' 