set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO

python3 -m examples.frozenlake.train_frozenlake_evaluator_optimizer \
    data.max_prompt_length=10240 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    trainer.project_name='rllm-workflow-MARL' \
    trainer.experiment_name='qwen3_0.6b-frozenlake-evaluator_optimizer' \
    trainer.n_gpus_per_node=4 \
    trainer.agent_names=['actor','evaluator'] \
    trainer.share_policy=False \
    +rllm.task=frozenlake

pkill -9 -f 'ray::WorkerDict'


# 1.7B
# actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \