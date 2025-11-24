#!/bin/bash
set -x

# Environment variables for VLLM optimization
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Clean up temporary LoRA directory
rm -rf /tmp/rllm_tmp_lora/

# Run multi-agent DeepCoder training
python3 -m examples.multi_agent.deepcoder.train_multi_agent_deepcoder \
    data.train_batch_size=16 \
    data.max_prompt_length=16384 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24577 \
    trainer.project_name='multi-agent-deepcoder' \
    trainer.experiment_name='qwen2.5_7b_deepcoder_3agents_share_policy' \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=4 \
    trainer.share_policy=True \
    trainer.agent_names=['generator','runner','refiner']

# Clean up Ray workers
pkill -9 -f 'ray::WorkerDict'
