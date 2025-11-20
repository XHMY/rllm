# Training Multi-Agent Workflows

This document explains how to train the multi-agent workflows using PPO (Proximal Policy Optimization).

## Overview

All multi-agent workflows can be trained using the `AgentTrainer` class, which provides a unified interface for training workflows with different reward functions and configurations.

## Training Scripts

Each multi-agent example includes a training script:

| Example | Training Script | Dataset | Reward Function |
|---------|----------------|---------|----------------|
| DeepCoder | `examples/multi_agent/deepcoder/train_multi_agent_deepcoder.py` | `deepcoder` | `code_reward_fn` |
| SWE | `examples/multi_agent/swe/train_multi_agent_swe.py` | `R2E_Gym_Subset` (train), `SWE_Bench_Verified` (val) | Test-based |
| Math Tool | `examples/multi_agent/math_tool/train_multi_agent_math_tool.py` | `deepscaler_math` (train), `aime2024` (test) | `math_reward_fn` |
| Search | `examples/multi_agent/search/train_multi_agent_search.py` | `hotpotqa` | `search_reward_fn` |
| DeepResearch | `examples/multi_agent/deepresearch/train_multi_agent_deepresearch.py` | Custom research dataset | Custom |

## Prerequisites

### 1. Prepare Datasets

Before training, ensure datasets are prepared:

**DeepCoder**:
```bash
python examples/deepcoder/prepare_deepcoder_data.py
```

**SWE**:
```bash
python examples/swe/prepare_swe_data.py
```

**Math Tool**:
```bash
python examples/math_tool/prepare_math_data.py
```

**Search**:
```bash
python examples/search/prepare_hotpotqa_data.py
```

### 2. Configure Training

Training uses Hydra for configuration management. The default config is `agent_ppo_trainer` from `rllm.trainer.config`.

You can override any configuration parameter via command line or config files.

## Running Training

### Basic Training

Run training with default configurations:

```bash
# DeepCoder
python examples/multi_agent/deepcoder/train_multi_agent_deepcoder.py

# SWE
python examples/multi_agent/swe/train_multi_agent_swe.py

# Math Tool
python examples/multi_agent/math_tool/train_multi_agent_math_tool.py

# Search
python examples/multi_agent/search/train_multi_agent_search.py
```

### Training with Custom Configurations

Override specific parameters:

```bash
# Change batch size
python examples/multi_agent/deepcoder/train_multi_agent_deepcoder.py \
    data.train_batch_size=16

# Change number of epochs
python examples/multi_agent/deepcoder/train_multi_agent_deepcoder.py \
    trainer.total_epochs=10

# Change learning rate
python examples/multi_agent/deepcoder/train_multi_agent_deepcoder.py \
    actor_rollout_ref.actor.optim.lr=1e-5

# Multiple overrides
python examples/multi_agent/swe/train_multi_agent_swe.py \
    data.train_batch_size=8 \
    trainer.total_epochs=5 \
    actor_rollout_ref.actor.optim.lr=5e-6
```

## Workflow-Specific Configuration

Each workflow has specific arguments that can be configured:

### DeepCoder

```python
workflow_args = {
    "reward_function": code_reward_fn,
    "max_refinement_iterations": 3,  # Number of refiner cycles
}
```

### SWE

```python
workflow_args = {
    "max_refinement_iterations": 3,  # Analyzer→Writer→Validator cycles
}
```

### Math Tool

```python
workflow_args = {
    "reward_function": math_reward_fn,
    "max_refinement_iterations": 3,  # Analyzer→Executor→Verifier cycles
}
```

### Search

```python
workflow_args = {
    "reward_function": search_reward_fn,
    "search_tool": search_tool,
    "max_refinement_iterations": 2,  # Query optimization cycles
}
```

## Multi-Agent Training Features

### Agent-Specific Rewards

Multi-agent workflows allow assigning different rewards to different agents:

```python
# In workflow.run()
generator_trajectory.steps[0].reward = generator_reward
test_runner_trajectory.steps[0].reward = test_runner_reward
refiner_trajectory.steps[0].reward = refiner_reward
```

This enables:
- **Joint training**: All agents trained together
- **Separate training**: Each agent can be trained independently
- **Specialized reward functions**: Different reward signals for different agent roles

### Metrics Tracking

Multi-agent workflows track per-agent metrics:

```python
metrics = {
    "generator_success": 0.85,
    "test_runner_analyses": 3,
    "refiner_attempts": 2,
    "refiner_success_rate": 0.5,
    "total_iterations": 2,
    "final_success": 1
}
```

These metrics help identify:
- Which agent is performing well/poorly
- Where improvements are needed
- Contribution of each agent to final success

## Monitoring Training

### TensorBoard

Training metrics are logged to TensorBoard (if configured):

```bash
tensorboard --logdir=logs
```

### Checkpoints

Model checkpoints are saved periodically. Location depends on Hydra configuration.

## Advanced Training

### Distributed Training

For large-scale training, use Ray for distributed execution:

```python
# Configured via Hydra
ray.cluster_config.num_workers=8
```

### Mixed Precision Training

Enable mixed precision for faster training:

```python
# In config
trainer.use_amp=true
```

### Gradient Accumulation

For larger effective batch sizes:

```python
# In config
trainer.gradient_accumulation_steps=4
```

## Troubleshooting

### Out of Memory

- Reduce `data.train_batch_size`
- Reduce `max_refinement_iterations`
- Enable gradient checkpointing
- Use smaller model

### Slow Training

- Increase `data.train_batch_size`
- Use distributed training
- Enable mixed precision
- Reduce `n_parallel_agents`

### Poor Convergence

- Adjust learning rate
- Modify reward function
- Increase training epochs
- Check dataset quality

## Comparison with Single-Agent Training

| Aspect | Single-Agent | Multi-Agent |
|--------|-------------|-------------|
| **Configuration** | `agent_class`, `env_class` | `workflow_class` |
| **Arguments** | `agent_args`, `env_args` | `workflow_args` |
| **Rewards** | Single reward per episode | Multiple rewards (one per agent) |
| **Metrics** | Episode-level | Agent-level + Episode-level |
| **Complexity** | Simpler | More complex but more interpretable |

## Example Training Session

Complete example training DeepCoder with custom settings:

```bash
# 1. Prepare dataset
python examples/deepcoder/prepare_deepcoder_data.py

# 2. Start training with custom config
python examples/multi_agent/deepcoder/train_multi_agent_deepcoder.py \
    data.train_batch_size=16 \
    trainer.total_epochs=10 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    trainer.save_freq=1000

# 3. Monitor with TensorBoard
tensorboard --logdir=logs

# 4. Evaluate checkpoints
python examples/multi_agent/deepcoder/run_multi_agent_deepcoder.py \
    --checkpoint=logs/checkpoint_5000
```

## References

- Training configuration: `rllm/trainer/config/`
- Agent trainer: `rllm/trainer/agent_trainer.py`
- Original single-agent training examples: `examples/*/train_*.py`
- Multi-agent workflows: `examples/multi_agent/*/multi_agent_*_workflow.py`
