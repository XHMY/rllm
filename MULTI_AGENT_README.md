# Multi-Agent RL Training with rLLM

## Table of Contents
- [Overview](#overview)
- [Design Philosophy](#design-philosophy)
- [Architecture](#architecture)
- [Data Preparation](#data-preparation)
- [Implementation Guide](#implementation-guide)
- [Configuration](#configuration)
- [Code Examples](#code-examples)
- [Implementation Roadmap](#implementation-roadmap)
- [Testing & Validation](#testing--validation)
- [Quick Reference](#quick-reference)

---

## Overview

This document provides a complete guide for implementing multi-agent RL training in rLLM, where each agent can have its own policy while sharing a single inference engine.

### Key Features

- **Shared Inference Engine**: One vLLM instance serves all agents using LoRA adapters
- **Independent Policies**: Each agent has its own LoRA (~1-2% model size overhead per agent)
- **Sequential Training**: Stable policy updates without gradient interference
- **Minimal Code Changes**: ~1300 lines across 11 files, extending rLLM's existing infrastructure

### Why This Approach?

✅ **Memory Efficient**: LoRA adapters are tiny compared to full models
✅ **Clean Architecture**: Extends AgentTrainer without modifying verl internals
✅ **Leverages Async Design**: Reuses AsyncAgentExecutionEngine for rollouts
✅ **User-Friendly**: Same API as single-agent training

---

## Design Philosophy

### Build on rLLM's Strengths

**Core Principle**: Extend rLLM's async agent infrastructure (AgentTrainer → AgentPPOTrainer → AsyncAgentExecutionEngine) rather than modifying verl directly.

```
User API: AgentTrainer (no changes to user code!)
    ↓
Internal: train_agent.remote() (parse multi_agent config)
    ↓
Training: AgentPPOTrainer (multi-agent training logic)
    ├─> Rollout: AsyncAgentExecutionEngine → Router → vLLM+LoRA
    └─> Training: Sequential LoRA updates
```

### Layer Separation

| Layer | Responsibility | Modifications |
|-------|---------------|---------------|
| **User API** | AgentTrainer | Accept multi_agent_config |
| **Training Logic** | AgentPPOTrainer | Trajectory separation, sequential updates |
| **Rollout** | AsyncAgentExecutionEngine, Router | Pass agent_id, LoRA routing |
| **verl Workers** | ActorRolloutRefWorker | LoRA adapter management |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Code (Unchanged!)                    │
│                                                              │
│  trainer = AgentTrainer(                                     │
│      agent_class=MultiRoleAgent,                             │
│      env_class=MultiAgentEnv,                                │
│      config=config  # includes multi_agent section           │
│  )                                                           │
│  trainer.train()                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Internal: AgentTrainer                       │
│  (rllm/trainer/agent_trainer.py)                            │
│                                                              │
│  - Passes config to train_agent.remote()                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              train_agent() (Ray remote)                      │
│  (rllm/trainer/verl/train_agent_ppo.py)                     │
│                                                              │
│  - Parses multi_agent config from Hydra                      │
│  - Initializes LoRA configurations                           │
│  - Creates AgentPPOTrainer with multi_agent_config           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  AgentPPOTrainer                             │
│  (rllm/trainer/verl/agent_ppo_trainer.py)                   │
│                                                              │
│  Rollout Phase:                                              │
│  ├─> AsyncAgentExecutionEngine.generate_trajectories()      │
│  │    └─> Router.generate_sequences(agent_id=...)           │
│  │         └─> vLLM.generate(lora_request=...)              │
│  │                                                           │
│  Training Phase:                                             │
│  └─> for agent_role in ["agent_0", "agent_1"]:              │
│       ├─> batches = _separate_by_agent_role(batch)          │
│       ├─> actor_worker.set_active_lora(agent_role)          │
│       └─> actor_worker.update_actor(batches[agent_role])    │
└─────────────────────────────────────────────────────────────┘
```

### Rollout Phase: Multi-LoRA Inference

```
MultiAgentEnv          Agent                  Router            vLLM Engine
┌──────────┐         ┌──────────┐         ┌──────────┐      ┌──────────────┐
│          │         │ Agent 0  │         │          │      │ Base Model   │
│ Agent 0's│◄────────┤ role:    │         │          │      │ Qwen3-32B    │
│   Turn   │         │ "nego"   │         │          │      │              │
│          │─────────►│          │         │          │      │ ┌─────────┐  │
│          │         └─────┬────┘         │          │      │ │LoRA "a0"│  │
│          │               │              │          │      │ │ ACTIVE  │  │
│          │        chat_completions      │          │      │ └─────────┘  │
│          │               │              │          │      │              │
│          │               ▼              │          │      │ ┌─────────┐  │
│          │         ┌──────────────┐     │          │      │ │LoRA "a1"│  │
│          │         │ agent_id=    │     │          │      │ │ loaded  │  │
│          │         │   "agent_0"  │────►│ Build    │─────►│ └─────────┘  │
│          │         │              │     │ LoRAReq  │      │              │
│          │         └──────────────┘     │ (a0, 1)  │      │              │
│          │◄─────────────response────────┴──────────┴──────┤              │
└──────────┘                                                 └──────────────┘
```

### Training Phase: Sequential LoRA Updates

```
Step 1: Separate Trajectories
┌─────────────────────────────────────────────────────────────┐
│  Combined Batch                                              │
│  ┌──────────────────────────────────────────────┐           │
│  │ agent_role: ["a0", "a1", "a0", "a1", ...]    │           │
│  │ rewards:    [0.5,  0.8,  0.3,  0.6,  ...]    │           │
│  └──────────────────────────────────────────────┘           │
│                       │                                      │
│                       ▼                                      │
│            _separate_by_agent_role()                         │
│                       │                                      │
│         ┌─────────────┴─────────────┐                        │
│         ▼                           ▼                        │
│  ┌────────────┐              ┌────────────┐                 │
│  │ Agent "a0" │              │ Agent "a1" │                 │
│  │ Batch      │              │ Batch      │                 │
│  └────────────┘              └────────────┘                 │
└─────────────────────────────────────────────────────────────┘

Step 2: Sequential Updates
┌─────────────────────────────────────────────────────────────┐
│  Update Agent "a0":                                          │
│  ┌────────────────────────────────────────────────┐         │
│  │ actor_worker.set_active_lora("a0")              │         │
│  │ actor_worker.update_actor(batch_a0)             │         │
│  │   → Only LoRA "a0" receives gradients           │         │
│  └────────────────────────────────────────────────┘         │
│                                                              │
│  Update Agent "a1":                                          │
│  ┌────────────────────────────────────────────────┐         │
│  │ actor_worker.set_active_lora("a1")              │         │
│  │ actor_worker.update_actor(batch_a1)             │         │
│  │   → Only LoRA "a1" receives gradients           │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Preparation

Multi-agent training follows rLLM's standard data pipeline where datasets are prepared in a separate step and saved to parquet files.

### Preparing the DeepMath Dataset

**Step 1: Run the preprocessing script**

```bash
# Prepare and register the dataset (one-time step)
python examples/math_reasoning/prepare_deepmath_data.py
```

This script will:
1. Download DeepMath-103K from HuggingFace
2. Filter by difficulty range (4-7)
3. Filter for integer-convertible answers
4. Split into train/test sets
5. Save to `rllm/data/datasets/deepmath/` as parquet files
6. Register with DatasetRegistry

**Step 2: Verify the dataset**

```bash
# Check registered datasets
python -c "from rllm.data import DatasetRegistry; print(DatasetRegistry.get_dataset_names())"
# Output: ['frozenlake', 'deepmath', ...]
```

### Dataset Format

The preprocessing script follows the same pattern as `examples/frozenlake/prepare_frozenlake_data.py`:

```python
from datasets import load_dataset
from rllm.data.dataset import DatasetRegistry

# Load and filter data
ds = load_dataset("zwhe99/DeepMath-103K")["train"]
ds = ds.filter(lambda x: 4 <= x["difficulty"] <= 7)
ds = ds.filter(lambda x: is_convertible_to_int(x["final_answer"]))

# Split and register
ds_split = ds.train_test_split(test_size=0.01, seed=42)
train_dataset = DatasetRegistry.register_dataset("deepmath", ds_split["train"], "train")
test_dataset = DatasetRegistry.register_dataset("deepmath", ds_split["test"], "test")
```

The training script then loads from the registry:

```python
# In train_multi_agent_math.py
from rllm.data import DatasetRegistry

train_dataset = DatasetRegistry.load_dataset("deepmath", "train")
val_dataset = DatasetRegistry.load_dataset("deepmath", "test")
```

### Custom Dataset Preparation

To prepare your own multi-agent dataset:

1. **Create a preprocessing script** (e.g., `prepare_my_data.py`):
   ```python
   from rllm.data.dataset import DatasetRegistry

   def prepare_my_dataset():
       # Load your data
       train_data = [{"question": q, "answer": a} for q, a in ...]
       test_data = [...]

       # Register with DatasetRegistry
       DatasetRegistry.register_dataset("mydataset", train_data, "train")
       DatasetRegistry.register_dataset("mydataset", test_data, "test")
   ```

2. **Run once before training**:
   ```bash
   python examples/my_task/prepare_my_data.py
   ```

3. **Load in training script**:
   ```python
   train_dataset = DatasetRegistry.load_dataset("mydataset", "train")
   ```

---

## Implementation Guide

### File Modifications Summary

#### Priority 1: rLLM User API (~110 lines)

**1. rllm/trainer/agent_trainer.py** (~50 lines)
```python
class AgentTrainer:
    def __init__(self, agent_class, env_class, config,
                 train_dataset, val_dataset,
                 agent_args=None, env_args=None):
        # Config can include multi_agent section
        # No other changes needed - just pass through!
        self.config = config
        # ... existing code ...
```

**2. rllm/trainer/verl/train_agent_ppo.py** (~60 lines)
```python
@ray.remote
def train_agent(config, agent_class, env_class, agent_args, env_args):
    # NEW: Parse multi_agent config
    multi_agent_config = {}
    if config.get("multi_agent", {}).get("enabled", False):
        multi_agent_config = OmegaConf.to_container(
            config.multi_agent, resolve=True
        )

    # Pass to AgentPPOTrainer
    trainer = AgentPPOTrainer(
        config=config,
        ...,
        multi_agent_config=multi_agent_config  # NEW
    )
```

#### Priority 2: rLLM Training Logic (~440 lines)

**3. rllm/trainer/verl/agent_ppo_trainer.py** (~400 lines) ⭐ **Main changes**
```python
class AgentPPOTrainer(RayPPOTrainer):
    def __init__(self, config, tokenizer, ..., multi_agent_config=None):
        super().__init__(...)
        self.multi_agent_config = multi_agent_config or {}
        self.is_multi_agent = len(self.multi_agent_config) > 0

    def init_workers(self):
        super().init_workers()
        # Pass LoRA configs to router
        if self.is_multi_agent:
            lora_configs = self.multi_agent_config.get("lora_configs", {})
            self.agent_execution_engine.router.lora_configs = lora_configs

    def fit_agent(self):
        # ... existing training loop ...
        for batch in dataloader:
            # Generate trajectories
            batch_output = self.generate_agent_trajectory(...)

            if self.is_multi_agent:
                # NEW: Multi-agent training
                agent_batches = self._separate_by_agent_role(batch)

                for agent_role in self.multi_agent_config["agent_roles"]:
                    self._update_agent_policy(
                        agent_role,
                        agent_batches[agent_role]
                    )
            else:
                # Existing single-agent update
                self.actor_rollout_wg.update_actor(batch)

    def _separate_by_agent_role(self, batch):
        """Separate trajectories by agent_role"""
        agent_batches = {}
        agent_roles = batch.non_tensor_batch.get("agent_role")

        for role in self.multi_agent_config["agent_roles"]:
            role_mask = agent_roles == role
            agent_batches[role] = batch[role_mask]

        return agent_batches

    def _update_agent_policy(self, agent_role, batch):
        """Update specific agent's LoRA"""
        lora_config = self.multi_agent_config["lora_configs"][agent_role]

        # Set active LoRA
        self.actor_rollout_wg.set_active_lora(agent_role, lora_config)

        # Update (only active LoRA receives gradients)
        self.actor_rollout_wg.update_actor(batch)
```

**4. rllm/trainer/config/ppo_trainer.yaml** (~40 lines)
```yaml
# NEW multi_agent section
multi_agent:
  enabled: false
  num_agents: 2
  agent_roles: ["agent_0", "agent_1"]

  lora_configs:
    agent_0:
      lora_path: null
      lora_int_id: 1
      lora_rank: 64
      lora_alpha: 16
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
      lora_dropout: 0.05

    agent_1:
      lora_path: null
      lora_int_id: 2
      lora_rank: 64
      lora_alpha: 16
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
      lora_dropout: 0.05
```

#### Priority 3: rLLM Inference/Routing (~150 lines)

**5. rllm/engine/agent_execution_engine.py** (~80 lines)
```python
class AsyncAgentExecutionEngine:
    def __init__(self, ..., lora_configs=None):
        # ... existing code ...
        self.lora_configs = lora_configs or {}

    async def run_agent_trajectory_async(self, idx, application_id, ...):
        agent = self.agents[idx]

        # NEW: Extract agent metadata
        agent_id = getattr(agent, 'agent_id', None)

        # ... trajectory loop ...
        response = await self.get_model_response(
            prompt,
            application_id,
            agent_id=agent_id,  # NEW: Pass to router
            **kwargs
        )

    async def _get_verl_async(self, prompt, application_id,
                              agent_id=None, **kwargs):  # NEW param
        # Pass agent_id to router for LoRA selection
        output = await self.router.generate_sequences(
            batch,
            application_id=application_id,
            agent_id=agent_id,  # NEW
            **kwargs
        )
```

**6. rllm/router/router.py** (~50 lines)
```python
class Router:
    def __init__(self, config, tokenizer, addresses, lora_configs=None):
        # ... existing code ...
        self.lora_configs = lora_configs or {}

    async def generate_sequences(self, batch, application_id,
                                  agent_id=None, **kwargs):
        # NEW: Build LoRA request if agent_id provided
        lora_request = None
        if agent_id and agent_id in self.lora_configs:
            lora_config = self.lora_configs[agent_id]
            lora_request = {
                "lora_name": agent_id,
                "lora_int_id": lora_config["lora_int_id"],
                "lora_path": lora_config.get("lora_path")
            }
            kwargs["lora_request"] = lora_request

        # ... existing generation code ...
```

**7. rllm/agents/agent.py** (~20 lines)
```python
@dataclass
class Step:
    observation: Any = None
    model_response: str = None
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)
    step: int = 0
    chat_completions: list = field(default_factory=list)

    # NEW: Multi-agent metadata
    agent_id: str = None
    agent_role: str = None
    lora_adapter: str = None
```

#### Priority 4: verl Worker (~150 lines)

**8. verl/verl/workers/fsdp_workers.py** (~150 lines)
```python
class ActorRolloutRefWorker(Worker):
    def __init__(self, config, ...):
        super().__init__()
        self.current_lora_name = None
        self.loaded_loras = {}

    def set_active_lora(self, lora_name, lora_config):
        """Switch active LoRA for training"""
        # Load if not already loaded
        if lora_name not in self.loaded_loras:
            self._load_lora_adapter(lora_name, lora_config)

        # Set as active
        self.actor_module.set_active_adapters([lora_name])
        self.current_lora_name = lora_name

    def _load_lora_adapter(self, lora_name, lora_config):
        """Load LoRA adapter using PEFT"""
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=lora_config.get("lora_rank", 64),
            lora_alpha=lora_config.get("lora_alpha", 16),
            target_modules=lora_config.get("target_modules"),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
        )

        # Add to model
        self.actor_module.add_adapter(lora_name, peft_config)

        # Load pretrained if provided
        if "lora_path" in lora_config and lora_config["lora_path"]:
            self.actor_module.load_adapter(
                lora_config["lora_path"],
                adapter_name=lora_name
            )

        self.loaded_loras[lora_name] = True

    def update_actor(self, batch):
        # Gradients only flow to active LoRA
        assert self.current_lora_name is not None
        return super().update_actor(batch)
```

#### Priority 5: Examples (~450 lines)

**9. rllm/agents/multi_role_agent.py** (~100 lines)
**10. examples/multi_agent/train_negotiation.py** (~150 lines)
**11. examples/multi_agent/negotiation_env.py** (~200 lines)

---

## Configuration

### Complete Multi-Agent Config

```yaml
# rllm/trainer/config/ppo_trainer.yaml

# Enable multi-agent training
multi_agent:
  enabled: true
  num_agents: 2
  agent_roles: ["negotiator", "responder"]

  # LoRA configurations per agent
  lora_configs:
    negotiator:
      lora_path: null  # or path to pretrained LoRA
      lora_int_id: 1
      lora_rank: 64
      lora_alpha: 16
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
      lora_dropout: 0.05

    responder:
      lora_path: null
      lora_int_id: 2
      lora_rank: 64
      lora_alpha: 16
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
      lora_dropout: 0.05

  # Optional: Per-agent settings
  agent_specific:
    negotiator:
      learning_rate: 1e-5
      system_prompt: "You are a skilled negotiator..."
    responder:
      learning_rate: 1e-5
      system_prompt: "You evaluate proposals carefully..."

# Rest of standard config
data:
  train_batch_size: 64
  max_prompt_length: 2048
  max_response_length: 4096

actor_rollout_ref:
  model:
    path: Qwen/Qwen3-32B
  hybrid_engine: true
  actor:
    optim:
      lr: 1e-6

trainer:
  n_gpus_per_node: 8
  total_epochs: 100
```

---

## Code Examples

### Multi-Agent Environment

```python
from rllm.environments.base.base_env import BaseEnv

class NegotiationEnv(BaseEnv):
    """Two agents negotiate over a resource allocation"""

    def __init__(self, task, agent_roles=None, **kwargs):
        self.task = task
        self.agent_roles = agent_roles or ["negotiator", "responder"]
        self.current_agent_id = 0
        self.history = []
        self.max_turns = 5

    def reset(self):
        self.current_agent_id = 0
        self.history = []

        obs = {
            "scenario": self.task["scenario"],
            "your_role": self.agent_roles[0]
        }
        return obs, {"agent_id": self.agent_roles[0]}

    def step(self, action):
        # Record current agent's action
        current_agent = self.agent_roles[self.current_agent_id]
        self.history.append({
            "agent": current_agent,
            "action": action
        })

        # Compute reward for current agent
        reward = self._evaluate_action(action, current_agent)

        # Switch to next agent
        self.current_agent_id = (self.current_agent_id + 1) % len(self.agent_roles)
        next_agent = self.agent_roles[self.current_agent_id]

        # Check termination
        done = len(self.history) >= self.max_turns * len(self.agent_roles)

        # Build observation for next agent
        obs = {
            "scenario": self.task["scenario"],
            "your_role": next_agent,
            "history": self.history
        }

        info = {
            "agent_id": current_agent,
            "next_agent_id": next_agent,
            "turn": len(self.history)
        }

        return obs, reward, done, info

    def _evaluate_action(self, action, agent_id):
        # Custom reward logic based on agent and action
        # e.g., check if proposal is fair, response is reasonable
        return 0.5  # Placeholder

    @staticmethod
    def from_dict(info: dict) -> "NegotiationEnv":
        return NegotiationEnv(**info)
```

### Multi-Role Agent

```python
from rllm.agents.agent import BaseAgent, Action, Step, Trajectory

class MultiRoleAgent(BaseAgent):
    """Agent that can take on different roles with different LoRAs"""

    def __init__(self, agent_id="agent_0", agent_role="default", **kwargs):
        self.agent_id = agent_id
        self.agent_role = agent_role
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0

    @property
    def lora_adapter_name(self):
        """LoRA adapter name for this agent's policy"""
        return f"lora_{self.agent_role}"

    def update_from_env(self, observation, reward, done, info, **kwargs):
        # Update previous step if exists
        if self.trajectory.steps:
            prior_step = self.trajectory.steps[-1]
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info

        if done:
            return

        # Build prompt based on role
        role = observation.get("your_role", self.agent_role)
        history = observation.get("history", [])

        system_prompt = self._get_role_prompt(role)
        context = self._format_history(history)

        formatted_obs = f"{system_prompt}\n\nScenario: {observation['scenario']}\n{context}"

        self.messages.append({"role": "user", "content": formatted_obs})

        cur_step = Step(observation=formatted_obs, step=self.step)
        self.trajectory.steps.append(cur_step)

    def update_from_model(self, response: str, **kwargs) -> Action:
        cur_step = self.get_current_state()

        # Store agent metadata
        cur_step.model_response = response
        cur_step.agent_id = self.agent_id
        cur_step.agent_role = self.agent_role
        cur_step.lora_adapter = self.lora_adapter_name

        self.messages.append({"role": "assistant", "content": response})
        self.step += 1

        return Action(action=response)

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0

    @property
    def chat_completions(self):
        return self.messages

    @property
    def trajectory(self):
        return self._trajectory

    def _get_role_prompt(self, role):
        prompts = {
            "negotiator": "You are a skilled negotiator...",
            "responder": "You carefully evaluate proposals..."
        }
        return prompts.get(role, "You are a helpful assistant.")

    def _format_history(self, history):
        if not history:
            return ""
        return "\n".join([
            f"{h['agent']}: {h['action']}" for h in history
        ])
```

### Training Script

```python
import hydra
from rllm.trainer import AgentTrainer
from rllm.data.dataset import DatasetRegistry

@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="ppo_trainer",
    version_base=None
)
def main(config):
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("negotiation", "train")
    val_dataset = DatasetRegistry.load_dataset("negotiation", "test")

    # Train (config includes multi_agent section)
    trainer = AgentTrainer(
        agent_class=MultiRoleAgent,
        env_class=NegotiationEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()
```

### Running Training

```bash
python -m examples.multi_agent.train_negotiation \
    multi_agent.enabled=True \
    multi_agent.num_agents=2 \
    data.train_batch_size=64 \
    trainer.n_gpus_per_node=8
```

---

## Implementation Roadmap

### Phase 1: User API Extension (Week 1-2)

**Goal**: Config passthrough without actual multi-agent logic

- [ ] Modify `AgentTrainer.__init__()` (no behavior change)
- [ ] Update `train_agent_ppo.py` to parse `multi_agent` config
- [ ] Add `multi_agent` section to `ppo_trainer.yaml`
- [ ] Test: Config loads and passes through correctly

**Validation**: Single-agent training still works, config is accessible

### Phase 2: Training Logic (Week 3)

**Goal**: Trajectory separation and update skeleton (no LoRA yet)

- [ ] Add `multi_agent_config` to `AgentPPOTrainer.__init__()`
- [ ] Implement `_separate_by_agent_role()` method
- [ ] Implement `_update_agent_policy()` skeleton
- [ ] Modify `fit_agent()` to call sequential updates
- [ ] Test: Trajectory separation works correctly

**Validation**: Can split batches by agent_role, sequential loop executes

### Phase 3: LoRA Integration (Week 4)

**Goal**: Full LoRA support in rollout and training

- [ ] Modify `Router.__init__()` to accept `lora_configs`
- [ ] Update `Router.generate_sequences()` to build LoRA requests
- [ ] Modify `AsyncAgentExecutionEngine` to extract `agent_id`
- [ ] Update `AsyncAgentExecutionEngine._get_verl_async()` to pass `agent_id`
- [ ] Implement `ActorRolloutRefWorker.set_active_lora()`
- [ ] Implement `ActorRolloutRefWorker._load_lora_adapter()`
- [ ] Test: LoRA loading and switching works

**Validation**: vLLM uses correct LoRA per agent, only active LoRA gets gradients

### Phase 4: Examples & Testing (Week 5)

**Goal**: End-to-end multi-agent training works

- [ ] Create `MultiRoleAgent` class
- [ ] Create example `NegotiationEnv`
- [ ] Create training script
- [ ] Add `agent_id`, `agent_role`, `lora_adapter` to `Step`
- [ ] Test: 2-agent training converges on toy task
- [ ] Test: Agents develop different policies

**Validation**: Gradient isolation confirmed, agents converge differently

### Phase 5: Documentation & Polish (Week 6)

**Goal**: Production-ready implementation

- [ ] Comprehensive unit tests
- [ ] Integration tests
- [ ] Performance benchmarking
- [ ] Update documentation
- [ ] Migration guide

**Validation**: All tests pass, documentation complete, performance acceptable

---

## Testing & Validation

### Unit Tests

```python
# Test 1: Multi-agent environment
def test_multi_agent_env():
    env = NegotiationEnv(
        task={"scenario": "test"},
        agent_roles=["agent_0", "agent_1"]
    )
    obs, info = env.reset()
    assert "agent_id" in info
    assert info["agent_id"] == "agent_0"

    for i in range(10):
        obs, reward, done, info = env.step("test action")
        assert "agent_id" in info
        assert "next_agent_id" in info

# Test 2: LoRA routing
async def test_lora_routing():
    router = Router(
        config=config,
        tokenizer=tokenizer,
        addresses=["localhost:8000"],
        lora_configs={
            "agent_0": {"lora_int_id": 1},
            "agent_1": {"lora_int_id": 2}
        }
    )

    # Should build LoRA request for agent_0
    batch = create_test_batch()
    output = await router.generate_sequences(
        batch,
        application_id="test",
        agent_id="agent_0"
    )
    # Verify correct LoRA was requested

# Test 3: Trajectory separation
def test_trajectory_separation():
    trainer = AgentPPOTrainer(
        config=config,
        multi_agent_config={
            "agent_roles": ["agent_0", "agent_1"]
        }
    )

    # Create batch with mixed agent roles
    batch = create_mixed_batch()

    batches = trainer._separate_by_agent_role(batch)

    assert len(batches) == 2
    assert "agent_0" in batches
    assert "agent_1" in batches
    assert len(batches["agent_0"]) > 0
    assert len(batches["agent_1"]) > 0

# Test 4: Gradient isolation
def test_gradient_isolation():
    worker = ActorRolloutRefWorker(config)

    # Load two LoRAs
    worker.set_active_lora("agent_0", lora_config_0)
    worker.set_active_lora("agent_1", lora_config_1)

    # Set agent_0 active
    worker.set_active_lora("agent_0", lora_config_0)

    # Update - only agent_0 should get gradients
    batch = create_test_batch()
    worker.update_actor(batch)

    # Verify agent_1 LoRA unchanged
    # Verify agent_0 LoRA updated
```

### Integration Test

```bash
# End-to-end 2-agent training
python -m examples.multi_agent.train_negotiation \
    multi_agent.enabled=True \
    multi_agent.num_agents=2 \
    data.train_batch_size=8 \
    trainer.total_epochs=2 \
    trainer.n_gpus_per_node=2

# Verify:
# 1. Training completes without errors
# 2. Both LoRAs are updated
# 3. Agents show different behaviors
# 4. Memory usage is acceptable
```

### Performance Benchmarks

| Metric | Target | Acceptable |
|--------|--------|------------|
| Rollout speed | Same as single-agent | ±5% |
| Training time (2 agents) | 2x single-agent | 2.5x |
| Memory overhead per agent | 2% | 5% |
| LoRA switching latency | <1ms | <5ms |

---

## Quick Reference

### User Code (No Changes!)

```python
from rllm.trainer import AgentTrainer

trainer = AgentTrainer(
    agent_class=MultiRoleAgent,
    env_class=MultiAgentEnv,
    config=config,  # Just add multi_agent section
)
trainer.train()
```

### File Modification Checklist

- [ ] `rllm/trainer/agent_trainer.py` - Accept config
- [ ] `rllm/trainer/verl/train_agent_ppo.py` - Parse multi_agent
- [ ] `rllm/trainer/verl/agent_ppo_trainer.py` - Multi-agent logic
- [ ] `rllm/trainer/config/ppo_trainer.yaml` - Add multi_agent section
- [ ] `rllm/engine/agent_execution_engine.py` - Pass agent_id
- [ ] `rllm/router/router.py` - LoRA routing
- [ ] `rllm/agents/agent.py` - Add agent metadata
- [ ] `verl/verl/workers/fsdp_workers.py` - LoRA management
- [ ] `rllm/agents/multi_role_agent.py` - Example agent
- [ ] `examples/multi_agent/train_negotiation.py` - Training script
- [ ] `examples/multi_agent/negotiation_env.py` - Example environment

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "LoRA not found" | Not loaded in vLLM | Check `lora_configs` in init |
| Same policy for both agents | Gradient interference | Verify `set_active_lora()` called |
| OOM error | Too many/large LoRAs | Reduce `lora_rank` or batch size |
| Wrong agent credited | Missing metadata | Add `agent_id` to Step |

### Memory Estimate (2 agents, Qwen3-32B)

- Base model (FP16): ~60GB
- LoRA agent_0 (rank 64): ~0.8GB
- LoRA agent_1 (rank 64): ~0.8GB
- FSDP overhead: ~10GB
- **Total: ~72GB** ✅ Fits on 8×H100 (80GB each)

---

## Additional Resources

- **vLLM LoRA Documentation**: https://docs.vllm.ai/en/latest/models/lora.html
- **PEFT Documentation**: https://huggingface.co/docs/peft
- **rLLM Documentation**: https://rllm-project.readthedocs.io

## Support

For questions or issues:
1. Check this README for implementation details
2. Review code examples in `examples/multi_agent/`
3. File an issue at https://github.com/rllm-org/rllm/issues
