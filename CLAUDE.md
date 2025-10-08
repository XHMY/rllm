# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

rLLM is an open-source framework for post-training language agents via reinforcement learning. It enables building custom agents and environments, training them with RL (PPO/GRPO), and deploying them for real-world workloads.

**Key Architecture**: The framework has three core abstractions:
- **Agents** ([rllm/agents/](rllm/agents/)): Define how language models interact with environments (e.g., MathAgent, ToolAgent, SWEAgent)
- **Environments** ([rllm/environments/](rllm/environments/)): Define tasks and reward functions (e.g., FrozenLakeEnv, CompetitionCodingEnv, ToolEnvironment)
- **Trainer** ([rllm/trainer/](rllm/trainer/)): Orchestrates RL training using the verl backend

The framework separates agent/environment logic from training infrastructure, allowing users to focus on defining custom agents without dealing with distributed training complexities.

## Environment Setup

**Note**: Use `conda activate verl` to activate the rLLM environment after installation.

## Common Development Commands

### Training

All training scripts follow the same pattern and use Hydra for configuration:

```bash
# General pattern for training scripts
python3 -m rllm.trainer.verl.train_agent_ppo <hydra_config_overrides>

# Or use example-specific scripts
bash examples/<domain>/train_<agent>.sh

# Examples:
bash examples/frozenlake/train_frozenlake_agent.sh
bash examples/deepcoder/train_deepcoder_16k.sh
bash examples/swe/train_deepswe_32b.sh
```

Training scripts use Hydra config overrides with dot notation:
- `data.train_batch_size=128`
- `actor_rollout_ref.model.path=Qwen/Qwen3-32B`
- `trainer.n_gpus_per_node=8`
- `algorithm.adv_estimator=grpo` (or `loop` for standard PPO)

### Testing

```bash
# Run tests with pytest
pytest tests/

# Run specific test files
pytest tests/rewards/test_math_reward.py
pytest tests/envs/test_tool_env.py
```

### Linting & Formatting

```bash
# Run ruff linting and auto-fix
ruff check --fix --show-fixes rllm/

# Run ruff formatting
ruff format rllm/

# Run pre-commit hooks
pre-commit run --all-files
```

**Note**: verl/ directory is excluded from linting/formatting checks.

### Documentation

```bash
# Build documentation with mkdocs
bash build_docs.sh
# Or manually:
mkdocs serve
```

## Core Concepts

### Agent-Environment Interaction Pattern

The interaction cycle follows a structured pattern:

1. **Environment Reset**: `env.reset()` provides initial observation
2. **Agent Processing**: Agent receives observation via `update_from_env()` and builds chat messages
3. **Model Inference**: Agent's `chat_completions` property triggers LLM to generate response
4. **Response Processing**: Agent processes model output with `update_from_model()` and returns `Action`
5. **Environment Feedback**: Environment executes action via `step()` and returns `(observation, reward, done, info)`
6. **Loop Continues**: Process repeats until `done=True`

This cycle enables sophisticated behaviors like self-correction, multi-turn reasoning, and adaptive problem-solving.

### Key Base Classes

**BaseAgent** ([rllm/agents/agent.py](rllm/agents/agent.py)): Abstract base for all agents

```python
class BaseAgent(ABC):
    @abstractmethod
    def update_from_env(self, observation, reward, done, info, **kwargs):
        """Updates agent state after receiving environment feedback."""
        pass

    @abstractmethod
    def update_from_model(self, response: str, **kwargs) -> Action:
        """Updates agent state after receiving model response. Returns Action."""
        pass

    @abstractmethod
    def reset(self):
        """Resets agent's internal state for new episode."""
        pass

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Returns messages formatted for chat completion."""
        return []

    @property
    def trajectory(self) -> Trajectory:
        """Returns complete interaction history."""
        return Trajectory()

    def get_current_state(self) -> Step | None:
        """Return the most recent step."""
        return None
```

**Key Responsibilities**:
- **State tracking**: Maintaining conversation history via `Trajectory` and `Step` objects
- **Model interaction**: Formatting messages for LLM via `chat_completions` property
- **Response processing**: Handling model outputs in `update_from_model()`
- **Environment adaptation**: Updating state based on feedback in `update_from_env()`

**BaseEnv** ([rllm/environments/base/base_env.py](rllm/environments/base/base_env.py)): Abstract base for environments

```python
class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> Tuple[Any, Dict]:
        """Reset environment and return initial observation and info."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Execute action and return (observation, reward, done, info)."""
        pass

    def close(self):
        """Clean up environment resources."""
        pass

    @staticmethod
    @abstractmethod
    def from_dict(env_args: Dict) -> "BaseEnv":
        """Create environment instance from dictionary.
        Used during both inference and training to instantiate environments."""
        pass
```

**Key Responsibilities**:
- **Task definition**: Providing problems for agents to solve
- **Observation generation**: Creating inputs for agent decision-making
- **Action evaluation**: Computing rewards based on agent responses
- **Episode management**: Determining when interactions should terminate

**Important Environment Types**:
- `SingleTurnEnvironment`: For single-step tasks (math, coding competitions)
- `ToolEnvironment`: For multi-turn tool-using agents

### Agent Execution Engine

The `AgentExecutionEngine` ([rllm/engine/agent_execution_engine.py](rllm/engine/agent_execution_engine.py)) orchestrates parallel agent-environment interactions:

**Key Features**:
- Manages `N=n_parallel_agents` agent-environment pairs running concurrently
- Supports fully asynchronous trajectory collection
- Handles task queue management: when a pair completes, it starts a new task
- Integrates with OpenAI-compatible API endpoints or verl for inference

**Usage Pattern**:
```python
engine = AgentExecutionEngine(
    agent_class=CustomAgent,
    env_class=CustomEnvironment,
    agent_args={},
    env_args={},
    engine_name="openai",  # or "verl"
    tokenizer=tokenizer,
    n_parallel_agents=64,
    max_steps=10,
    max_response_length=4096,
    max_prompt_length=2048,
    sampling_params={"temperature": 0.7},
    rollout_engine_args={"base_url": "http://localhost:8000/v1"}
)

# Execute tasks asynchronously
results = await engine.execute_tasks(tasks)
```

**Task Initialization**: Each task is a dictionary merged with `env_args` to create environment via `env_class.from_dict({**env_args, **task})`.

### Training Infrastructure

Training uses the `AgentTrainer` which wraps verl's distributed RL training:

**Training Flow**:
1. **Initialization**: `AgentPPOTrainer` inherits from verl's `RayPPOTrainer`
2. **Setup**: Ray workers, AgentExecutionEngine, and dataset batching initialized
3. **Training Loop**: For each mini-batch:
   - Data passed to AgentExecutionEngine
   - Agent-environment pairs process batch in parallel
   - Trajectories collected through interactions
4. **Update Phase**: Trajectories transformed to verl format, gradients computed

**Configuration** (Hydra YAML in [rllm/trainer/config/](rllm/trainer/config/)):
- `data.*`: Dataset paths, batch sizes, sequence lengths
- `actor_rollout_ref.*`: Model paths, training hyperparameters, FSDP/vLLM configs
- `algorithm.*`: PPO/GRPO settings (advantage estimation, KL penalties)
- `trainer.*`: Training logistics (epochs, GPUs, logging, checkpointing)
- `agent.*`: Agent settings (max_steps, n_parallel_agents, timeouts)
- `env.*`: Environment selection and configuration

### RL Algorithms & Agent Types

rLLM categorizes agents by context management:

**1. Cumulative Agents**: Accumulate full interaction history in a single trajectory
- Use **GRPO with Observation Masking**: Mask non-model tokens (system prompts, observations), compute loss only on model-generated tokens
- Each trajectory gets scalar reward, advantages computed by grouping trajectories over same task
- Examples: DeepSWE, ReTool, Search-R1, RAGEN

**2. Non-Cumulative Agents**: Manage context via summarized state (MDP formulation)
- Each step is independent prompt-response interaction
- Trajectory is sequence of steps with varying lengths

**Two Approaches for Non-Cumulative Agents**:
- **Stepwise GRPO with Advantage Broadcasting**: Compute advantage on final step using terminal reward, broadcast to all previous steps. Best when earlier actions contribute to final outcome without fine-grained rewards.
- **Stepwise GRPO with Per-Step Grouping**: Each step has own reward, grouped by position across trajectories. Best for symmetric trajectories (e.g., iterative self-refinement like Kevin-32B).

**Configuration**:
- Set `algorithm.adv_estimator=grpo` for GRPO
- Set `algorithm.adv_estimator=loop` for standard PPO
- Set `agent.use_stepwise_advantage=true` for stepwise advantage calculation

### Data Pipeline

The `Dataset` class ([rllm/data/dataset.py](rllm/data/dataset.py)) wraps training data:
- Supports loading from Parquet, JSONL, or JSON files
- `DatasetRegistry.load_dataset(name, split)` provides pre-registered datasets
- `DatasetRegistry.register_dataset(name, data, split)` registers datasets and saves to parquet
- Each example should have fields needed by the environment's `from_dict()` method

**Data Preparation Pattern**: Datasets are prepared in a separate preprocessing step before training:

1. **Create preprocessing script** (e.g., `examples/math_reasoning/prepare_deepmath_data.py`):
   ```python
   from datasets import load_dataset
   from rllm.data.dataset import DatasetRegistry

   # Load and filter data
   ds = load_dataset("my-dataset")["train"]
   ds = ds.filter(...)  # Apply filters

   # Split and register
   ds_split = ds.train_test_split(test_size=0.01)
   DatasetRegistry.register_dataset("mydataset", ds_split["train"], "train")
   DatasetRegistry.register_dataset("mydataset", ds_split["test"], "test")
   ```

2. **Run preprocessing once** before training:
   ```bash
   python examples/my_task/prepare_my_data.py
   ```

3. **Training script loads from registry**:
   ```python
   train_dataset = DatasetRegistry.load_dataset("mydataset", "train")
   val_dataset = DatasetRegistry.load_dataset("mydataset", "test")
   ```

This pattern:
- Separates data preprocessing from training
- Saves filtered data to `rllm/data/datasets/` as parquet files
- Enables faster training startup (no repeated filtering)
- Follows the same pattern as FrozenLake and DeepMath examples

### Registry Pattern

The framework uses registries for extensibility:
- `DatasetRegistry` ([rllm/data/dataset.py](rllm/data/dataset.py)): Register dataset loaders
- Environment/Agent mappings in [rllm/trainer/env_agent_mappings.py](rllm/trainer/env_agent_mappings.py)

### Environment Variables for Training

Training scripts typically set:
```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
```

## Project Structure

```
rllm/
├── agents/          # Agent implementations (BaseAgent subclasses)
├── environments/    # Environment implementations (BaseEnv subclasses)
│   ├── base/       # Base classes (BaseEnv, SingleTurnEnvironment)
│   ├── tools/      # Tool-using environment
│   ├── code/       # Coding competition environment
│   ├── frozenlake/ # Grid world environment
│   └── browsergym/ # Web browsing environment
├── data/           # Dataset utilities and registry
│   ├── datasets/   # Built-in dataset loaders
│   └── preprocess/ # Data preprocessing utilities
├── rewards/        # Reward functions (math_reward_fn, code_reward_fn)
├── trainer/        # Training infrastructure
│   ├── config/     # Hydra YAML configuration files
│   └── verl/       # verl backend integration
├── tools/          # External tool integrations
├── parser/         # Response parsing utilities
├── router/         # Model routing and inference
└── engine/         # Asynchronous trajectory generation

examples/          # Training examples for different domains
├── frozenlake/    # Simple grid world for testing
├── deepcoder/     # Code generation with long context (16k-32k)
├── deepscaler/    # Math reasoning with extended context (8k-24k)
├── swe/           # Software engineering agents (DeepSWE)
├── math_tool/     # Math with calculator tools
├── search/        # Question answering with search
└── miniwob/       # Web interaction tasks

verl/              # Training backend (submodule, excluded from linting)

tests/             # Unit tests
docs/              # MkDocs documentation source
scripts/           # Utility scripts
```

## Creating Custom Agents & Environments

### Custom Agent Example

```python
from rllm.agents.agent import BaseAgent, Action, Step, Trajectory

class MyAgent(BaseAgent):
    def __init__(self, **kwargs):
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0

    def update_from_env(self, observation, reward, done, info, **kwargs):
        # Build chat messages from environment observation
        if not self.trajectory.steps:
            # Initial observation
            formatted_obs = f"Task: {observation}"
        else:
            # Update previous step's outcome
            prior_step = self._trajectory.steps[-1]
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info
            formatted_obs = f"Feedback: {observation}"

        if done:
            return

        self.messages.append({"role": "user", "content": formatted_obs})
        cur_step = Step(observation=formatted_obs, step=self.step)
        self.trajectory.steps.append(cur_step)

    def update_from_model(self, response: str, **kwargs) -> Action:
        # Process model response and return action
        cur_step = self.get_current_state()
        cur_step.model_response = response

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
```

### Custom Environment Example

```python
from rllm.environments.base.base_env import BaseEnv

class MyEnv(BaseEnv):
    def __init__(self, task, reward_fn, max_steps=5, **kwargs):
        self.task = task
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        # Return initial observation and info
        self.current_step = 0
        return {"question": self.task["question"]}, {}

    def step(self, action):
        # Execute action, compute reward
        self.current_step += 1
        reward = self.reward_fn(self.task, action)
        done = reward > 0.0 or self.current_step >= self.max_steps
        return None, reward, done, {}

    @staticmethod
    def from_dict(info: dict) -> "MyEnv":
        return MyEnv(**info)
```

### Register and Train

```python
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer
import hydra

# Register dataset
DatasetRegistry.register_dataset("mydataset", loader_fn)

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    # Train
    trainer = AgentTrainer(
        agent_class=MyAgent,
        env_class=MyEnv,
        agent_args={},
        env_args={"reward_fn": my_reward_fn},
        config=config,
        train_dataset=DatasetRegistry.load_dataset("mydataset", "train"),
        val_dataset=DatasetRegistry.load_dataset("mydataset", "test"),
    )
    trainer.train()

if __name__ == "__main__":
    main()
```

## Multi-Agent Environments (Advanced)

For multi-agent scenarios where each agent can have its own policy, see **[MULTI_AGENT_README.md](MULTI_AGENT_README.md)** for a comprehensive implementation guide. See **[MULTI_AGENT_IMPLEMENTATION_STATUS.md](MULTI_AGENT_IMPLEMENTATION_STATUS.md)** for the completed 3-agent math reasoning implementation.

### Recommended Architecture

**Key Design**: Use **shared vLLM inference engine with LoRA adapters** for agent-specific policies, and **sequential policy updates** during training. This minimizes modifications to the existing rLLM pipeline.

**Benefits**:
- Single vLLM instance serves all agents (memory efficient)
- Each agent has independent policy via LoRA (~1-2% model size per agent)
- Sequential updates ensure training stability
- No need for multiple rollout workers or trainers

### Design Considerations

**1. Multi-Agent Environment Structure**:
- Environment manages multiple agents simultaneously
- Each agent may have different observation/action spaces
- Reward can be individual or shared across agents
- Environment tracks which agent's turn it is via `agent_id` in `info` dict

**2. Agent Policy Management (LoRA-based)**:
- Each agent has a unique LoRA adapter (e.g., "lora_agent_0", "lora_agent_1")
- vLLM loads multiple LoRA adapters simultaneously
- Router passes `agent_id` to select correct LoRA during inference
- ActorRolloutRefWorker switches active LoRA for sequential updates

**3. Implementation Approach**:

**Recommended: Single Environment, Multiple Agent Roles with LoRA**
```python
class MultiAgentEnv(BaseEnv):
    def __init__(self, task, num_agents=2, agent_roles=None, **kwargs):
        self.num_agents = num_agents
        self.current_agent_id = 0
        self.agent_roles = agent_roles or ["agent_0", "agent_1"]
        self.agent_states = {}

    def step(self, action):
        # Identify which agent took action
        agent_id = self.agent_roles[self.current_agent_id]

        # Compute agent-specific reward
        reward = self.evaluate_action(action, agent_id)

        # Rotate to next agent
        self.current_agent_id = (self.current_agent_id + 1) % self.num_agents
        done = self.check_termination()

        next_agent_id = self.agent_roles[self.current_agent_id]
        obs = self.get_observation(next_agent_id)

        return obs, reward, done, {
            "agent_id": agent_id,           # Who just acted
            "next_agent_id": next_agent_id  # Who acts next
        }

    def reset(self):
        self.current_agent_id = 0
        obs = self.get_observation(self.agent_roles[0])
        return obs, {"agent_id": self.agent_roles[0]}
```

**Agent Implementation with LoRA Support**
```python
class MultiRoleAgent(BaseAgent):
    def __init__(self, agent_id="agent_0", agent_role="default", **kwargs):
        super().__init__()
        self.agent_id = agent_id        # Unique instance ID
        self.agent_role = agent_role    # Role type (maps to LoRA)
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def lora_adapter_name(self):
        """Return LoRA adapter name for this agent's policy"""
        return f"lora_{self.agent_role}"

    def update_from_model(self, response: str, **kwargs) -> Action:
        # Store agent metadata in step
        cur_step = self.get_current_state()
        cur_step.agent_id = self.agent_id
        cur_step.agent_role = self.agent_role
        cur_step.lora_adapter = self.lora_adapter_name

        # ... rest of implementation ...
```

**4. Trajectory Management**:
- Environment passes `agent_id` in `info` dict at each step
- Each `Step` stores `agent_id`, `agent_role`, `lora_adapter` metadata
- During training, `AgentPPOTrainer._separate_by_agent_role()` splits trajectories
- Each agent's policy (LoRA) is updated sequentially using its subset of trajectories

**5. Training Workflow**:
```python
# Rollout: All agents share one vLLM engine
for step in trajectory:
    agent_id = env.get_current_agent_id()
    response = vllm.generate(
        prompt,
        lora_request=LoRARequest(name=f"lora_{agent_id}", ...)
    )

# Training: Sequential policy updates
agent_batches = separate_by_agent_role(batch)

for agent_role in ["agent_0", "agent_1"]:
    # Switch active LoRA adapter
    actor_worker.set_active_lora(agent_role, lora_config)

    # Update this agent's LoRA weights
    actor_worker.update_actor(agent_batches[agent_role])
```

**6. Configuration Example**:
```yaml
multi_agent:
  enabled: true
  num_agents: 2
  agent_roles: ["agent_0", "agent_1"]

  lora_configs:
    agent_0:
      lora_path: null  # Initialize from scratch or load pretrained
      lora_int_id: 1
      lora_rank: 64
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    agent_1:
      lora_path: null
      lora_int_id: 2
      lora_rank: 64
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Note**: See [MULTI_AGENT_README.md](MULTI_AGENT_README.md) for complete implementation guide including:
- Detailed architecture diagrams
- Code examples for all components
- Phase-by-phase implementation roadmap
- Testing strategy and validation

For a working example, see [MULTI_AGENT_IMPLEMENTATION_STATUS.md](MULTI_AGENT_IMPLEMENTATION_STATUS.md) which documents the completed 3-agent math reasoning system (~1,630 lines across 12 files).

## Important Configuration Notes

- **Sequence lengths**: `data.max_prompt_length` + `data.max_response_length` must fit in GPU memory
- **Batch sizing**: `data.train_batch_size` should be divisible by `actor_rollout_ref.actor.ppo_mini_batch_size`
- **Multi-GPU training**: Set `trainer.n_gpus_per_node` and `trainer.nnodes`, adjust `actor_rollout_ref.rollout.tensor_model_parallel_size`
- **Algorithm choice**: Use `algorithm.adv_estimator=grpo` for GRPO (simpler, no critic) or `loop` for standard PPO
- **Hybrid engine**: `actor_rollout_ref.hybrid_engine=True` uses vLLM for fast rollout + FSDP for training
- **Rejection sampling**: Enable with `trainer.rejection_sample=True` and set `trainer.rejection_sample_multiplier`

## Key Python Modules

- **System prompts**: [rllm/system_prompts.py](rllm/system_prompts.py) contains prompt templates for different agents
- **Trajectory visualization**: [rllm/trajectory_visualizer.py](rllm/trajectory_visualizer.py) for debugging agent rollouts
- **Reward functions**: [rllm/rewards/](rllm/rewards/) contains domain-specific reward computation
- **Model routing**: [rllm/router/](rllm/router/) handles model inference and API routing
- **Agent utilities**: [rllm/agents/utils.py](rllm/agents/utils.py) for message formatting and tokenization

## Development Workflow

1. **Define agent**: Subclass `BaseAgent`, implement `update_from_env()`, `update_from_model()`, `reset()`, `chat_completions`, `trajectory`
2. **Define environment**: Subclass `BaseEnv`, implement `reset()`, `step()`, `from_dict()`
3. **Prepare data**: Create dataset loader and register with `DatasetRegistry`
4. **Configure training**: Create or modify Hydra config, typically starting from `ppo_trainer.yaml`
5. **Create training script**: Use `AgentTrainer` with `@hydra.main` decorator
6. **Run training**: Execute training script, monitor with wandb (set `trainer.logger=['console','wandb']`)
7. **Evaluate**: Training automatically runs validation at intervals specified by `trainer.test_freq`

## Related Projects

- **verl**: Training backend ([github.com/volcengine/verl](https://github.com/volcengine/verl))
- **DeepSWE**: SWE agent trained with rLLM ([HuggingFace model](https://huggingface.co/agentica-org/DeepSWE-Preview))
- **DeepCoder**: Coding model trained with rLLM
- **DeepScaleR**: Math reasoning model with extended context
