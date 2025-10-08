"""
Training script for multi-agent collaborative math problem solving.

This script trains a 3-agent system where:
- generator_initial: Proposes initial solutions
- evaluator_critique: Evaluates and critiques solutions
- generator_refinement: Refines solutions based on feedback

Each agent has its own LoRA adapter for independent policy learning.
"""

import json
import hydra
from pathlib import Path

from rllm.agents.multi_role_math_agent import MultiRoleMathAgent
from rllm.data import DatasetRegistry
from rllm.environments.math.multi_agent_math_env import MultiAgentMathEnv
from rllm.trainer.agent_trainer import AgentTrainer


def load_prompts(prompts_file: str) -> dict:
    """Load prompt templates from JSON file."""
    prompts_path = Path(prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)

    if "multi_agent_math_prompts" not in prompts_data:
        raise KeyError(f"Required key 'multi_agent_math_prompts' not found in {prompts_file}")

    return prompts_data["multi_agent_math_prompts"]


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    """
    Main training function for multi-agent math reasoning.

    The config should include:
        multi_agent.enabled: true
        multi_agent.agent_roles: ["generator_initial", "evaluator_critique", "generator_refinement"]
        multi_agent.lora_configs: {...}  # LoRA configs for each role
    """
    print("=" * 80)
    print("Multi-Agent Math Reasoning Training")
    print("=" * 80)

    # Load prompt templates
    prompts_file = "examples/math_reasoning/prompt.json"
    prompts = load_prompts(prompts_file)
    print(f"\nLoaded prompts from {prompts_file}")
    print(f"Available prompts: {list(prompts.keys())}")

    # Check if dataset is already prepared
    if not DatasetRegistry.dataset_exists("deepmath", "train"):
        raise RuntimeError(
            "DeepMath dataset not found in registry. "
            "Please run 'python examples/math_reasoning/prepare_deepmath_data.py' first to prepare the dataset."
        )

    # Load datasets from registry
    print("\nLoading datasets from registry...")
    train_dataset = DatasetRegistry.load_dataset("deepmath", "train")
    val_dataset = DatasetRegistry.load_dataset("deepmath", "test")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # Create trainer
    print("\nInitializing AgentTrainer...")
    trainer = AgentTrainer(
        agent_class=MultiRoleMathAgent,
        env_class=MultiAgentMathEnv,
        agent_args={"prompts": prompts},
        env_args={"prompts": prompts},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Start training
    print("\nStarting multi-agent training...")
    print(f"Model: {config.actor_rollout_ref.model.path}")
    print(f"Batch size: {config.data.train_batch_size}")
    print(f"Max steps per episode: {config.agent.max_steps}")
    print(f"Multi-agent enabled: {config.multi_agent.get('enabled', False)}")
    if config.multi_agent.get('enabled', False):
        print(f"Agent roles: {config.multi_agent.get('agent_roles', [])}")
    print("=" * 80)

    trainer.train()


if __name__ == "__main__":
    main()
