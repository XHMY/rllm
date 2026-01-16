"""FrozenLake training using workflow-based trainer.

This script trains a FrozenLake agent using the EnvSingleAgentWorkflow,
which provides identical behavior to the original agent-based training
but uses the AgentWorkflowPPOTrainer (workflow-based trainer).

This serves as Step 1 in the migration path:
- Step 0: Original agent-based training (train_frozenlake_agent.py)
- Step 1: Workflow-based training with same behavior (this file)
- Step 2: Multi-agent workflow training (train_frozenlake_evaluator_optimizer.py)

Usage:
    python -m examples.frozenlake.train_frozenlake_workflow
"""

import hydra

from rllm.agents.frozenlake_agent import FrozenLakeAgent
from rllm.data import DatasetRegistry
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.env_single_agent_workflow import EnvSingleAgentWorkflow


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="multi_agent_ppo_trainer",
    version_base=None,
)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("frozenlake", "train")
    val_dataset = DatasetRegistry.load_dataset("frozenlake", "test")

    # Agent configuration
    agent_args = {
        "use_accumulate_thinking": True,
        "use_multistep_prompt": False,
        "use_accumulate_history": True,
    }

    # Environment configuration
    env_args = {}

    # Get max_steps from config if specified
    max_steps = config.rllm.get("agent", {}).get("max_steps", 20)

    trainer = AgentTrainer(
        workflow_class=EnvSingleAgentWorkflow,
        workflow_args={
            "agent_cls": FrozenLakeAgent,
            "env_cls": FrozenLakeEnv,
            "agent_args": agent_args,
            "env_args": env_args,
            "max_steps": max_steps,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
