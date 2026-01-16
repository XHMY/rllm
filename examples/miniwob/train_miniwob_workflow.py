"""MiniWob training using workflow-based trainer.

This script trains a MiniWob agent using the EnvSingleAgentWorkflow,
which provides identical behavior to the original agent-based training
but uses the AgentWorkflowPPOTrainer (workflow-based trainer).

This serves as Step 1 in the migration path:
- Step 0: Original agent-based training (train_miniwob_agent.py)
- Step 1: Workflow-based training with same behavior (this file)
- Step 2: Multi-agent workflow training (train_miniwob_evaluator_optimizer.py)

Usage:
    python -m examples.miniwob.train_miniwob_workflow

Environment Variables:
    MINIWOB_URL: URL for MiniWob environment (required)
"""

import os

import hydra

from rllm.agents.miniwob_agent import MiniWobAgent
from rllm.data import DatasetRegistry
from rllm.environments.browsergym.browsergym import BrowserGymEnv
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.env_single_agent_workflow import EnvSingleAgentWorkflow


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="multi_agent_ppo_trainer",
    version_base=None,
)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("miniwob", "train")
    val_dataset = DatasetRegistry.load_dataset("miniwob", "test")

    url = os.getenv("MINIWOB_URL")
    if url is None:
        raise Exception("MINIWOB_URL is not set.")
    else:
        print(f"MINIWOB_URL is set to: {url}")

    # Agent configuration
    agent_args = {
        "use_html": True,
        "use_axtree": True,
        "use_screenshot": False,
        "use_accumulate_thinking": True,
        "cot_prompt": False,
        "use_full_conversation": True,
    }

    # Environment configuration
    env_args = {
        "subtask": "miniwob",
        "miniwob_url": url,
    }

    # Get max_steps from config if specified
    max_steps = config.rllm.get("agent", {}).get("max_steps", 10)

    trainer = AgentTrainer(
        workflow_class=EnvSingleAgentWorkflow,
        workflow_args={
            "agent_cls": MiniWobAgent,
            "env_cls": BrowserGymEnv,
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
