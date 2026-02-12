"""Training entry point for the Orchestrator-Workers Math Workflow.

This script trains a 2-agent orchestrator-workers workflow for mathematical
problem solving using reinforcement learning with task decomposition.

Usage:
    python -m examples.math_reasoning.train_orchestrator_workers_math \
        trainer.agent_names=['orchestrator','worker'] \
        +rllm.workflow.max_subtasks=4
"""

import hydra

from examples.math_reasoning.orchestrator_workers_math_workflow import (
    OrchestratorWorkersMathWorkflow,
)
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="multi_agent_ppo_trainer",
    version_base=None,
)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("dapo_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2025", "test")

    assert train_dataset is not None, (
        "Failed to load train dataset. "
        "Please run examples/math_reasoning/prepare_dapo_math_data.py first."
    )
    assert test_dataset is not None, (
        "Failed to load test dataset. "
        "Please run examples/math_reasoning/prepare_aime_test_data.py first."
    )

    # Get workflow config parameters
    max_subtasks = 4
    use_final_outcome_reward = True
    share_context_with_workers = True
    if hasattr(config, "rllm") and hasattr(config.rllm, "workflow"):
        max_subtasks = getattr(config.rllm.workflow, "max_subtasks", 4)
        use_final_outcome_reward = getattr(
            config.rllm.workflow, "use_final_outcome_reward", True
        )
        share_context_with_workers = getattr(
            config.rllm.workflow, "share_context_with_workers", True
        )

    trainer = AgentTrainer(
        workflow_class=OrchestratorWorkersMathWorkflow,
        workflow_args={
            "max_subtasks": max_subtasks,
            "reward_function": math_reward_fn,
            "use_final_outcome_reward": use_final_outcome_reward,
            "share_context_with_workers": share_context_with_workers
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
