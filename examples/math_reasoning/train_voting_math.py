"""Training entry point for the Voting Math Workflow.

This script trains a 2-agent voting workflow for mathematical
problem solving using reinforcement learning with parallel generation.

Usage:
    python -m examples.math_reasoning.train_voting_math \
        trainer.agent_names=['generator','aggregator'] \
        +rllm.workflow.n_votes=3
"""

import hydra

from examples.math_reasoning.voting_math_workflow import VotingMathWorkflow
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
    test_dataset = DatasetRegistry.load_dataset("dapo_math", "test")

    assert train_dataset is not None, (
        "Failed to load train dataset. "
        "Please run examples/math_reasoning/prepare_dapo_math_data.py first."
    )
    assert test_dataset is not None, (
        "Failed to load test dataset. "
        "Please run examples/math_reasoning/prepare_dapo_math_data.py first."
    )

    # Get n_votes from config if specified, default to 3
    n_votes = 3
    if hasattr(config, "rllm") and hasattr(config.rllm, "workflow"):
        n_votes = getattr(config.rllm.workflow, "n_votes", 3)

    trainer = AgentTrainer(
        workflow_class=VotingMathWorkflow,
        workflow_args={
            "n_votes": n_votes,
            "reward_function": math_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
