import hydra

from examples.multi_agent.math_tool.multi_agent_math_tool_workflow import MultiAgentMathToolWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="multi_agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")

    workflow_args = {
        "reward_function": math_reward_fn,
        "max_refinement_iterations": 3,
    }

    trainer = AgentTrainer(
        workflow_class=MultiAgentMathToolWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
