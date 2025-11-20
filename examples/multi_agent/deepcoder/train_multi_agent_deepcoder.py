import hydra

from examples.multi_agent.deepcoder.multi_agent_deepcoder_workflow import MultiAgentDeepCoderWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import code_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("deepcoder", "train")
    test_dataset = DatasetRegistry.load_dataset("deepcoder", "test")

    workflow_args = {
        "reward_function": code_reward_fn,
        "max_refinement_iterations": 3,
    }

    trainer = AgentTrainer(
        workflow_class=MultiAgentDeepCoderWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
