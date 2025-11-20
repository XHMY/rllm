import hydra
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from examples.multi_agent.search.multi_agent_search_workflow import MultiAgentSearchWorkflow
from examples.search.local_retrieval_tool import LocalRetrievalTool
from rllm.data import DatasetRegistry
from rllm.rewards.reward_fn import search_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
    val_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")

    # Use the local retrieval tool (same as single-agent version)
    search_tool = LocalRetrievalTool()

    workflow_args = {
        "reward_function": search_reward_fn,
        "search_tool": search_tool,
        "max_refinement_iterations": 2,
    }

    trainer = AgentTrainer(
        workflow_class=MultiAgentSearchWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
