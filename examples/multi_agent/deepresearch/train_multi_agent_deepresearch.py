import hydra

from examples.multi_agent.deepresearch.multi_agent_deepresearch_workflow import MultiAgentDeepResearchWorkflow
from rllm.data import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Note: DeepResearch typically uses evaluation datasets like HLE (Humanity's Last Exam)
    # For training, you would need to prepare a suitable research-question dataset
    # This is a template - adjust dataset names based on your prepared data
    train_dataset = DatasetRegistry.load_dataset("research_questions", "train")
    val_dataset = DatasetRegistry.load_dataset("research_questions", "val")

    # Tools configuration - adapt based on your needs
    # In practice, you would configure search, scholar, and other research tools here
    tools = {}  # e.g., {"search": SearchTool(), "scholar": ScholarTool()}

    workflow_args = {
        "tools": tools,
        "max_subqueries": 5,
    }

    trainer = AgentTrainer(
        workflow_class=MultiAgentDeepResearchWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
