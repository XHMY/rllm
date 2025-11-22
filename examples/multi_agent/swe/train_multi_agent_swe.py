import hydra

from examples.multi_agent.swe.multi_agent_swe_workflow import MultiAgentSWEWorkflow
from rllm.data import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="multi_agent_ppo_trainer", version_base=None)
def main(config):
    # Load SWE datasets - using names from prepare_swe_data.py
    train_dataset = DatasetRegistry.load_dataset("R2E_Gym_Subset", "train")
    val_dataset = DatasetRegistry.load_dataset("SWE_Bench_Verified", "test")

    workflow_args = {
        "max_refinement_iterations": 3,
    }

    trainer = AgentTrainer(
        workflow_class=MultiAgentSWEWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
