"""FrozenLake training with multi-agent evaluator-optimizer workflow.

This script trains a FrozenLake agent using the EnvEvaluatorOptimizerWorkflow,
which adds an evaluator-optimizer loop at each action step. This enables
iterative refinement of movement actions.

At each step:
1. Actor (FrozenLakeAgent) proposes a movement action
2. Evaluator reviews the action (considers holes, goal position, etc.)
3. If rejected, Actor refines based on feedback
4. Loop until approved or max iterations
5. Execute final action in FrozenLakeEnv

This produces trajectories for 2 agents:
- actor: Primary agent that proposes, refines, and executes actions
- evaluator: Reviews proposed movements and provides feedback

Each agent can have its own LoRA adapter for specialized training.

Usage:
    python -m examples.frozenlake.train_frozenlake_evaluator_optimizer
"""

import hydra

from rllm.agents.frozenlake_agent import FrozenLakeAgent
from rllm.data import DatasetRegistry
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.env_evaluator_optimizer_workflow import EnvEvaluatorOptimizerWorkflow


# Custom prompt templates for FrozenLake
FROZENLAKE_EVAL_PROMPT = """Review the proposed movement action on the frozen lake.

Current Grid State:
{observation}

Symbols: _ Frozen | O Hole | G Goal | P Player

Proposed Action: {action}

Is this movement safe and helpful? Consider:
- Will this movement lead into a hole (O)?
- Does this movement progress toward the goal (G)?
- Is there a safer or more efficient path?

Respond with:
- GOOD: <brief reason> if the action is safe and helpful
- BAD: <brief explanation> if there's a problem (e.g., leads to hole, wrong direction)"""

FROZENLAKE_REFINE_PROMPT = """The proposed movement was rejected. Please suggest a better action.

Current Grid State:
{observation}

Symbols: _ Frozen | O Hole | G Goal | P Player

Rejected Action: {action}
Feedback: {feedback}

Propose a better movement action. Think about:
1. Where is the player (P)?
2. Where is the goal (G)?
3. Where are the holes (O) to avoid?
4. What is the safest path to the goal?

Output the action as one of: ```Up```, ```Down```, ```Left```, ```Right```"""


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

    # Get workflow parameters from config
    max_steps = config.rllm.get("agent", {}).get("max_steps", 20)
    max_refine_iterations = config.rllm.get("workflow", {}).get("max_refine_iterations", 2)

    trainer = AgentTrainer(
        workflow_class=EnvEvaluatorOptimizerWorkflow,
        workflow_args={
            "agent_cls": FrozenLakeAgent,
            "env_cls": FrozenLakeEnv,
            "agent_args": agent_args,
            "env_args": env_args,
            "max_steps": max_steps,
            "max_refine_iterations": max_refine_iterations,
            "eval_prompt_template": FROZENLAKE_EVAL_PROMPT,
            "refine_prompt_template": FROZENLAKE_REFINE_PROMPT,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
