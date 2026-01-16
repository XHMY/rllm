"""MiniWob training with multi-agent evaluator-optimizer workflow.

This script trains a MiniWob agent using the EnvEvaluatorOptimizerWorkflow,
which adds an evaluator-optimizer loop at each action step. This enables
iterative refinement of browser actions.

At each step:
1. Actor (MiniWobAgent) proposes an action
2. Evaluator reviews the action
3. If rejected, Actor refines based on feedback
4. Loop until approved or max iterations
5. Execute final action in BrowserGymEnv

This produces trajectories for 2 agents:
- actor: Primary agent that proposes, refines, and executes actions
- evaluator: Reviews proposed actions and provides feedback

Each agent can have its own LoRA adapter for specialized training.

This serves as Step 2 in the migration path:
- Step 0: Original agent-based training (train_miniwob_agent.py)
- Step 1: Workflow-based training (train_miniwob_workflow.py)
- Step 2: Multi-agent workflow training (this file)

Usage:
    python -m examples.miniwob.train_miniwob_evaluator_optimizer

Environment Variables:
    MINIWOB_URL: URL for MiniWob environment (required)
"""

import os

import hydra

from rllm.agents.miniwob_agent import MiniWobAgent
from rllm.data import DatasetRegistry
from rllm.environments.browsergym.browsergym import BrowserGymEnv
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.env_evaluator_optimizer_workflow import EnvEvaluatorOptimizerWorkflow


# Custom prompt templates for MiniWob
MINIWOB_EVAL_PROMPT = """Review the proposed browser action for the current page state.

Current Page State (Accessibility Tree):
{observation}

Proposed Action: {action}

Is this action appropriate for the current task? Consider:
- Does the action target the correct element (by element ID)?
- Is this the right type of action (click, type, scroll, etc.)?
- Will this help progress toward completing the task?

Respond with:
- GOOD: <brief reason> if the action is appropriate
- BAD: <brief explanation of the issue> if there's a problem"""

MINIWOB_REFINE_PROMPT = """The proposed browser action was rejected. Please suggest a better action.

Current Page State (Accessibility Tree):
{observation}

Rejected Action: {action}
Feedback: {feedback}

Based on the feedback, propose a better action. Think step by step about:
1. What element should be targeted?
2. What action should be taken (click, type, etc.)?
3. Why is this action better?

Output the action in the format: ```action_name("element_id")```"""


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

    # Get workflow parameters from config
    max_steps = config.rllm.get("agent", {}).get("max_steps", 10)
    max_refine_iterations = config.rllm.get("workflow", {}).get("max_refine_iterations", 2)

    trainer = AgentTrainer(
        workflow_class=EnvEvaluatorOptimizerWorkflow,
        workflow_args={
            "agent_cls": MiniWobAgent,
            "env_cls": BrowserGymEnv,
            "agent_args": agent_args,
            "env_args": env_args,
            "max_steps": max_steps,
            "max_refine_iterations": max_refine_iterations,
            "eval_prompt_template": MINIWOB_EVAL_PROMPT,
            "refine_prompt_template": MINIWOB_REFINE_PROMPT,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
