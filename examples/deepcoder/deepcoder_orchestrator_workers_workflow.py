"""Deepcoder Orchestrator-Workers Workflow.

This module provides a concrete implementation of OrchestratorWorkersWorkflow
for code generation where complex problems are decomposed into subtasks,
solved by workers, and synthesized into a final solution.
"""

import json
import re
from typing import Any, Dict, List

from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_fn import code_reward_fn
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.orchestrator_workers_workflow import (
    DecompositionResult,
    OrchestratorWorkersWorkflow,
    SubtaskResult,
)


class DeepcodeOrchestratorWorkersWorkflow(OrchestratorWorkersWorkflow):
    """Code-specific orchestrator-workers workflow.

    This workflow implements the orchestrator-workers pattern for code
    generation:

    1. Orchestrator: Decomposes coding problem into subtasks
    2. Workers: Solve each subtask in parallel, producing code snippets
    3. Synthesizer: Combines worker solutions into a final complete program

    Example:
        workflow = DeepcodeOrchestratorWorkersWorkflow(
            rollout_engine=engine,
            max_subtasks=3,
        )
        episode = await workflow.run(task, uid)
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        prompts: Dict[str, Any] = None,
        prompt_file: str = "examples/deepcoder/prompt.json",
        max_subtasks: int = 3,
        use_final_outcome_reward: bool = True,
        share_main_task_with_workers: bool = True,
        **kwargs,
    ):
        """Initialize the code orchestrator-workers workflow.

        Args:
            rollout_engine: Engine for LLM inference
            prompts: Optional pre-loaded prompt templates
            prompt_file: Path to JSON file with prompt templates
            max_subtasks: Maximum number of subtasks allowed (default: 3)
            use_final_outcome_reward: If True, assign final reward to all trajectories
            share_main_task_with_workers: Whether to share original task with workers
            **kwargs: Additional arguments passed to parent workflow
        """
        super().__init__(
            rollout_engine=rollout_engine,
            max_subtasks=max_subtasks,
            default_execution_mode="parallel",
            use_final_outcome_reward=use_final_outcome_reward,
            share_main_task_with_workers=share_main_task_with_workers,
            **kwargs,
        )
        self._prompts = prompts
        self._prompt_file = prompt_file

    @property
    def prompts(self) -> Dict[str, Any]:
        """Lazy load prompts from file if not provided."""
        if self._prompts is None:
            self._prompts = self._load_prompts(self._prompt_file)
        return self._prompts

    def _load_prompts(self, prompt_file: str) -> Dict[str, Any]:
        """Load prompt templates from JSON file."""
        with open(prompt_file, "r") as f:
            data = json.load(f)
        return data.get("deepcoder_orchestrator_workers_prompts", {})

    # ===== Required abstract method implementations =====

    def build_decomposition_prompt(self, task: Dict[str, Any], max_subtasks: int) -> str:
        """Build prompt for decomposing coding problem into subtasks."""
        problem = task["question"]
        template = self.prompts.get("orchestrator_decompose", {}).get(
            "template",
            "You are an expert programming coordinator. Your task is to analyze "
            "the following programming problem and break it down into at most "
            "{max_subtasks} simpler subtasks that can be solved independently.\n\n"
            "Problem:\n{problem}\n\n"
            "Instructions:\n"
            "1. Identify the key components or steps needed to solve this problem\n"
            "2. Break it down into at most {max_subtasks} independent subtasks\n"
            "3. Each subtask should describe a specific function or module to implement\n"
            "4. Format your response as:\n"
            "   SUBTASK 1: [description of first subtask]\n"
            "   SUBTASK 2: [description of second subtask]\n"
            "   (and so on, up to {max_subtasks} subtasks maximum)\n\n"
            "Decompose the problem now:",
        )
        return template.format(problem=problem, max_subtasks=max_subtasks)

    def build_worker_prompt(
        self,
        task: Dict[str, Any],
        subtask: str,
        subtask_id: int,
        previous_results: List[SubtaskResult],
    ) -> str:
        """Build prompt for a worker to solve a specific subtask."""
        problem = task["question"]

        template = self.prompts.get("worker_solve", {}).get(
            "template",
            "You are an expert competitive programmer working on a specific part "
            "of a larger problem.\n\n"
            "{context}\n\n"
            "Your assigned subtask: {subtask}\n\n"
            "Instructions:\n"
            "1. Solve this specific subtask step by step\n"
            "2. Write clean, efficient Python code\n"
            "3. Handle all edge cases\n"
            "4. Output your code in a markdown code block with ```python\n\n"
            "Provide your solution:",
        )

        if self.share_main_task_with_workers:
            full_context = f"Original problem: {problem}"
        else:
            full_context = ""

        return template.format(context=full_context, subtask=subtask)

    def build_synthesis_prompt(
        self,
        task: Dict[str, Any],
        decomposition: DecompositionResult,
        worker_results: List[SubtaskResult],
    ) -> str:
        """Build prompt for synthesizing worker solutions into final code."""
        problem = task["question"]

        solutions_text = ""
        for result in worker_results:
            solutions_text += (
                f"\n--- Subtask {result.subtask_id + 1} ---\n"
                f"Task: {result.subtask_description}\n"
                f"Solution: {result.response}\n"
            )

        template = self.prompts.get("orchestrator_synthesize", {}).get(
            "template",
            "You are an expert programming coordinator. Your task is to synthesize "
            "the solutions from multiple subtasks into a single, complete solution.\n\n"
            "Original problem:\n{problem}\n\n"
            "Subtask solutions:{solutions}\n\n"
            "Instructions:\n"
            "1. Review all subtask solutions\n"
            "2. Combine them into a single, complete, runnable Python program\n"
            "3. Ensure proper integration between components\n"
            "4. Handle all edge cases\n"
            "5. Output your final code in a markdown code block with ```python\n\n"
            "Provide the complete solution:",
        )

        return template.format(problem=problem, solutions=solutions_text)

    def parse_decomposition(self, orchestrator_response: str) -> DecompositionResult:
        """Parse orchestrator response to extract subtasks.

        Looks for patterns like:
        - SUBTASK 1: description
        - SUBTASK 2: description
        Or numbered lists:
        - 1. description
        - 2. description
        """
        subtasks = []

        # Try to parse SUBTASK N: pattern
        subtask_pattern = r"SUBTASK\s*(\d+)\s*:\s*(.+?)(?=SUBTASK\s*\d+\s*:|$)"
        matches = re.findall(subtask_pattern, orchestrator_response, re.IGNORECASE | re.DOTALL)

        if matches:
            sorted_matches = sorted(matches, key=lambda x: int(x[0]))
            subtasks = [match[1].strip() for match in sorted_matches]
        else:
            # Try numbered list pattern (1. description)
            numbered_pattern = r"^\s*(\d+)[.)]\s*(.+?)(?=^\s*\d+[.)]|\Z)"
            matches = re.findall(numbered_pattern, orchestrator_response, re.MULTILINE | re.DOTALL)

            if matches:
                sorted_matches = sorted(matches, key=lambda x: int(x[0]))
                subtasks = [match[1].strip() for match in sorted_matches]
            else:
                # Fallback: split by newlines and filter non-empty lines
                lines = [line.strip() for line in orchestrator_response.split('\n') if line.strip()]
                subtasks = [line for line in lines if len(line) > 10][:self.max_subtasks]

        # Ensure we have at least one subtask
        if not subtasks:
            subtasks = [orchestrator_response.strip()]

        # Limit to max_subtasks
        subtasks = subtasks[:self.max_subtasks]

        return DecompositionResult(
            subtasks=subtasks,
            strategy="code_decomposition",
            execution_mode="parallel",
            metadata={"raw_response": orchestrator_response},
        )

    def compute_final_reward(
        self,
        task: Dict[str, Any],
        final_response: str,
    ) -> RewardOutput:
        """Compute reward using test execution.

        The base class calls this synchronously, so we use code_reward_fn
        directly (not async).
        """
        return code_reward_fn(task, final_response)

    # ===== Optional customizations =====

    def extract_response(self, model_output) -> str:
        """Extract content from model output."""
        return model_output.content or model_output.text or ""

    def compute_worker_reward(
        self,
        task: Dict[str, Any],
        subtask: str,
        response: str,
        subtask_id: int,
    ) -> RewardOutput:
        """Compute reward for a worker trajectory.

        For code subtasks, we don't have ground truth for individual subtasks,
        so we return 0.0 and rely on final outcome reward.
        """
        return RewardOutput(reward=0.0, is_correct=False)

    def compute_decomposition_reward(
        self,
        task: Dict[str, Any],
        decomposition: DecompositionResult,
    ) -> RewardOutput:
        """Compute reward for the decomposition trajectory.

        Default returns 0.0 and relies on final outcome reward.
        """
        return RewardOutput(reward=0.0, is_correct=False)
