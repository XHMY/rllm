"""Math-specific implementation of the Orchestrator-Workers workflow.

This module provides a concrete implementation of OrchestratorWorkersWorkflow
for mathematical problem solving where complex problems are decomposed into
subproblems, solved by workers, and synthesized into a final answer.
"""

import json
import re
from typing import Any, Dict, List

from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.orchestrator_workers_workflow import (
    DecompositionResult,
    OrchestratorWorkersWorkflow,
    SubtaskResult,
)


class OrchestratorWorkersMathWorkflow(OrchestratorWorkersWorkflow):
    """Math-specific orchestrator-workers workflow.

    This workflow implements the orchestrator-workers pattern for mathematical
    problem solving:

    1. Orchestrator: Decomposes complex math problem into 2-4 subproblems
    2. Workers: Solve each subproblem in parallel, outputting in \\boxed{} format
    3. Orchestrator: Synthesizes solutions into final answer in \\boxed{} format

    Example:
        workflow = OrchestratorWorkersMathWorkflow(
            rollout_engine=engine,
            reward_function=math_reward_fn,
            max_subtasks=4,
        )
        episode = await workflow.run(task, uid)
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        reward_function: RewardFunction,
        prompts: Dict[str, Any] = None,
        prompt_file: str = "examples/math_reasoning/prompt.json",
        max_subtasks: int = 4,
        use_final_outcome_reward: bool = True,
        share_context_with_workers: bool = True,
        **kwargs,
    ):
        """Initialize the math orchestrator-workers workflow.

        Args:
            rollout_engine: Engine for LLM inference
            reward_function: Function to compute rewards based on ground truth
            prompts: Optional pre-loaded prompt templates
            prompt_file: Path to JSON file with prompt templates
            max_subtasks: Maximum number of subtasks allowed (default: 4)
            use_final_outcome_reward: If True, assign final reward to all trajectories
            **kwargs: Additional arguments passed to parent workflow
        """
        super().__init__(
            rollout_engine=rollout_engine,
            max_subtasks=max_subtasks,
            default_execution_mode="parallel",
            use_final_outcome_reward=use_final_outcome_reward,
            share_context_with_workers=share_context_with_workers,
            **kwargs,
        )
        self.reward_function = reward_function
        self._prompts = prompts
        self._prompt_file = prompt_file

    @property
    def prompts(self) -> Dict[str, Any]:
        """Lazy load prompts from file if not provided."""
        if self._prompts is None:
            self._prompts = self._load_prompts(self._prompt_file)
        return self._prompts

    def _load_prompts(self, prompt_file: str) -> Dict[str, Any]:
        """Load prompt templates from JSON file.

        Args:
            prompt_file: Path to JSON file

        Returns:
            Dictionary of prompt templates
        """
        with open(prompt_file, "r") as f:
            data = json.load(f)
        return data.get("orchestrator_workers_prompts", {})

    # ===== Required abstract method implementations =====

    def build_decomposition_prompt(self, task: Dict[str, Any], max_subtasks: int) -> str:
        """Build prompt for decomposing math problem into subproblems.

        Args:
            task: Task dictionary with 'question' field
            max_subtasks: Maximum number of subtasks allowed

        Returns:
            Formatted prompt for the orchestrator to decompose the task
        """
        problem = task["question"]
        template = self.prompts.get("orchestrator_decompose", {}).get(
            "template",
            "You are a math problem-solving coordinator. Your task is to analyze "
            "the following complex math problem and break it down into at most "
            "{max_subtasks} simpler subproblems that can be solved independently.\n\n"
            "Problem: {problem}\n\n"
            "Instructions:\n"
            "1. Identify the key components or steps needed to solve this problem\n"
            "2. Break it down into at most {max_subtasks} independent subproblems\n"
            "3. Each subproblem should be self-contained and solvable on its own\n"
            "4. Format your response as:\n"
            "   SUBTASK 1: [description of first subproblem]\n"
            "   SUBTASK 2: [description of second subproblem]\n"
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
        """Build prompt for a worker to solve a specific subproblem.

        Args:
            task: Original task dictionary
            subtask: Description of the subproblem to solve
            subtask_id: Index of this subtask (0-indexed)
            previous_results: Results from previous subtasks (for sequential mode)

        Returns:
            Formatted prompt for the worker
        """
        problem = task["question"]

        # Build context from previous results if in sequential mode
        context = ""
        if previous_results:
            context = "\n\nPrevious subproblem solutions:\n"
            for prev in previous_results:
                context += f"- Subproblem {prev.subtask_id + 1}: {prev.response}\n"

        template = self.prompts.get("worker_solve", {}).get(
            "template",
            "You are a math problem solver working on a specific part of a larger "
            "problem.\n\n"
            "Original problem context: {context}\n\n"
            "Your assigned subproblem: {subtask}\n\n"
            "Instructions:\n"
            "1. Solve this specific subproblem step by step\n"
            "2. Show your work clearly\n"
            "3. Provide your answer in \\boxed{{}} format\n\n"
            "Solve the subproblem:",
        )

        # Include original problem in context
        full_context = f"Original problem: {problem}"
        if context:
            full_context += context

        return template.format(context=full_context, subtask=subtask)

    def build_synthesis_prompt(
        self,
        task: Dict[str, Any],
        decomposition: DecompositionResult,
        worker_results: List[SubtaskResult],
    ) -> str:
        """Build prompt for synthesizing worker solutions into final answer.

        Args:
            task: Original task dictionary
            decomposition: The decomposition result from phase 1
            worker_results: All results from workers

        Returns:
            Formatted prompt for the orchestrator to synthesize
        """
        problem = task["question"]

        # Build solutions section
        solutions_text = ""
        for result in worker_results:
            solutions_text += (
                f"\n--- Subproblem {result.subtask_id + 1} ---\n"
                f"Task: {result.subtask_description}\n"
                f"Solution: {result.response}\n"
            )

        template = self.prompts.get("orchestrator_synthesize", {}).get(
            "template",
            "You are a math problem-solving coordinator. Your task is to synthesize "
            "the solutions from multiple subproblems into a final answer.\n\n"
            "Original problem: {problem}\n\n"
            "Subproblem solutions:{solutions}\n\n"
            "Instructions:\n"
            "1. Review all subproblem solutions\n"
            "2. Combine them logically to solve the original problem\n"
            "3. Verify the final answer makes sense\n"
            "4. Provide the final answer in \\boxed{{}} format\n\n"
            "Synthesize the final answer:",
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

        Args:
            orchestrator_response: Raw text response from orchestrator

        Returns:
            DecompositionResult with extracted subtasks
        """
        subtasks = []

        # Try to parse SUBTASK N: pattern
        subtask_pattern = r"SUBTASK\s*(\d+)\s*:\s*(.+?)(?=SUBTASK\s*\d+\s*:|$)"
        matches = re.findall(subtask_pattern, orchestrator_response, re.IGNORECASE | re.DOTALL)

        if matches:
            # Sort by subtask number and extract descriptions
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
                # Take lines that look like task descriptions (longer than 10 chars)
                subtasks = [line for line in lines if len(line) > 10][:self.max_subtasks]

        # Ensure we have at least one subtask
        if not subtasks:
            subtasks = [orchestrator_response.strip()]

        # Limit to max_subtasks
        subtasks = subtasks[:self.max_subtasks]

        return DecompositionResult(
            subtasks=subtasks,
            strategy="math_decomposition",
            execution_mode="parallel",
            metadata={"raw_response": orchestrator_response},
        )

    def compute_final_reward(
        self,
        task: Dict[str, Any],
        final_response: str,
    ) -> RewardOutput:
        """Compute reward using ground truth math evaluation.

        Args:
            task: Task dictionary with ground truth answer
            final_response: Final synthesized response from orchestrator

        Returns:
            RewardOutput from reward function
        """
        # Ensure ground_truth is set for the reward function
        if "ground_truth" not in task:
            task["ground_truth"] = task.get("final_answer", "")
        return self.reward_function(task, final_response)

    # ===== Optional customizations =====

    def extract_response(self, model_output) -> str:
        """Extract content from model output.

        Args:
            model_output: ModelOutput from rollout engine

        Returns:
            Full content for context
        """
        return model_output.content or model_output.text or ""

    def compute_worker_reward(
        self,
        task: Dict[str, Any],
        subtask: str,
        response: str,
        subtask_id: int,
    ) -> RewardOutput:
        """Compute reward for a worker trajectory.

        For math problems, we don't have ground truth for subtasks,
        so we return 0.0 and rely on final outcome reward.

        Args:
            task: Task dictionary
            subtask: The subtask description
            response: Worker's response
            subtask_id: Index of the subtask

        Returns:
            RewardOutput with 0.0 reward (relies on final outcome)
        """
        return RewardOutput(reward=0.0, is_correct=False)

    def compute_decomposition_reward(
        self,
        task: Dict[str, Any],
        decomposition: DecompositionResult,
    ) -> RewardOutput:
        """Compute reward for the decomposition trajectory.

        Default returns 0.0 and relies on final outcome reward.

        Args:
            task: Task dictionary
            decomposition: The parsed decomposition result

        Returns:
            RewardOutput with 0.0 reward
        """
        return RewardOutput(reward=0.0, is_correct=False)
