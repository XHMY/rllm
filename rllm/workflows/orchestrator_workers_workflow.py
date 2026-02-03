"""Orchestrator-Workers Workflow.

This module provides an abstract base class for implementing orchestrator-workers
workflows. The pattern involves:
1. Orchestrator (Decomposition): Analyzes complex task, breaks into subtasks
2. Workers (Execution): Execute subtasks in parallel or sequential
3. Orchestrator (Synthesis): Synthesizes worker outputs into final response
"""

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.workflow import Workflow


@dataclass
class SubtaskResult:
    """Result from a worker executing a subtask.

    Attributes:
        subtask_id: Index of the subtask (0-indexed)
        subtask_description: Text description of the subtask
        response: Worker's response to the subtask
        success: Whether the worker completed successfully
        metadata: Additional subtask-specific data
    """

    subtask_id: int
    subtask_description: str
    response: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionResult:
    """Result from orchestrator's task decomposition.

    Attributes:
        subtasks: List of subtask descriptions
        strategy: Optional description of the decomposition strategy
        execution_mode: "parallel" or "sequential" execution of subtasks
        metadata: Additional decomposition-specific data
    """

    subtasks: List[str]
    strategy: str = ""
    execution_mode: str = "parallel"  # or "sequential"
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrchestratorWorkersWorkflow(Workflow):
    """Abstract base class for orchestrator-workers workflows.

    This workflow pattern uses two agents:
    - Orchestrator: Decomposes complex tasks and synthesizes results
    - Worker: Executes individual subtasks

    The workflow:
    1. Orchestrator analyzes the task and breaks it into subtasks
    2. Workers execute subtasks (in parallel or sequential)
    3. Orchestrator synthesizes worker outputs into final response

    Subclasses must implement:
    - build_decomposition_prompt(): Create prompt for task decomposition
    - build_worker_prompt(): Create prompt for worker to solve subtask
    - build_synthesis_prompt(): Create prompt to combine worker results
    - parse_decomposition(): Parse orchestrator response into DecompositionResult
    - compute_final_reward(): Calculate reward based on final answer

    Optional overrides:
    - extract_response(): Extract relevant content from model output
    - should_execute_parallel(): Determine execution mode
    - on_decomposition_complete(): Hook after decomposition
    - on_worker_complete(): Hook after each worker completes
    - on_synthesis_complete(): Hook after synthesis
    - compute_worker_reward(): Calculate per-worker reward (default: 0.0)
    - compute_decomposition_reward(): Calculate decomposition reward (default: 0.0)

    Example:
        class MathOrchestratorWorkflow(OrchestratorWorkersWorkflow):
            def build_decomposition_prompt(self, task):
                return f"Break down: {task['question']}"
            # ... implement other abstract methods ...
    """

    # Agent names (no underscores per CLAUDE.md conventions)
    ORCHESTRATOR_NAME = "orchestrator"
    WORKER_NAME = "worker"

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        max_subtasks: int = 5,
        default_execution_mode: str = "parallel",
        use_final_outcome_reward: bool = True,
        share_context_with_workers: bool = True,
        **kwargs,
    ):
        """Initialize the OrchestratorWorkersWorkflow.

        Args:
            rollout_engine: Engine for LLM inference
            max_subtasks: Maximum number of subtasks allowed
            default_execution_mode: Default execution mode ("parallel" or "sequential")
            use_final_outcome_reward: If True, assign the final outcome reward to
                ALL trajectories in the episode
            share_context_with_workers: Whether to share original task context with workers
            **kwargs: Additional arguments passed to parent Workflow
        """
        super().__init__(rollout_engine, **kwargs)
        self.max_subtasks = max_subtasks
        self.default_execution_mode = default_execution_mode
        self.use_final_outcome_reward = use_final_outcome_reward
        self.share_context_with_workers = share_context_with_workers

    # ===== Abstract methods that subclasses MUST implement =====

    @abstractmethod
    def build_decomposition_prompt(self, task: Dict[str, Any], max_subtasks: int) -> str:
        """Build the prompt for task decomposition.

        Args:
            task: Task dictionary containing problem information
            max_subtasks: Maximum number of subtasks allowed

        Returns:
            Formatted prompt string for the orchestrator to decompose the task
        """
        pass

    @abstractmethod
    def build_worker_prompt(
        self,
        task: Dict[str, Any],
        subtask: str,
        subtask_id: int,
        previous_results: List[SubtaskResult],
    ) -> str:
        """Build the prompt for a worker to solve a subtask.

        Args:
            task: Original task dictionary
            subtask: Description of the subtask to solve
            subtask_id: Index of this subtask (0-indexed)
            previous_results: Results from previous subtasks (for sequential mode)

        Returns:
            Formatted prompt string for the worker
        """
        pass

    @abstractmethod
    def build_synthesis_prompt(
        self,
        task: Dict[str, Any],
        decomposition: DecompositionResult,
        worker_results: List[SubtaskResult],
    ) -> str:
        """Build the prompt for synthesizing worker results.

        Args:
            task: Original task dictionary
            decomposition: The decomposition result from phase 1
            worker_results: All results from workers

        Returns:
            Formatted prompt string for the orchestrator to synthesize
        """
        pass

    @abstractmethod
    def parse_decomposition(self, orchestrator_response: str) -> DecompositionResult:
        """Parse orchestrator response into structured decomposition result.

        Args:
            orchestrator_response: Raw text response from orchestrator

        Returns:
            DecompositionResult with subtasks and execution mode
        """
        pass

    @abstractmethod
    def compute_final_reward(
        self,
        task: Dict[str, Any],
        final_response: str,
    ) -> RewardOutput:
        """Compute reward based on the final synthesized response.

        Args:
            task: Task dictionary (may contain ground truth)
            final_response: Final synthesized response from orchestrator

        Returns:
            RewardOutput with reward value and metadata
        """
        pass

    # ===== Optional hooks for customization =====

    def extract_response(self, model_output) -> str:
        """Extract the relevant response content from model output.

        Default: returns model_output.content

        Override for custom extraction (e.g., parsing specific format).

        Args:
            model_output: ModelOutput from rollout engine

        Returns:
            Extracted response string
        """
        return model_output.content or model_output.text or ""

    def should_execute_parallel(
        self,
        decomposition: DecompositionResult,
        task: Dict[str, Any],
    ) -> bool:
        """Determine if subtasks should be executed in parallel.

        Default: uses decomposition.execution_mode or default_execution_mode

        Override for custom logic based on task properties.

        Args:
            decomposition: The decomposition result
            task: Task dictionary

        Returns:
            True for parallel execution, False for sequential
        """
        mode = decomposition.execution_mode or self.default_execution_mode
        return mode == "parallel"

    def on_decomposition_complete(
        self,
        decomposition: DecompositionResult,
        trajectory: Trajectory,
    ):
        """Hook called after task decomposition completes.

        Override to add custom logging, metrics, or state updates.

        Args:
            decomposition: The parsed decomposition result
            trajectory: Trajectory for the decomposition step
        """
        pass

    def on_worker_complete(
        self,
        subtask_id: int,
        result: SubtaskResult,
        trajectory: Trajectory,
    ):
        """Hook called after each worker completes.

        Override to add custom logging or intermediate processing.

        Args:
            subtask_id: Index of the completed subtask
            result: The worker's result
            trajectory: Trajectory for this worker step
        """
        pass

    def on_synthesis_complete(
        self,
        final_response: str,
        trajectory: Trajectory,
    ):
        """Hook called after synthesis completes.

        Override to add custom logging or post-processing.

        Args:
            final_response: The synthesized final response
            trajectory: Trajectory for the synthesis step
        """
        pass

    def compute_worker_reward(
        self,
        task: Dict[str, Any],
        subtask: str,
        response: str,
        subtask_id: int,
    ) -> RewardOutput:
        """Compute reward for a worker trajectory.

        Default: returns 0.0 reward (relies on final outcome reward).

        Override to provide intermediate feedback to workers.

        Args:
            task: Task dictionary
            subtask: The subtask description
            response: Worker's response
            subtask_id: Index of the subtask

        Returns:
            RewardOutput with reward value
        """
        return RewardOutput(reward=0.0, is_correct=False)

    def compute_decomposition_reward(
        self,
        task: Dict[str, Any],
        decomposition: DecompositionResult,
    ) -> RewardOutput:
        """Compute reward for the decomposition trajectory.

        Default: returns 0.0 reward (relies on final outcome reward).

        Override to provide feedback on decomposition quality.

        Args:
            task: Task dictionary
            decomposition: The parsed decomposition result

        Returns:
            RewardOutput with reward value
        """
        return RewardOutput(reward=0.0, is_correct=False)

    # ===== Core workflow implementation =====

    async def _execute_worker(
        self,
        task: Dict[str, Any],
        subtask: str,
        subtask_id: int,
        previous_results: List[SubtaskResult],
    ) -> tuple[SubtaskResult, Trajectory]:
        """Execute a single worker task.

        Args:
            task: Original task dictionary
            subtask: Description of the subtask
            subtask_id: Index of this subtask
            previous_results: Results from previous subtasks (for sequential mode)

        Returns:
            Tuple of (SubtaskResult, Trajectory)
        """
        prompt = self.build_worker_prompt(task, subtask, subtask_id, previous_results)
        messages = [{"role": "user", "content": prompt}]

        output = await self.rollout_engine.get_model_response(
            messages,
            agent_name=self.WORKER_NAME,
        )

        response = self.extract_response(output)
        reward = self.compute_worker_reward(task, subtask, response, subtask_id)

        result = SubtaskResult(
            subtask_id=subtask_id,
            subtask_description=subtask,
            response=response,
            success=True,
            metadata={"reward": reward.reward},
        )

        trajectory = Trajectory(
            name=self.WORKER_NAME,
            steps=[
                Step(
                    chat_completions=messages + [{
                        "role": "assistant",
                        "content": output.content,
                        "reasoning": output.reasoning,
                    }],
                    thought=output.reasoning,
                    action=response,
                    model_output=output,
                    reward=reward.reward,
                )
            ],
        )
        trajectory.reward = reward.reward

        return result, trajectory

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        """Execute the orchestrator-workers workflow.

        Args:
            task: Task dictionary containing problem information
            uid: Unique identifier for this episode

        Returns:
            Episode with all orchestrator and worker trajectories
        """
        self.reset(task, uid)

        all_trajectories = []

        # Phase 1: Orchestrator decomposes the task
        decomp_prompt = self.build_decomposition_prompt(task, self.max_subtasks)
        decomp_messages = [{"role": "user", "content": decomp_prompt}]

        decomp_output = await self.rollout_engine.get_model_response(
            decomp_messages,
            agent_name=self.ORCHESTRATOR_NAME,
        )

        decomposition = self.parse_decomposition(decomp_output.content)

        # Build decomposition trajectory (needed for both success and failure cases)
        decomp_trajectory = Trajectory(
            name=self.ORCHESTRATOR_NAME,
            steps=[
                Step(
                    chat_completions=decomp_messages + [{
                        "role": "assistant",
                        "content": decomp_output.content,
                        "reasoning": decomp_output.reasoning,
                    }],
                    thought=decomp_output.reasoning,
                    action={
                        "phase": "decomposition",
                        "subtasks": decomposition.subtasks,
                        "strategy": decomposition.strategy,
                        "execution_mode": decomposition.execution_mode,
                    },
                    model_output=decomp_output,
                    reward=0.0,  # Will be updated below
                )
            ],
        )

        # Check if orchestrator exceeded max_subtasks limit
        # This is a negative example - orchestrator failed to follow instructions
        if len(decomposition.subtasks) > self.max_subtasks:
            # Return episode with only decomposition trajectory and 0 reward
            decomp_trajectory.reward = 0.0
            for step in decomp_trajectory.steps:
                step.reward = 0.0

            return Episode(
                id=uid,
                task=task,
                trajectories=[decomp_trajectory],
                is_correct=False,
                metrics={
                    "exceeded_max_subtasks": 1,
                    "n_subtasks_generated": len(decomposition.subtasks),
                    "max_subtasks_allowed": self.max_subtasks,
                    "success": 0,
                },
            )

        decomp_reward = self.compute_decomposition_reward(task, decomposition)

        # Update decomposition trajectory with computed reward
        decomp_trajectory.reward = decomp_reward.reward
        for step in decomp_trajectory.steps:
            step.reward = decomp_reward.reward

        all_trajectories.append(decomp_trajectory)

        # Commit decomposition trajectory immediately to preserve it if later steps fail
        self.commit(trajectory=decomp_trajectory)

        # Hook for decomposition complete
        self.on_decomposition_complete(decomposition, decomp_trajectory)

        # Phase 2: Workers execute subtasks
        worker_results: List[SubtaskResult] = []
        worker_trajectories: List[Trajectory] = []

        if self.should_execute_parallel(decomposition, task):
            # Parallel execution using asyncio.gather()
            worker_tasks = [
                self._execute_worker(task, subtask, i, [])
                for i, subtask in enumerate(decomposition.subtasks)
            ]
            results_and_trajectories = await asyncio.gather(*worker_tasks)

            for i, (result, trajectory) in enumerate(results_and_trajectories):
                worker_results.append(result)
                worker_trajectories.append(trajectory)
                # Commit worker trajectory immediately
                self.commit(trajectory=trajectory)
                self.on_worker_complete(i, result, trajectory)
        else:
            # Sequential execution, passing previous results
            for i, subtask in enumerate(decomposition.subtasks):
                result, trajectory = await self._execute_worker(
                    task, subtask, i, worker_results
                )
                worker_results.append(result)
                worker_trajectories.append(trajectory)
                # Commit worker trajectory immediately
                self.commit(trajectory=trajectory)
                self.on_worker_complete(i, result, trajectory)

        all_trajectories.extend(worker_trajectories)

        # Phase 3: Orchestrator synthesizes results
        synth_prompt = self.build_synthesis_prompt(task, decomposition, worker_results)
        synth_messages = [{"role": "user", "content": synth_prompt}]

        synth_output = await self.rollout_engine.get_model_response(
            synth_messages,
            agent_name=self.ORCHESTRATOR_NAME,
        )

        final_response = self.extract_response(synth_output)
        final_reward = self.compute_final_reward(task, final_response)

        synth_trajectory = Trajectory(
            name=self.ORCHESTRATOR_NAME,
            steps=[
                Step(
                    chat_completions=synth_messages + [{
                        "role": "assistant",
                        "content": synth_output.content,
                        "reasoning": synth_output.reasoning,
                    }],
                    thought=synth_output.reasoning,
                    action={"phase": "synthesis", "final_response": final_response},
                    model_output=synth_output,
                    reward=final_reward.reward,
                )
            ],
        )
        synth_trajectory.reward = final_reward.reward
        all_trajectories.append(synth_trajectory)

        # Commit synthesis trajectory immediately
        self.commit(trajectory=synth_trajectory)

        # Hook for synthesis complete
        self.on_synthesis_complete(final_response, synth_trajectory)

        # Phase 4: Apply final outcome reward to ALL trajectories if enabled
        if self.use_final_outcome_reward:
            final_reward_value = final_reward.reward
            for trajectory in all_trajectories:
                trajectory.reward = final_reward_value
                for step in trajectory.steps:
                    step.reward = final_reward_value

        # Compute metrics
        metrics = self._compute_workflow_metrics(
            all_trajectories,
            decomposition,
            worker_results,
            final_reward.is_correct,
        )

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=final_reward.is_correct,
            metrics=metrics,
        )

    def _compute_workflow_metrics(
        self,
        trajectories: List[Trajectory],
        decomposition: DecompositionResult,
        worker_results: List[SubtaskResult],
        final_correct: bool,
    ) -> Dict[str, Any]:
        """Compute standard workflow metrics.

        Args:
            trajectories: All trajectories in the episode
            decomposition: The decomposition result
            worker_results: All worker results
            final_correct: Whether final response is correct

        Returns:
            Dictionary of metrics
        """
        n_subtasks = len(decomposition.subtasks)
        n_workers = len(worker_results)
        successful_workers = sum(1 for r in worker_results if r.success)

        # Count orchestrator calls (decomposition + synthesis = 2)
        orchestrator_calls = 2
        worker_calls = n_workers

        return {
            "n_subtasks": n_subtasks,
            "n_workers": n_workers,
            "successful_workers": successful_workers,
            "worker_success_rate": successful_workers / n_workers if n_workers > 0 else 0.0,
            f"{self.ORCHESTRATOR_NAME}_calls": orchestrator_calls,
            f"{self.WORKER_NAME}_calls": worker_calls,
            "success": int(final_correct),
            "total_trajectories": len(trajectories),
        }
