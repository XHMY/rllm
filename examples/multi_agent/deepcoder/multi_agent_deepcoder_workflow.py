"""
Multi-Agent DeepCoder Workflow

This workflow implements a multi-agent approach to competitive programming:
1. CodeGenerator: Creates initial code solution
2. TestRunner: Executes tests and identifies failures
3. CodeRefiner: Fixes bugs based on test feedback

Pattern: Generator → TestRunner → Refiner (iterative refinement loop)
"""

import asyncio
import re
from typing import Any

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow


class CodeGenerator:
    """Agent that generates initial code solutions."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def generate_code(self, problem: str) -> Trajectory:
        """Generate initial code solution for the problem."""
        messages = [
            {
                "role": "user",
                "content": f"{problem}\n\nPlease provide a complete solution in Python. "
                "Wrap your code in ```python and ``` tags.",
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="generator")

        return Trajectory(
            name="generator",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._extract_code(output.content),
                    model_output=output,
                )
            ],
        )

    def _extract_code(self, response: str) -> str:
        """Extract code from markdown code blocks."""
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()


class TestRunner:
    """Agent that analyzes test results and provides feedback."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def analyze_tests(self, problem: str, code: str, test_results: list[dict]) -> Trajectory:
        """Analyze test failures and provide diagnostic feedback."""
        messages = [{"role": "user", "content": self._create_test_analysis_prompt(problem, code, test_results)}]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="test_runner")

        return Trajectory(
            name="test_runner",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_analysis(output.content),
                    model_output=output,
                )
            ],
        )

    def _create_test_analysis_prompt(self, problem: str, code: str, test_results: list[dict]) -> str:
        """Create prompt for test analysis."""
        prompt = f"""You are a code testing expert. Analyze the following code and test results:

Problem:
{problem}

Code:
```python
{code}
```

Test Results:
"""
        for i, test in enumerate(test_results[:5]):  # Limit to first 5 failures
            if not test.get("passed", False):
                prompt += f"\nTest {i + 1} FAILED:\n"
                prompt += f"  Input: {test.get('input', 'N/A')}\n"
                prompt += f"  Expected: {test.get('expected', 'N/A')}\n"
                prompt += f"  Actual: {test.get('output', 'N/A')}\n"
                if test.get("error_message"):
                    prompt += f"  Error: {test['error_message']}\n"

        prompt += """
Provide a detailed analysis of:
1. What is causing the test failures?
2. What specific bugs or edge cases are not being handled?
3. What changes are needed to fix the code?

Wrap your analysis in <analysis>...</analysis> tags.
"""
        return prompt

    def _parse_analysis(self, response: str) -> dict:
        """Parse test analysis from response."""
        analysis_match = re.search(r"<analysis>(.*?)</analysis>", response, re.IGNORECASE | re.DOTALL)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        else:
            analysis = response

        return {"analysis": analysis, "full_response": response}


class CodeRefiner:
    """Agent that refines code based on test feedback."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def refine_code(
        self, problem: str, previous_code: str, test_analysis: str, test_results: list[dict]
    ) -> Trajectory:
        """Generate improved code based on test analysis."""
        messages = [
            {
                "role": "user",
                "content": self._create_refinement_prompt(problem, previous_code, test_analysis, test_results),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="refiner")

        return Trajectory(
            name="refiner",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._extract_code(output.content),
                    model_output=output,
                )
            ],
        )

    def _create_refinement_prompt(
        self, problem: str, previous_code: str, test_analysis: str, test_results: list[dict]
    ) -> str:
        """Create prompt for code refinement."""
        prompt = f"""You are an expert programmer. Refine the following code based on test analysis:

Problem:
{problem}

Previous Code:
```python
{previous_code}
```

Test Analysis:
{test_analysis}

Failed Tests Summary:
"""
        for i, test in enumerate(test_results[:3]):
            if not test.get("passed", False):
                prompt += f"\nTest {i + 1}: Input={test.get('input', 'N/A')}, Expected={test.get('expected', 'N/A')}, Got={test.get('output', 'N/A')}\n"

        prompt += """
Please provide an improved solution that fixes the identified issues.
Wrap your code in ```python and ``` tags.
"""
        return prompt

    def _extract_code(self, response: str) -> str:
        """Extract code from markdown code blocks."""
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()


class MultiAgentDeepCoderWorkflow(Workflow):
    """Multi-agent workflow for competitive coding with iterative refinement."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        reward_function: RewardFunction = None,
        max_refinement_iterations: int = 3,
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.reward_function = reward_function
        self.max_refinement_iterations = max_refinement_iterations

        # Initialize agents
        self.generator = CodeGenerator(rollout_engine)
        self.test_runner = TestRunner(rollout_engine)
        self.refiner = CodeRefiner(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute multi-agent competitive coding workflow.

        Flow:
        1. CodeGenerator creates initial solution
        2. Execute tests and check results
        3. If tests fail: TestRunner analyzes → CodeRefiner improves
        4. Repeat until tests pass or max iterations reached

        Args:
            task: Dictionary with 'question' and test cases
            uid: Unique identifier for this episode

        Returns:
            Episode with all trajectories and metrics
        """
        self.reset(task, uid)

        problem = task.get("question", task.get("prompt", ""))
        all_trajectories = []

        # Step 1: Generator creates initial solution
        generator_trajectory = await self.generator.generate_code(problem)
        current_code = generator_trajectory.steps[0].action

        # Evaluate initial solution
        reward_result = self._evaluate_code(task, current_code)
        generator_trajectory.steps[0].reward = reward_result["reward"]
        all_trajectories.append(generator_trajectory)

        # Track metrics
        generator_correct = int(reward_result["all_passed"])
        test_runner_analyses = 0
        refiner_attempts = 0
        refiner_correct = 0

        # Iterative refinement loop
        iteration = 0
        while not reward_result["all_passed"] and iteration < self.max_refinement_iterations:
            test_results = reward_result.get("test_results", [])

            # Step 2: TestRunner analyzes failures
            test_runner_trajectory = await self.test_runner.analyze_tests(problem, current_code, test_results)
            test_analysis = test_runner_trajectory.steps[0].action["analysis"]

            # Reward test runner based on quality of analysis (simplified: 0.5 for now)
            test_runner_trajectory.steps[0].reward = 0.5
            test_runner_analyses += 1
            all_trajectories.append(test_runner_trajectory)

            # Step 3: CodeRefiner improves solution
            refiner_trajectory = await self.refiner.refine_code(problem, current_code, test_analysis, test_results)
            current_code = refiner_trajectory.steps[0].action

            # Evaluate refined solution
            reward_result = self._evaluate_code(task, current_code)
            refiner_trajectory.steps[0].reward = reward_result["reward"]
            refiner_attempts += 1
            refiner_correct += int(reward_result["all_passed"])

            all_trajectories.append(refiner_trajectory)

            iteration += 1

        # Final metrics
        final_correct = reward_result["all_passed"]
        metrics = {
            "generator_success": generator_correct,
            "test_runner_analyses": test_runner_analyses,
            "refiner_attempts": refiner_attempts,
            "refiner_success_rate": refiner_correct / refiner_attempts if refiner_attempts > 0 else 0.0,
            "total_iterations": iteration,
            "final_success": int(final_correct),
            "pass_rate": reward_result.get("pass_rate", 0.0),
        }

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=final_correct,
            metrics=metrics,
        )

    def _evaluate_code(self, task: dict, code: str) -> dict:
        """
        Evaluate code against test cases.

        Args:
            task: Task dictionary with test cases
            code: Code to evaluate

        Returns:
            Dictionary with reward, test results, and pass rate
        """
        if self.reward_function:
            # Use the provided reward function
            result = self.reward_function(task, code)
            return {
                "reward": result.reward,
                "all_passed": result.is_correct,
                "test_results": getattr(result, "test_results", []),
                "pass_rate": result.reward,
            }
        else:
            # Fallback: simple evaluation
            return {"reward": 0.0, "all_passed": False, "test_results": [], "pass_rate": 0.0}
