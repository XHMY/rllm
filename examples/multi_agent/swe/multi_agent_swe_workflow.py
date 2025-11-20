"""
Multi-Agent SWE Workflow

This workflow implements a multi-agent approach to software engineering tasks:
1. IssueAnalyzer: Analyzes the bug/feature request and creates a plan
2. CodeWriter: Implements the changes based on the plan
3. TestValidator: Validates changes by running tests

Pattern: Analyzer → Writer → Validator (iterative refinement loop)
"""

import asyncio
import re
from typing import Any

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.workflows.workflow import Workflow


class IssueAnalyzer:
    """Agent that analyzes issues and creates implementation plans."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def analyze_issue(self, issue_description: str, codebase_context: str = "") -> Trajectory:
        """Analyze the issue and create an implementation plan."""
        messages = [
            {
                "role": "user",
                "content": self._create_analysis_prompt(issue_description, codebase_context),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="analyzer")

        return Trajectory(
            name="analyzer",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_plan(output.content),
                    model_output=output,
                )
            ],
        )

    def _create_analysis_prompt(self, issue_description: str, codebase_context: str) -> str:
        """Create prompt for issue analysis."""
        prompt = f"""You are a software engineering expert. Analyze the following issue and create a detailed implementation plan.

Issue Description:
{issue_description}
"""
        if codebase_context:
            prompt += f"\nCodebase Context:\n{codebase_context}\n"

        prompt += """
Provide:
1. Root cause analysis - what is the actual problem?
2. Affected components - which files/modules need changes?
3. Implementation plan - step-by-step approach to fix
4. Potential risks - what could go wrong?

Wrap your plan in <plan>...</plan> tags.
"""
        return prompt

    def _parse_plan(self, response: str) -> dict:
        """Parse implementation plan from response."""
        plan_match = re.search(r"<plan>(.*?)</plan>", response, re.IGNORECASE | re.DOTALL)
        if plan_match:
            plan = plan_match.group(1).strip()
        else:
            plan = response

        return {"plan": plan, "full_response": response}


class CodeWriter:
    """Agent that implements code changes based on the plan."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def write_code(self, issue_description: str, plan: str, previous_attempt: str = None) -> Trajectory:
        """Generate code changes based on the implementation plan."""
        messages = [
            {
                "role": "user",
                "content": self._create_writing_prompt(issue_description, plan, previous_attempt),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="writer")

        return Trajectory(
            name="writer",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_code_changes(output.content),
                    model_output=output,
                )
            ],
        )

    def _create_writing_prompt(self, issue_description: str, plan: str, previous_attempt: str = None) -> str:
        """Create prompt for code writing."""
        prompt = f"""You are an expert software engineer. Implement the following changes:

Issue:
{issue_description}

Implementation Plan:
{plan}
"""
        if previous_attempt:
            prompt += f"""
Previous Attempt (that failed tests):
{previous_attempt}

Please fix the issues and provide an improved implementation.
"""

        prompt += """
Provide the complete code changes needed. Use the format:
<function=function_name>
<parameter=param_name>value</parameter>
</function>

Or provide the changes in a structured format that can be applied.
"""
        return prompt

    def _parse_code_changes(self, response: str) -> dict:
        """Parse code changes from response."""
        # Try to extract function calls (SWE-style format)
        function_pattern = r"<function=(.*?)>(.*?)</function>"
        matches = re.findall(function_pattern, response, re.DOTALL)

        if matches:
            changes = []
            for func_name, params in matches:
                changes.append({"function": func_name.strip(), "parameters": params.strip()})
            return {"changes": changes, "raw_response": response}
        else:
            return {"changes": [], "raw_response": response}


class TestValidator:
    """Agent that validates code changes by analyzing test results."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def validate_changes(self, issue_description: str, code_changes: str, test_results: dict) -> Trajectory:
        """Validate changes and provide feedback on test results."""
        messages = [
            {
                "role": "user",
                "content": self._create_validation_prompt(issue_description, code_changes, test_results),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="validator")

        return Trajectory(
            name="validator",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_validation(output.content, test_results),
                    model_output=output,
                )
            ],
        )

    def _create_validation_prompt(self, issue_description: str, code_changes: str, test_results: dict) -> str:
        """Create prompt for validation."""
        prompt = f"""You are a software testing expert. Validate the following code changes:

Original Issue:
{issue_description}

Code Changes:
{code_changes}

Test Results:
"""
        if test_results.get("passed", False):
            prompt += "✓ All tests PASSED\n"
        else:
            prompt += "✗ Tests FAILED\n"
            if "error" in test_results:
                prompt += f"Error: {test_results['error']}\n"
            if "failed_tests" in test_results:
                prompt += f"Failed Tests: {test_results['failed_tests']}\n"

        prompt += """
Provide:
1. Whether the fix successfully addresses the original issue
2. Any remaining problems or edge cases
3. Suggestions for improvement if tests failed

Wrap your verdict in <verdict>PASS</verdict> or <verdict>FAIL</verdict> tags.
"""
        return prompt

    def _parse_validation(self, response: str, test_results: dict) -> dict:
        """Parse validation result from response."""
        verdict_match = re.search(r"<verdict>(PASS|FAIL)</verdict>", response, re.IGNORECASE)
        if verdict_match:
            verdict = verdict_match.group(1).upper()
        else:
            # Fallback to test results
            verdict = "PASS" if test_results.get("passed", False) else "FAIL"

        return {
            "verdict": verdict,
            "passed": verdict == "PASS",
            "feedback": response,
            "test_passed": test_results.get("passed", False),
        }


class MultiAgentSWEWorkflow(Workflow):
    """Multi-agent workflow for software engineering tasks with iterative refinement."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        swe_env=None,
        max_refinement_iterations: int = 3,
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.swe_env = swe_env
        self.max_refinement_iterations = max_refinement_iterations

        # Initialize agents
        self.analyzer = IssueAnalyzer(rollout_engine)
        self.writer = CodeWriter(rollout_engine)
        self.validator = TestValidator(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute multi-agent SWE workflow.

        Flow:
        1. IssueAnalyzer analyzes the issue and creates a plan
        2. CodeWriter implements changes based on the plan
        3. TestValidator runs tests and validates
        4. If tests fail: loop back to CodeWriter with feedback
        5. Repeat until tests pass or max iterations reached

        Args:
            task: Dictionary with issue description and repository info
            uid: Unique identifier for this episode

        Returns:
            Episode with all trajectories and metrics
        """
        self.reset(task, uid)

        issue_description = task.get("problem_statement", task.get("issue", ""))
        codebase_context = task.get("codebase_context", "")
        all_trajectories = []

        # Step 1: Analyzer analyzes the issue
        analyzer_trajectory = await self.analyzer.analyze_issue(issue_description, codebase_context)
        plan = analyzer_trajectory.steps[0].action["plan"]

        # Reward analyzer (simplified: based on plan quality, default 0.5)
        analyzer_trajectory.steps[0].reward = 0.5
        all_trajectories.append(analyzer_trajectory)

        # Track metrics
        writer_attempts = 0
        validator_checks = 0
        tests_passed = False
        previous_code = None

        # Iterative refinement loop
        iteration = 0
        while not tests_passed and iteration < self.max_refinement_iterations:
            # Step 2: Writer implements changes
            writer_trajectory = await self.writer.write_code(issue_description, plan, previous_code)
            code_changes = writer_trajectory.steps[0].action["raw_response"]
            previous_code = code_changes
            writer_attempts += 1

            # Simulate test execution (in real scenario, this would interact with SWE environment)
            test_results = self._execute_tests(task, code_changes)

            # Reward writer based on test results
            writer_trajectory.steps[0].reward = 1.0 if test_results.get("passed", False) else 0.0
            all_trajectories.append(writer_trajectory)

            # Step 3: Validator validates the changes
            validator_trajectory = await self.validator.validate_changes(issue_description, code_changes, test_results)
            validation_result = validator_trajectory.steps[0].action

            # Reward validator based on correct assessment
            validator_correct = validation_result["passed"] == test_results.get("passed", False)
            validator_trajectory.steps[0].reward = 1.0 if validator_correct else 0.0
            validator_checks += 1
            all_trajectories.append(validator_trajectory)

            tests_passed = test_results.get("passed", False)
            iteration += 1

        # Final metrics
        metrics = {
            "analyzer_runs": 1,
            "writer_attempts": writer_attempts,
            "validator_checks": validator_checks,
            "total_iterations": iteration,
            "final_success": int(tests_passed),
            "tests_passed": tests_passed,
        }

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=tests_passed,
            metrics=metrics,
        )

    def _execute_tests(self, task: dict, code_changes: str) -> dict:
        """
        Execute tests for the code changes.

        In a real implementation, this would interact with the SWE environment
        to apply changes and run tests. For now, it's a placeholder.

        Args:
            task: Task dictionary
            code_changes: Code changes to test

        Returns:
            Dictionary with test results
        """
        # Placeholder: In real scenario, use self.swe_env to execute tests
        # For now, return a mock result
        return {
            "passed": False,  # Would be determined by actual test execution
            "error": "Mock test execution - integrate with SWE environment",
            "failed_tests": [],
        }
