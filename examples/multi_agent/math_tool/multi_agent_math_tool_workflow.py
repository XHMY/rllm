"""
Multi-Agent Math Tool Workflow

This workflow implements a multi-agent approach to mathematical problem solving with tools:
1. ProblemAnalyzer: Analyzes the problem and plans solution approach
2. CodeExecutor: Writes and executes Python code to solve the problem
3. AnswerVerifier: Validates the solution and checks correctness

Pattern: Analyzer → Executor → Verifier (iterative refinement loop)
"""

import asyncio
import re
from typing import Any

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow


class ProblemAnalyzer:
    """Agent that analyzes math problems and plans solution approach."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def analyze_problem(self, problem: str) -> Trajectory:
        """Analyze the math problem and create a solution plan."""
        messages = [
            {
                "role": "user",
                "content": self._create_analysis_prompt(problem),
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

    def _create_analysis_prompt(self, problem: str) -> str:
        """Create prompt for problem analysis."""
        prompt = f"""You are a mathematics expert. Analyze the following problem and create a solution plan:

Problem:
{problem}

Provide:
1. Problem type - what kind of math problem is this?
2. Key concepts - what mathematical concepts are needed?
3. Solution approach - step-by-step plan to solve
4. Python tools needed - what Python libraries or functions would help?

Wrap your plan in <plan>...</plan> tags.
"""
        return prompt

    def _parse_plan(self, response: str) -> dict:
        """Parse solution plan from response."""
        plan_match = re.search(r"<plan>(.*?)</plan>", response, re.IGNORECASE | re.DOTALL)
        if plan_match:
            plan = plan_match.group(1).strip()
        else:
            plan = response

        return {"plan": plan, "full_response": response}


class CodeExecutor:
    """Agent that writes and executes Python code to solve math problems."""

    def __init__(self, rollout_engine: RolloutEngine, python_tool=None, **kwargs):
        self.rollout_engine = rollout_engine
        self.python_tool = python_tool

    async def execute_solution(self, problem: str, plan: str, previous_attempt: dict = None) -> Trajectory:
        """Generate and execute Python code based on the solution plan."""
        messages = [
            {
                "role": "user",
                "content": self._create_execution_prompt(problem, plan, previous_attempt),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="executor")

        # Extract and execute code
        code = self._extract_code(output.content)
        execution_result = await self._execute_code(code)

        return Trajectory(
            name="executor",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action={
                        "code": code,
                        "execution_result": execution_result,
                        "answer": self._extract_answer(output.content, execution_result),
                    },
                    model_output=output,
                )
            ],
        )

    def _create_execution_prompt(self, problem: str, plan: str, previous_attempt: dict = None) -> str:
        """Create prompt for code execution."""
        prompt = f"""You are a Python coding expert specializing in mathematical computation.

Problem:
{problem}

Solution Plan:
{plan}
"""
        if previous_attempt:
            prompt += f"""
Previous Attempt (incorrect):
Code: {previous_attempt.get('code', 'N/A')}
Result: {previous_attempt.get('execution_result', 'N/A')}
Issue: {previous_attempt.get('issue', 'Answer was incorrect')}

Please fix the issues and provide an improved solution.
"""

        prompt += """
Write Python code to solve this problem. You can use:
- numpy, scipy, sympy for mathematical computations
- Standard library functions

Wrap your code in ```python and ``` tags.
After the code, provide your final answer in \\boxed{answer} format.
"""
        return prompt

    def _extract_code(self, response: str) -> str:
        """Extract Python code from markdown code blocks."""
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    async def _execute_code(self, code: str) -> dict:
        """Execute Python code and return results."""
        if not code:
            return {"success": False, "output": None, "error": "No code to execute"}

        if self.python_tool:
            # Use the actual Python tool to execute code
            try:
                result = await self.python_tool.execute(code)
                return {"success": True, "output": result, "error": None}
            except Exception as e:
                return {"success": False, "output": None, "error": str(e)}
        else:
            # Placeholder: mock execution
            return {"success": True, "output": "Mock execution result", "error": None}

    def _extract_answer(self, response: str, execution_result: dict) -> str:
        """Extract final answer from response."""
        # Try to find boxed answer
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Fallback to execution result
        if execution_result.get("success") and execution_result.get("output"):
            return str(execution_result["output"])

        return ""


class AnswerVerifier:
    """Agent that verifies the correctness of mathematical solutions."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def verify_solution(
        self, problem: str, code: str, execution_result: dict, proposed_answer: str
    ) -> Trajectory:
        """Verify the solution and check for correctness."""
        messages = [
            {
                "role": "user",
                "content": self._create_verification_prompt(problem, code, execution_result, proposed_answer),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="verifier")

        return Trajectory(
            name="verifier",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_verification(output.content),
                    model_output=output,
                )
            ],
        )

    def _create_verification_prompt(
        self, problem: str, code: str, execution_result: dict, proposed_answer: str
    ) -> str:
        """Create prompt for solution verification."""
        prompt = f"""You are a mathematics verification expert. Verify the following solution:

Problem:
{problem}

Code:
```python
{code}
```

Execution Result:
{execution_result}

Proposed Answer:
{proposed_answer}

Check:
1. Is the code logically correct?
2. Does the execution have errors?
3. Is the proposed answer reasonable for this problem?
4. Are there any calculation errors or edge cases missed?

Provide your verdict in <verdict>CORRECT</verdict> or <verdict>INCORRECT</verdict> tags.
If incorrect, explain what needs to be fixed in <feedback>...</feedback> tags.
"""
        return prompt

    def _parse_verification(self, response: str) -> dict:
        """Parse verification result from response."""
        verdict_match = re.search(r"<verdict>(CORRECT|INCORRECT)</verdict>", response, re.IGNORECASE)
        if verdict_match:
            verdict = verdict_match.group(1).upper()
        else:
            verdict = "UNKNOWN"

        feedback_match = re.search(r"<feedback>(.*?)</feedback>", response, re.IGNORECASE | re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else ""

        return {
            "verdict": verdict,
            "is_correct": verdict == "CORRECT",
            "feedback": feedback,
            "full_response": response,
        }


class MultiAgentMathToolWorkflow(Workflow):
    """Multi-agent workflow for mathematical problem solving with tools."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        reward_function: RewardFunction = None,
        python_tool=None,
        max_refinement_iterations: int = 3,
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.reward_function = reward_function
        self.max_refinement_iterations = max_refinement_iterations

        # Initialize agents
        self.analyzer = ProblemAnalyzer(rollout_engine)
        self.executor = CodeExecutor(rollout_engine, python_tool)
        self.verifier = AnswerVerifier(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute multi-agent math tool workflow.

        Flow:
        1. ProblemAnalyzer analyzes problem and creates plan
        2. CodeExecutor writes and executes Python code
        3. AnswerVerifier validates the solution
        4. If incorrect: loop back to CodeExecutor with feedback
        5. Repeat until correct or max iterations reached

        Args:
            task: Dictionary with 'question' and 'ground_truth'
            uid: Unique identifier for this episode

        Returns:
            Episode with all trajectories and metrics
        """
        self.reset(task, uid)

        problem = task.get("question", task.get("problem", ""))
        all_trajectories = []

        # Step 1: Analyzer analyzes the problem
        analyzer_trajectory = await self.analyzer.analyze_problem(problem)
        plan = analyzer_trajectory.steps[0].action["plan"]

        # Reward analyzer (simplified: default 0.5)
        analyzer_trajectory.steps[0].reward = 0.5
        all_trajectories.append(analyzer_trajectory)

        # Track metrics
        executor_attempts = 0
        verifier_checks = 0
        solution_correct = False
        previous_attempt = None

        # Iterative refinement loop
        iteration = 0
        while not solution_correct and iteration < self.max_refinement_iterations:
            # Step 2: Executor writes and executes code
            executor_trajectory = await self.executor.execute_solution(problem, plan, previous_attempt)
            executor_step = executor_trajectory.steps[0]
            code = executor_step.action["code"]
            execution_result = executor_step.action["execution_result"]
            proposed_answer = executor_step.action["answer"]
            executor_attempts += 1

            # Evaluate executor's answer using ground truth
            if self.reward_function:
                reward_result = self.reward_function(task, proposed_answer)
                executor_step.reward = reward_result.reward
                ground_truth_correct = reward_result.is_correct
            else:
                executor_step.reward = 0.0
                ground_truth_correct = False

            all_trajectories.append(executor_trajectory)

            # Step 3: Verifier validates the solution
            verifier_trajectory = await self.verifier.verify_solution(
                problem, code, execution_result, proposed_answer
            )
            verification_result = verifier_trajectory.steps[0].action

            # Reward verifier based on correct assessment
            verifier_correct = verification_result["is_correct"] == ground_truth_correct
            verifier_trajectory.steps[0].reward = 1.0 if verifier_correct else 0.0
            verifier_checks += 1
            all_trajectories.append(verifier_trajectory)

            # Check if solution is correct
            solution_correct = ground_truth_correct

            # Prepare for next iteration if needed
            if not solution_correct and iteration < self.max_refinement_iterations - 1:
                previous_attempt = {
                    "code": code,
                    "execution_result": execution_result,
                    "issue": verification_result.get("feedback", "Answer was incorrect"),
                }

            iteration += 1

        # Final metrics
        metrics = {
            "analyzer_runs": 1,
            "executor_attempts": executor_attempts,
            "verifier_checks": verifier_checks,
            "total_iterations": iteration,
            "final_success": int(solution_correct),
            "solution_correct": solution_correct,
        }

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=solution_correct,
            metrics=metrics,
        )
