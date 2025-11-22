"""
Multi-Agent DeepResearch Workflow

This workflow implements a multi-agent approach to research tasks:
1. QueryPlanner: Breaks complex questions into sub-queries and creates research plan
2. InformationGatherer: Retrieves information using available tools (search, web, etc.)
3. AnswerSynthesizer: Combines findings into a coherent, comprehensive answer

Pattern: Planner → Gatherer(s) → Synthesizer
"""

import asyncio
import re
from typing import Any, Dict, List

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.workflows.workflow import Workflow


class QueryPlanner:
    """Agent that analyzes research questions and creates sub-queries."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def plan_research(self, question: str) -> Trajectory:
        """Analyze the question and create a research plan with sub-queries."""
        messages = [
            {
                "role": "user",
                "content": self._create_planning_prompt(question),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="planner")

        return Trajectory(
            name="planner",
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

    def _create_planning_prompt(self, question: str) -> str:
        """Create prompt for research planning."""
        prompt = f"""You are a research planning expert. Analyze the following question and create a comprehensive research plan.

Question:
{question}

Provide:
1. Question type - what kind of research is needed?
2. Key aspects - what are the main components to investigate?
3. Sub-queries - break down into 3-5 specific sub-questions that need to be answered
4. Information sources - what types of sources would be most useful?

Format your sub-queries as:
<subquery>Sub-question 1</subquery>
<subquery>Sub-question 2</subquery>
...

Wrap your complete plan in <plan>...</plan> tags.
"""
        return prompt

    def _parse_plan(self, response: str) -> dict:
        """Parse research plan and extract sub-queries."""
        plan_match = re.search(r"<plan>(.*?)</plan>", response, re.IGNORECASE | re.DOTALL)
        if plan_match:
            plan = plan_match.group(1).strip()
        else:
            plan = response

        # Extract sub-queries
        subqueries = re.findall(r"<subquery>(.*?)</subquery>", response, re.IGNORECASE | re.DOTALL)
        subqueries = [sq.strip() for sq in subqueries]

        return {"plan": plan, "subqueries": subqueries, "full_response": response}


class InformationGatherer:
    """Agent that gathers information using available tools."""

    def __init__(self, rollout_engine: RolloutEngine, tools: dict = None, **kwargs):
        self.rollout_engine = rollout_engine
        self.tools = tools or {}

    async def gather_information(self, subquery: str, tool_types: List[str] = None) -> Trajectory:
        """Gather information for a specific sub-query using available tools."""
        messages = [
            {
                "role": "user",
                "content": self._create_gathering_prompt(subquery, tool_types),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="gatherer")

        # Extract and execute tool calls
        tool_calls = self._extract_tool_calls(output.content)
        gathered_info = await self._execute_tools(tool_calls)

        return Trajectory(
            name="gatherer",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action={
                        "subquery": subquery,
                        "tool_calls": tool_calls,
                        "gathered_info": gathered_info,
                        "summary": self._summarize_findings(gathered_info),
                    },
                    model_output=output,
                )
            ],
        )

    def _create_gathering_prompt(self, subquery: str, tool_types: List[str] = None) -> str:
        """Create prompt for information gathering."""
        prompt = f"""You are a research assistant. Gather information to answer the following sub-question:

Sub-question:
{subquery}

Available tools:
- Search: Web search for current information
- Scholar: Academic paper search
- Visit: Visit and analyze web pages
- Python: Execute Python code for calculations

Use tool calls in this format:
<tool_call>
{{"name": "Search", "arguments": {{"query": "your search query"}}}}
</tool_call>

Gather comprehensive information from multiple sources to fully answer the sub-question.
"""
        return prompt

    def _extract_tool_calls(self, response: str) -> List[Dict]:
        """Extract tool calls from response."""
        import json

        tool_calls = []
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        return tool_calls

    async def _execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results."""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})

            if tool_name in self.tools:
                try:
                    result = await self.tools[tool_name].execute(**arguments)
                    results.append({"tool": tool_name, "success": True, "result": result})
                except Exception as e:
                    results.append({"tool": tool_name, "success": False, "error": str(e)})
            else:
                # Mock execution for demonstration
                results.append(
                    {
                        "tool": tool_name,
                        "success": True,
                        "result": f"Mock result for {tool_name} with args {arguments}",
                    }
                )

        return results

    def _summarize_findings(self, gathered_info: List[Dict]) -> str:
        """Summarize the gathered information."""
        if not gathered_info:
            return "No information gathered."

        summary = "Gathered Information:\n"
        for i, info in enumerate(gathered_info, 1):
            tool = info.get("tool", "Unknown")
            if info.get("success"):
                result = info.get("result", "N/A")
                summary += f"{i}. {tool}: {result[:200]}...\n"
            else:
                summary += f"{i}. {tool}: Error - {info.get('error', 'Unknown error')}\n"

        return summary


class AnswerSynthesizer:
    """Agent that synthesizes gathered information into a final answer."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def synthesize_answer(
        self, original_question: str, research_plan: str, gathered_findings: List[Dict]
    ) -> Trajectory:
        """Synthesize final answer from all gathered information."""
        messages = [
            {
                "role": "user",
                "content": self._create_synthesis_prompt(original_question, research_plan, gathered_findings),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="synthesizer")

        return Trajectory(
            name="synthesizer",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._extract_final_answer(output.content),
                    model_output=output,
                )
            ],
        )

    def _create_synthesis_prompt(
        self, original_question: str, research_plan: str, gathered_findings: List[Dict]
    ) -> str:
        """Create prompt for answer synthesis."""
        prompt = f"""You are an expert research synthesizer. Combine all gathered information into a comprehensive answer.

Original Question:
{original_question}

Research Plan:
{research_plan}

Gathered Findings:
"""
        for i, finding in enumerate(gathered_findings, 1):
            subquery = finding.get("subquery", f"Sub-query {i}")
            summary = finding.get("summary", "No information")
            prompt += f"\n{i}. {subquery}\n   {summary}\n"

        prompt += """
Synthesize a comprehensive, well-structured answer that:
1. Directly answers the original question
2. Integrates information from all sub-queries
3. Provides citations or sources where applicable
4. Highlights any limitations or uncertainties

Wrap your final answer in <answer>...</answer> tags.
"""
        return prompt

    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response."""
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        return response.strip()


class MultiAgentDeepResearchWorkflow(Workflow):
    """Multi-agent workflow for comprehensive research tasks."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        tools: dict = None,
        max_subqueries: int = 5,
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.tools = tools or {}
        self.max_subqueries = max_subqueries

        # Initialize agents
        self.planner = QueryPlanner(rollout_engine)
        self.gatherer = InformationGatherer(rollout_engine, tools=self.tools)
        self.synthesizer = AnswerSynthesizer(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute multi-agent deep research workflow.

        Flow:
        1. QueryPlanner breaks down the question into sub-queries
        2. InformationGatherer gathers information for each sub-query (parallel)
        3. AnswerSynthesizer combines all findings into final answer

        Args:
            task: Dictionary with 'question' and optional 'answer'
            uid: Unique identifier for this episode

        Returns:
            Episode with all trajectories and metrics
        """
        self.reset(task, uid)

        question = task.get("question", task.get("query", ""))
        ground_truth = task.get("answer", "")
        all_trajectories = []

        # Step 1: Planner creates research plan
        planner_trajectory = await self.planner.plan_research(question)
        research_plan = planner_trajectory.steps[0].action["plan"]
        subqueries = planner_trajectory.steps[0].action["subqueries"][: self.max_subqueries]

        # Reward planner (simplified: based on number of subqueries generated)
        planner_trajectory.steps[0].reward = min(1.0, len(subqueries) / 3.0)
        all_trajectories.append(planner_trajectory)

        # Step 2: Gatherer collects information for each sub-query (in parallel)
        gatherer_tasks = [self.gatherer.gather_information(sq) for sq in subqueries]
        gatherer_trajectories = await asyncio.gather(*gatherer_tasks)

        gathered_findings = []
        for gatherer_traj in gatherer_trajectories:
            # Reward gatherer based on successful tool executions
            action = gatherer_traj.steps[0].action
            gathered_info = action["gathered_info"]
            success_rate = (
                sum(1 for info in gathered_info if info.get("success", False)) / len(gathered_info)
                if gathered_info
                else 0.0
            )
            gatherer_traj.steps[0].reward = success_rate

            gathered_findings.append(action)
            all_trajectories.append(gatherer_traj)

        # Step 3: Synthesizer creates final answer
        synthesizer_trajectory = await self.synthesizer.synthesize_answer(
            question, research_plan, gathered_findings
        )
        final_answer = synthesizer_trajectory.steps[0].action

        # Evaluate synthesizer (simplified: compare with ground truth if available)
        if ground_truth:
            # Simple substring matching (in practice, use more sophisticated evaluation)
            is_correct = ground_truth.lower() in final_answer.lower() or final_answer.lower() in ground_truth.lower()
            synthesizer_trajectory.steps[0].reward = 1.0 if is_correct else 0.0
        else:
            is_correct = False
            synthesizer_trajectory.steps[0].reward = 0.5  # Default reward when no ground truth

        all_trajectories.append(synthesizer_trajectory)

        # Final metrics
        metrics = {
            "planner_runs": 1,
            "num_subqueries": len(subqueries),
            "gatherer_runs": len(gatherer_trajectories),
            "synthesizer_runs": 1,
            "total_tool_calls": sum(len(f.get("tool_calls", [])) for f in gathered_findings),
            "answer_length": len(final_answer),
            "has_ground_truth": bool(ground_truth),
        }

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=is_correct,
            metrics=metrics,
        )
