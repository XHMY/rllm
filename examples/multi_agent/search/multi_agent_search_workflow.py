"""
Multi-Agent Search Workflow

This workflow implements a multi-agent approach to information retrieval and question answering:
1. QueryOptimizer: Analyzes question and generates optimized search queries
2. DocumentRetriever: Uses search tools to retrieve relevant documents
3. AnswerExtractor: Extracts and synthesizes the final answer from retrieved documents

Pattern: QueryOptimizer → DocumentRetriever → AnswerExtractor (with optional refinement)
"""

import asyncio
import re
from typing import Any, Dict, List

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow


class QueryOptimizer:
    """Agent that analyzes questions and generates optimized search queries."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def optimize_queries(self, question: str) -> Trajectory:
        """Analyze the question and generate optimized search queries."""
        messages = [
            {
                "role": "user",
                "content": self._create_optimization_prompt(question),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="query_optimizer")

        return Trajectory(
            name="query_optimizer",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action=self._parse_queries(output.content),
                    model_output=output,
                )
            ],
        )

    def _create_optimization_prompt(self, question: str) -> str:
        """Create prompt for query optimization."""
        prompt = f"""You are a search query optimization expert. Analyze the following question and generate optimized search queries.

Question:
{question}

Your task:
1. Identify key entities, concepts, and relationships in the question
2. Determine if this requires single-hop or multi-hop reasoning
3. Generate 2-4 optimized search queries that will help find the answer
4. Each query should be concise and focused on specific information needed

Format your queries as:
<query>First search query</query>
<query>Second search query</query>
...

Also provide a brief strategy in <strategy>...</strategy> tags explaining your search approach.
"""
        return prompt

    def _parse_queries(self, response: str) -> dict:
        """Parse optimized queries from response."""
        queries = re.findall(r"<query>(.*?)</query>", response, re.IGNORECASE | re.DOTALL)
        queries = [q.strip() for q in queries]

        strategy_match = re.search(r"<strategy>(.*?)</strategy>", response, re.IGNORECASE | re.DOTALL)
        strategy = strategy_match.group(1).strip() if strategy_match else ""

        return {"queries": queries, "strategy": strategy, "full_response": response}


class DocumentRetriever:
    """Agent that retrieves documents using search tools."""

    def __init__(self, rollout_engine: RolloutEngine, search_tool=None, **kwargs):
        self.rollout_engine = rollout_engine
        self.search_tool = search_tool

    async def retrieve_documents(self, question: str, query: str, context: str = "") -> Trajectory:
        """Retrieve documents for a specific query."""
        messages = [
            {
                "role": "user",
                "content": self._create_retrieval_prompt(question, query, context),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(
            messages, agent_name="document_retriever"
        )

        # Execute search if tool is available
        search_results = await self._execute_search(query)

        return Trajectory(
            name="document_retriever",
            steps=[
                Step(
                    chat_completions=messages
                    + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                    thought=output.reasoning,
                    action={
                        "query": query,
                        "search_results": search_results,
                        "summary": self._summarize_results(search_results),
                    },
                    model_output=output,
                )
            ],
        )

    def _create_retrieval_prompt(self, question: str, query: str, context: str = "") -> str:
        """Create prompt for document retrieval."""
        prompt = f"""You are a document retrieval expert. Execute a search and analyze the results.

Original Question:
{question}

Search Query:
{query}
"""
        if context:
            prompt += f"\nPrevious Context:\n{context}\n"

        prompt += """
After executing the search:
1. Identify the most relevant documents
2. Extract key information that helps answer the question
3. Note any gaps that require additional searches

Summarize your findings in <findings>...</findings> tags.
"""
        return prompt

    async def _execute_search(self, query: str) -> List[Dict]:
        """Execute search using the search tool."""
        if self.search_tool:
            try:
                # Execute the search tool
                results = await self.search_tool.execute(query=query)
                if isinstance(results, dict):
                    # Handle different result formats
                    if "documents" in results:
                        return results["documents"]
                    elif "results" in results:
                        return results["results"]
                    else:
                        return [results]
                elif isinstance(results, list):
                    return results
                else:
                    return [{"content": str(results)}]
            except Exception as e:
                return [{"error": str(e), "query": query}]
        else:
            # Mock search results
            return [
                {
                    "title": f"Mock result for: {query}",
                    "content": f"This is a mock search result for query: {query}",
                    "relevance": 0.8,
                }
            ]

    def _summarize_results(self, search_results: List[Dict]) -> str:
        """Summarize search results."""
        if not search_results:
            return "No results found."

        summary = "Search Results:\n"
        for i, result in enumerate(search_results[:5], 1):  # Limit to top 5
            if "error" in result:
                summary += f"{i}. Error: {result['error']}\n"
            else:
                title = result.get("title", "No title")
                content = result.get("content", "No content")[:200]
                summary += f"{i}. {title}\n   {content}...\n"

        return summary


class AnswerExtractor:
    """Agent that extracts and synthesizes the final answer."""

    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        self.rollout_engine = rollout_engine

    async def extract_answer(
        self, question: str, all_search_results: List[Dict], previous_attempt: str = None
    ) -> Trajectory:
        """Extract final answer from all gathered information."""
        messages = [
            {
                "role": "user",
                "content": self._create_extraction_prompt(question, all_search_results, previous_attempt),
            }
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="answer_extractor")

        return Trajectory(
            name="answer_extractor",
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

    def _create_extraction_prompt(
        self, question: str, all_search_results: List[Dict], previous_attempt: str = None
    ) -> str:
        """Create prompt for answer extraction."""
        prompt = f"""You are an expert at extracting answers from search results. Synthesize the information to answer the question.

Question:
{question}

Search Results:
"""
        for i, result_info in enumerate(all_search_results, 1):
            query = result_info.get("query", "N/A")
            summary = result_info.get("summary", "No information")
            prompt += f"\nQuery {i}: {query}\n{summary}\n"

        if previous_attempt:
            prompt += f"""
Previous Answer Attempt (incorrect):
{previous_attempt}

Please provide an improved answer based on all available information.
"""

        prompt += """
Provide:
1. A clear, concise answer to the question
2. Supporting evidence from the search results
3. Confidence level in your answer

Wrap your final answer in <answer>...</answer> tags.
"""
        return prompt

    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response."""
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        return response.strip()


class MultiAgentSearchWorkflow(Workflow):
    """Multi-agent workflow for search-based question answering."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        reward_function: RewardFunction = None,
        search_tool=None,
        max_refinement_iterations: int = 2,
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.reward_function = reward_function
        self.max_refinement_iterations = max_refinement_iterations

        # Initialize agents
        self.query_optimizer = QueryOptimizer(rollout_engine)
        self.document_retriever = DocumentRetriever(rollout_engine, search_tool=search_tool)
        self.answer_extractor = AnswerExtractor(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute multi-agent search workflow.

        Flow:
        1. QueryOptimizer generates optimized search queries
        2. DocumentRetriever retrieves documents for each query (parallel)
        3. AnswerExtractor synthesizes final answer from all results
        4. If incorrect and iterations remain: refine and retry

        Args:
            task: Dictionary with 'question' and 'answer' (ground truth)
            uid: Unique identifier for this episode

        Returns:
            Episode with all trajectories and metrics
        """
        self.reset(task, uid)

        question = task.get("question", "")
        ground_truth = task.get("answer", "")
        all_trajectories = []

        # Step 1: QueryOptimizer generates search queries
        optimizer_trajectory = await self.query_optimizer.optimize_queries(question)
        queries = optimizer_trajectory.steps[0].action["queries"]
        strategy = optimizer_trajectory.steps[0].action["strategy"]

        # Reward optimizer based on number of queries generated
        optimizer_trajectory.steps[0].reward = min(1.0, len(queries) / 3.0) if queries else 0.0
        all_trajectories.append(optimizer_trajectory)

        # Track metrics
        total_searches = 0
        answer_correct = False
        extractor_attempts = 0
        previous_answer = None

        # Iterative search and refinement loop
        iteration = 0
        while not answer_correct and iteration < self.max_refinement_iterations:
            # Step 2: DocumentRetriever retrieves documents for each query (in parallel)
            retrieval_tasks = [self.document_retriever.retrieve_documents(question, query) for query in queries]
            retrieval_trajectories = await asyncio.gather(*retrieval_tasks)

            search_results_summary = []
            for retrieval_traj in retrieval_trajectories:
                action = retrieval_traj.steps[0].action
                search_results = action["search_results"]

                # Reward retriever based on getting results
                has_results = len(search_results) > 0 and "error" not in search_results[0]
                retrieval_traj.steps[0].reward = 1.0 if has_results else 0.0

                search_results_summary.append(action)
                all_trajectories.append(retrieval_traj)
                total_searches += 1

            # Step 3: AnswerExtractor synthesizes the final answer
            extractor_trajectory = await self.answer_extractor.extract_answer(
                question, search_results_summary, previous_answer
            )
            final_answer = extractor_trajectory.steps[0].action
            extractor_attempts += 1

            # Evaluate the answer using reward function
            if self.reward_function:
                reward_result = self.reward_function(task, final_answer)
                extractor_trajectory.steps[0].reward = reward_result.reward
                answer_correct = reward_result.is_correct
            else:
                # Fallback: simple string matching
                answer_correct = ground_truth.lower() in final_answer.lower() if ground_truth else False
                extractor_trajectory.steps[0].reward = 1.0 if answer_correct else 0.0

            all_trajectories.append(extractor_trajectory)

            # If not correct and iterations remain, prepare for refinement
            if not answer_correct and iteration < self.max_refinement_iterations - 1:
                previous_answer = final_answer
                # Could potentially generate new queries here for refinement

            iteration += 1

        # Final metrics
        metrics = {
            "optimizer_runs": 1,
            "num_queries_generated": len(queries),
            "total_searches": total_searches,
            "extractor_attempts": extractor_attempts,
            "total_iterations": iteration,
            "final_success": int(answer_correct),
            "answer_correct": answer_correct,
        }

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=answer_correct,
            metrics=metrics,
        )
