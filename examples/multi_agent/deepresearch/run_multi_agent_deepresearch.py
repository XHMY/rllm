"""
Run Multi-Agent DeepResearch Workflow

This script demonstrates how to run the multi-agent research workflow
with planner, gatherer, and synthesizer agents.
"""

import asyncio
import os

from transformers import AutoTokenizer

from rllm.engine.workflow_execution_engine import WorkflowExecutionEngine
from rllm.utils import save_trajectories

# Import the multi-agent workflow
import sys

sys.path.append(os.path.dirname(__file__))
from multi_agent_deepresearch_workflow import MultiAgentDeepResearchWorkflow


def load_research_tasks(num_tasks: int = 10):
    """
    Create sample research tasks.

    In a real scenario, these would come from a dataset of research questions.
    """
    sample_tasks = [
        {
            "question": "What are the recent advancements in quantum computing and their potential applications?",
            "answer": "",  # No ground truth for open-ended research
        },
        {
            "question": "How has climate change affected global agriculture in the past decade?",
            "answer": "",
        },
        {
            "question": "What are the key differences between various COVID-19 vaccines and their efficacy rates?",
            "answer": "",
        },
        {
            "question": "What are the main causes and potential solutions for the global semiconductor shortage?",
            "answer": "",
        },
        {
            "question": "How do large language models work and what are their current limitations?",
            "answer": "",
        },
        {
            "question": "What are the latest developments in renewable energy storage technologies?",
            "answer": "",
        },
        {
            "question": "How has artificial intelligence impacted the healthcare industry?",
            "answer": "",
        },
        {
            "question": "What are the main challenges in developing autonomous vehicles?",
            "answer": "",
        },
        {
            "question": "What is the current state of fusion energy research and when might it be commercially viable?",
            "answer": "",
        },
        {
            "question": "How do blockchain and cryptocurrencies work, and what are their real-world applications?",
            "answer": "",
        },
    ]

    return sample_tasks[:num_tasks]


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 8  # Reduced due to multi-agent complexity and tool usage

    # For research tasks, using a more capable model
    model_name = "gpt-4o-mini"  # Or use local model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")  # For token counting

    # Note: Tool integration would require actual tool implementations
    # For now, using mock tools (see InformationGatherer._execute_tools)
    tools = {}  # In practice: {"Search": SearchTool(), "Scholar": ScholarTool(), ...}

    workflow_args = {
        "tools": tools,
        "max_subqueries": 5,  # Maximum number of sub-queries to explore
    }

    sampling_params = {"temperature": 0.7, "top_p": 0.95, "model": model_name}

    # Check for OpenAI API key if using OpenAI model
    if "gpt" in model_name.lower() and not os.getenv("OPENAI_API_KEY"):
        print("Warning: Using GPT model but OPENAI_API_KEY not set.")
        print("Please set OPENAI_API_KEY environment variable or use a local model.")
        print("Exiting...")
        sys.exit(1)

    rollout_engine_args = (
        {"base_url": "https://api.openai.com/v1", "api_key": os.getenv("OPENAI_API_KEY")}
        if "gpt" in model_name.lower()
        else {"base_url": "http://localhost:30000/v1", "api_key": "None"}
    )

    engine = WorkflowExecutionEngine(
        workflow_class=MultiAgentDeepResearchWorkflow,
        workflow_args=workflow_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args=rollout_engine_args,
        max_response_length=16384,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
    )

    # Load tasks
    tasks = load_research_tasks(num_tasks=5)  # Start with 5 tasks

    # Run multi-agent workflow
    print(f"Running multi-agent DeepResearch workflow on {len(tasks)} tasks...")
    print(f"Using {n_parallel_agents} parallel agents")
    print(f"Max sub-queries per question: {workflow_args['max_subqueries']}")

    results = asyncio.run(engine.execute_tasks(tasks))

    # Save results
    save_trajectories(results, filename="multi_agent_deepresearch_trajectories.pt")

    # Print detailed statistics
    print("\n" + "=" * 80)
    print("Multi-Agent DeepResearch Results")
    print("=" * 80)

    for i, episode in enumerate(results, 1):
        question = episode.task.get("question", "")[:80]
        num_subqueries = episode.metrics.get("num_subqueries", 0)
        num_tool_calls = episode.metrics.get("total_tool_calls", 0)
        answer_length = episode.metrics.get("answer_length", 0)

        print(f"\n{i}. {question}...")
        print(f"   Sub-queries: {num_subqueries}")
        print(f"   Tool calls: {num_tool_calls}")
        print(f"   Answer length: {answer_length} chars")

    # Aggregate metrics
    avg_subqueries = sum(ep.metrics.get("num_subqueries", 0) for ep in results) / len(results)
    avg_tool_calls = sum(ep.metrics.get("total_tool_calls", 0) for ep in results) / len(results)
    avg_answer_length = sum(ep.metrics.get("answer_length", 0) for ep in results) / len(results)

    print("\n" + "=" * 80)
    print("Aggregate Statistics")
    print("=" * 80)
    print(f"Average Sub-queries per Question: {avg_subqueries:.2f}")
    print(f"Average Tool Calls per Question: {avg_tool_calls:.2f}")
    print(f"Average Answer Length: {avg_answer_length:.0f} characters")
    print("=" * 80)
