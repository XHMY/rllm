"""
Run Multi-Agent Search Workflow

This script demonstrates how to run the multi-agent search workflow
with query optimizer, document retriever, and answer extractor agents.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.workflow_execution_engine import WorkflowExecutionEngine
from rllm.rewards.reward_fn import search_reward_fn
from rllm.utils import save_trajectories

# Import the multi-agent workflow
sys.path.append(os.path.dirname(__file__))
from multi_agent_search_workflow import MultiAgentSearchWorkflow


def load_search_data(train_size=3000, test_size=100):
    """
    Load search data, preparing it if not already available.
    Returns the test dataset data for evaluation.
    """
    test_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
    if test_dataset is None:
        print("Dataset not found, preparing search dataset...")
        sys.path.append("../../search")
        from prepare_hotpotqa_data import prepare_hotpotqa_data

        _, test_dataset = prepare_hotpotqa_data(train_size=train_size, test_size=test_size)

    return test_dataset.get_data()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if "RETRIEVAL_SERVER_URL" not in os.environ:
        os.environ["RETRIEVAL_SERVER_URL"] = "http://127.0.0.1:8000"

    load_dotenv()

    n_parallel_agents = 16  # Reduced due to multi-agent complexity

    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Import search tool
    try:
        sys.path.append("../../search")
        from local_retrieval_tool import LocalRetrievalTool

        search_tool = LocalRetrievalTool()
        print("✓ LocalRetrievalTool loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load LocalRetrievalTool: {e}")
        print("Using mock search instead")
        search_tool = None

    workflow_args = {
        "reward_function": search_reward_fn,
        "search_tool": search_tool,
        "max_refinement_iterations": 2,  # Allow up to 2 refinement cycles
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = WorkflowExecutionEngine(
        workflow_class=MultiAgentSearchWorkflow,
        workflow_args=workflow_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=16384,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
    )

    # Load dataset
    try:
        tasks = load_search_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please prepare the HotpotQA dataset first.")
        sys.exit(1)

    # Run multi-agent workflow
    print(f"\nRunning multi-agent search workflow on {len(tasks)} tasks...")
    print(f"Using {n_parallel_agents} parallel agents")
    print(f"Max refinement iterations: {workflow_args['max_refinement_iterations']}")

    results = asyncio.run(engine.execute_tasks(tasks))

    # Save results
    save_trajectories(results, filename="multi_agent_search_trajectories.pt")

    # Print detailed statistics
    print("\n" + "=" * 80)
    print("Multi-Agent Search Results")
    print("=" * 80)

    total_correct = sum(1 for ep in results if ep.is_correct)
    total_tasks = len(results)

    print(f"Overall Success Rate: {total_correct}/{total_tasks} ({100*total_correct/total_tasks:.1f}%)")

    # Aggregate metrics
    avg_queries = sum(ep.metrics.get("num_queries_generated", 0) for ep in results) / len(results)
    avg_searches = sum(ep.metrics.get("total_searches", 0) for ep in results) / len(results)
    avg_extractor_attempts = sum(ep.metrics.get("extractor_attempts", 0) for ep in results) / len(results)
    avg_iterations = sum(ep.metrics.get("total_iterations", 0) for ep in results) / len(results)

    print(f"\nAverage Queries Generated: {avg_queries:.2f}")
    print(f"Average Total Searches: {avg_searches:.2f}")
    print(f"Average Extractor Attempts: {avg_extractor_attempts:.2f}")
    print(f"Average Iterations: {avg_iterations:.2f}")

    # Show agent contribution
    first_try_success = sum(1 for ep in results if ep.is_correct and ep.metrics.get("extractor_attempts", 0) == 1)
    improved_by_refinement = sum(
        1 for ep in results if ep.is_correct and ep.metrics.get("extractor_attempts", 0) > 1
    )

    print(f"\nFirst Attempt Success: {first_try_success}")
    print(f"Improved by Refinement: {improved_by_refinement}")

    # Show sample results
    print("\n" + "=" * 80)
    print("Sample Results (first 5)")
    print("=" * 80)

    for i, episode in enumerate(results[:5], 1):
        question = episode.task.get("question", "")[:80]
        num_queries = episode.metrics.get("num_queries_generated", 0)
        num_searches = episode.metrics.get("total_searches", 0)
        is_correct = "✓" if episode.is_correct else "✗"

        print(f"\n{i}. {is_correct} {question}...")
        print(f"   Queries: {num_queries}, Searches: {num_searches}")

    print("=" * 80)
