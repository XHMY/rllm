"""
Run Multi-Agent SWE Workflow

This script demonstrates how to run the multi-agent software engineering workflow
with analyzer, writer, and validator agents.
"""

import asyncio
import os

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.workflow_execution_engine import WorkflowExecutionEngine
from rllm.utils import compute_pass_at_k, save_trajectories

# Import the multi-agent workflow
import sys

sys.path.append(os.path.dirname(__file__))
from multi_agent_swe_workflow import MultiAgentSWEWorkflow


def load_swe_data():
    """Load SWE-Bench dataset."""
    if DatasetRegistry.dataset_exists("SWE_Bench_Verified", "test"):
        test_dataset = DatasetRegistry.load_dataset("SWE_Bench_Verified", "test")
        return test_dataset.get_data()
    raise ValueError(
        "SWE_Bench_Verified dataset not found. Please run `python ../../swe/prepare_swe_data.py` to create the dataset."
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name = "agentica-org/DeepSWE-Preview"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = {"temperature": 1, "model": model_name}

    workflow_args = {
        "max_refinement_iterations": 3,  # Allow up to 3 refinement cycles
    }

    engine = WorkflowExecutionEngine(
        workflow_class=MultiAgentSWEWorkflow,
        workflow_args=workflow_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        n_parallel_agents=16,  # Reduced due to multi-agent complexity
        max_response_length=65536,
        max_prompt_length=4096,
    )

    # Load tasks
    try:
        tasks = load_swe_data()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please prepare the SWE-Bench dataset first.")
        sys.exit(1)

    # Run multi-agent workflow
    print(f"Running multi-agent SWE workflow on {len(tasks)} tasks...")
    print(f"Max refinement iterations: {workflow_args['max_refinement_iterations']}")

    results = asyncio.run(engine.execute_tasks(tasks))

    # Save results
    save_trajectories(results, filename="multi_agent_swe_trajectories.pt")

    # Compute and display results
    compute_pass_at_k(results)

    # Print detailed statistics
    print("\n" + "=" * 80)
    print("Multi-Agent SWE Results")
    print("=" * 80)

    total_correct = sum(1 for ep in results if ep.is_correct)
    total_tasks = len(results)

    print(f"Overall Success Rate: {total_correct}/{total_tasks} ({100*total_correct/total_tasks:.1f}%)")

    # Aggregate metrics
    avg_iterations = sum(ep.metrics.get("total_iterations", 0) for ep in results) / len(results)
    avg_writer_attempts = sum(ep.metrics.get("writer_attempts", 0) for ep in results) / len(results)
    avg_validator_checks = sum(ep.metrics.get("validator_checks", 0) for ep in results) / len(results)

    print(f"\nAverage Iterations: {avg_iterations:.2f}")
    print(f"Average Writer Attempts: {avg_writer_attempts:.2f}")
    print(f"Average Validator Checks: {avg_validator_checks:.2f}")

    print("=" * 80)
