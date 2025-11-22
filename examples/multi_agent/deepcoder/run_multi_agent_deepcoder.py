"""
Run Multi-Agent DeepCoder Workflow

This script demonstrates how to run the multi-agent competitive coding workflow
with generator, test runner, and refiner agents.
"""

import asyncio
import os
from datetime import datetime

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.workflow_execution_engine import WorkflowExecutionEngine
from rllm.rewards.reward_fn import code_reward_fn
from rllm.utils import save_trajectories

# Import the multi-agent workflow
import sys

sys.path.append(os.path.dirname(__file__))
from multi_agent_deepcoder_workflow import MultiAgentDeepCoderWorkflow

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 16  # Reduced from 64 due to multi-agent complexity

    model_name = "agentica-org/DeepCoder-14B-Preview"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reward_fn = code_reward_fn

    workflow_args = {
        "reward_function": reward_fn,
        "max_refinement_iterations": 3,  # Allow up to 3 refinement cycles
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = WorkflowExecutionEngine(
        workflow_class=MultiAgentDeepCoderWorkflow,
        workflow_args=workflow_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=65536,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
    )

    # Load dataset
    test_dataset = DatasetRegistry.load_dataset("deepcoder", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        # Import from the original deepcoder example
        sys.path.append("../../deepcoder")
        from prepare_deepcoder_data import prepare_deepcoder_data

        _, test_dataset = prepare_deepcoder_data()

    tasks = test_dataset.get_data()

    # Run multi-agent workflow
    print(f"Running multi-agent DeepCoder workflow on {len(tasks)} tasks...")
    print(f"Using {n_parallel_agents} parallel agents")
    print(f"Max refinement iterations: {workflow_args['max_refinement_iterations']}")

    results = asyncio.run(engine.execute_tasks(tasks))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_trajectories(results, filename=f"multi_agent_deepcoder_trajectories_{len(tasks)}_{timestamp}.pt")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Multi-Agent DeepCoder Results")
    print("=" * 80)

    total_correct = sum(1 for ep in results if ep.is_correct)
    total_tasks = len(results)

    print(f"Overall Success Rate: {total_correct}/{total_tasks} ({100*total_correct/total_tasks:.1f}%)")

    # Aggregate metrics
    avg_iterations = sum(ep.metrics.get("total_iterations", 0) for ep in results) / len(results)
    generator_success = sum(ep.metrics.get("generator_success", 0) for ep in results) / len(results)
    refiner_success = (
        sum(ep.metrics.get("refiner_success_rate", 0) for ep in results) / len(results) if results else 0
    )

    print(f"\nAverage Iterations: {avg_iterations:.2f}")
    print(f"Generator Success Rate: {100*generator_success:.1f}%")
    print(f"Refiner Success Rate: {100*refiner_success:.1f}%")

    # Show agent contribution breakdown
    improved_by_refiner = sum(
        1 for ep in results if ep.is_correct and ep.metrics.get("generator_success", 0) == 0
    )
    print(f"\nTasks Solved by Generator: {sum(ep.metrics.get('generator_success', 0) for ep in results)}")
    print(f"Tasks Improved by Refiner: {improved_by_refiner}")

    print("=" * 80)
