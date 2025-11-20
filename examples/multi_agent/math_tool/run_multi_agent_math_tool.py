"""
Run Multi-Agent Math Tool Workflow

This script demonstrates how to run the multi-agent math problem solving workflow
with analyzer, executor, and verifier agents.
"""

import asyncio
import os

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.workflow_execution_engine import WorkflowExecutionEngine
from rllm.rewards.reward_fn import math_reward_fn
from rllm.tools.python_tool import PythonTool
from rllm.utils import compute_pass_at_k, save_trajectories

# Import the multi-agent workflow
import sys

sys.path.append(os.path.dirname(__file__))
from multi_agent_math_tool_workflow import MultiAgentMathToolWorkflow

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 16  # Reduced due to multi-agent complexity

    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize Python tool for code execution
    python_tool = PythonTool()

    workflow_args = {
        "reward_function": math_reward_fn,
        "python_tool": python_tool,
        "max_refinement_iterations": 3,  # Allow up to 3 refinement cycles
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = WorkflowExecutionEngine(
        workflow_class=MultiAgentMathToolWorkflow,
        workflow_args=workflow_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=16384,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )

    # Load dataset
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        sys.path.append("../../math_tool")
        from prepare_math_data import prepare_math_data

        _, test_dataset = prepare_math_data()

    tasks = test_dataset.get_data()

    # Run multi-agent workflow
    print(f"Running multi-agent math tool workflow on {len(tasks)} tasks...")
    print(f"Using {n_parallel_agents} parallel agents")
    print(f"Max refinement iterations: {workflow_args['max_refinement_iterations']}")

    results = asyncio.run(engine.execute_tasks(tasks))

    # Save results
    save_trajectories(results, filename="multi_agent_math_tool_trajectories.pt")

    # Compute pass@k
    compute_pass_at_k(results)

    # Print detailed statistics
    print("\n" + "=" * 80)
    print("Multi-Agent Math Tool Results")
    print("=" * 80)

    total_correct = sum(1 for ep in results if ep.is_correct)
    total_tasks = len(results)

    print(f"Overall Success Rate: {total_correct}/{total_tasks} ({100*total_correct/total_tasks:.1f}%)")

    # Aggregate metrics
    avg_iterations = sum(ep.metrics.get("total_iterations", 0) for ep in results) / len(results)
    avg_executor_attempts = sum(ep.metrics.get("executor_attempts", 0) for ep in results) / len(results)
    avg_verifier_checks = sum(ep.metrics.get("verifier_checks", 0) for ep in results) / len(results)

    print(f"\nAverage Iterations: {avg_iterations:.2f}")
    print(f"Average Executor Attempts: {avg_executor_attempts:.2f}")
    print(f"Average Verifier Checks: {avg_verifier_checks:.2f}")

    # Show agent contribution
    first_try_success = sum(1 for ep in results if ep.is_correct and ep.metrics.get("executor_attempts", 0) == 1)
    improved_by_refinement = sum(
        1 for ep in results if ep.is_correct and ep.metrics.get("executor_attempts", 0) > 1
    )

    print(f"\nFirst Attempt Success: {first_try_success}")
    print(f"Improved by Refinement: {improved_by_refinement}")

    print("=" * 80)
