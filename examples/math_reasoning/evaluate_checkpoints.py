"""Evaluate math reasoning checkpoints on the AIME 2025 dataset.

This script evaluates all math-related checkpoints using a simplified vLLM pipeline
with native data parallelism and dynamic LoRA loading.

Usage:
    python -m examples.math_reasoning.evaluate_checkpoints \
        --checkpoints-dir /path/to/checkpoints/rllm-workflow-MARL \
        --output-csv aime2025_eval_results.csv \
        --dataset aime2025
"""

import argparse
import asyncio
import csv
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from examples.math_reasoning.evaluator_optimizer_math_workflow import (
    EvaluatorOptimizerMathWorkflow,
)
from examples.math_reasoning.single_agent_math_workflow import SingleAgentMathWorkflow
from examples.math_reasoning.voting_math_workflow import VotingMathWorkflow
from rllm.agents.agent import Episode
from rllm.data.dataset import DatasetRegistry
from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.rewards.reward_fn import math_reward_fn


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CheckpointInfo:
    """Information about a discovered checkpoint."""

    experiment_name: str  # e.g., "voting-qwen3_0.6b-math"
    workflow_type: str  # "single_agent", "evaluator_optimizer", "voting"
    model_size: str  # "0.6b", "1.7b"
    base_model: str  # "Qwen/Qwen3-0.6B"
    checkpoint_step: int  # e.g., 100
    actor_path: str  # Full path to actor directory (contains lora_adapter or lora_adapter_{agent})
    share_policy: bool  # Whether share_policy mode was used


@dataclass
class EvalResult:
    """Evaluation result for a single checkpoint."""

    experiment_name: str
    checkpoint_step: int
    workflow_type: str
    model_size: str
    share_policy: bool
    accuracy: float
    num_correct: int
    num_total: int
    eval_duration_seconds: float
    problem_results: list[dict] = field(default_factory=list)

    def to_csv_row(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            "experiment_name": self.experiment_name,
            "checkpoint_step": self.checkpoint_step,
            "workflow_type": self.workflow_type,
            "model_size": self.model_size,
            "share_policy": self.share_policy,
            "accuracy": f"{self.accuracy:.4f}",
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "eval_duration_seconds": f"{self.eval_duration_seconds:.1f}",
        }


# ============================================================================
# Constants and Mappings
# ============================================================================

WORKFLOW_MAP = {
    "single_agent": SingleAgentMathWorkflow,
    "evaluator_optimizer": EvaluatorOptimizerMathWorkflow,
    "voting": VotingMathWorkflow,
}

MODEL_MAP = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
}

AGENT_NAMES_MAP = {
    "single_agent": ["generator"],
    "evaluator_optimizer": ["generator", "evaluator"],
    "voting": ["generator", "aggregator"],
}


# ============================================================================
# Checkpoint Discovery
# ============================================================================


def parse_experiment_name(experiment_name: str) -> dict | None:
    """Parse experiment name to extract workflow type, model size, and share_policy."""
    share_policy = "share_policy" in experiment_name

    workflow_types = ["single_agent", "evaluator_optimizer", "voting"]
    workflow_type = None
    for wt in workflow_types:
        if experiment_name.startswith(wt):
            workflow_type = wt
            break

    if workflow_type is None:
        return None

    model_size_match = re.search(r"qwen3_(\d+\.?\d*b)", experiment_name, re.IGNORECASE)
    if model_size_match:
        model_size = model_size_match.group(1).lower()
    else:
        return None

    base_model = MODEL_MAP.get(model_size)
    if base_model is None:
        return None

    return {
        "workflow_type": workflow_type,
        "model_size": model_size,
        "base_model": base_model,
        "share_policy": share_policy,
    }


def discover_checkpoints(
    checkpoints_dir: str,
    experiment_filter: str = None,
    step_filter: list[int] = None,
) -> list[CheckpointInfo]:
    """Discover all math checkpoints in the directory.

    LoRA directory structure:
    - share_policy=True: actor/lora_adapter/
    - share_policy=False (multi-agent): actor/lora_adapter_{agent_name}/
    """
    checkpoints = []
    checkpoints_path = Path(checkpoints_dir)

    if not checkpoints_path.exists():
        raise ValueError(f"Checkpoints directory not found: {checkpoints_dir}")

    for experiment_dir in sorted(checkpoints_path.iterdir()):
        if not experiment_dir.is_dir():
            continue

        experiment_name = experiment_dir.name

        if "math" not in experiment_name.lower():
            continue
        if "deepcoder" in experiment_name.lower():
            continue

        if experiment_filter and not re.search(experiment_filter, experiment_name):
            continue

        parsed = parse_experiment_name(experiment_name)
        if parsed is None:
            print(f"Warning: Could not parse experiment name: {experiment_name}")
            continue

        for step_dir in sorted(experiment_dir.iterdir()):
            if not step_dir.is_dir():
                continue

            step_match = re.match(r"global_step_(\d+)", step_dir.name)
            if not step_match:
                continue

            checkpoint_step = int(step_match.group(1))

            if step_filter and checkpoint_step not in step_filter:
                continue

            actor_path = step_dir / "actor"
            if not actor_path.exists():
                print(f"Warning: Actor directory not found: {actor_path}")
                continue

            # Check for LoRA adapter(s)
            if parsed["share_policy"]:
                # share_policy mode: single lora_adapter/ directory
                lora_adapter_path = actor_path / "lora_adapter"
                if not lora_adapter_path.exists():
                    print(f"Warning: LoRA adapter not found: {lora_adapter_path}")
                    continue
            else:
                # Multi-agent mode: lora_adapter_{agent}/ directories
                agent_names = AGENT_NAMES_MAP.get(parsed["workflow_type"], ["generator"])
                missing_adapters = [
                    agent for agent in agent_names
                    if not (actor_path / f"lora_adapter_{agent}").exists()
                ]
                if missing_adapters:
                    print(f"Warning: LoRA adapters not found for agents {missing_adapters} in: {actor_path}")
                    continue

            checkpoints.append(
                CheckpointInfo(
                    experiment_name=experiment_name,
                    workflow_type=parsed["workflow_type"],
                    model_size=parsed["model_size"],
                    base_model=parsed["base_model"],
                    checkpoint_step=checkpoint_step,
                    actor_path=str(actor_path),
                    share_policy=parsed["share_policy"],
                )
            )

    return checkpoints


def group_checkpoints_by_model(
    checkpoints: list[CheckpointInfo],
) -> dict[str, list[CheckpointInfo]]:
    """Group checkpoints by base model for efficient evaluation."""
    grouped = defaultdict(list)
    for checkpoint in checkpoints:
        grouped[checkpoint.base_model].append(checkpoint)
    return dict(grouped)


# ============================================================================
# vLLM Server Management
# ============================================================================


class VLLMServerManager:
    """Manages vLLM server lifecycle with LoRA support."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        port: int = 8000,
        max_loras: int = 8,
        max_lora_rank: int = 64,
        max_model_len: int = None,
    ):
        self.model = model
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.process = None
        self.loaded_loras: set[str] = set()
        self.config = {
            "tensor_parallel_size": tensor_parallel_size,
            "data_parallel_size": data_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_loras": max_loras,
            "max_lora_rank": max_lora_rank,
            "max_model_len": max_model_len,
        }

    def start(self):
        """Start vLLM server with LoRA support."""
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--enable-lora",
            "--max-loras", str(self.config["max_loras"]),
            "--max-lora-rank", str(self.config["max_lora_rank"]),
            "--tensor-parallel-size", str(self.config["tensor_parallel_size"]),
            "--gpu-memory-utilization", str(self.config["gpu_memory_utilization"]),
            "--port", str(self.port),
            "--trust-remote-code",
        ]

        # Add data parallel size if > 1
        if self.config["data_parallel_size"] > 1:
            cmd.extend(["--data-parallel-size", str(self.config["data_parallel_size"])])

        # Add max model length if specified
        if self.config["max_model_len"]:
            cmd.extend(["--max-model-len", str(self.config["max_model_len"])])

        print(f"Starting vLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=None,  # Output goes directly to terminal
            stderr=None,  # Errors go directly to terminal
            text=True,
        )
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: int = 600):
        """Wait for server to be ready."""
        print(f"Waiting for vLLM server to be ready (timeout: {timeout}s)...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"vLLM server ready at {self.base_url}")
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass

            # Check if process died
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server process died with exit code: {self.process.returncode}. "
                    "Check the terminal output above for details."
                )

            time.sleep(5)

        raise TimeoutError(f"vLLM server failed to start within {timeout}s")

    def load_lora(self, lora_name: str, lora_path: str):
        """Load LoRA adapter dynamically."""
        if lora_name in self.loaded_loras:
            print(f"LoRA adapter already loaded: {lora_name}")
            return

        response = requests.post(
            f"{self.base_url}/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to load LoRA {lora_name}: {response.text}")

        self.loaded_loras.add(lora_name)
        print(f"Loaded LoRA adapter: {lora_name} from {lora_path}")

    def unload_lora(self, lora_name: str):
        """Unload LoRA adapter."""
        if lora_name not in self.loaded_loras:
            return

        try:
            response = requests.post(
                f"{self.base_url}/v1/unload_lora_adapter",
                json={"lora_name": lora_name},
                timeout=30,
            )
            if response.status_code == 200:
                self.loaded_loras.discard(lora_name)
                print(f"Unloaded LoRA adapter: {lora_name}")
            else:
                print(f"Warning: Failed to unload LoRA {lora_name}: {response.text}")
        except Exception as e:
            print(f"Warning: Error unloading LoRA {lora_name}: {e}")

    def unload_all_loras(self):
        """Unload all LoRA adapters."""
        for lora_name in list(self.loaded_loras):
            self.unload_lora(lora_name)

    def stop(self):
        """Stop vLLM server."""
        if self.process:
            print("Stopping vLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            self.loaded_loras.clear()
            print("vLLM server stopped")


# ============================================================================
# LoRA-Aware OpenAI Engine
# ============================================================================


class LoRAOpenAIEngine(OpenAIEngine):
    """OpenAIEngine with LoRA adapter support via model parameter."""

    def __init__(
        self,
        base_model: str,
        tokenizer,
        base_url: str = "http://localhost:8000/v1",
        lora_names: dict[str, str] = None,
        **kwargs,
    ):
        # Remove parameters not supported by OpenAIEngine
        kwargs.pop("validate", None)

        super().__init__(
            model=base_model,
            tokenizer=tokenizer,
            base_url=base_url,
            api_key="EMPTY",  # vLLM doesn't need API key
            **kwargs,
        )
        self.base_model_name = base_model
        self.lora_names = lora_names or {}

    def set_lora_names(self, lora_names: dict[str, str]):
        """Update LoRA name mapping."""
        self.lora_names = lora_names

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """Override to use LoRA model name based on agent_name."""
        agent_name = kwargs.pop("agent_name", None)

        # Determine which model/lora to use
        if agent_name and agent_name in self.lora_names:
            # Use the LoRA adapter for this agent
            self.model = self.lora_names[agent_name]
        elif "default" in self.lora_names:
            # Use default LoRA
            self.model = self.lora_names["default"]
        else:
            # Use base model
            self.model = self.base_model_name

        return await super().get_model_response(messages, **kwargs)


# ============================================================================
# Evaluation Functions
# ============================================================================


async def evaluate_checkpoint(
    checkpoint: CheckpointInfo,
    server: VLLMServerManager,
    engine: LoRAOpenAIEngine,
    dataset: list[dict],
    n_parallel: int = 32,
) -> EvalResult:
    """Evaluate a single checkpoint on the dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint.experiment_name} step {checkpoint.checkpoint_step}")
    print(f"  Workflow: {checkpoint.workflow_type}")
    print(f"  Actor Path: {checkpoint.actor_path}")
    print(f"  Share Policy: {checkpoint.share_policy}")
    print(f"{'='*60}")

    start_time = time.time()

    # Load LoRA adapter(s) for this checkpoint
    # lora_prefix creates unique names for vLLM when loading multiple checkpoints
    lora_prefix = f"{checkpoint.experiment_name}_step{checkpoint.checkpoint_step}"

    if checkpoint.share_policy:
        # Single shared adapter at lora_adapter/
        lora_name = f"{lora_prefix}_shared"
        lora_path = os.path.join(checkpoint.actor_path, "lora_adapter")
        server.load_lora(lora_name, lora_path)
        # Map all agent names to the same shared adapter
        agent_names = AGENT_NAMES_MAP.get(checkpoint.workflow_type, ["generator"])
        lora_mapping = {agent: lora_name for agent in agent_names}
        engine.set_lora_names(lora_mapping)
    else:
        # Per-agent adapters at lora_adapter_{agent_name}/
        agent_names = AGENT_NAMES_MAP.get(checkpoint.workflow_type, ["generator"])
        lora_mapping = {}
        for agent_name in agent_names:
            lora_name = f"{lora_prefix}_{agent_name}"
            agent_lora_path = os.path.join(checkpoint.actor_path, f"lora_adapter_{agent_name}")

            if not os.path.exists(agent_lora_path):
                raise FileNotFoundError(
                    f"LoRA adapter not found for agent '{agent_name}' at: {agent_lora_path}"
                )

            server.load_lora(lora_name, agent_lora_path)
            lora_mapping[agent_name] = lora_name

        engine.set_lora_names(lora_mapping)

    # Get workflow class and kwargs
    workflow_cls = WORKFLOW_MAP.get(checkpoint.workflow_type)
    if workflow_cls is None:
        raise ValueError(f"Unknown workflow type: {checkpoint.workflow_type}")

    workflow_kwargs = {"reward_function": math_reward_fn}
    if checkpoint.workflow_type == "voting":
        workflow_kwargs["n_votes"] = 3
    if checkpoint.workflow_type == "evaluator_optimizer":
        workflow_kwargs["max_iterations"] = 3

    # Run parallel evaluation with semaphore
    semaphore = asyncio.Semaphore(n_parallel)
    progress_bar = tqdm(total=len(dataset), desc="Evaluating")

    async def evaluate_single(task: dict, uid: str) -> Episode | Exception:
        async with semaphore:
            try:
                workflow = workflow_cls(rollout_engine=engine, **workflow_kwargs)
                result = await workflow.run(task, uid)
                return result
            except Exception as e:
                print(f"Error evaluating task {uid}: {e}")
                return e
            finally:
                progress_bar.update(1)

    # Execute all tasks in parallel
    tasks = [
        evaluate_single(task, f"eval_{i}")
        for i, task in enumerate(dataset)
    ]
    episodes = await asyncio.gather(*tasks)
    progress_bar.close()

    # Collect results
    problem_results = []
    num_correct = 0

    for i, episode in enumerate(episodes):
        if isinstance(episode, Episode):
            is_correct = episode.is_correct
            metrics = episode.metrics
        else:
            is_correct = False
            metrics = {"error": str(episode)}

        problem_results.append({
            "uid": f"eval_{i}",
            "is_correct": is_correct,
            "metrics": metrics,
        })
        if is_correct:
            num_correct += 1

    eval_duration = time.time() - start_time
    accuracy = num_correct / len(dataset) if dataset else 0.0

    print(f"\nResults: {num_correct}/{len(dataset)} correct ({accuracy:.2%})")
    print(f"Duration: {eval_duration:.1f}s")

    return EvalResult(
        experiment_name=checkpoint.experiment_name,
        checkpoint_step=checkpoint.checkpoint_step,
        workflow_type=checkpoint.workflow_type,
        model_size=checkpoint.model_size,
        share_policy=checkpoint.share_policy,
        accuracy=accuracy,
        num_correct=num_correct,
        num_total=len(dataset),
        eval_duration_seconds=eval_duration,
        problem_results=problem_results,
    )


# ============================================================================
# CSV Output
# ============================================================================


def save_results_to_csv(results: list[EvalResult], output_path: str):
    """Save evaluation results to CSV file."""
    if not results:
        return

    fieldnames = [
        "experiment_name",
        "checkpoint_step",
        "workflow_type",
        "model_size",
        "share_policy",
        "accuracy",
        "num_correct",
        "num_total",
        "eval_duration_seconds",
    ]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())

    print(f"\nResults saved to: {output_path}")


# ============================================================================
# Main Evaluation Loop
# ============================================================================


def main(args):
    """Main evaluation function."""
    print("=" * 60)
    print("Math Checkpoint Evaluation (vLLM Pipeline)")
    print("=" * 60)

    # Discover checkpoints
    print(f"\nDiscovering checkpoints in: {args.checkpoints_dir}")
    step_filter = args.step_filter if args.step_filter else None
    checkpoints = discover_checkpoints(
        args.checkpoints_dir,
        experiment_filter=args.experiment_filter,
        step_filter=step_filter,
    )

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    for cp in checkpoints:
        print(f"  - {cp.experiment_name} step {cp.checkpoint_step}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = DatasetRegistry.load_dataset(args.dataset, "test")
    if dataset is None:
        raise ValueError(f"Dataset not found: {args.dataset}")

    # Convert dataset to list of dicts
    dataset_list = []
    for i in range(len(dataset)):
        task = dict(dataset[i])
        if "ground_truth" not in task and "final_answer" in task:
            task["ground_truth"] = task["final_answer"]
        elif "final_answer" not in task and "ground_truth" in task:
            task["final_answer"] = task["ground_truth"]
        dataset_list.append(task)

    print(f"Dataset size: {len(dataset_list)} problems")

    # Group checkpoints by model
    checkpoints_by_model = group_checkpoints_by_model(checkpoints)
    print(f"Models to evaluate: {list(checkpoints_by_model.keys())}")

    all_results = []
    server = None

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, cleaning up...")
        if server:
            server.stop()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Process each model group (restart server per model)
    for base_model, model_checkpoints in checkpoints_by_model.items():
        print(f"\n{'#'*60}")
        print(f"Processing model: {base_model}")
        print(f"Checkpoints to evaluate: {len(model_checkpoints)}")
        print(f"{'#'*60}")

        try:
            # Start vLLM server for this model
            server = VLLMServerManager(
                model=base_model,
                tensor_parallel_size=args.tensor_parallel,
                data_parallel_size=args.data_parallel,
                gpu_memory_utilization=args.gpu_memory_utilization,
                port=args.port,
                max_loras=args.max_loras,
                max_lora_rank=args.max_lora_rank,
                max_model_len=args.max_prompt_length + args.max_tokens,
            )
            server.start()

            # Create engine
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            engine = LoRAOpenAIEngine(
                base_model=base_model,
                tokenizer=tokenizer,
                base_url=f"http://localhost:{args.port}/v1",
                max_prompt_length=args.max_prompt_length,
                max_response_length=args.max_tokens,
                sampling_params={
                    "temperature": args.temperature,
                },
            )

            # Evaluate all checkpoints for this model
            for checkpoint in tqdm(model_checkpoints, desc="Evaluating checkpoints", total=len(model_checkpoints)):
                try:
                    result = asyncio.run(
                        evaluate_checkpoint(
                            checkpoint=checkpoint,
                            server=server,
                            engine=engine,
                            dataset=dataset_list,
                            n_parallel=args.n_parallel,
                        )
                    )
                    all_results.append(result)

                    # Save intermediate results
                    save_results_to_csv(all_results, args.output_csv)

                except Exception as e:
                    print(f"Error evaluating {checkpoint.experiment_name}: {e}")
                    import traceback
                    traceback.print_exc()

                # Unload LoRAs between checkpoints to manage memory
                server.unload_all_loras()

        finally:
            # Stop server before switching models
            if server:
                server.stop()
                server = None

    # Final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total checkpoints evaluated: {len(all_results)}")
    print(f"Results saved to: {args.output_csv}")

    if all_results:
        print("\nSummary:")
        print(f"{'Experiment':<50} {'Step':>6} {'Accuracy':>10}")
        print("-" * 70)
        for result in sorted(
            all_results, key=lambda x: (x.experiment_name, x.checkpoint_step)
        ):
            print(
                f"{result.experiment_name:<50} "
                f"{result.checkpoint_step:>6} "
                f"{result.accuracy:>10.2%}"
            )


# ============================================================================
# CLI Entry Point
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate math reasoning checkpoints using vLLM pipeline"
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Path to checkpoints directory",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="eval_results.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aime2025",
        help="Dataset name to evaluate on",
    )
    parser.add_argument(
        "--experiment-filter",
        type=str,
        default=None,
        help="Regex pattern to filter experiments",
    )
    parser.add_argument(
        "--step-filter",
        type=int,
        nargs="+",
        default=None,
        help="Only evaluate specific checkpoint steps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=16384,
        help="Max prompt length in tokens",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=32,
        help="Number of parallel workflow instances",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--data-parallel",
        type=int,
        default=1,
        help="Data parallel size for load balancing",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port",
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=5,
        help="Maximum number of LoRA adapters to load",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=64,
        help="Maximum LoRA rank",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
