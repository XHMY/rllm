"""Central task-type registry for math and deepcoder experiments."""

TASK_CONFIGS = {
    "math": {
        "workflow_map": {
            "single_agent": "examples.math_reasoning.single_agent_math_workflow.SingleAgentMathWorkflow",
            "evaluator_optimizer": "examples.math_reasoning.evaluator_optimizer_math_workflow.EvaluatorOptimizerMathWorkflow",
            "voting": "examples.math_reasoning.voting_math_workflow.VotingMathWorkflow",
            "orchestrator_workers": "examples.math_reasoning.orchestrator_workers_math_workflow.OrchestratorWorkersMathWorkflow",
        },
        "reward_fn": "rllm.rewards.reward_fn.math_reward_fn",
        "entry_points": {
            "evaluator_optimizer": "examples.math_reasoning.train_evaluator_optimizer_math",
            "voting": "examples.math_reasoning.train_voting_math",
            "orchestrator_workers": "examples.math_reasoning.train_orchestrator_workers_math",
            "single_agent": "examples.math_reasoning.train_single_agent_math",
        },
        "default_eval_dataset": "aime2025",
        "experiment_suffix": "math",
        "prompt_response_lengths": {
            "evaluator_optimizer": (30720, 5120),
            "voting": (20480, 5120),
            "orchestrator_workers": (20480, 3072),
            "single_agent": (15360, 5120),
        },
        "workflow_params": {
            "evaluator_optimizer": {
                "max_iterations": 3,
                "use_final_outcome_reward": True,
            },
            "voting": {
                "n_votes": 3,
                "use_final_outcome_reward": True,
            },
            "orchestrator_workers": {
                "max_subtasks": 3,
                "use_final_outcome_reward": True,
                "share_main_task_with_workers": False,
            },
            "single_agent": {},
        },
        "extra_sbatch_cmds": "",
        "experiment_filter_include": "math",
        "experiment_filter_exclude": "deepcoder",
    },
    "deepcoder": {
        "workflow_map": {
            "single_agent": "examples.deepcoder.single_agent_deepcoder_workflow.SingleAgentDeepcodeWorkflow",
            "evaluator_optimizer": "examples.deepcoder.deepcoder_evaluator_optimizer_workflow.DeepcodeEvaluatorOptimizerWorkflow",
            "voting": "examples.deepcoder.deepcoder_voting_workflow.DeepcodeVotingWorkflow",
            "orchestrator_workers": "examples.deepcoder.deepcoder_orchestrator_workers_workflow.DeepcodeOrchestratorWorkersWorkflow",
        },
        "reward_fn": "rllm.rewards.reward_fn.code_reward_fn",
        "entry_points": {
            "evaluator_optimizer": "examples.deepcoder.train_deepcoder_evaluator_optimizer",
            "voting": "examples.deepcoder.train_deepcoder_voting",
            "orchestrator_workers": "examples.deepcoder.train_deepcoder_orchestrator_workers",
            "single_agent": "examples.deepcoder.train_single_agent_deepcoder",
        },
        "default_eval_dataset": "deepcoder",
        "experiment_suffix": "deepcoder",
        "prompt_response_lengths": {
            "evaluator_optimizer": (10240, 2048),
            "voting": (10240, 2048),
            "orchestrator_workers": (10240, 2048),
            "single_agent": (4096, 2048),
        },
        "workflow_params": {
            "evaluator_optimizer": {
                "max_iterations": 2,
                "use_final_outcome_reward": True,
                "enable_test_loop": False,
            },
            "voting": {
                "n_votes": 3,
                "use_final_outcome_reward": True,
                "enable_test_loop": False,
            },
            "orchestrator_workers": {
                "max_subtasks": 3,
                "use_final_outcome_reward": True,
                "share_main_task_with_workers": False,
                "enable_test_loop": False,
            },
            "single_agent": {
                "enable_test_loop": False,
            },
        },
        "extra_sbatch_cmds": "ulimit -n 1048576",
        "experiment_filter_include": "deepcoder",
        "experiment_filter_exclude": None,
    },
}

AGENT_NAMES_MAP = {
    "single_agent": ["generator"],
    "evaluator_optimizer": ["generator", "evaluator"],
    "voting": ["generator", "aggregator"],
    "orchestrator_workers": ["orchestrator", "worker", "synthesizer"],
}

MODEL_MAP = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
}

EVAL_DATASETS = {
    "Math": ["dapo_math", "aime2025"],
    "Code": ["deepcoder"],
}


def infer_task_type(experiment_name: str) -> str:
    """Infer task type from experiment directory name."""
    name_lower = experiment_name.lower()
    if "deepcoder" in name_lower:
        return "deepcoder"
    return "math"


def get_task_config(task_type: str) -> dict:
    """Return config dict for the given task type."""
    if task_type not in TASK_CONFIGS:
        raise ValueError(f"Unknown task type: {task_type!r}. Expected one of: {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[task_type]


# ── Hydra override helpers ──────────────────────────────────────────────────

# Keys that already exist in the base Hydra config (no '+' prefix needed)
_HYDRA_BASE_KEYS = {"use_final_outcome_reward"}


def workflow_params_to_hydra(params: dict) -> str:
    """Convert a workflow_params dict to a Hydra override string."""
    parts = []
    for key, value in params.items():
        val_str = str(value).lower() if isinstance(value, bool) else str(value)
        prefix = "" if key in _HYDRA_BASE_KEYS else "+"
        parts.append(f"{prefix}rllm.workflow.{key}={val_str}")
    return " ".join(parts)


def resolve_launch_params(task_type: str, workflow: str, model_key: str) -> dict:
    """Resolve all launch parameters from the central config.

    Returns a dict with: entry_point, agent_names, workflow_params (Hydra string),
    max_prompt_length, max_response_length, model_path.
    """
    config = get_task_config(task_type)
    model_lower = model_key.lower()
    if workflow not in config["entry_points"]:
        raise ValueError(f"Unknown workflow {workflow!r} for task {task_type!r}")
    if model_lower not in MODEL_MAP:
        raise ValueError(f"Unknown model {model_key!r}")
    prompt_len, response_len = config["prompt_response_lengths"][workflow]
    return {
        "entry_point": config["entry_points"][workflow],
        "agent_names": AGENT_NAMES_MAP[workflow],
        "workflow_params": workflow_params_to_hydra(config["workflow_params"].get(workflow, {})),
        "max_prompt_length": prompt_len,
        "max_response_length": response_len,
        "model_path": MODEL_MAP[model_lower],
    }
