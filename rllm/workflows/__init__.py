"""Engine module for rLLM.

This module contains the core execution infrastructure for agent trajectory rollout.
"""

from .workflow import TerminationEvent, TerminationReason, Workflow

__all__ = [
    "Workflow",
    "TerminationReason",
    "TerminationEvent",
    "SingleTurnWorkflow",
    "MultiTurnWorkflow",
    "CumulativeWorkflow",
    "TimingTrackingMixin",
    "EvaluatorOptimizerWorkflow",
    "EvaluationResult",
    "VotingWorkflow",
    "CodeTestLoopMixin",
    "TestRoundResult",
    # Environment-based workflows (generic, work with any agent/env)
    "EnvSingleAgentWorkflow",
    "EnvEvaluatorOptimizerWorkflow",
    "ActionEvaluation",
]


def __getattr__(name):
    if name == "SingleTurnWorkflow":
        from .single_turn_workflow import SingleTurnWorkflow as _Single

        return _Single
    if name == "MultiTurnWorkflow":
        from .multi_turn_workflow import MultiTurnWorkflow as _Multi

        return _Multi
    if name == "CumulativeWorkflow":
        from .cumulative_workflow import CumulativeWorkflow as _Cumulative

        return _Cumulative
    if name == "TimingTrackingMixin":
        from .timing_mixin import TimingTrackingMixin as _Mixin

        return _Mixin
    if name == "EvaluatorOptimizerWorkflow":
        from .evaluator_optimizer_workflow import EvaluatorOptimizerWorkflow as _EvalOpt

        return _EvalOpt
    if name == "EvaluationResult":
        from .evaluator_optimizer_workflow import EvaluationResult as _EvalResult

        return _EvalResult
    if name == "VotingWorkflow":
        from .voting_workflow import VotingWorkflow as _Voting

        return _Voting
    if name == "CodeTestLoopMixin":
        from .code_test_loop_mixin import CodeTestLoopMixin as _CodeTestLoop

        return _CodeTestLoop
    if name == "TestRoundResult":
        from .code_test_loop_mixin import TestRoundResult as _TestRoundResult

        return _TestRoundResult
    if name == "EnvSingleAgentWorkflow":
        from .env_single_agent_workflow import EnvSingleAgentWorkflow as _EnvSingle

        return _EnvSingle
    if name == "EnvEvaluatorOptimizerWorkflow":
        from .env_evaluator_optimizer_workflow import EnvEvaluatorOptimizerWorkflow as _EnvEvalOpt

        return _EnvEvalOpt
    if name == "ActionEvaluation":
        from .env_evaluator_optimizer_workflow import ActionEvaluation as _ActionEval

        return _ActionEval
    raise AttributeError(name)
