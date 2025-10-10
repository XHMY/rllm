"""
Multi-Agent Math Environment for collaborative problem solving.

This environment manages turn-based interaction between three agent roles:
1. generator_initial - Proposes initial solution
2. evaluator_critique - Evaluates and critiques solutions
3. generator_refinement - Refines solution based on feedback

The environment maintains full conversation history across all agent turns,
allowing each agent to see the complete context of previous interactions.
"""

from typing import Any, Dict, Tuple

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.math_reward import extract_boxed_answer


class MultiAgentMathEnv(BaseEnv):
    """
    Environment managing 3-agent collaboration with shared conversation history.

    The environment alternates between generator and evaluator agents, maintaining
    the full conversation history so each agent can see all previous exchanges.

    Interaction Flow:
        Problem → generator_initial → evaluator_critique
                → [if incorrect] → generator_refinement → evaluator_critique
                → [repeat until correct or max_steps]
    """

    def __init__(
        self,
        question: str,
        final_answer: str,
        prompts: Dict[str, Dict[str, Any]],
        max_steps: int = 6,
        **kwargs
    ):
        """
        Initialize the multi-agent math environment.

        Args:
            question: The math problem to solve
            final_answer: The correct answer for reward calculation
            prompts: Dictionary of prompt templates for each agent role
            max_steps: Maximum number of interaction steps
            **kwargs: Additional arguments (for compatibility)
        """
        self.question = question
        self.ground_truth = str(final_answer).strip()
        self.prompts = prompts
        self.max_steps = max_steps

        # State tracking
        self.current_step = 0
        self.current_role = "generator_initial"

        # Shared conversation history across all agents
        self.conversation_history = []

        # Track last solution and verdict
        self.last_solution = None
        self.last_verdict = None

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset environment to initial state.

        Returns:
            observation: Dict containing question and empty conversation history
            info: Dict containing current agent_role
        """
        self.current_step = 0
        self.current_role = "generator_initial"
        self.conversation_history = []
        self.last_solution = None
        self.last_verdict = None

        obs = {
            "question": self.question,
            "conversation_history": []
        }
        info = {"agent_role": "generator_initial"}

        return obs, info

    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: The agent's response (solution or critique)

        Returns:
            observation: Next observation (or None if done)
            reward: Reward for this step
            done: Whether the episode is finished
            info: Additional information including next agent_role
        """
        self.current_step += 1

        # Add current agent's response to shared history
        self.conversation_history.append({
            "role": self.current_role,
            "content": action
        })

        if self.current_role == "generator_initial":
            return self._handle_generator_initial(action)
        elif self.current_role == "evaluator_critique":
            return self._handle_evaluator_critique(action)
        elif self.current_role == "generator_refinement":
            return self._handle_generator_refinement(action)
        else:
            raise ValueError(f"Unknown role: {self.current_role}")

    def _handle_generator_initial(self, action: str) -> Tuple[Any, float, bool, Dict]:
        """Handle generator_initial agent's action."""
        # Generator proposed initial solution
        self.last_solution = action

        # Assign reward based on correctness to provide variance for GRPO
        is_correct = self._check_correctness(action)
        reward = 1.0 if is_correct else -0.2

        self.current_role = "evaluator_critique"

        obs = {
            "question": self.question,
            "conversation_history": self.conversation_history.copy()
        }
        # Add metadata for rejection sampling
        info = {
            "agent_role": "evaluator_critique",
            "generator_solution": action,
            "is_solution_correct": is_correct,
        }
        done = False

        return obs, reward, done, info

    def _handle_evaluator_critique(self, action: str) -> Tuple[Any, float, bool, Dict]:
        """Handle evaluator_critique agent's action."""
        # Evaluator just critiqued
        verdict = self._parse_verdict(action)
        self.last_verdict = verdict

        # Check if solution is actually correct
        is_correct = self._check_correctness(self.last_solution)

        if is_correct:
            # Solution is correct
            if verdict == "Correct":
                reward = 1.0  # Perfect: correct solution, correct verdict
            else:
                reward = 0.5  # Solution correct but evaluator said incorrect
            done = True
            obs = None
            info = {}

        else:
            # Solution is wrong
            if verdict == "Incorrect":
                reward = 0.2  # Good: evaluator correctly identified error

                # Check if we should continue to refinement
                done = self.current_step >= self.max_steps

                if not done:
                    # Move to refinement phase
                    self.current_role = "generator_refinement"
                    obs = {
                        "question": self.question,
                        "conversation_history": self.conversation_history.copy()
                    }
                    info = {"agent_role": "generator_refinement"}
                else:
                    obs = None
                    info = {}
            else:
                # Evaluator said "Correct" when it's actually wrong
                reward = 0.0  # Bad: evaluator gave wrong verdict
                done = True  # Stop when evaluator says "Correct"
                obs = None
                info = {}

        return obs, reward, done, info

    def _handle_generator_refinement(self, action: str) -> Tuple[Any, float, bool, Dict]:
        """Handle generator_refinement agent's action."""
        # Generator refined solution
        previous_solution = self.last_solution
        self.last_solution = action

        # Assign reward based on correctness AND change to provide variance for GRPO
        is_correct = self._check_correctness(action)
        solution_changed = self._check_solution_changed(action, previous_solution)

        # Reward: +1.0 if (correct AND changed), -0.2 otherwise
        reward = 1.0 if (is_correct and solution_changed) else -0.2

        self.current_role = "evaluator_critique"

        obs = {
            "question": self.question,
            "conversation_history": self.conversation_history.copy()
        }

        # Add metadata for rejection sampling
        info = {
            "agent_role": "evaluator_critique",
            "generator_solution": action,
            "previous_solution": previous_solution,
            "is_solution_correct": is_correct,
            "solution_changed": solution_changed,
        }
        done = False

        return obs, reward, done, info

    def _parse_verdict(self, response: str) -> str:
        """
        Extract verdict (Correct/Incorrect) from evaluator response.

        Args:
            response: The evaluator's response text

        Returns:
            "Correct", "Incorrect", or "Unknown"
        """
        if "\\boxed{Correct}" in response or "boxed{Correct}" in response:
            return "Correct"
        elif "\\boxed{Incorrect}" in response or "boxed{Incorrect}" in response:
            return "Incorrect"
        return "Unknown"

    def _check_correctness(self, solution: str) -> bool:
        """
        Check if solution matches ground truth answer.

        Args:
            solution: The proposed solution text

        Returns:
            True if solution is correct, False otherwise
        """
        if solution is None:
            return False

        predicted = extract_boxed_answer(solution)
        if predicted is None:
            return False

        # Normalize both answers
        predicted = str(predicted).strip()
        ground_truth = str(self.ground_truth).strip()

        return predicted == ground_truth

    def _check_solution_changed(self, new_solution: str, old_solution: str) -> bool:
        """
        Check if the extracted answer changed between solutions.

        Args:
            new_solution: New solution text
            old_solution: Previous solution text

        Returns:
            True if answers are different, False if same
        """
        try:
            from rllm.rewards.math_reward import extract_boxed_answer

            new_ans = extract_boxed_answer(new_solution)
            old_ans = extract_boxed_answer(old_solution)

            # If we can't extract either answer, consider it changed
            if new_ans is None or old_ans is None:
                return True

            return str(new_ans).strip() != str(old_ans).strip()
        except Exception:
            # If extraction fails, assume changed
            return True

    @staticmethod
    def from_dict(info: Dict[str, Any]) -> "MultiAgentMathEnv":
        """
        Create environment instance from dictionary.

        This method is required by rLLM's agent execution engine.

        Args:
            info: Dictionary containing environment parameters

        Returns:
            MultiAgentMathEnv instance
        """
        return MultiAgentMathEnv(**info)

    @staticmethod
    def is_multithread_safe() -> bool:
        """
        Indicate whether this environment is safe for multithreaded access.

        Returns:
            True (this environment has no shared mutable state)
        """
        return True
