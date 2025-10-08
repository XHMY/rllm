"""
Multi-Role Math Agent for collaborative problem solving.

This agent can take on any of three roles with independent policies:
1. generator_initial - Proposes initial solutions
2. evaluator_critique - Evaluates and critiques solutions
3. generator_refinement - Refines solutions based on feedback

Each role has its own LoRA adapter, and the agent builds messages from
the full conversation history shared by all agents.
"""

import copy
from typing import Any, Dict, List

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class MultiRoleMathAgent(BaseAgent):
    """
    Agent that can act as any of the 3 roles with full conversation history.

    The agent receives its current role from the environment and formats
    prompts accordingly. All agents share the same conversation history,
    enabling collaborative problem solving.
    """

    def __init__(
        self,
        agent_id: str = "agent_0",
        prompts: Dict[str, Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the multi-role math agent.

        Args:
            agent_id: Unique identifier for this agent instance
            prompts: Dictionary of prompt templates for each role
            **kwargs: Additional arguments (for compatibility)
        """
        self.agent_id = agent_id
        self.agent_role = None  # Will be set by environment
        self.prompts = prompts or {}
        self._trajectory = Trajectory()
        self.messages = []
        self.conversation_history = []

    @property
    def lora_adapter_name(self) -> str:
        """
        Map agent role directly to LoRA adapter name.

        Returns:
            LoRA adapter name (e.g., "lora_generator_initial")
        """
        if self.agent_role:
            return f"lora_{self.agent_role}"
        return None

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        **kwargs
    ):
        """
        Process environment feedback and build messages from shared history.

        The agent receives the full conversation history from the environment
        and constructs appropriate messages based on its current role.

        Args:
            observation: Environment observation containing question and history
            reward: Reward from previous action (unused in message building)
            done: Whether episode is finished
            info: Dict containing agent_role for this turn
            **kwargs: Additional arguments
        """
        if done:
            return

        # Get role from environment
        self.agent_role = info.get("agent_role")

        # Get conversation history from environment
        self.conversation_history = observation.get("conversation_history", [])

        # Build messages from scratch using full history
        self.messages = []

        # Add all previous exchanges to message history
        for exchange in self.conversation_history:
            # All previous agent responses are shown as assistant messages
            self.messages.append({
                "role": "assistant",
                "content": exchange["content"]
            })

        # Add current role's prompt as user message
        current_prompt = self._format_current_prompt(observation)
        self.messages.append({"role": "user", "content": current_prompt})

    def _format_current_prompt(self, observation: Dict[str, Any]) -> str:
        """
        Format the prompt for the current agent role.

        Args:
            observation: Environment observation

        Returns:
            Formatted prompt string
        """
        if self.agent_role == "generator_initial":
            template = self.prompts["generator_initial"]["template"]
            return template.format(problem=observation["question"])

        elif self.agent_role == "evaluator_critique":
            template = self.prompts["evaluator_critique"]["template"]
            return template

        elif self.agent_role == "generator_refinement":
            template = self.prompts["generator_refinement"]["template"]
            return template

        else:
            raise ValueError(f"Unknown agent role: {self.agent_role}")

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Process model response and update agent state.

        Stores the model response in messages and creates a Step with
        multi-agent metadata for training.

        Args:
            response: The model's generated response
            **kwargs: Additional arguments

        Returns:
            Action containing the response
        """
        # Add model response to messages
        self.messages.append({"role": "assistant", "content": response})

        # Create step with multi-agent metadata
        step = Step(
            model_response=response,
            chat_completions=copy.deepcopy(self.messages),
            agent_id=self.agent_id,
            agent_role=self.agent_role,
            lora_adapter=self.lora_adapter_name
        )
        self._trajectory.steps.append(step)

        return Action(action=response)

    def reset(self):
        """Reset agent state for new episode."""
        self._trajectory = Trajectory()
        self.messages = []
        self.conversation_history = []
        self.agent_role = None

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """
        Return conversation history for model interaction.

        Returns:
            List of message dicts in OpenAI chat format
        """
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """
        Return complete interaction trajectory.

        Returns:
            Trajectory object containing all steps
        """
        return self._trajectory

    def get_current_state(self) -> Step:
        """
        Return the most recent step.

        Returns:
            The latest Step object, or None if no steps exist
        """
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]
