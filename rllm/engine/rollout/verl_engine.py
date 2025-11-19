import os.path
import uuid
from glob import glob

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
from vllm.lora.request import LoRARequest
from os.path import join


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.rollout_manager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, rollout_manager.async_llm_servers)
        self.tokenizer = tokenizer
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, disable_thinking=config.get("rllm", {}).get("disable_thinking", False))

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)

        self.train_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.do_sample is False else config.actor_rollout_ref.rollout.temperature,
            top_k=config.actor_rollout_ref.rollout.top_k,
            top_p=config.actor_rollout_ref.rollout.top_p,
        )

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
        )

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks_verl

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        agent_name = kwargs.pop("agent_name")
        sampling_params = self.val_sampling_params.copy() if self.validate or validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_ids)

        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        share_policy = self.config.trainer.get("share_policy", False)
        agent_name = "default" if share_policy else agent_name
        agent_lora_dir = join(self.config.trainer.get("lora_adapter_path"), "step_*", agent_name)

        if os.path.exists(agent_lora_dir) and (not self.config.trainer.get("ori_single_policy_no_lora_mode", False)):
            print("Loading LoRA adapter for agent:", agent_name)
            target_lora_file_path = glob(agent_lora_dir)
            assert len(target_lora_file_path) == 1, f"Expected one LoRA adapter file in {agent_lora_dir}, but found {len(target_lora_file_path)}"
            lora_request = LoRARequest(agent_name,
                                       self.config.trainer.get("agent_names").index(agent_name) + 1,
                                       target_lora_file_path[0])
        else:
            lora_request = None

        completion_ids: list[int] = await self.server_manager.generate(
            request_id=application_id, prompt_ids=prompt_ids, sampling_params=sampling_params,
            lora_request=lora_request
        )

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        parsed_output = self.chat_parser.parse_completion(completion_ids)

        return ModelOutput(
            text=completion_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    def wake_up(self):
        self.rollout_manager.wake_up()

    def sleep(self):
        self.rollout_manager.sleep()
