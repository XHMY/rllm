from __future__ import annotations

import json
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from rllm.data.dataset import DatasetRegistry
from verl.utils.model import compute_position_id_with_mask


class DeepMathDataset:
    """
    Dataset handler for DeepMath-103K with math problem formatting.
    Integrates with multi-agent math workflow by providing problems
    and ground truth answers for reward computation.

    Loads pre-processed data from the DatasetRegistry (parquet files).
    Data should be prepared using examples/math_reasoning/prepare_deepmath_data.py

    Args:
        config: Dataset configuration dictionary
        tokenizer: Tokenizer for text processing
        split: Dataset split to use ('train' or 'test'). Defaults to 'train'.
    """

    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizer, split: str = 'train'):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split

        # Tokenization config - now directly from the data config
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # Load prompt templates
        self.prompts = self._load_prompt_templates(config.get("prompts", {}).get("config_file", "prompts.json"))

        # Load pre-processed dataset from registry
        dataset = DatasetRegistry.load_dataset("deepmath", split)
        if dataset is None:
            raise ValueError(
                f"DeepMath dataset '{split}' split not found in registry. "
                "Please run examples/math_reasoning/prepare_deepmath_data.py first."
            )

        self.active_dataset = dataset.get_data()
        print(f"Loaded DeepMath {split} dataset: {len(self.active_dataset)} examples")

    def _load_prompt_templates(self, prompts_file: str) -> Dict[str, Dict[str, str]]:
        """Load prompt templates from JSON file."""
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)

        if "multi_agent_math_prompts" not in prompts_data:
            raise KeyError(f"Required key 'multi_agent_math_prompts' not found in {prompts_file}")

        return prompts_data["multi_agent_math_prompts"]
    
    def format_problem_for_generator_initial(self, problem: str) -> List[Dict[str, Any]]:
        """Format a math problem for the initial generator agent."""
        if "generator_initial" not in self.prompts:
            raise KeyError("'generator_initial' prompt template not found in prompts")
        
        if "template" not in self.prompts["generator_initial"]:
            raise KeyError("'template' not found in generator_initial prompt config")
        
        template = self.prompts["generator_initial"]["template"]
        
        # Check if the template requires the problem variable
        required_vars = self.prompts["generator_initial"].get("required_variables", [])
        if "problem" in required_vars and "{problem}" not in template:
            raise ValueError("Template requires 'problem' variable but doesn't contain {problem} placeholder")
        
        # Format the template with the problem
        formatted_content = template.format(problem=problem)
        
        return [{"role": "user", "content": formatted_content}]
    
    def get_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """Get prompt templates for agents."""
        return self.prompts

    def __len__(self):
        return len(self.active_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample by index with proper tokenization.
        Required for PyTorch DataLoader compatibility.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Dictionary containing tokenized data following Verl patterns
        """
        item = self.active_dataset[idx]
        problem = item["question"]
        ground_truth = str(item["final_answer"]).strip()

        # Format problem for initial generator
        messages = self.format_problem_for_generator_initial(problem)

        # Apply chat template to get raw prompt string
        if self.apply_chat_template_kwargs.get("chat_template") is None:
            assert hasattr(self.tokenizer, "chat_template"), (
                "chat_template should be provided in apply_chat_template_kwargs or tokenizer config"
            )

        raw_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs
        )

        # Tokenize the prompt
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        # Postprocess data with padding/truncation following Verl pattern
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Compute position IDs
        position_ids = compute_position_id_with_mask(attention_mask)

        # Store raw prompt IDs for compatibility with Verl
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "question": item["question"],  # Pass question for multi-agent environment
            "ground_truth_answer": ground_truth,
            "prompts": self.prompts,  # Pass prompts to environment
            "messages": messages,  # Include original chat messages for multi-agent worker
            "index": idx,
        }
