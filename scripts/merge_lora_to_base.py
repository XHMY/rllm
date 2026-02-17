#!/usr/bin/env python3
"""
Merge a LoRA adapter into its base model and save the merged model.

Uses PEFT's built-in merge_and_unload() to produce a standalone model
that can be loaded without PEFT.

Usage:
    python scripts/merge_lora_to_base.py \
        --base_model Qwen/Qwen3-1.7B \
        --adapter_path checkpoints/.../lora_adapter_generator \
        --output_dir merged_models/qwen3_1.7b_generator

    # Copy tokenizer files from a checkpoint's huggingface/ dir:
    python scripts/merge_lora_to_base.py \
        --base_model Qwen/Qwen3-1.7B \
        --adapter_path checkpoints/.../lora_adapter_generator \
        --output_dir merged_models/qwen3_1.7b_generator \
        --tokenizer_path checkpoints/.../actor/huggingface

    # Use bf16 (default) or fp16:
    python scripts/merge_lora_to_base.py \
        --base_model Qwen/Qwen3-1.7B \
        --adapter_path checkpoints/.../lora_adapter_generator \
        --output_dir merged_models/qwen3_1.7b_generator \
        --dtype fp16
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the base model and save"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or path (e.g., Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory (containing adapter_config.json and adapter_model.safetensors)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged model",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (default: same as base_model). "
        "Useful if the checkpoint has a custom tokenizer in its huggingface/ dir.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for the model (default: bf16)",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=False,
        help="Save in safetensors format (default: False, saves in PyTorch bin format)",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        default=True,
        help="Save in pytorch bin format instead of safetensors",
    )

    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)

    # Validate adapter path
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_path}"
        )

    # Fix adapter_config.json if base_model_name_or_path is null
    config_path = adapter_path / "adapter_config.json"
    with open(config_path) as f:
        adapter_config = json.load(f)

    if not adapter_config.get("base_model_name_or_path"):
        print(f"Patching adapter_config.json: setting base_model_name_or_path to '{args.base_model}'")
        adapter_config["base_model_name_or_path"] = args.base_model
        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=2, ensure_ascii=False)

    # Determine dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    safe_serialization = args.safe_serialization and not args.no_safe_serialization

    # Step 1: Load base model
    print(f"Loading base model: {args.base_model} (dtype={args.dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Step 2: Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch_dtype,
    )

    # Step 3: Merge and unload
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    # Step 4: Save merged model
    print(f"Saving merged model to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=safe_serialization,
    )

    # Step 5: Save tokenizer
    tokenizer_source = args.tokenizer_path or args.base_model
    print(f"Saving tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(output_dir))

    # Print summary
    print("\nDone!")
    print(f"  Merged model saved to: {output_dir}")
    print(f"  Files:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
