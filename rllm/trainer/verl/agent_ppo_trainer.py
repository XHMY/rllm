import asyncio
import json
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    _timer,
    compute_advantage,
    compute_data_metrics,
    compute_response_mask,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.utils.torch_functional import pad_sequence_to_length


class AgentPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
        multi_agent_config=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_args = env_args or {}
        self.agent_args = agent_args or {}

        # Multi-agent configuration
        self.multi_agent_config = multi_agent_config
        self.is_multi_agent = multi_agent_config is not None and multi_agent_config.get("enabled", False)
        self.lora_configs = multi_agent_config.get("lora_configs", {}) if self.is_multi_agent else {}
        self.agent_roles = multi_agent_config.get("agent_roles", []) if self.is_multi_agent else []

        if self.is_multi_agent:
            print(f"Multi-agent mode enabled with {len(self.agent_roles)} roles: {self.agent_roles}")

        if self.config.agent.use_stepwise_advantage:
            print("Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied episode-wise")

    def init_workers(self):
        super().init_workers()

        # Initialize additional agent class
        # Number of agents is set to be 0 initially
        if self.hybrid_engine:
            agent_rollout_wg = self.actor_rollout_wg
        else:
            agent_rollout_wg = self.rollout_wg

        if self.config.actor_rollout_ref.rollout.mode == "async":
            rollout_engine = self.async_rollout_manager
        else:
            rollout_engine = agent_rollout_wg

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=rollout_engine,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.agent.use_stepwise_advantage,
            trajectory_timeout=self.config.agent.trajectory_timeout,
            overlong_filter=self.config.agent.overlong_filter,
            lora_configs=self.lora_configs if self.is_multi_agent else None,
            **self.config.agent.get("engine_args", {}),
        )

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        env_args = batch.non_tensor_batch["extra_info"].tolist()

        full_agent_args = dict(self.config.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.env.get("env_args", {})) | self.env_args

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Create agents in parallel while preserving order
        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(envs))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def _separate_by_agent_role(self, batch: DataProto) -> dict[str, DataProto]:
        """
        Separate batch by agent_role for multi-agent training.

        Extracts agent_role from trajectory steps and groups data by role.

        Args:
            batch: Combined batch containing data from all agent roles

        Returns:
            Dictionary mapping agent_role to role-specific DataProto batch
        """
        import torch
        from verl import DataProto

        # Extract agent roles from trajectory metadata
        # Assuming batch has trajectory_data with steps containing agent_role
        role_indices = {role: [] for role in self.agent_roles}

        batch_size = len(batch.batch)
        for idx in range(batch_size):
            # Get agent_role from batch metadata
            # This should be stored during trajectory generation
            agent_role = batch.non_tensor_batch.get("agent_role")[idx]

            if agent_role and agent_role in role_indices:
                role_indices[agent_role].append(idx)

        # Create separate batches for each role
        role_batches = {}
        for role, indices in role_indices.items():
            if len(indices) == 0:
                continue

            # Select data for this role
            role_batch_dict = {}
            for key, value in batch.batch.items():
                if isinstance(value, torch.Tensor):
                    role_batch_dict[key] = value[indices]
                elif isinstance(value, np.ndarray):
                    role_batch_dict[key] = value[indices]
                else:
                    role_batch_dict[key] = [value[i] for i in indices]

            role_non_tensor = {}
            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    role_non_tensor[key] = value[indices]
                else:
                    role_non_tensor[key] = [value[i] for i in indices]

            # Create role-specific batch
            role_batch = DataProto.from_dict(
                tensors=role_batch_dict, non_tensors=role_non_tensor, meta_info=batch.meta_info)

            # Apply token masking: mask out tokens from other agents
            role_batch = self._mask_non_agent_tokens(role_batch, role)

            role_batches[role] = role_batch

        return role_batches

    def _mask_non_agent_tokens(self, batch: DataProto, target_role: str) -> DataProto:
        """
        Mask out tokens not generated by target_role.

        Sets traj_mask=0 for all tokens from other agents, ensuring
        the policy gradient only applies to this agent's own tokens.

        Args:
            batch: Batch containing only steps from target_role
            target_role: The agent role to keep (e.g., "generator_initial")

        Returns:
            Batch with updated traj_mask
        """
        # Get step-level agent_role information
        step_agent_roles = batch.non_tensor_batch.get("step_agent_role", None)

        if step_agent_roles is None:
            # Fallback: assume all steps in this batch are from target_role
            # (This happens if agent_role is only tracked at episode level)
            print(f"Warning: No step_agent_role metadata, assuming all steps from {target_role}")
            return batch

        # Create mask: 1 for target agent's tokens, 0 for others
        traj_mask = batch.batch["traj_mask"].clone()

        for i, step_role in enumerate(step_agent_roles):
            if step_role != target_role:
                # Zero out entire response for this step (it's from another agent)
                traj_mask[i, :] = 0

        # Update batch with masked trajectory
        batch.batch["traj_mask"] = traj_mask

        # Log masking statistics
        total_tokens = traj_mask.numel()
        masked_tokens = (traj_mask == 0).sum().item()
        active_tokens = (traj_mask == 1).sum().item()
        print(f"  {target_role}: Active tokens={active_tokens}, Masked tokens={masked_tokens}/{total_tokens} ({masked_tokens/total_tokens*100:.1f}%)")

        return batch

    def _split_trajectory_by_agent_role(self, trajectory_dict: dict) -> list[dict]:
        """
        Split a multi-agent trajectory into per-agent training samples.

        Each agent gets a separate sample with:
        - Full trajectory tokens (all steps visible for context)
        - Modified response_masks: 1 only for this agent's tokens, 0 for others
        - Agent-specific rewards and metadata

        Args:
            trajectory_dict: Dict from run_agent_trajectory_async Token mode containing:
                - trajectory: Trajectory object with steps
                - step_token_boundaries: List of (start_idx, end_idx, step_idx)
                - response_masks, response_tokens, etc.

        Returns:
            List of per-agent trajectory dicts (one per unique agent_role)
        """
        from collections import defaultdict

        trajectory = trajectory_dict.get("trajectory")
        step_token_boundaries = trajectory_dict.get("step_token_boundaries", [])
        original_masks = torch.tensor(trajectory_dict["response_masks"], dtype=torch.long)

        if trajectory is None or not step_token_boundaries:
            # No trajectory metadata, return original as-is (single-agent mode)
            return [trajectory_dict]

        # Group steps by agent_role
        role_to_step_indices = defaultdict(list)
        for step_idx, step in enumerate(trajectory.steps):
            if hasattr(step, "agent_role") and step.agent_role:
                role_to_step_indices[step.agent_role].append(step_idx)

        if not role_to_step_indices:
            # No agent_role metadata, return original trajectory as-is
            return [trajectory_dict]

        per_agent_samples = []

        for agent_role, agent_step_indices in role_to_step_indices.items():
            # Create agent-specific mask: 1 only for this agent's tokens
            agent_mask = torch.zeros_like(original_masks)

            # Find token boundaries for this agent's steps
            for start_idx, end_idx, step_idx in step_token_boundaries:
                if step_idx in agent_step_indices:
                    # Unmask this agent's tokens
                    agent_mask[start_idx:end_idx] = 1

            # Extract agent-specific metadata
            agent_steps = [trajectory.steps[idx] for idx in agent_step_indices]
            agent_rewards = [step.reward for step in agent_steps]

            # Create per-agent sample
            agent_sample = {
                "prompt_tokens": trajectory_dict["prompt_tokens"],
                "response_tokens": trajectory_dict["response_tokens"],
                "response_masks": agent_mask,  # MODIFIED: Only this agent's tokens
                "trajectory_reward": sum(agent_rewards),  # Agent-specific total
                "idx": trajectory_dict["idx"],
                "chat_completions": trajectory_dict["chat_completions"],
                "agent_role": agent_role,
                "agent_steps": agent_steps,  # For rejection sampling
                "agent_step_indices": agent_step_indices,  # For debugging
                "metrics": trajectory_dict["metrics"],
            }

            per_agent_samples.append(agent_sample)

        return per_agent_samples

    def _update_multi_agent_policies(self, batch: DataProto, actor_rollout_wg):
        """
        Update each agent's policy sequentially using LoRA adapters.

        Args:
            batch: Combined batch containing all agent data
            actor_rollout_wg: Actor rollout worker group
        """
        # Separate trajectories by agent role (includes token masking)
        role_batches = self._separate_by_agent_role(batch)

        print(f"Separated batch into {len(role_batches)} role-specific batches:")
        for role, role_batch in role_batches.items():
            print(f"  {role}: {len(role_batch.batch)} samples")

        # Store outputs for metric aggregation
        self._multi_agent_outputs = {}

        # Update each policy sequentially
        for agent_role in self.agent_roles:
            if agent_role not in role_batches:
                print(f"No data for agent role {agent_role}, skipping update")
                continue

            role_batch = role_batches[agent_role]
            lora_config = self.lora_configs.get(agent_role, {})

            print(f"Updating policy for agent role: {agent_role}")

            # Switch active LoRA adapter
            actor_rollout_wg.set_active_lora(agent_role, lora_config)

            # Update this agent's policy with its data (only trains on its own tokens due to masking!)
            actor_output = actor_rollout_wg.update_actor(role_batch)

            # Store for metric aggregation
            self._multi_agent_outputs[agent_role] = actor_output

            print(f"Finished updating {agent_role}")

    def _aggregate_multi_agent_metrics(self) -> dict:
        """
        Aggregate metrics from all agent roles.

        Returns:
            Dictionary of aggregated metrics with role-specific prefixes
        """
        import ray

        all_metrics = {}

        for agent_role, actor_output in self._multi_agent_outputs.items():
            # Get remote result
            output = ray.get(actor_output)
            role_metrics = reduce_metrics(output.meta_info["metrics"])

            # Prefix with role name
            for key, value in role_metrics.items():
                all_metrics[f"{agent_role}/{key}"] = value

        # Compute aggregated stats
        if all_metrics:
            # Average policy loss across roles
            policy_losses = [v for k, v in all_metrics.items() if "policy_loss" in k]
            if policy_losses:
                all_metrics["multi_agent/avg_policy_loss"] = sum(policy_losses) / len(policy_losses)

            # Average value loss across roles (if using critic)
            value_losses = [v for k, v in all_metrics.items() if "value_loss" in k]
            if value_losses:
                all_metrics["multi_agent/avg_value_loss"] = sum(value_losses) / len(value_losses)

        return all_metrics

    def _apply_agent_level_rejection_sampling(self, batch: DataProto) -> DataProto:
        """
        Filter training samples based on agent-role-specific informativeness.

        Uses binary reward criteria to identify informative samples for each agent:
        - generator_initial: Keep if solution eventually became correct (trajectory got reward >= 1.0)
        - evaluator_critique: Keep if gave informative feedback (reward > 0)
        - generator_refinement: Keep if refinement led to correct solution (reward >= 1.0)

        Args:
            batch: DataProto with non_tensor_batch containing:
                - agent_role: Which agent produced this sample
                - agent_steps: List of Step objects for this agent

        Returns:
            Filtered DataProto with only informative samples
        """
        agent_roles = batch.non_tensor_batch["agent_role"]
        agent_steps_list = batch.non_tensor_batch.get("agent_steps", None)

        if agent_steps_list is None:
            print("WARNING: No agent_steps metadata for rejection sampling, returning original batch")
            return batch

        keep_indices = []
        stats = {role: {"total": 0, "kept": 0} for role in self.agent_roles}

        for idx, (agent_role, agent_steps) in enumerate(zip(agent_roles, agent_steps_list)):
            if agent_role not in stats:
                stats[agent_role] = {"total": 0, "kept": 0}

            stats[agent_role]["total"] += 1
            keep = False

            if agent_role == "generator_initial":
                # Keep if this generator's solution was eventually correct
                # Check: did ANY step in this agent's trajectory get reward >= 1.0?
                # (This means evaluator said "Correct" at some point)
                if agent_steps and any(step.reward >= 1.0 for step in agent_steps):
                    keep = True

            elif agent_role == "evaluator_critique":
                # Keep if evaluator gave informative feedback
                # reward > 0 means: either correctly said "Correct" (1.0)
                #                   or correctly said "Incorrect" (0.2)
                # reward = 0 means: incorrectly said "Correct" when wrong (not informative)
                if agent_steps and any(step.reward > 0 for step in agent_steps):
                    keep = True

            elif agent_role == "generator_refinement":
                # Keep if refinement led to correct solution
                # Check: Did trajectory get reward >= 1.0 after this refinement?
                # (Subsequent evaluator gave high reward)
                if agent_steps and any(step.reward >= 1.0 for step in agent_steps):
                    keep = True

            if keep:
                keep_indices.append(idx)
                stats[agent_role]["kept"] += 1

        # Log rejection statistics
        print("Agent-level rejection sampling:")
        total_samples = len(agent_roles)
        total_kept = len(keep_indices)
        for role, s in stats.items():
            if s["total"] > 0:
                keep_rate = s["kept"] / s["total"] * 100
                print(f"  {role}: {s['kept']}/{s['total']} ({keep_rate:.1f}%)")
        print(f"  Overall: {total_kept}/{total_samples} ({total_kept/total_samples*100:.1f}%)")

        if not keep_indices:
            print("WARNING: All samples rejected! Returning original batch to avoid empty batch.")
            return batch

        # Filter batch
        keep_mask = torch.zeros(len(agent_roles), dtype=torch.bool)
        keep_mask[keep_indices] = True
        return batch[keep_mask]

    def _apply_stepwise_rejection_sampling(self, batch: DataProto) -> DataProto:
        """
        Stratified rejection sampling maintaining balanced positive/negative samples.

        Strategy:
        1. Separate samples into positive/negative per agent role
        2. Keep ALL positive samples (count = P)
        3. Calculate target negatives: N = (negative_ratio * P) / (1 - negative_ratio)
        4. Randomly sample N negatives from available negatives
        5. Final batch has exactly negative_ratio% negative samples

        Positive/Negative criteria per agent:
        - generator_initial: positive = correct, negative = incorrect
        - evaluator_critique: positive = step_reward > 0, negative = step_reward ≤ 0
        - generator_refinement: positive = (correct AND changed), negative = else

        Args:
            batch: DataProto with non_tensor_batch containing:
                - agent_role: Agent role for each step
                - step_reward: Reward for this individual step
                - is_solution_correct: Ground truth correctness (Boolean)
                - solution_changed: Whether answer changed (Boolean)

        Returns:
            Filtered DataProto with balanced positive/negative samples

        Raises:
            ValueError: If required metadata is missing
        """
        agent_roles = batch.non_tensor_batch["agent_role"]
        step_rewards = batch.non_tensor_batch.get("step_reward")
        is_solution_correct = batch.non_tensor_batch.get("is_solution_correct")
        solution_changed = batch.non_tensor_batch.get("solution_changed")

        # NO FALLBACK - raise error if metadata missing
        if step_rewards is None:
            raise ValueError("Missing 'step_reward' in non_tensor_batch - required for rejection sampling")
        if is_solution_correct is None:
            raise ValueError("Missing 'is_solution_correct' in non_tensor_batch - check environment metadata")
        if solution_changed is None:
            raise ValueError("Missing 'solution_changed' in non_tensor_batch - check environment metadata")

        # Get negative sample ratio from config (default 0.3)
        negative_ratio = getattr(self.config.trainer, "negative_sample_ratio", 0.3)

        positive_indices = []
        negative_indices = []
        stats = {role: {"total": 0, "positive": 0, "negative": 0} for role in self.agent_roles}

        # Classify each sample as positive or negative
        for idx, agent_role in enumerate(agent_roles):
            if agent_role not in stats:
                stats[agent_role] = {"total": 0, "positive": 0, "negative": 0}

            stats[agent_role]["total"] += 1
            is_positive = False

            if agent_role == "generator_initial":
                # Positive if correct, negative if incorrect
                is_positive = is_solution_correct[idx] is True

            elif agent_role == "evaluator_critique":
                # Positive if gave correct feedback
                is_positive = step_rewards[idx] > 0

            elif agent_role == "generator_refinement":
                # Positive if (correct AND changed)
                is_positive = is_solution_correct[idx] is True and solution_changed[idx] is True

            if is_positive:
                positive_indices.append(idx)
                stats[agent_role]["positive"] += 1
            else:
                negative_indices.append(idx)
                stats[agent_role]["negative"] += 1

        # Calculate how many negatives to keep
        num_positives = len(positive_indices)
        num_negatives_available = len(negative_indices)

        if num_positives == 0:
            raise ValueError(
                "No positive samples found in batch! Cannot perform stratified sampling. "
                "Check your environment rewards and metadata."
            )

        # Target negatives: N = (negative_ratio * P) / (1 - negative_ratio)
        # This ensures N/(P+N) = negative_ratio
        num_negatives_target = int((negative_ratio * num_positives) / (1 - negative_ratio))

        # Sample negatives (or take all if not enough)
        if num_negatives_available <= num_negatives_target:
            # Not enough negatives, take all
            sampled_negative_indices = negative_indices
            num_negatives_kept = num_negatives_available
        else:
            # Randomly sample target number of negatives
            import random
            sampled_negative_indices = random.sample(negative_indices, num_negatives_target)
            num_negatives_kept = num_negatives_target

        # Combine positive and sampled negative indices
        keep_indices = positive_indices + sampled_negative_indices

        # Log sampling statistics
        total_samples = len(agent_roles)
        total_kept = len(keep_indices)
        actual_negative_ratio = num_negatives_kept / total_kept if total_kept > 0 else 0

        print("Stratified rejection sampling:")
        print(f"  Target negative ratio: {negative_ratio:.1%}")
        print(f"  Actual negative ratio: {actual_negative_ratio:.1%}")
        print(f"  Positive samples: {num_positives}")
        print(f"  Negative samples: {num_negatives_kept}/{num_negatives_available}")
        print(f"  Total kept: {total_kept}/{total_samples} ({total_kept/total_samples*100:.1f}%)")
        print("  Per-agent breakdown:")
        for role, s in stats.items():
            if s["total"] > 0:
                print(f"    {role}: {s['total']} total ({s['positive']} pos, {s['negative']} neg)")

        if not keep_indices:
            raise ValueError(
                "No samples kept after stratified sampling! "
                "Check your environment rewards and metadata."
            )

        # Filter batch
        keep_mask = torch.zeros(len(agent_roles), dtype=torch.bool)
        keep_mask[keep_indices] = True
        return batch[keep_mask]

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics = {}
                timing_raw = {}

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                batch.meta_info = {
                    "agent_rollout": True,  # no need to generate multiple ones since environment is repeated already
                }

                with _timer("step", timing_raw):
                    self.init_envs_and_agents(batch)

                    if self.config.agent.use_stepwise_advantage:
                        final_gen_batch_output = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info, uids=batch.non_tensor_batch["uid"])
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        # need to repeat to make shape match
                        batch = batch.repeat_by_counts(repeat_counts, interleave=True)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                        # batch needs to be padded to divisor of world size, we will pad with everything masked out
                        batch = batch.union(final_gen_batch_output)
                        batch = self._pad_dataproto_to_world_size(batch=batch)
                    else:
                        final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
                        batch = batch.union(final_gen_batch_output)
                        metrics.update(generate_metrics)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]  # filled in by environment collected trajectory transformation

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            # Check if all rewards are <= 0 or all are 1 >= for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                        if self.config.trainer.rejection_sample:
                            # log the actual complete training rewards before rejection sampling
                            token_level_rewards = None  # for metrics calculation
                            if self.config.agent.use_stepwise_advantage:
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                non_pad_steps = batch.select_idxs(non_pad_step_indices)
                                is_last_step = non_pad_steps.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)
                                token_level_rewards = last_step_batch.batch["token_level_scores"]
                            else:
                                token_level_rewards = batch.batch["token_level_scores"]
                            full_sequence_score = token_level_rewards.sum(-1)
                            metrics["critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                            metrics["critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                            metrics["critic/full-score/min"] = torch.min(full_sequence_score).detach().item()

                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]

                            if self.config.agent.use_stepwise_advantage and self.config.agent.stepwise_advantage_mode == "broadcast":
                                # batch now only contains steps with valid uids
                                # filter out padding steps
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps

                                # need to make sure both number of last steps (number of uids) and number of total steps in the batch (batch size after processing) are all multiples of world size
                                # separate out last step and intermediate steps
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)

                                # filter last_step_batch to make sure its multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    last_step_batch.batch["input_ids"].shape[0]  # 1 per trajectory
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                last_step_batch = last_step_batch[size_mask]  # filtered last steps

                                # now we go through all the non_last_step_batch and keep everything that has same idxs that exists in the filtered last steps
                                valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                non_last_step_batch = non_last_step_batch[non_last_step_mask]

                                # concatenate then pad
                                batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                batch = self._pad_dataproto_to_world_size(batch)
                            else:
                                # Round down to the nearest multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                batch = batch[size_mask]

                        # NEW: Agent-level rejection sampling for multi-agent training
                        if self.is_multi_agent and self.config.trainer.get("agent_level_rejection_sampling", False):
                            with _timer("agent_rejection", timing_raw):
                                # Dispatch to appropriate rejection sampling method based on mode
                                if self.config.agent.use_stepwise_advantage:
                                    batch = self._apply_stepwise_rejection_sampling(batch)
                                else:
                                    batch = self._apply_agent_level_rejection_sampling(batch)

                                # Re-pad after filtering
                                batch = self._pad_dataproto_to_world_size(batch)

                        # recompute old_log_probs
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.agent.use_stepwise_advantage:
                            if self.config.agent.stepwise_advantage_mode == "mc_return":
                                batch.batch["token_level_rewards"] = batch.batch["mc_returns"]
                                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
                            elif self.config.agent.stepwise_advantage_mode == "broadcast":
                                # In case of step-wise advantage broadcast, we would split out the final steps, then merge again
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                last_step_indices = np.where(is_last_step == True)[0]
                                other_step_indices = np.where(is_last_step == False)[0]
                                other_step_batch = batch.select_idxs(other_step_indices)
                                batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                            else:
                                raise ValueError(f"Stepwise advantage mode {self.config.agent.stepwise_advantage_mode} not supported")

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            mask_truncated_samples=self.config.algorithm.mask_truncated_samples,
                            clip_advantages=self.config.algorithm.clip_advantages,
                        )

                        if self.config.agent.use_stepwise_advantage and self.config.agent.stepwise_advantage_mode == "broadcast":
                            # remove the padded last steps
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, other_step_batch=other_step_batch)
                            # batch = batch.merge(other_step_batch)
                            batch = DataProto.concat([batch, other_step_batch])

                    batch = self._pad_dataproto_to_world_size(batch=batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            if self.is_multi_agent:
                                # Multi-agent: update each policy sequentially with masked data
                                self._update_multi_agent_policies(batch, self.actor_rollout_wg)
                                # Aggregate metrics from all roles
                                actor_output_metrics = self._aggregate_multi_agent_metrics()
                            else:
                                # Single-agent: normal update
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _validate_agent(self):
        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "agent_rollout": True,
            }
            self.init_envs_and_agents(test_batch)

            if self.config.agent.use_stepwise_advantage:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            else:
                test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info)

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor = test_batch.batch["token_level_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            uid_lst.append(test_batch.non_tensor_batch["uid"])

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}

        # to group for pass@k
        uid_tensor = np.concatenate(uid_lst, axis=0)
        data_source_uid_pass_rates = {}  # data source to {uid: pass or not}

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0  # default to not pass
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            # clip rewards to be between 0 and 1
            rewards_array = np.array(rewards)
            rewards_array = np.clip(rewards_array, 0, 1)
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards_array)

        for data_source, pass_rates in data_source_uid_pass_rates.items():
            pass_k_lst = []
            for uid, pass_score in pass_rates.items():
                pass_k_lst.append(pass_score >= 1)  # assuming 1 means passed
            metric_dict[f"val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)

        return metric_dict

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
            Dict[str:float]: Metrics for the generation process.
        """
        if timing_raw is None:
            timing_raw = {}
        with _timer("collect_trajectory", timing_raw):
            trajectories = []
            if self.config.agent.async_engine:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for _, trajectory in enumerate(gen_seq_generator):
                    trajectories.append(trajectory)
            else:
                # generate_trajectories returns list of trajectories.
                trajectories = self.agent_execution_engine.generate_trajectories(timing_raw=timing_raw, mode="Token", meta_info=meta_info)
        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])

        with _timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)
        return final_gen_batch_output, metrics

    def generate_agent_steps(self, timing_raw=None, meta_info=None, uids=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        if timing_raw is None:
            timing_raw = {}
        if uids is None:
            uids = []
        with _timer("collect_trajectory", timing_raw):
            steps = []
            if self.config.agent.async_engine:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
                for _, trajectory in enumerate(gen_seq_generator):
                    steps.append(trajectory)
            else:
                # generate_trajectories returns list of trajectories.
                steps = self.agent_execution_engine.generate_trajectories(timing_raw=timing_raw, mode="Step", meta_info=meta_info)
        # Sort trajectories by their idx, to ensure they are in order.
        steps.sort(key=lambda x: x["idx"])

        with _timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_steps(steps, uids=uids)
        return final_gen_batch_output

    def _transform_agent_trajectories(self, trajectories: list[dict]):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """


        # NEW: Split multi-agent trajectories into per-agent samples
        if self.is_multi_agent:
            split_trajectories = []
            for traj in trajectories:
                split_trajectories.extend(self._split_trajectory_by_agent_role(traj))
            trajectories = split_trajectories
            print(f"Multi-agent: Split {len(trajectories)} original trajectories into {len(split_trajectories)} per-agent samples")

        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        chat_completions = []
        traj_metrics = []
        all_agent_roles = []  # Track agent_role for each sample
        all_agent_steps = []  # Track agent steps for rejection sampling
        metrics = {}

        for traj in trajectories:
            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            # test if trajectory is empty
            assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(traj["response_masks"])
            traj_scores.append(traj["trajectory_reward"])
            chat_completions.append(traj["chat_completions"])
            traj_metrics.append(traj["metrics"])
            all_agent_roles.append(traj.get("agent_role", None))  # Extract agent_role (None if not present)
            all_agent_steps.append(traj.get("agent_steps", []))  # Extract agent steps for rejection sampling

        # Flatten traj_metrics into a dict of lists
        traj_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
        # Aggregate metrics (mean, min, max)
        for k, v_list in traj_metrics.items():
            v_list = [v for v in v_list if v is not None and v >= 0]
            if not v_list:
                continue
            v_list = np.array(v_list)
            metrics.update(
                {
                    f"traj/{k}_mean": v_list.mean(),
                    f"traj/{k}_min": v_list.min(),
                    f"traj/{k}_max": v_list.max(),
                }
            )

        # Save chat completions to a file
        save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
        os.makedirs(save_dir, exist_ok=True)
        # Save it into a jsonl files (self.global_steps)
        with open(os.path.join(save_dir, f"{self.global_steps}.jsonl"), "w") as f:
            for chat_completion in chat_completions:
                f.write(json.dumps(chat_completion) + "\n")

        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])

        prompts_batch = pad_sequence_to_length(prompts_batch, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)

        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)

        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)

        # Compute position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        for i, traj_score in enumerate(traj_scores):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                score_batch[i, last_valid_idx] = traj_score

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "traj_mask": traj_mask,
        }

        # Create non_tensor_batch with agent_role for multi-agent training support
        non_tensor_batch = {
            "agent_role": np.array(all_agent_roles, dtype=object),  # For policy separation in _separate_by_agent_role()
            "step_agent_role": np.array(all_agent_roles, dtype=object),  # For token masking (trajectory-level: entire traj from one agent)
            "agent_steps": np.array(all_agent_steps, dtype=object),  # For agent-level rejection sampling
        }

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch))

        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch), metrics

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="traj_mask"):
        """
        Visualize the trajectory from tensor_batch by detokenizing prompts and responses,
        and highlighting the masked parts with color.

        Args:
            tensor_batch: The tensor batch containing trajectory data
            sample_idx: Starting index of samples to visualize
            max_samples: Maximum number of samples to visualize
        """
        from rllm.misc import colorful_print

        # Get the relevant tensors
        prompts = tensor_batch.batch["prompts"]
        responses = tensor_batch.batch["responses"]
        traj_mask = tensor_batch.batch[mask_key]
        token_level_scores = tensor_batch.batch["token_level_scores"]

        batch_size = prompts.shape[0]
        end_idx = min(sample_idx + max_samples, batch_size)

        for i in range(sample_idx, end_idx):
            colorful_print(f"\n===== Sample {i} =====", fg="cyan", bold=True)

            # Detokenize prompt
            prompt_tokens = prompts[i]
            prompt_mask = prompt_tokens != self.tokenizer.pad_token_id
            valid_prompt_tokens = prompt_tokens[prompt_mask]
            prompt_text = self.tokenizer.decode(valid_prompt_tokens)

            colorful_print("Prompt:", fg="green", bold=True)
            colorful_print(f"{prompt_text}\n", fg="green")

            # Detokenize response with color highlighting for masked tokens
            response_tokens = responses[i]
            response_mask = traj_mask[i]

            # Get non-padding tokens
            valid_indices = response_tokens != self.tokenizer.pad_token_id
            valid_response_tokens = response_tokens[valid_indices]
            valid_response_mask = response_mask[valid_indices]

            # Then show token-by-token with masking
            colorful_print("Response with masking:", fg="yellow", bold=True)

            for j, (token, mask) in enumerate(zip(valid_response_tokens, valid_response_mask, strict=False)):
                token_text = self.tokenizer.decode(token)

                # Check if this token has a reward
                has_reward = token_level_scores[i, j] != 0

                # Apply different colors based on mask and rewards
                if mask == 0:
                    # Masked token (not used in training)
                    colorful_print(token_text, fg="red", end="")
                elif has_reward:
                    # Token with reward
                    colorful_print(token_text, bg="green", end="")

                    reward_info = ""
                    if has_reward:
                        reward_info += f" R:{token_level_scores[i, j].item():.2f}"

                    colorful_print(reward_info, fg="magenta", end="")
                else:
                    # Normal token used in training
                    colorful_print(token_text, fg="blue", end="")

            print()  # New line after all tokens

            # Print reward summary
            total_reward = token_level_scores[i].sum().item()
            colorful_print("Rewards:", fg="green", bold=True)
            print(f" Trajectory Reward={total_reward:.2f}")

    def generate_agent_trajectories_async(self, timing_raw=None, meta_info=None, mode="Token"):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode=mode, meta_info=meta_info):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done

            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _transform_agent_steps(self, steps: list[dict], uids: np.ndarray):

        all_prompts_list = []
        all_responses_list = []

        step_numbers = []  # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = []  # total number of steps the trajectory this step belongs to have
        all_steps_step_ids = []
        all_steps_agent_roles = []  # agent_role for each step (for token masking)
        all_step_rewards = []  # per-step rewards from environment
        training_rewards = []
        all_mc_returns = []  # Monte Carlo returns for each episode
        all_episode_final_rewards = []  # Final reward of each trajectory (for stepwise rejection sampling)
        all_episode_max_rewards = []  # Max reward in each trajectory (for stepwise rejection sampling)
        all_is_solution_correct = []  # Ground truth correctness per step (for rejection sampling)
        all_solution_changed = []  # Whether refinement changed answer (for rejection sampling)
        # the last step will have reward assigned and be used for advantage calculation

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode["trajectory_reward"]  # Keep for metrics
            mc_returns = episode["mc_returns"]
            step_rewards = episode.get("step_rewards", [0.0] * len(episode_steps))  # Per-step rewards from env

            all_prompts_list.extend([torch.tensor(self.tokenizer.encode(s["prompt"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])
            all_responses_list.extend([torch.tensor(self.tokenizer.encode(s["response"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])

            step_numbers.append(len(episode_steps) - 1)
            training_rewards.append(training_reward)
            all_mc_returns.extend(mc_returns)

            # Extract agent_role and rewards for each step
            all_steps_agent_roles.extend([s.get("agent_role", "unknown") for s in episode_steps])
            all_step_rewards.extend(step_rewards)

            # Extract solution correctness metadata from episode steps
            all_is_solution_correct.extend([s.get("is_solution_correct") for s in episode_steps])
            all_solution_changed.extend([s.get("solution_changed") for s in episode_steps])

            # Compute trajectory-level success metrics for rejection sampling
            final_reward = step_rewards[-1] if step_rewards else 0.0
            max_reward = max(step_rewards) if step_rewards else 0.0
            all_episode_final_rewards.extend([final_reward] * len(episode_steps))
            all_episode_max_rewards.extend([max_reward] * len(episode_steps))

            all_steps_idx_list.extend([idx for _ in range(len(episode_steps))])
            all_steps_is_last_step_list.extend([False for _ in range(len(episode_steps))])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([len(episode_steps) for _ in range(len(episode_steps))])
            all_steps_step_ids.extend([f"{uids[idx]}_step{i}" for i in range(len(episode_steps))])

        # Convert all steps into token tensors
        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])

        prompts_batch = pad_sequence_to_length(prompts_batch, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)

        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(complete_step_batch != self.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # same as regular repsonse_mask, padded tensors will have this zeroed out
        traj_mask = torch.where(response_batch != self.tokenizer.pad_token_id, 1, 0)

        # Place all rewards to last response token of the last_step response
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        mc_return_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        # Assign per-step rewards (not trajectory-level reward!)
        # This ensures each agent gets credit for its own actions
        for step_index in range(len(all_step_rewards)):
            last_valid_idx = valid_response_length_sequences[step_index] - 1
            if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                # Use per-step reward from environment
                score_batch[step_index, last_valid_idx] = all_step_rewards[step_index]
                mc_return_batch[step_index, last_valid_idx] = all_mc_returns[step_index]

        assert len(all_step_rewards) == score_batch.shape[0], f"Number of step rewards {len(all_step_rewards)} should equal batch size {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "mc_returns": mc_return_batch,
            "traj_mask": traj_mask,
        }

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list),
            "step_nums": np.array(all_steps_step_num),
            "is_last_step": np.array(all_steps_is_last_step_list),
            "is_pad_step": np.array([False for _ in range(len(all_steps_idx_list))]),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))]),  # in case need to differentiate which iteration the step is coming from
            "step_ids": np.array(all_steps_step_ids),
            "agent_role": np.array(all_steps_agent_roles, dtype=object),  # For policy separation in _separate_by_agent_role()
            "step_agent_role": np.array(all_steps_agent_roles, dtype=object),  # For token masking in _mask_non_agent_tokens()
            "step_reward": np.array(all_step_rewards, dtype=np.float32),  # For stepwise rejection sampling
            "episode_final_reward": np.array(all_episode_final_rewards, dtype=np.float32),  # For stepwise rejection sampling
            "episode_max_reward": np.array(all_episode_max_rewards, dtype=np.float32),  # For stepwise rejection sampling
            "is_solution_correct": np.array(all_is_solution_correct, dtype=object),  # Ground truth correctness
            "solution_changed": np.array(all_solution_changed, dtype=object),  # Whether answer changed
        }

        meta_info = {"repeat_counts": [x + 1 for x in step_numbers]}

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

        # Find indices of last steps for visualization
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        if last_step_indices:
            sample_indices = np.random.choice(last_step_indices, size=min(2, len(last_step_indices)), replace=False)
            for idx in sample_indices:
                self.visualize_trajectory(result, sample_idx=idx, max_samples=1)
        return result

    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]
        src_mask = last_step_batch.batch["response_mask"]
        tgt_mask = other_step_batch.batch["response_mask"]

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.agent.normalize_step_advantage:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32) for i, idx in enumerate(tgt_indices)])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch
