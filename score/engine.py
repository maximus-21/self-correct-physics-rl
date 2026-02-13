"""
DeepSpeed engine initialization for the SCoRe pipeline.

Creates and wraps the three models used during training:
  - **Actor**: the policy model being trained (with LoRA)
  - **Reward**: a frozen evaluator model (base LLM + LoRA adapter)
"""

import os
import time
import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM, get_scheduler
from peft import PeftModel

from score.utils import (
    get_train_ds_config,
    get_eval_ds_config,
    create_hf_model,
    convert_linear_layer_to_lora,
    only_optimize_lora_parameters,
    make_model_gradient_checkpointing_compatible,
    get_optimizer_grouped_parameters,
    print_rank_0,
)


def _log_init(model_name, stime=None):
    """Log model initialization start/end on rank 0."""
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if rank == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = f"(duration: {time.time() - stime:.2f}s)" if stime else ""
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        print("*" * stars + msg + "*" * (90 - len(msg) - stars))
    return time.time()


class DeepSpeedScoreEngine:
    """
    Holds the DeepSpeed-wrapped actor and reward model engines.

    Args:
        actor_model_name_or_path: HuggingFace model id or local path for the actor.
        reward_model_name_or_path: HuggingFace model id or local path for the reward base model.
        actor_tokenizer: Tokenizer for the actor model.
        reward_tokenizer: Tokenizer for the reward model.
        args: Training arguments namespace.
        num_total_iters: Total number of training iterations (for LR scheduler).
    """

    def __init__(
        self,
        actor_model_name_or_path,
        reward_model_name_or_path,
        actor_tokenizer,
        reward_tokenizer,
        args,
        num_total_iters,
    ):
        self.args = args
        self.num_total_iters = num_total_iters
        self.actor_tokenizer = actor_tokenizer
        self.reward_tokenizer = reward_tokenizer

        self.actor = self._init_actor(actor_model_name_or_path)
        self.ref = None  # LoRA disable trick is used instead of a separate ref model
        self.reward = self._init_reward(reward_model_name_or_path)

    # -----------------------------------------------------------------
    # Actor
    # -----------------------------------------------------------------
    def _init_actor(self, actor_model_name_or_path):
        stime = _log_init("Actor")
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        ds_config = get_train_ds_config(
            offload=self.args.offload,
            dtype=self.args.dtype,
            stage=self.args.actor_zero_stage,
            enable_hybrid_engine=self.args.enable_hybrid_engine,
            inference_tp_size=self.args.inference_tp_size,
            release_inference_cache=self.args.release_inference_cache,
            pin_parameters=not self.args.unpin_actor_parameters,
            tp_gather_partition_size=self.args.tp_gather_partition_size,
            max_out_tokens=self.args.max_prompt_seq_len + self.args.max_answer_seq_len,
            enable_mixed_precision_lora=self.args.enable_mixed_precision_lora,
            tb_name="step3_actor",
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.args.per_device_training_batch_size
        ds_config["train_batch_size"] = (
            self.args.per_device_training_batch_size
            * world_size
            * self.args.gradient_accumulation_steps_actor
        )

        actor_model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=actor_model_name_or_path,
            tokenizer=self.actor_tokenizer,
            ds_config=ds_config,
            dropout=self.args.actor_dropout,
        )

        # LoRA
        if self.args.actor_lora_dim > 0:
            actor_model = convert_linear_layer_to_lora(
                actor_model, self.args.actor_lora_module_name, self.args.actor_lora_dim
            )
            if self.args.only_optimize_lora:
                actor_model = only_optimize_lora_parameters(actor_model)
                actor_model = make_model_gradient_checkpointing_compatible(actor_model)

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            actor_model, self.args.actor_weight_decay, self.args.actor_lora_learning_rate
        )
        optim = AdamOptimizer(optim_params, lr=self.args.actor_learning_rate, betas=(0.9, 0.95))

        # LR scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        actor_engine, *_ = deepspeed.initialize(
            model=actor_model, optimizer=optim, lr_scheduler=lr_scheduler, config=ds_config
        )

        _log_init("Actor", stime=stime)
        return actor_engine

    # -----------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------
    def _init_reward(self, reward_model_name_or_path):
        stime = _log_init("Reward")
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        zero_stage = self.args.reward_zero_stage
        if zero_stage != 3:
            zero_stage = 0

        ds_config = get_eval_ds_config(
            offload=self.args.offload_reward_model,
            dtype=self.args.dtype,
            stage=zero_stage,
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.args.per_device_training_batch_size
        ds_config["train_batch_size"] = (
            self.args.per_device_training_batch_size
            * world_size
            * self.args.gradient_accumulation_steps
        )

        assert os.path.exists(self.args.reward_lora_path), (
            f"Reward LoRA adapter not found: {self.args.reward_lora_path}"
        )

        reward_base_model = AutoModelForCausalLM.from_pretrained(reward_model_name_or_path)
        reward_model = PeftModel.from_pretrained(reward_base_model, self.args.reward_lora_path)

        reward_model.eval()
        for param in reward_model.parameters():
            param.requires_grad = False

        reward_engine, *_ = deepspeed.initialize(model=reward_model, config=ds_config)

        _log_init("Reward", stime=stime)
        return reward_engine
