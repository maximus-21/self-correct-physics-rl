# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utility functions for the SCoRe training pipeline.

Includes:
  - Distributed printing helpers
  - Seed initialization
  - DeepSpeed config builders
  - LoRA layer definitions and conversion utilities
  - HuggingFace model creation helpers
  - Optimizer parameter grouping
  - Model saving utilities
"""

import os
import math
import random
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.modeling_utils import no_init_weights

import deepspeed
import deepspeed.comm as dist
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
from deepspeed.accelerator import get_accelerator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_rank_0():
    """Return True if this is rank 0 or if distributed is not initialized."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def print_rank_0(msg, rank=None):
    """Print *msg* only on rank 0."""
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)


# ---------------------------------------------------------------------------
# DeepSpeed config builders
# ---------------------------------------------------------------------------

def get_train_ds_config(
    offload,
    dtype,
    stage=2,
    enable_hybrid_engine=False,
    inference_tp_size=1,
    release_inference_cache=False,
    pin_parameters=True,
    tp_gather_partition_size=8,
    max_out_tokens=512,
    enable_tensorboard=False,
    enable_mixed_precision_lora=False,
    tb_path="",
    tb_name="",
):
    """Build a DeepSpeed training config dictionary."""
    device = "cpu" if offload else "none"

    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    zero_opt_dict = {
        "stage": stage,
        "overlap_comm": True,
        "offload_param": {"device": device},
        "offload_optimizer": {"device": device},
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False,
    }

    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != get_accelerator().device_count():
            zero_opt_dict["zero_hpz_partition_size"] = (
                get_accelerator().device_count()
            )

    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard",
        },
    }


def get_eval_ds_config(offload, dtype, stage=0):
    """Build a DeepSpeed evaluation (inference) config dictionary."""
    device = "cpu" if offload else "none"

    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {"device": device},
        "memory_efficient_linear": False,
    }

    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


# ---------------------------------------------------------------------------
# LoRA implementation
# ---------------------------------------------------------------------------

class LinearLayer_LoRA(nn.Module):
    """
    A simple LoRA wrapper around ``nn.Linear``.

    Output = Linear(x) + Dropout(x) @ B @ A * scaling
    where A (left) is [lora_dim, out_features] and B (right) is [in_features, lora_dim].
    """

    def __init__(self, weight, lora_dim=0, lora_scaling=1, lora_dropout=0, bias=None):
        super().__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError("LoRA reduced dim must be > 0")

        try:
            rows, columns = weight.ds_shape  # ZeRO-3
        except AttributeError:
            rows, columns = weight.shape

        self.lora_right_weight = nn.Parameter(torch.zeros(columns, lora_dim))
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        self.weight.requires_grad = False
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    def train(self, mode=True):
        self.lora_dropout.train(mode)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        """Fuse LoRA deltas into the base weight (for inference)."""
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t()
            )
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        """Remove fused LoRA deltas from the base weight."""
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t()
            )
        self.fuse_lora = False

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        return F.linear(input, self.weight, self.bias) + (
            self.lora_dropout(input) @ self.lora_right_weight @ self.lora_left_weight
        ) * self.lora_scaling


def _z3_params_to_fetch(param_list):
    """Return ZeRO-3 params that need gathering."""
    return [
        p
        for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def convert_linear_layer_to_lora(
    model, part_module_name, lora_dim=0, lora_scaling=1, lora_dropout=0
):
    """Replace ``nn.Linear`` layers whose name contains *part_module_name* with LoRA layers."""
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_LoRA(
            module.weight, lora_dim, lora_scaling, lora_dropout, module.bias
        ).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model


def convert_lora_to_linear_layer(model):
    """Fuse LoRA weights into base linear layers (ZeRO-3 aware)."""
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_LoRA):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, "ds_id")
        with deepspeed.zero.GatheredParameters(
            _z3_params_to_fetch(
                [module.weight, module.bias, module.lora_left_weight, module.lora_right_weight]
            ),
            modifier_rank=0,
            enabled=zero_stage_3,
        ):
            module.fuse_lora_weight()
    return model


def only_optimize_lora_parameters(model, force_optimize_params=None):
    """Freeze everything except LoRA parameters (and optionally *force_optimize_params*)."""
    if force_optimize_params is None:
        force_optimize_params = []
    for name, param in model.named_parameters():
        if (
            "lora_right_weight" in name
            or "lora_left_weight" in name
            or name in force_optimize_params
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def make_model_gradient_checkpointing_compatible(model):
    """Enable gradient checkpointing for LoRA-only training."""
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model


@contextlib.contextmanager
def disable_lora_weights(model):
    """
    Context manager that temporarily fuses LoRA weights, effectively
    turning the model into its base (reference) version.
    """
    original_flags = []
    for module in model.modules():
        if isinstance(module, LinearLayer_LoRA):
            original_flags.append((module, module.fuse_lora))
            module.fuse_lora = True  # disable residual LoRA path
    try:
        yield
    finally:
        for module, was_fused in original_flags:
            module.fuse_lora = was_fused


# ---------------------------------------------------------------------------
# HuggingFace model creation
# ---------------------------------------------------------------------------

def configure_dropout(model_config, dropout):
    """Set all dropout-related config fields to *dropout*."""
    if dropout is not None:
        for key in ("dropout", "attention_dropout", "hidden_dropout", "activation_dropout"):
            if hasattr(model_config, key):
                setattr(model_config, key, dropout)


def create_hf_model(
    model_class, model_name_or_path, tokenizer, ds_config=None, dropout=None
):
    """
    Create a HuggingFace causal LM from pretrained weights.

    Handles ZeRO-3 initialization, dropout configuration, and token
    embedding resizing.
    """
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    configure_dropout(model_config, dropout)

    # ZeRO-3 requires HfDeepSpeedConfig in scope
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841
    else:
        dschf = None  # noqa: F841

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
    )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    return model


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------

def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=None,
    lora_name_list=None,
):
    """
    Group model parameters for the optimizer:
      1. Base params with weight decay
      2. LoRA params with separate LR and weight decay
      3. Norm / bias params without weight decay
    """
    if no_decay_name_list is None:
        no_decay_name_list = [
            "bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"
        ]
    if lora_name_list is None:
        lora_name_list = ["lora_right_weight", "lora_left_weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (
                    not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                    and not any(nd in n.lower() for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (
                    not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                    and any(nd in n.lower() for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
            "lr": lora_lr,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n.lower() for nd in no_decay_name_list) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    return [g for g in optimizer_grouped_parameters if g["params"]]


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------

def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    """
    Save model weights to disk, handling ZeRO-3 parameter partitioning.

    LoRA parameters are excluded from the saved state dict.
    """
    zero_stage_3 = zero_stage == 3
    os.makedirs(save_dir, exist_ok=True)
    output_model_file = os.path.join(save_dir, "pytorch_model.bin")

    model_to_save = model_ema.module if hasattr(model_ema, "module") else model_ema

    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, "ds_id"):
                with deepspeed.zero.GatheredParameters(
                    _z3_params_to_fetch([v]), enabled=zero_stage_3
                ):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict
