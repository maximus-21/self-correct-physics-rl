"""
Main training script for the SCoRe (Self-Correcting Reinforcement) pipeline.

Launch with DeepSpeed::

    deepspeed --num_gpus <N> scripts/train.py \\
        --actor_model meta-llama/Llama-3.2-3B-Instruct \\
        --reward_model meta-llama/Llama-3.2-3B-Instruct \\
        --reward_lora_path /path/to/reward_lora_adapter \\
        --dataset_path data/final_dpo_data.json \\
        --output_dir outputs/
"""

import argparse
import gc
import json
import os
import time

import torch
import deepspeed
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from deepspeed.accelerator import get_accelerator

from score.engine import DeepSpeedScoreEngine
from score.trainer import DeepSpeedScoreTrainer
from score.utils import print_rank_0, set_random_seed, save_zero_three_model


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a physics expert assistant. "
    "You are given a physics problem. Your task is to solve it in a clear and rigorous step-by-step format. "
    "Each step should be clearly numbered.\n"
    "Strictly follow this format:\n"
    "## Step 1: <step_1>\n"
    "## Step 2: <step_2>\n"
    "... and so on until the final answer is derived.\n\n"
)


def generate_prompt(sample, tokenizer):
    """Format a single physics problem into a chat-style prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample.get("question", "")},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def format_example(sample, tokenizer):
    """Add a ``prompt`` field to *sample*."""
    sample["prompt"] = generate_prompt(sample, tokenizer)
    return sample


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SCoRe RLHF Training")

    # Distributed training
    parser.add_argument("--local_rank", type=int, default=-1)

    # Model paths
    parser.add_argument("--actor_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="HuggingFace model id or local path for the actor model.")
    parser.add_argument("--reward_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="HuggingFace model id or local path for the reward base model.")
    parser.add_argument("--reward_lora_path", type=str, required=True,
                        help="Path to the LoRA adapter for the reward model.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the training dataset JSON file.")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Directory for saving checkpoints and logs.")

    # DeepSpeed / offloading
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--actor_zero_stage", type=int, default=2)
    parser.add_argument("--reward_zero_stage", type=int, default=0)
    parser.add_argument("--offload_reward_model", action="store_true")
    parser.add_argument("--actor_gradient_checkpointing", action="store_true", default=True)

    # LoRA
    parser.add_argument("--actor_lora_dim", type=int, default=32,
                        help="LoRA rank. Set to 0 to disable LoRA.")
    parser.add_argument("--actor_lora_module_name", type=str, default="layers.",
                        help="Substring to match when converting linear layers to LoRA.")
    parser.add_argument("--only_optimize_lora", action="store_true", default=True)

    # Tokenization / generation
    parser.add_argument("--max_prompt_seq_len", type=int, default=2480)
    parser.add_argument("--max_answer_seq_len", type=int, default=1024)
    parser.add_argument("--max_prompt_len_attempt1", type=int, default=512,
                        help="Max prompt length for Attempt 1.")
    parser.add_argument("--max_new_tokens_attempt1", type=int, default=1240,
                        help="Max new tokens to generate for Attempt 1.")
    parser.add_argument("--max_prompt_len_attempt2", type=int, default=2480,
                        help="Max prompt length for Attempt 2 (prompt + answer1 + feedback).")
    parser.add_argument("--max_new_tokens_attempt2", type=int, default=1240,
                        help="Max new tokens to generate for Attempt 2.")
    parser.add_argument("--temperature", type=float, default=0.7)

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_training_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--actor_dropout", type=float, default=0.005)

    # Learning rate
    parser.add_argument("--actor_learning_rate", type=float, default=5e-5)
    parser.add_argument("--actor_lora_learning_rate", type=float, default=5e-5)
    parser.add_argument("--actor_weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=50)

    # Hybrid engine
    parser.add_argument("--enable_hybrid_engine", action="store_true", default=True)
    parser.add_argument("--inference_tp_size", type=int, default=2)
    parser.add_argument("--release_inference_cache", action="store_true")
    parser.add_argument("--unpin_actor_parameters", action="store_true")
    parser.add_argument("--tp_gather_partition_size", type=int, default=128)

    # Mixed precision
    parser.add_argument("--enable_mixed_precision_lora", action="store_true")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Distributed setup ---
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()

    args.device = device
    args.global_rank = torch.distributed.get_rank()
    args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    set_random_seed(args.seed)
    torch.distributed.barrier()

    # --- Tokenizers ---
    actor_tokenizer = AutoTokenizer.from_pretrained(args.actor_model, use_fast=False)
    actor_tokenizer.pad_token = actor_tokenizer.eos_token

    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=False)
    reward_tokenizer.pad_token = reward_tokenizer.eos_token

    # --- Engine ---
    rlhf_engine = DeepSpeedScoreEngine(
        actor_model_name_or_path=args.actor_model,
        reward_model_name_or_path=args.reward_model,
        actor_tokenizer=actor_tokenizer,
        reward_tokenizer=reward_tokenizer,
        args=args,
        num_total_iters=1,
    )

    rlhf_engine.reward.eval()
    for param in rlhf_engine.reward.parameters():
        param.requires_grad = False

    # --- Dataset ---
    raw_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    formatted_dataset = raw_dataset.map(format_example, fn_kwargs={"tokenizer": actor_tokenizer})

    dataset = formatted_dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = dataset["train"]

    sampler = DistributedSampler(train_dataset) if torch.distributed.is_initialized() else None
    dataloader = DataLoader(train_dataset, batch_size=1, sampler=sampler)

    # --- Trainer ---
    trainer = DeepSpeedScoreTrainer(rlhf_engine, args)

    # --- Training loop ---
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        print_rank_0(f"\n==== Epoch {epoch + 1}/{args.num_epochs} ====\n")

        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            start_time = time.time()

            out = trainer.generate_experience(batch)

            # --- Logging ---
            rank = torch.distributed.get_rank()
            log_path = os.path.join(args.output_dir, f"rlhf_outputs_rank{rank}.jsonl")
            for i in range(len(out["answer1"])):
                log_item = {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "answer1": out["answer1"][i],
                    "reward1": float(out["reward1"][i]),
                    "feedback": out["feedback"][i],
                    "answer2": out["answer2"][i],
                    "reward2": float(out["reward2"][i]),
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_item) + "\n")

            # --- Training step ---
            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_enable()

            loss = trainer.train_rlhf(out)
            print_rank_0(f"[Epoch {epoch + 1} | Step {step + 1}] Loss: {loss:.4f}")

            elapsed = time.time() - start_time
            print_rank_0(f"Step time: {elapsed:.2f}s")

            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()

            del out, loss
            gc.collect()
            torch.cuda.empty_cache()

        # --- Save checkpoint ---
        if torch.distributed.get_rank() == 0:
            save_path = os.path.join(args.output_dir, f"actor_epoch_{epoch + 1}")
            os.makedirs(save_path, exist_ok=True)
            save_zero_three_model(
                rlhf_engine.actor,
                global_rank=0,
                save_dir=save_path,
                zero_stage=args.actor_zero_stage,
            )
            actor_tokenizer.save_pretrained(save_path)
            print_rank_0(f"Saved actor model at {save_path}")


if __name__ == "__main__":
    main()
