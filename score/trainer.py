"""
Core training logic for the SCoRe pipeline.

Implements the two-attempt self-correction loop:
  1. Generate Attempt 1 from the actor model.
  2. Evaluate it with the reward model (error localization).
  3. Produce natural-language feedback from the evaluation.
  4. Generate Attempt 2 using the original prompt + Attempt 1 + feedback.
  5. Evaluate Attempt 2 for the final reward.
  6. Compute RL loss and update the actor model.
"""

import re
import ast

import torch
import torch.nn.functional as F
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from score.utils import LinearLayer_LoRA, disable_lora_weights, print_rank_0


# ---------------------------------------------------------------------------
# Reward model evaluation prompt
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = (
    "You are a physics error localization assistant. You are given a physics question, "
    "a correct step-by-step solution (ground truth), and an LLM-generated solution to the same question. "
    "Your job is to identify the first step in the LLM-generated solution that contains an error (if any).\n"
    "\nYou must compare the LLM-generated solution with the ground truth solution and determine:\n"
    "1. The first erroneous step (i.e., the first step where the LLM diverges from correct reasoning).\n"
    "2. The error type, which must be one of the following:\n"
    '   - "conceptual error": Incorrect use of physics principles, laws, or formulas.\n'
    '   - "computational error": Mistake in arithmetic, algebra, calculus, or unit handling.\n'
    '   - "problem miscomprehension": Misunderstanding of the problem statement or given data.\n'
    "3. A brief error explanation describing what the mistake is and why it is incorrect.\n"
    "4. If there are no errors and the LLM-generated solution is completely correct, "
    "mention this in error_explanation and return 0 in error step.\n"
    "\nReturn your result strictly in this JSON format:\n"
    "{\n"
    '    "error_step": <step_number>,\n'
    '    "error_type": "<conceptual error | computational error | problem miscomprehension | no error>",\n'
    '    "error_explanation": "<brief explanation>"\n'
    "}"
)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DeepSpeedScoreTrainer:
    """
    Orchestrates the SCoRe self-correction training loop.

    Args:
        rlhf_engine: A ``DeepSpeedScoreEngine`` instance.
        args: Training arguments namespace.
    """

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = rlhf_engine.actor
        self.ref_model = rlhf_engine.ref
        self.reward_model = rlhf_engine.reward
        self.actor_tokenizer = rlhf_engine.actor_tokenizer
        self.reward_tokenizer = rlhf_engine.reward_tokenizer
        self.args = args
        self.z3_enabled = args.actor_zero_stage == 3
        self.kl_ctl = 0.1
        self.beta2 = 0.1

    # -----------------------------------------------------------------
    # Experience generation (the two-attempt loop)
    # -----------------------------------------------------------------

    def generate_experience(self, batch):
        """
        Run the full two-attempt self-correction loop on *batch*.

        Returns a dictionary with log-probs, rewards, masks, KL, and
        decoded answers/feedback for downstream training and logging.
        """
        self.eval()

        prompt_texts = batch["prompt"]
        ground_truths = batch["cot_solution"]
        questions = batch["question"]

        # --- Attempt 1 ---
        x1 = self.actor_tokenizer.batch_encode_plus(
            prompt_texts,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_prompt_len_attempt1,
            return_tensors="pt",
        )
        x1 = {k: v.to(self.args.device) for k, v in x1.items()}

        with torch.no_grad():
            action1_token = self.actor_model.module.generate(
                x1["input_ids"],
                attention_mask=x1["attention_mask"],
                max_new_tokens=self.args.max_new_tokens_attempt1,
                pad_token_id=self.actor_tokenizer.pad_token_id,
                temperature=self.args.temperature,
                synced_gpus=self.z3_enabled,
                do_sample=False,
            )

        x1_len = x1["input_ids"].shape[1]
        attempt1_answer_tokens = action1_token[:, x1_len:]
        attempt1_answer_mask = self._get_eos_mask(attempt1_answer_tokens)
        answer1 = self.actor_tokenizer.batch_decode(attempt1_answer_tokens, skip_special_tokens=True)

        print_rank_0(f"\n[Attempt 1] {answer1[0][:200]}...")

        # --- Reward 1 ---
        reward_1, decoded_batch = [], []
        for q, gt, a in zip(questions, ground_truths, answer1):
            r1, decoded = self._generate_reward(
                {"question": q, "cot_solution": gt, "incorrect_solution": a}
            )
            reward_1.append(r1)
            decoded_batch.append(decoded)

        print_rank_0(f"[Reward 1] {reward_1[0]:.4f}")

        # --- Feedback ---
        feedbacks = [self._generate_feedback(d) for d in decoded_batch]
        print_rank_0(f"[Feedback] {feedbacks[0][:200]}...")

        # --- Attempt 2 ---
        revised_prompts = [
            self.actor_tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": pt},
                    {"role": "assistant", "content": ans},
                    {"role": "user", "content": fb},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for pt, ans, fb in zip(prompt_texts, answer1, feedbacks)
        ]

        x2 = self.actor_tokenizer.batch_encode_plus(
            revised_prompts,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_prompt_len_attempt2,
            return_tensors="pt",
        )
        x2 = {k: v.to(self.args.device) for k, v in x2.items()}

        with torch.no_grad():
            action2_token = self.actor_model.module.generate(
                x2["input_ids"],
                attention_mask=x2["attention_mask"],
                max_new_tokens=self.args.max_new_tokens_attempt2,
                pad_token_id=self.actor_tokenizer.pad_token_id,
                temperature=self.args.temperature,
                synced_gpus=self.z3_enabled,
                do_sample=False,
            )

        x2_len = x2["input_ids"].shape[1]
        attempt2_answer_tokens = action2_token[:, x2_len:]
        attempt2_answer_mask = self._get_eos_mask(attempt2_answer_tokens)
        answer2 = self.actor_tokenizer.batch_decode(attempt2_answer_tokens, skip_special_tokens=True)

        print_rank_0(f"\n[Attempt 2] {answer2[0][:200]}...")

        # --- Reward 2 ---
        reward_2 = []
        for q, gt, a in zip(questions, ground_truths, answer2):
            r2, _ = self._generate_reward(
                {"question": q, "cot_solution": gt, "incorrect_solution": a}
            )
            reward_2.append(r2)

        print_rank_0(f"[Reward 2] {reward_2[0]:.4f}")

        # --- Log-probs and KL ---
        action1_token = action1_token.clone()

        with torch.no_grad():
            # Reference log-probs via LoRA-disable trick
            with disable_lora_weights(self.actor_model):
                _, base_probs = self._get_log_probs(
                    self.actor_model, action1_token, x1_len, return_probs=True
                )

        att1_log_probs, att1_probs = self._get_log_probs(
            self.actor_model, action1_token, x1_len, return_probs=True
        )
        kl_div = self._get_kl_div(base_probs, att1_probs.detach(), attempt1_answer_mask)

        att2_log_probs = self._get_log_probs_incremental(
            model=self.actor_model.module,
            input_ids=x2["input_ids"],
            generated_ids=attempt2_answer_tokens,
            device=self.args.device,
        )

        return {
            "logprob1": att1_log_probs,
            "logprob2": att2_log_probs,
            "reward1": reward_1,
            "reward2": reward_2,
            "mask1": attempt1_answer_mask,
            "mask2": attempt2_answer_mask,
            "kl_div": kl_div,
            "answer1": answer1,
            "feedback": feedbacks,
            "answer2": answer2,
        }

    # -----------------------------------------------------------------
    # RL training step
    # -----------------------------------------------------------------

    def train_rlhf(self, inputs):
        """
        Compute the SCoRe RL loss and update the actor model.

        Loss = -( (mean_logprob_att1 + mean_logprob_att2) * reward2 - beta * KL )
        """
        att1_log_probs = inputs["logprob1"]
        att2_log_probs = inputs["logprob2"]
        reward2_tensor = torch.tensor(inputs["reward2"], dtype=torch.float32).to(self.args.device)
        mask1 = inputs["mask1"]
        mask2 = inputs["mask2"]
        kl_div = inputs["kl_div"]

        loss = -(
            (
                (att1_log_probs[:, 1:] * mask1[:, 1:]).sum(-1) / mask1[:, 1:].sum(-1)
                + (att2_log_probs * mask2[:, 1:]).sum(-1) / mask2[:, 1:].sum(-1)
            )
            * reward2_tensor
            - self.beta2 * kl_div
        ).mean()

        self.actor_model.backward(loss)
        self.actor_model.step()

        return loss

    # -----------------------------------------------------------------
    # Mode switching
    # -----------------------------------------------------------------

    def eval(self):
        self.actor_model.eval()
        self.reward_model.eval()

    def train(self):
        self.actor_model.train()

    # -----------------------------------------------------------------
    # Reward computation
    # -----------------------------------------------------------------

    def _generate_reward(self, sample):
        """
        Use the reward model to evaluate a solution.

        Returns:
            reward (float): Proportional to the first error step.
                1.0 = no errors, 0.01 = parse failure / no steps found.
            decoded: Parsed JSON dict from the reward model, or raw string on failure.
        """
        prompt = self._build_eval_prompt(sample)
        inputs = self.reward_tokenizer(prompt, return_tensors="pt", padding=True).to(self.args.device)

        with torch.no_grad():
            outputs = self.reward_model.module.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                pad_token_id=self.reward_tokenizer.pad_token_id,
                temperature=0.7,
                synced_gpus=self.z3_enabled,
                do_sample=False,
            )

        decoded = self.reward_tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            decoded = ast.literal_eval(decoded.split("assistant\n\n")[-1])
            error_step = decoded["error_step"]
            if error_step == 0:
                return 1.0, decoded

            step_pattern = r"(?:#+\s*)?Step\s+\d+:"
            matches = re.findall(step_pattern, sample["incorrect_solution"])
            if matches:
                total_steps = len(matches)
                return error_step / (total_steps + 1), decoded

            print_rank_0("No steps found in the solution. Assigning minimal reward.")
            return 0.01, decoded

        except Exception:
            print_rank_0("Failed to parse reward model output. Assigning minimal reward.")
            decoded_str = decoded.split("assistant\n\n")[-1]
            return 0.01, decoded_str

    def _build_eval_prompt(self, sample):
        """Format the evaluation prompt for the reward model."""
        prompt = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{sample['question']}\n\n"
                    f"GROUND TRUTH SOLUTION:\n{sample['cot_solution']}\n\n"
                    f"LLM GENERATED SOLUTION:\n{sample['incorrect_solution']}\n\n"
                    "Compare the LLM-generated solution with the ground truth. "
                    "Identify and explain the first error step (if any) in the LLM-generated solution. "
                    "Return your response in the specified JSON format."
                ),
            },
        ]
        return self.reward_tokenizer.apply_chat_template(prompt, tokenize=False)

    @staticmethod
    def _generate_feedback(decoded):
        """Convert the reward model's parsed output into natural-language feedback."""
        try:
            if decoded["error_step"] == 0:
                return "There is no error in your solution."

            return (
                f"There is a {decoded['error_type']} in Step {decoded['error_step']}.\n"
                f"The mistake occurred because: {decoded['error_explanation']}.\n"
                "Please revise the solution by correcting this step and continuing from there.\n\n"
                "Strictly follow this format:\n"
                "## Step 1: <step_1>\n"
                "## Step 2: <step_2>\n"
                "... and so on until the final answer is derived.\n"
            )
        except Exception:
            print_rank_0("Failed to generate structured feedback; returning raw output.")
            return str(decoded)

    # -----------------------------------------------------------------
    # Log-probability computation
    # -----------------------------------------------------------------

    def _get_log_probs(self, model, input_ids, prompt_len, return_probs=False):
        """
        Compute per-token log-probabilities for the generated portion of *input_ids*.

        Args:
            model: The language model (DeepSpeed engine).
            input_ids: Full sequence [prompt + generation].
            prompt_len: Length of the prompt prefix.
            return_probs: If True, also return the full log-prob distribution.
        """
        logits = model(input_ids, use_cache=False).logits
        logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:]

        per_token_logps = []
        all_log_probs = [] if return_probs else None

        for logits_row, ids_row in zip(logits, target_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            if return_probs:
                all_log_probs.append(log_probs)
            token_log_prob = torch.gather(log_probs, dim=1, index=ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)

        stacked = torch.stack(per_token_logps)[:, prompt_len - 1:]
        if not return_probs:
            return stacked
        return stacked, torch.stack(all_log_probs)[:, prompt_len - 1:]

    @torch.no_grad()
    def _get_log_probs_incremental(self, model, input_ids, generated_ids, device):
        """
        Compute log-probs for *generated_ids* given *input_ids* using KV caching.

        More memory-efficient for long prompts (Attempt 2).
        """
        model.eval().to(device)

        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True)
            past_key_values = out.past_key_values

        input_ids_gen = generated_ids[:, :-1]
        targets_gen = generated_ids[:, 1:]

        out = model(input_ids=input_ids_gen, past_key_values=past_key_values, use_cache=False)
        log_probs = F.log_softmax(out.logits, dim=-1)
        logprobs = torch.gather(log_probs, dim=-1, index=targets_gen.unsqueeze(-1)).squeeze(-1)

        return logprobs

    # -----------------------------------------------------------------
    # Masking and KL
    # -----------------------------------------------------------------

    def _get_eos_mask(self, answer_ids):
        """Create a binary mask that is 1 up to and including the first EOS token."""
        is_eos = answer_ids == self.actor_tokenizer.eos_token_id
        mask = (torch.cumsum(is_eos, dim=1) <= 1).int()
        return mask

    @staticmethod
    def _get_kl_div(base_logprobs, logprobs, mask):
        """Compute masked KL divergence between two log-probability distributions."""
        kl_div = F.kl_div(logprobs, base_logprobs, reduction="none", log_target=True)
        return (kl_div.mean(-1) * mask).sum(-1) / mask.sum(-1)
