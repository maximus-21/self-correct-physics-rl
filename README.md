# Self-Correcting Reinforcement Learning for Physics Problem Solving

An implementation of [SCoRe (Training Language Models to Self-Correct via Reinforcement Learning)](https://arxiv.org/abs/2409.12917) adapted for **physics problem solving**. This pipeline trains language models to generate step-by-step physics solutions, receive structured error feedback from a reward model, and improve their answers through multi-turn reinforcement learning.

The original SCoRe method (Kumar et al., 2024) demonstrated that multi-turn online RL can significantly improve an LLM's self-correction ability using entirely self-generated data, achieving state-of-the-art results on MATH and HumanEval. This repository applies the core SCoRe framework to physics reasoning, with a domain-specific reward model that performs **error localization** -- identifying the first incorrect step and its error type (conceptual, computational, or problem miscomprehension).

## Method Overview

SCoRe combines on-policy generation with an error-localization reward model to teach LLMs **self-correction**:

```
Physics Question
       |
       v
 +-----------+
 |   Actor   | ----> Generate Attempt 1 ----> Answer 1
 |   Model   |                                    |
 +-----------+                                    v
                                         +--------------+
                                         | Reward Model | ---> error_step, error_type
                                         +--------------+           |
                                                                    v
                                                           Generate Feedback
                                                           ("Step 3 has a conceptual error...")
                                                                    |
 +-----------+                                                      |
 |   Actor   | <--- Prompt + Answer 1 + Feedback <------------------+
 |   Model   | ----> Generate Attempt 2 ----> Answer 2
 +-----------+                                    |
                                                  v
                                         +--------------+
                                         | Reward Model | ---> reward_2
                                         +--------------+
                                                  |
                                                  v
                                          RL Loss & Backprop
```

### Reward Computation

The reward model acts as an **error localizer** -- it identifies the first incorrect step in a step-by-step physics solution:

| Outcome | Reward |
|---------|--------|
| No errors found (`error_step = 0`) | **1.0** |
| Error at step *k* of *N* total steps | `k / (N + 1)` |
| Failed to parse output | **0.01** |

This encourages the model to push errors later in the reasoning chain (or eliminate them entirely).

### Training Loss

```
loss = -( mean_logprob(attempt_1) + mean_logprob(attempt_2) ) * reward_2 + beta * KL
```

The loss maximizes log-probabilities of both attempts weighted by the quality of the corrected answer, regularized by KL divergence from the base policy to prevent mode collapse.

### Reference Model Trick

Instead of maintaining a separate frozen reference model for KL computation, the code temporarily disables LoRA weights (fusing them into the identity), effectively turning the actor into its own reference. This halves the memory footprint.

## Project Structure

```
self-correct-physics-rl/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .env.example
├── configs/
│   └── default.yaml          # Default hyperparameters
├── score/
│   ├── __init__.py
│   ├── utils.py               # Shared utilities (LoRA, DeepSpeed configs, model creation, etc.)
│   ├── engine.py              # DeepSpeed engine initialization (actor + reward models)
│   └── trainer.py             # Core two-attempt training logic and reward computation
├── scripts/
│   └── train.py               # Main training entry point
└── data/
    └── .gitkeep               # Place your dataset JSON here
```

## Installation

```bash
git clone https://github.com/<your-username>/self-correct-physics-rl.git
cd self-correct-physics-rl

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

## Prerequisites

1. **Hugging Face access**: You need access to gated models (e.g., Llama). Set your token:
   ```bash
   export HF_TOKEN=hf_your_token_here
   # or
   huggingface-cli login
   ```

2. **Reward model adapter**: Train or obtain a LoRA adapter for the error-localization reward model. The adapter should be fine-tuned to output JSON in this format:
   ```json
   {
       "error_step": 3,
       "error_type": "conceptual error",
       "error_explanation": "Incorrectly applied Newton's third law..."
   }
   ```

3. **Dataset**: Prepare a JSON file with the following fields per example:
   ```json
   {
       "question": "A ball is thrown upward...",
       "cot_solution": "## Step 1: Identify given values..."
   }
   ```

## Usage

### Training

```bash
deepspeed --num_gpus 2 scripts/train.py \
    --actor_model meta-llama/Llama-3.2-3B-Instruct \
    --reward_model meta-llama/Llama-3.2-3B-Instruct \
    --reward_lora_path /path/to/reward_lora_adapter \
    --dataset_path data/final_dpo_data.json \
    --output_dir outputs/ \
    --num_epochs 3 \
    --actor_lora_dim 32 \
    --actor_zero_stage 2 \
    --enable_hybrid_engine \
    --inference_tp_size 2 \
    --actor_gradient_checkpointing
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--actor_model` | `meta-llama/Llama-3.2-3B-Instruct` | Actor (policy) model |
| `--reward_model` | `meta-llama/Llama-3.2-3B-Instruct` | Reward model base |
| `--reward_lora_path` | (required) | Path to reward LoRA adapter |
| `--dataset_path` | (required) | Training data JSON |
| `--actor_lora_dim` | 32 | LoRA rank (0 = full fine-tuning) |
| `--actor_zero_stage` | 2 | DeepSpeed ZeRO stage for actor |
| `--enable_hybrid_engine` | True | Use DeepSpeed hybrid engine for generation |
| `--num_epochs` | 3 | Training epochs |
| `--temperature` | 0.7 | Generation temperature |
| `--actor_learning_rate` | 5e-5 | Learning rate |

See `configs/default.yaml` for a complete list of configurable parameters.

## How It Works

1. **Prompt**: Each physics question is formatted with a system prompt instructing step-by-step reasoning.

2. **Attempt 1**: The actor model generates a solution using greedy decoding.

3. **Evaluation**: The reward model (a separate LLM with a fine-tuned LoRA adapter) identifies the first erroneous step and classifies the error type.

4. **Feedback**: The error analysis is converted to natural-language feedback (e.g., *"There is a conceptual error in Step 3..."*).

5. **Attempt 2**: The actor generates a revised solution given the original prompt, its first answer, and the feedback as a multi-turn conversation.

6. **Training**: Both attempts' log-probabilities are used in the RL loss, weighted by the reward of Attempt 2 and regularized by KL divergence.

## Hardware Requirements

- Minimum: 2x GPUs with 24GB VRAM (e.g., RTX 3090/4090) for 3B parameter models with LoRA
- Recommended: 2x A100 40GB for comfortable training with the hybrid engine

## Citation

This project is an implementation of the SCoRe method adapted for physics problem solving. If you use this code, please cite the original paper:

```bibtex
@article{kumar2024training,
  title={Training Language Models to Self-Correct via Reinforcement Learning},
  author={Kumar, Aviral and Zhuang, Vincent and Agarwal, Rishabh and Su, Yi and Co-Reyes, John D and Singh, Avi and Baumli, Kate and Iqbal, Shariq and Bishop, Colton and Roelofs, Rebecca and Zhang, Lei M and McKinney, Kay and Shrivastava, Disha and Paduraru, Cosmin and Tucker, George and Precup, Doina and Behbahani, Feryal and Faust, Aleksandra},
  journal={arXiv preprint arXiv:2409.12917},
  year={2024}
}
```

## Acknowledgements

- Based on [SCoRe](https://arxiv.org/abs/2409.12917) by Kumar et al. (Google DeepMind)
- Built on [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) by Microsoft
- Uses the [PEFT](https://github.com/huggingface/peft) library for LoRA adapters
