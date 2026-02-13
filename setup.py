from setuptools import setup, find_packages

setup(
    name="self-correct-physics-rl",
    version="0.1.0",
    description="SCoRe: Self-Correcting Reinforcement Learning for Physics Problem Solving",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "deepspeed>=0.14.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "huggingface_hub>=0.22.0",
        "tqdm",
        "numpy",
    ],
)
