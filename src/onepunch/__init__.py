"""OnePunchBench: one prompt to predict LLM benchmark scores."""

from onepunch.utils.env import load_env as _load_env

_load_env()  # .env (HF_TOKEN, HF_HOME, ...) before anything imports huggingface_hub
