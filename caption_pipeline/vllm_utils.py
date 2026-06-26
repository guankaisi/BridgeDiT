"""vLLM helper utilities for local / multi-GPU runs."""
import os


def get_tensor_parallel_size(default: int = 1) -> int:
    """Read tensor parallel size from env, falling back to *default*."""
    return int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", default))
