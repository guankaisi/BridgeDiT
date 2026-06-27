import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def model_root() -> str:
    return os.environ.get("T2SV_MODEL_ROOT", os.path.join(REPO_ROOT, "models"))


def model_path(*parts: str) -> str:
    return os.path.join(model_root(), *parts)
