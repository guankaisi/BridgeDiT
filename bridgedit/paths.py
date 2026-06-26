import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BRIDGEDIT_ROOT = os.path.dirname(__file__)


def model_root() -> str:
    return os.environ.get("T2SV_MODEL_ROOT", os.path.join(REPO_ROOT, "models"))


def model_path(*parts: str) -> str:
    return os.path.join(model_root(), *parts)


def data_root() -> str:
    return os.environ.get("T2SV_DATA_ROOT", os.path.join(REPO_ROOT, "data"))


def caption_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, "caption_pipeline", *parts)


def default_ckpt_path(*parts: str) -> str:
    return model_path("bridgedit", *parts)
