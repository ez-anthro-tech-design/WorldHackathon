"""Environment-backed configuration for the demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEMO_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    load_dotenv(DEMO_ROOT / ".env.local")


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy demo/.env.local.example to demo/.env.local and fill it in."
        )
    return value


@dataclass(frozen=True)
class Config:
    reactor_api_key: str
    reactor_model: str
    vllm_base_url: str
    vllm_api_key: str
    vllm_model: str
    seed_image: Path

    @classmethod
    def from_env(cls) -> Config:
        load_env()
        seed_image = Path(os.environ.get("SEED_IMAGE", "assets/seed.jpg"))
        if not seed_image.is_absolute():
            seed_image = DEMO_ROOT / seed_image
        return cls(
            reactor_api_key=_required("REACTOR_API_KEY"),
            reactor_model=os.environ.get("REACTOR_MODEL", "reactor/lingbot-world-2"),
            vllm_base_url=_required("VLLM_BASE_URL"),
            vllm_api_key=os.environ.get("VLLM_API_KEY", "local-dev-key"),
            vllm_model=os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
            seed_image=seed_image,
        )
