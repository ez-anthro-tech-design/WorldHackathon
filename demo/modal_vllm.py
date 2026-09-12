"""Modal-hosted vLLM server exposing an OpenAI-compatible API for the controller.

    modal secret create neuro-vllm VLLM_API_KEY=<your-key>
    modal deploy modal_vllm.py

The printed URL plus `/v1` is VLLM_BASE_URL in demo/.env.local.
"""

from __future__ import annotations

import subprocess

import modal

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
VLLM_PORT = 8000
GPU = "A100-40GB"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.7.3", "huggingface_hub[hf_transfer]==0.29.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "0"})
)

hf_cache = modal.Volume.from_name("neuro-vllm-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("neuro-vllm-cache", create_if_missing=True)

app = modal.App("neuro-vllm")


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=15 * 60,
    timeout=30 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("neuro-vllm")],
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * 60)
def serve() -> None:
    import os

    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--api-key",
            os.environ["VLLM_API_KEY"],
            "--max-model-len",
            "8192",
            # One image per request; keeps KV cache small enough for a single A100.
            "--limit-mm-per-prompt",
            "image=1",
        ]
    )
