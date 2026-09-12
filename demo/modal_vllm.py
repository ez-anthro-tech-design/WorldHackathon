"""Modal-hosted vLLM server exposing an OpenAI-compatible API for the controller.

    modal secret create neuro-vllm VLLM_API_KEY=<your-key>
    modal deploy modal_vllm.py

The printed URL plus `/v1` is VLLM_BASE_URL in demo/.env.local. The endpoint is
network-public but vLLM itself rejects requests without VLLM_API_KEY.
"""

from __future__ import annotations

import json

import modal

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MINUTES = 60
VLLM_PORT = 8000
N_GPU = 1

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("neuro-vllm")


@app.server(
    image=vllm_image,
    gpu=f"A100-40GB:{N_GPU}",
    scaledown_window=15 * MINUTES,
    startup_timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    port=VLLM_PORT,
    secrets=[modal.Secret.from_name("neuro-vllm")],
    target_concurrency=4,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self) -> None:
        import os
        import subprocess

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--api-key",
            os.environ["VLLM_API_KEY"],
            "--tensor-parallel-size",
            str(N_GPU),
            "--max-model-len",
            "8192",
            # The control loop sends exactly one frame per turn.
            "--limit-mm-per-prompt",
            json.dumps({"image": 1, "video": 0, "audio": 0}),
        ]
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
