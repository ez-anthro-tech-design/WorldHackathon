"""Vision-language controller backed by vLLM on Modal (OpenAI-compatible API)."""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass

import numpy as np
from openai import AsyncOpenAI
from PIL import Image

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are steering a surgical-arm camera through a simulated brain.

Each turn you see the current view and reply with a single JSON object:
{"observation": "<one sentence>", "keys": [<held keys>], "prompt": "<scene prompt or null>"}

Valid keys: "w" (advance), "s" (retract), "a" (strafe left), "d" (strafe right),
"left", "right" (yaw), "up", "down" (pitch). An empty list holds position.

Keys are held state, not taps: whatever you list stays applied until your next
turn, so list only what should be active now. Set "prompt" only when the scene
itself should change (entering a new structure, bleeding, stimulation applied);
otherwise use null. Reply with JSON only."""

VALID_KEYS = {"w", "a", "s", "d", "left", "right", "up", "down"}


@dataclass
class Decision:
    observation: str
    keys: list[str]
    prompt: str | None


def _encode(frame: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(frame).convert("RGB").save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class VLMController:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def decide(self, frame: np.ndarray, goal: str) -> Decision:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Goal: {goal}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_encode(frame)}"},
                        },
                    ],
                },
            ],
            max_tokens=200,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return self._parse(response.choices[0].message.content or "")

    @staticmethod
    def _parse(content: str) -> Decision:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Controller returned non-JSON, holding position: %s", content)
            return Decision(observation="unparseable response", keys=[], prompt=None)

        raw_keys = payload.get("keys") or []
        keys = [k.lower() for k in raw_keys if isinstance(k, str) and k.lower() in VALID_KEYS]
        prompt = payload.get("prompt")
        return Decision(
            observation=str(payload.get("observation", "")),
            keys=keys,
            prompt=prompt if isinstance(prompt, str) and prompt.strip() else None,
        )
