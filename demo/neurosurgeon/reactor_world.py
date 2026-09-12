"""Reactor world session: image-anchored generation plus WASD-style controls.

Wraps the base `reactor_sdk.Reactor` client for `reactor/lingbot-world-2`. The
model emits `main_video`; every client-to-model interaction is a command.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
from reactor_sdk import Reactor, Track

logger = logging.getLogger(__name__)

# WASD, as the reference frontend binds it. Movement is persistent state on the
# model, so "release" is an explicit idle rather than a keyup event.
KEY_BINDINGS: dict[str, tuple[str, str]] = {
    "w": ("set_move_longitudinal", "forward"),
    "s": ("set_move_longitudinal", "back"),
    "a": ("set_move_lateral", "strafe_left"),
    "d": ("set_move_lateral", "strafe_right"),
    "left": ("set_look_horizontal", "left"),
    "right": ("set_look_horizontal", "right"),
    "up": ("set_look_vertical", "up"),
    "down": ("set_look_vertical", "down"),
}

AXIS_PARAMS: dict[str, str] = {
    "set_move_longitudinal": "move_longitudinal",
    "set_move_lateral": "move_lateral",
    "set_look_horizontal": "look_horizontal",
    "set_look_vertical": "look_vertical",
}


class ReactorWorld:
    """A live Reactor session with the latest rendered frame kept on hand."""

    def __init__(self, model_name: str, api_key: str, video_track: str = "main_video") -> None:
        self._reactor = Reactor(model_name=model_name, api_key=api_key)
        self._video_track = video_track
        self._latest_frame: np.ndarray | None = None
        self._first_frame = asyncio.Event()
        self._reactor.on_track(self._attach_track)

    def _attach_track(self, track: Track) -> None:
        if track.name != self._video_track:
            return

        @track.on_frame
        def _on_frame(frame: np.ndarray) -> None:
            self._latest_frame = frame
            self._first_frame.set()

    async def connect(self) -> None:
        await self._reactor.connect()

    async def disconnect(self) -> None:
        await self._reactor.disconnect()

    async def stage(self, seed_image: Path, prompt: str) -> None:
        """Set the reference image and prompt, then start generation.

        `start` fails until both are set.
        """
        ref = await self._reactor.upload_file(str(seed_image))
        await self._reactor.send_command("set_image", {"image": ref})
        await self._reactor.send_command("set_prompt", {"prompt": prompt})
        await self._reactor.send_command("start", {})

    async def set_prompt(self, prompt: str) -> None:
        await self._reactor.send_command("set_prompt", {"prompt": prompt})

    async def press(self, key: str) -> None:
        """Hold a WASD/arrow key until the matching axis is idled."""
        command, value = KEY_BINDINGS[key.lower()]
        await self._reactor.send_command(command, {AXIS_PARAMS[command]: value})

    async def release_all(self) -> None:
        for command, param in AXIS_PARAMS.items():
            await self._reactor.send_command(command, {param: "idle"})

    async def set_rotation_speed(self, degrees: float) -> None:
        await self._reactor.send_command("set_rotation_speed_deg", {"rotation_speed_deg": degrees})

    async def wait_for_first_frame(self, timeout: float = 120.0) -> None:
        await asyncio.wait_for(self._first_frame.wait(), timeout)

    @property
    def latest_frame(self) -> np.ndarray | None:
        return self._latest_frame
