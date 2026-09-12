"""Offline checks for the control loop: no Reactor or vLLM credentials needed."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from neurosurgeon import agent
from neurosurgeon.reactor_world import AXIS_PARAMS, KEY_BINDINGS
from neurosurgeon.vlm_client import Decision, VLMController


class FakeWorld:
    def __init__(self) -> None:
        self.latest_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.commands: list[tuple[str, str]] = []

    async def wait_for_first_frame(self, timeout: float = 120.0) -> None:
        return None

    async def press(self, key: str) -> None:
        command, value = KEY_BINDINGS[key]
        self.commands.append((command, value))

    async def release_all(self) -> None:
        for command in AXIS_PARAMS:
            self.commands.append((command, "idle"))

    async def set_prompt(self, prompt: str) -> None:
        self.commands.append(("set_prompt", prompt))


class FakeController:
    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = decisions

    async def decide(self, frame: np.ndarray, goal: str) -> Decision:
        return self._decisions.pop(0)


def test_bindings_cover_every_axis() -> None:
    assert {command for command, _ in KEY_BINDINGS.values()} == set(AXIS_PARAMS)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"observation": "grey tissue", "keys": ["w", "a"], "prompt": null}', ["w", "a"]),
        ('{"observation": "hold", "keys": [], "prompt": null}', []),
        ('{"observation": "junk", "keys": ["space", "W"], "prompt": null}', ["w"]),
        ("not json at all", []),
    ],
)
def test_parse_filters_to_valid_keys(content: str, expected: list[str]) -> None:
    assert VLMController._parse(content).keys == expected


def test_blank_prompt_is_dropped() -> None:
    decision = VLMController._parse('{"observation": "x", "keys": [], "prompt": "   "}')
    assert decision.prompt is None


def test_loop_idles_every_axis_before_applying_keys() -> None:
    world = FakeWorld()
    controller = FakeController(
        [
            Decision(observation="advance", keys=["w"], prompt=None),
            Decision(observation="entered nucleus", keys=[], prompt="pale nucleus in view"),
        ]
    )

    asyncio.run(agent.run(world, controller, steps=2, step_seconds=0.0))

    idles = [("set_move_longitudinal", "idle")] + [
        (command, "idle") for command in AXIS_PARAMS if command != "set_move_longitudinal"
    ]
    assert world.commands[: len(idles)] == idles
    assert ("set_move_longitudinal", "forward") in world.commands
    assert ("set_prompt", "pale nucleus in view") in world.commands
    # The loop leaves the world stationary rather than mid-move.
    assert world.commands[-len(AXIS_PARAMS) :] == [(command, "idle") for command in AXIS_PARAMS]
