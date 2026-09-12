"""Control loop: Reactor frames -> VLM -> WASD commands back into Reactor."""

from __future__ import annotations

import asyncio
import logging

from .reactor_world import ReactorWorld
from .vlm_client import VLMController

logger = logging.getLogger(__name__)

DEFAULT_GOAL = (
    "Advance the electrode along the surgical corridor toward the pale nucleus "
    "at the centre of the field, keeping it centred in view."
)

SEED_PROMPT = (
    "Endoscopic view inside a human brain during deep brain stimulation surgery, "
    "a slender electrode advancing through soft grey tissue toward a pale nucleus, "
    "wet surgical lighting, shallow depth of field"
)


async def run(
    world: ReactorWorld,
    controller: VLMController,
    goal: str = DEFAULT_GOAL,
    steps: int = 20,
    step_seconds: float = 2.0,
) -> None:
    """Drive the world for `steps` decisions.

    Commands land on the next chunk boundary and movement persists, so each step
    idles every axis before applying the new key set and then lets the world run
    for `step_seconds` before observing again.
    """
    await world.wait_for_first_frame()

    for step in range(1, steps + 1):
        frame = world.latest_frame
        if frame is None:
            await asyncio.sleep(step_seconds)
            continue

        decision = await controller.decide(frame, goal)
        logger.info(
            "step %d/%d keys=%s | %s", step, steps, decision.keys or ["hold"], decision.observation
        )

        await world.release_all()
        for key in decision.keys:
            await world.press(key)
        if decision.prompt:
            logger.info("re-prompting world: %s", decision.prompt)
            await world.set_prompt(decision.prompt)

        await asyncio.sleep(step_seconds)

    await world.release_all()
