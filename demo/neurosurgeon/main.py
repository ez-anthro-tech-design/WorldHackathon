"""Entry point: python -m neurosurgeon.main [--steps N] [--goal ...]"""

from __future__ import annotations

import argparse
import asyncio
import logging

from . import agent
from .config import Config
from .reactor_world import ReactorWorld
from .vlm_client import VLMController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="number of controller decisions")
    parser.add_argument("--step-seconds", type=float, default=2.0, help="seconds between decisions")
    parser.add_argument("--goal", default=agent.DEFAULT_GOAL, help="task given to the controller")
    parser.add_argument("--prompt", default=agent.SEED_PROMPT, help="initial world prompt")
    return parser.parse_args()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = Config.from_env()

    if not config.seed_image.exists():
        raise SystemExit(
            f"Reference image {config.seed_image} not found; set SEED_IMAGE in demo/.env.local."
        )

    world = ReactorWorld(model_name=config.reactor_model, api_key=config.reactor_api_key)
    controller = VLMController(
        base_url=config.vllm_base_url, api_key=config.vllm_api_key, model=config.vllm_model
    )

    await world.connect()
    try:
        await world.stage(config.seed_image, args.prompt)
        await agent.run(
            world,
            controller,
            goal=args.goal,
            steps=args.steps,
            step_seconds=args.step_seconds,
        )
    finally:
        await world.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
