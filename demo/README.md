# LLM-steered Reactor world (look-and-feel demo)

A vision-language model watches a Reactor world model and drives it with the same
commands the demo page's WASD keys send. Scope is deliberately visual: this is
Layer B + Layer C of [PLAN.md](../PLAN.md) with no physiology (Layer A), so it
renders a plausible surgical corridor but cannot tell you whether a tremor was
suppressed or a vessel was hit.

```
Reactor (reactor/lingbot-world-2)  --main_video-->  frame
        ^                                             |
        |  set_move_* / set_look_* / set_prompt        v
  control loop  <--- {"keys": ["w"], ...} ---  vLLM on Modal (Qwen2.5-VL)
```

## Layout

| Path | Role |
| --- | --- |
| `neurosurgeon/reactor_world.py` | Reactor session: staging, WASD bindings, latest frame |
| `neurosurgeon/vlm_client.py` | Frame -> vLLM -> parsed `Decision` |
| `neurosurgeon/agent.py` | The loop: observe, decide, idle axes, apply keys |
| `neurosurgeon/main.py` | CLI entry point |
| `modal_vllm.py` | Modal app serving vLLM's OpenAI-compatible API |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.local.example .env.local   # then fill in; .env.local is gitignored
```

Deploy the controller model (Qwen2.5-VL-7B on vLLM, one A100, scales to zero
after 15 min idle):

```bash
pip install modal && modal setup   # one-time, opens a browser to authenticate
modal secret create neuro-vllm VLLM_API_KEY=$(openssl rand -hex 16)
modal deploy modal_vllm.py
```

Put the printed URL + `/v1` in `VLLM_BASE_URL`, the same key in `VLLM_API_KEY`,
your Reactor key (`rk_...`) in `REACTOR_API_KEY`, and a reference image at
`assets/seed.jpg` — an endoscopic/surgical-corridor still works best, since
LingBot World 2 anchors the whole world on it.

## Run

```bash
python -m neurosurgeon.main --steps 20 --goal "Advance toward the pale nucleus"
```

Each step logs the model's one-line observation and the keys it is holding.

## Why it is shaped this way

- **Movement is held state, not keystrokes.** `set_move_longitudinal: "forward"`
  keeps driving until an explicit `"idle"`, so every step idles all four axes
  before applying the new key set; the loop cannot leave the world drifting.
- **Commands land on the next chunk boundary**, and a VLM round-trip is
  hundreds of ms, so the loop steers at ~0.5 Hz. Fine electrode placement would
  need a low-level controller underneath — the VLM picks goals, not increments.
- **`set_prompt` for events, WASD for motion.** Re-prompting reinterprets the
  whole scene, so it is reserved for state changes ("bleeding at entry point")
  rather than nudges.
- **No key ever reaches the client.** The API key stays in `.env.local` and is
  used server-side only, per Reactor's auth guidance.

## Tests

```bash
pytest tests    # runs offline; no Reactor or Modal credentials required
```
