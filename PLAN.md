# Initial Plan — DBS Tremor-Mapping World Model (WorldHackathon)

## 1. What we are actually building

Goal as stated: a simulated world where brain stimulation is applied, tremor response is observed,
the tremor-controlling targets are localized, and a robotic arm policy is trained in that world
before doing real surgery.

Important framing constraint discovered while reading the Reactor docs: **Reactor models are
real-time generative *video* world models, not physics or physiology simulators.** They take a
reference image + prompt and emit a steerable video stream (`main_video`) driven by commands like
`set_prompt`, `set_image`, `set_camera_pose`, `set_move_longitudinal/lateral` (see
`reactor/lingbot-world-2` schema). They do not compute forces, tissue deformation, or neural
dynamics, and nothing they render is ground truth.

So the system splits into three layers, and Reactor owns exactly one of them:

| Layer | Responsibility | Tech |
| --- | --- | --- |
| A. Physiology core | Stimulation → neural response → tremor signal. The *ground truth* the policy is scored against. | Our own simulator (Python) |
| B. Perception / world rendering | Photoreal surgical view (microscope / endoscopic / burr-hole view) that reacts to the arm's motion and to stimulation events | Reactor (`reactor/lingbot-world-2` or `helios`) |
| C. Agent + robotic arm | Chooses electrode trajectory and stimulation parameters, reads tremor feedback, is trained/evaluated | Gym-style env + RL / search |

Layer A must be the reward signal. Layer B must never be the reward signal — a video model will
happily hallucinate a "successful" outcome.

## 2. Layer A — physiology core (the ground truth)

Minimal viable version, no ML needed:

- A coarse 3D brain atlas volume (start with a simplified subcortical mesh: STN, GPi, VIM thalamus,
  internal capsule, plus "everything else"). Candidate free data: MNI152 template + a subcortical
  atlas; if licensing is slow, ship a synthetic voxel phantom with the same structure so the loop
  runs day one.
- Each voxel gets: `tremor_suppression_gain`, `side_effect_gain` (e.g. capsule → dysarthria /
  muscle contraction), `tissue_type`, `vessel_mask` (hemorrhage risk).
- Stimulation model: electrode at position `p`, amplitude `mA`, frequency `Hz`, pulse width →
  activated volume of tissue (spherical/ellipsoidal VTA approximation, radius monotone in amplitude).
- Tremor model: a 4–6 Hz oscillator whose amplitude is reduced as a function of the overlap between
  the activated volume and the suppression map, with latency, habituation, and noise.
- Outputs per step: tremor amplitude time series, side-effect score, tissue-damage score.

This is the piece that makes the project scientifically meaningful, and it is also the piece that is
entirely under our control, so build it first.

## 3. Layer B — the Reactor world

Used for the *visual* world the arm sees and for demo-quality output.

- Model: start with `reactor/lingbot-world-2` (image-anchored, two-axis navigation, native
  `set_camera_pose`, live `set_prompt`). Fall back to `helios` for image-to-video if we prefer a
  simpler surface. Take the exact slug from each model's own docs page; do not guess it.
- Reference image (`set_image`): a real surgical-field still (microscope view of a burr hole /
  stereotactic frame) so the world is anchored to plausible anatomy rather than invented.
- Camera = end-effector camera. Map the arm's tool-tip pose delta each control tick onto
  `set_camera_pose` (per-frame motion deltas), with `set_move_longitudinal` for insertion depth.
  Commands land on the next chunk boundary, so the control loop must be chunk-aligned, not per-frame.
- Stimulation and tissue events are rendered by hot-swapping the prompt, e.g. "electrode tip enters
  pale grey nucleus, faint contact sheen" → keeps continuity without resetting the session.
- Auth: API key (`rk_...`) stays server-side; mint short-lived JWTs via
  `POST https://api.reactor.inc/tokens`. Never in the client bundle.
- Python control path: `pip install reactor-sdk`, `Reactor(model_name=...)`, `@reactor.on_status`,
  frames arrive as `(H, W, 3)` uint8 arrays — feed straight into the agent's vision stack.

Known risks to design around: ~48 fps but chunk-boundary latency, no guarantee of geometric
consistency, cost per session, and the fact that the same command sequence will not reproduce the
same pixels. Treat Layer B as a perception-domain randomizer, not a simulator.

## 4. Layer C — agent and robotic arm

- Gym-style env: `observation = {video_frame, tremor_signal, proprioception}`,
  `action = {Δpose, insertion_depth, amplitude, frequency}`.
- Reward: `tremor_suppression − λ₁·side_effects − λ₂·vessel_proximity − λ₃·path_length`.
- Two-phase training, because video is expensive: train mostly against Layer A + a cheap
  geometric renderer, and periodically evaluate/fine-tune the vision encoder against Reactor frames
  so the policy is robust to realistic imagery.
- Target-identification output: a per-voxel posterior over "controls tremor", produced by the
  agent's exploration history — this is the actual deliverable to a clinician, not the arm motion.

## 5. Milestones

1. **Physiology core + headless loop** — atlas/phantom, VTA model, tremor oscillator, scripted
   electrode sweep producing a suppression heatmap. No Reactor, no UI.
2. **Reactor spike** — minimal Python script: connect, `set_image`, `set_prompt`, `start`, drive
   `set_camera_pose` from a canned trajectory, save frames. Confirms latency, cost, and control feel.
3. **Env wrapper** — join 1 and 2 behind one Gym env; verify a random policy runs end to end.
4. **Baseline agent** — Bayesian-optimization / bandit search over electrode positions that recovers
   the planted tremor target in the phantom. This is the "we can identify the region" proof.
5. **Web demo** — `npx create-reactor-app --model=lingbot-world-2`, live view of the arm's camera
   plus a tremor trace and the evolving target heatmap side by side.
6. **Sim-to-real hooks** — record trajectories in a format a real arm controller could replay; no
   real hardware in scope for the hackathon.

Milestones 1–3 are one working session each; 4–5 are the bulk of the demo value.

## 5b. External data we can pull in (answers "can we collect brain data elsewhere?")

Yes — and it maps cleanly onto Layer A. Three kinds of data, in the order they unblock work:

**Anatomy (voxel geometry — replaces the synthetic phantom)**
- ICBM152 2009b NLIN asym template (MNI) — the standard space everything below is registered to.
- DISTAL / Lead-DBS subcortical atlas — STN, GPi, VIM, RN, internal capsule as labelled meshes in
  MNI space. This is literally the atlas clinical DBS planning uses; free, redistributable with
  attribution.
- Harvard-Oxford subcortical atlas (FSL) as a fallback/cross-check.

**Stimulation → outcome (calibrates `tremor_suppression_gain` — the core of our reward)**
- Published probabilistic stimulation / "sweet spot" maps for tremor: the multicentre essential-
  tremor map (Nowacki et al. 2022) localises >50% tremor improvement to a cluster spanning the
  posterior subthalamic area into VIM, along the cerebello-thalamic tract. Several such maps are
  released as MNI-space NIfTI volumes on OSF/netstim.org.
- Lead-Tutor (Aperture Neuro, 2025): open-access pre/post-op MRI + CT from 10 DBS patients with
  reference electrode localisations — the only openly downloadable set of real electrode positions
  I found, useful for validating our electrode-placement geometry.
- Normative connectomes (HCP-derived, shipped with Lead-DBS) — lets us score a candidate target by
  which tracts its activated volume engages, not just which nucleus it sits in.

**Tremor dynamics (calibrates the oscillator and the DBS-on/off response)**
- MRC BNDU (Oxford) open data: bilateral EEG + thalamic LFP + tremor recordings in essential-tremor
  patients under DBS-ON vs DBS-OFF; separate externalised-DBS LFP set across three upper-limb tasks
  (2048 Hz, HDF5); and EMG/accelerometer tremor recordings in ET and PD. These give us real tremor
  amplitude envelopes, 4–6 Hz spectral structure, and — critically — real suppression latency and
  magnitude when stimulation turns on. Registration/request required, not instant download.
- PPMI (Parkinson's Progression Markers Initiative) for imaging + clinical scores if we want
  population variation; data-use agreement required, so treat as stretch.

**How it lands in the sim**
1. Register the sweet-spot maps and atlas into one voxel grid → `tremor_suppression_gain` becomes a
   real probability map instead of a planted Gaussian, and eloquent structures (internal capsule,
   vessels) become the penalty map.
2. Fit the oscillator's amplitude, frequency band, latency, and habituation to the ET DBS-ON/OFF
   recordings rather than hand-tuning them.
3. Hold out one published map (or a subset of patients) so "can the agent find the tremor target?"
   is measured against data the simulator never saw — otherwise we are just grading our own planting.
4. Reactor's reference image (`set_image`) can come from the Lead-Tutor imaging, giving the rendered
   world real anatomy as its visual anchor.

Caveats: every one of these is normalised group-level data, so it captures *average* anatomy, not a
specific patient's; the tremor recordings are predominantly essential tremor rather than Parkinson's
tremor; and the request-based sets (MRC BNDU, PPMI) have lead time, so milestone 1 should still ship
on the synthetic phantom with the real maps swapped in behind the same interface.

## 6. Open questions

- Do we have (or can we get) a Reactor API key, and what is the per-minute budget?
- Is there a real robotic arm / URDF we should target, or is the arm purely simulated?
- Do we have access to real DBS imaging or tremor recordings to calibrate Layer A, or do we stay
  fully synthetic (which keeps the hackathon honest but limits clinical claims)?
- Python or TypeScript as the primary stack? The training loop argues Python; the demo argues TS.

## 7. Scope honesty

Nothing here is clinically validated, and a generative video model cannot validate a surgical
approach. The defensible claim for the hackathon is: *a closed-loop simulated environment in which a
stimulation-target-search policy can be trained and evaluated, with a photoreal steerable view of the
procedure* — not *a system that plans real Parkinson's surgery*.
