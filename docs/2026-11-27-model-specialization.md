# The best viable model for every task: assessment, inventory, shortlist and roadmap (2026-11-27)

The owner's principle (2026-11-27) is the best economically viable model for every task, inside a modular architecture that keeps improving.

This document draws on:
- the code inventory of every model dependency;
- six research tracks, with the full report at `reports/Specialist models for creative production.md` (notes in `research_notes/`);
- what we built today.

## 1. Assessment of the philosophy

It's right, with two conditions.

**Why it's right.** The evidence supports specialisation, but mostly not where a general reasoner already leads.
- **Text: the gains are narrow.** The strongest open models (GLM-5.3, Kimi K3.1 Pro, Qwen 4 Max) sit in the same band as Opus 5.5 on creative-writing boards. Only GPT-6.2 Astra and Anthropic's own Fable 5.x are clearly ahead. The text gains are mostly cost, and a different voice for variety.
- **Media: the gains are large.** There, the specialised tools change what we can execute:
  - structural controls (pose, depth, camera);
  - each model's native prompt format;
  - measured geometry (depth, pose, free space);
  - shot-type classifiers.

  No general model can make Z-Image move a body; a pose ControlNet can.

**Condition 1: the finished asset is the benchmark.** A model that writes more impressive direction has won nothing until the renders it produces are better. Text benchmarks are the cheap first gate, and render A/Bs decide.

**Condition 2: specialist judges rank, they don't gate.** The image reward models favour a glossy, saturated look, the opposite of what we want. ShotVL's accuracy falls from 70% to 34% when asked to reason step by step. Omni models give no reliable video timestamps. These models give evidence. They must never become rejection gates, and each needs a calibration set of our own labelled assets first.

## 2. What we use today: the inventory

The full inventory has file references. In summary:

**What is already modular:**
- Every LLM call goes through `llm.py` with a role. No model string is hardcoded.
- Roles map to `{model, fallback}`, with Claude or open hosts (Together, DeepInfra). Approved overrides switch a role without a deploy.
- Shadow tests, model selection (agreement, quality, blind preference) and ledger costs work per call.
- Production engines are job kinds chosen by policy, with routing by cost per accepted output. The ComfyUI graph-plus-sidecar path (LTX, EchoMimic) is fully declarative.

**What was coupled** (the first two are now fixed):
1. **One role held many tasks.** The director's shots and edit; strategist, analyst and thesis choice inside `brain`; image and clip screens; four evaluator rubrics; nine dossier sections. No single task could be moved to a specialist.
2. **No metadata on models.** There were prices and a vision flag, but no licence, data terms, status or evidence.
3. **One judge call does everything.** Identity, realism, hands, teeth, lip closure, performance, brand and safety come from one vision call. No specialist can replace one of those sub-judgements without splitting the prompt.
4. **The prompt compiler is keyed by target, not by model.** A second photo model would inherit Z-Image's choices. Also, several model quirks live in code: Wan frame counts, "InfiniteTalk is the only V2V model", a LoRA trainer that only works with Z-Image.
5. **Claude-only conveniences are silently lost on a switch.** Batching, caching, web research and effort don't exist on open models.
6. **Model selection is narrow.** Only 9 of about 45 roles can be proposed for a switch. Review roles are scored only by agreement with the incumbent, so a better specialist scores as "disagreeing".
7. **Weight lists for the ComfyUI paths are in code,** not declarative like the graph path.

**Current models by area:**

| Area | Models |
|---|---|
| Research | Haiku, Sonnet, Opus (web search Claude-only) |
| Casting | Opus, with a Sonnet judge |
| Talent / style / market | Sonnet; style director candidate Qwen4-Max won blind 3/3 at ~53% of the cost, not switched |
| Showrunner | Qwen4-Max |
| Director | Sonnet; GLM-5.3 and Qwen4-Max lost 4/4 blind |
| Evaluator and screens | Haiku |
| Confirms | Sonnet |
| Production | Z-Image Turbo + LoRA, Wan 2.2 I2V, InfiniteTalk, LongCat, LTX-2.5, Qwen3-TTS, Whisper, ECAPA, UTMOS, MediaPipe lip-sync |

## 3. Research-backed shortlist, by creative impact per dollar

| # | Candidate | What it would improve | Licence | Cost | Evidence |
|---|---|---|---|---|---|
| 1 | Each production model's native prompt format | Wan I2V (≤100 words, motion and camera only), LTX one-paragraph present tense, Z-Image's own enhancer | — | code only | vendor docs |
| 2 | Z-Image-Turbo Fun ControlNet Union (pose, depth, canny) | Directs the body and composition, not just the face: our largest measured gap | Apache-2.0 | GPU time only | vendor; LoRA compatibility only third-party |
| 3 | Wan 2.2 Fun Control / Fun Camera | Directed motion and camera in clips | Apache-2.0 | GPU | vendor; not verified with lightx2v |
| 4 | Measurement pass on stills: MoGe-2, SAM 3 / SAM 3D Body, RTMPose or ViTPose, Depth Anything 3 base | The director knows posture, hands, free space and camera before directing motion; code can check plans | Mostly permissive; Sapiens and DA3-giant are non-commercial | small GPU | papers |
| 5 | Grounding VLM: Qwen3-VL (hosted on DeepInfra) or GLM-4.6V-Flash | Points and boxes on our own stills | Apache / MIT | ~$0.15–0.88/M tokens | vendor plus one independent paper |
| 6 | Image preference judges: HPSv3, UnifiedReward | Ranking takes cheaply (never rejecting) | Apache / MIT | small GPU | benchmark; biased toward glossy |
| 7 | ShotVL-7B | Verifies shot size, angle and lens against the shot sheet | Apache-2.0 | 24 GB GPU | benchmark plus independent replication; direct answers only |
| 8 | Timed AV checks: SyncNet-class scorer, emotion2vec+, Whisper word times | Intended vs delivered performance, deterministic | Mixed: original SyncNet weights are research-only | small | papers |
| 9 | Nemotron 3 Nano Omni (FP8, one L40S) | Holistic video read, rank-only | NVIDIA Open Model Agreement | GPU | vendor |
| 10 | Writing: GLM-5.3, Kimi K3.1 Pro, Qwen 4 Max, plus a Grok 4.7 humour probe | Dialogue, humour, variety, cost | MIT-style / modified MIT / API | GLM ~$0.0024 a call on DeepInfra vs Sonnet ~$0.009 | benchmark plus our own: GLM lost 4/4 on direction |
| 11 | HY-Motion 1.0 → rendered skeleton → pose-conditioned video | Synthetic motion without anyone's footage | Tencent Hunyuan licence (unread) plus SMPL terms | GPU | vendor |

**Licence traps, never in production:**
- Non-commercial: Hemmingway-1, OpenFace 3, LibreFace, Sapiens, the DA3 giant model, Audio Flamingo 3, InsightFace's pretrained models.
- Research framing: MolmoPoint and Molmo2.
- Provenance: PaintedFantasy and similar roleplay merges.
- Non-commercial data: CineTechBench, evaluation only.

Twelve Labs' self-serve terms train on uploads unless we opt out. No current role uses Kimi K2, which retires at the end of December.

## 4. Recommended modular architecture: what we keep and what we added

**Keep:** `llm.py` routing, overrides, shadows, selection and the ledger; job kinds and routing; the declarative ComfyUI graph path; the analytical spine; the trial ledger.

**Added today** (tested, deployed, no behaviour change unless configured):
- **Tasks inside roles.** `as_task(llm, "director_edit")`; `llm.tasks` declares 26 tasks with their parent roles. A task routes as its role until `llm.roles.<task>` or an approved override gives it its own model. `llm.shadow.<task>` tests that task alone, and selection can then propose a switch per task.
- **A model registry.** `llm.registry` records family, provider, modalities, licence, data terms, status (approved, candidate or retired) and graded evidence (vendor, benchmark, community, own). A test fails on any model in use without an entry, and on any role held by an unapproved model. Run `mw llm registry` to see it.
- **Per-model prompting for open models:** `system_prefix`, `temperature`, `top_p` and `min_p` per model.
- **Per-model prompt formats for production:** `direction.model_formats.<target>` sets the sections kept, their order and a word cap. Wan 2.2's official I2V shape is staged, off until a render A/B.
- **Benchmarks on our own tasks:** `manyworlds/benchmarks.py` and `policy/benchmarks/creative-v1.yaml`, with seven scenarios on our creator: bartender monologue, low and high behavioural temperature, series ideas, photo direction, house tour, narrative scene.
  - Every candidate gets the same assignment, several times.
  - Each answer is compared blind with the incumbent, in a hashed A/B order.
  - **Judges never come from a contestant's model family.**
  - There's a hard dollar cap and a dry-run estimate, and every answer and verdict goes to the trial ledger.
  - Run it with `mw llm bench --models … --dry-run`.

**Next, in order of value:**
1. **Split the fused review judge.** Identity, technical defects, direction adherence and performance become separate tasks, so deterministic or specialist checks can replace one part each.
2. **A per-model layer for photo prompts** (realism and style words per engine, not per target).
3. **Measured facts on the shot sheet:** depth, pose and free space from the measurement pass, which the director sees and code checks.
4. **Selection modes for review tasks against a calibration set,** not agreement with the incumbent.
5. **Move the Wan, InfiniteTalk and LongCat weight lists into declarative sidecars,** as the graph path already does.

**Behavioural temperature:** the performance language (14 controls, levels 0–4, context shifts) already represents it. Two pieces are missing:
- an embodiment step ("what would she do?") kept apart from direction ("how do we show it?");
- a benchmark that checks range without losing identity. Two scenarios test this now (`embodiment_low` and `embodiment_high`).

## 5. Roadmap and benchmarks

| Step | What | Cost | Needs |
|---|---|---|---|
| 1 | Text bench: GLM-5.3, Kimi K3.1 Pro, Qwen 4 Max vs Sonnet on all 7 scenarios, 2 samples each, judged by Opus plus a non-Claude family | ~$1–2 LLM (estimate from `--dry-run` first) | your yes |
| 2 | Render A/B of Wan's native I2V format vs ours: same 6 shots, 2 seeds | ~$1–1.5 GPU | your yes |
| 3 | Z-Image pose/depth ControlNet with Maren's LoRA: identity kept? does pose follow the skeleton? 10 skeletons × 2 | ~$1–2 GPU, plus a worker image with the ControlNet | your yes; image build |
| 4 | Measurement pass (MoGe-2, RTMPose, SAM 3) on 50 of our stills, compared with the director's text description | ~$0.5 GPU | your yes |
| 5 | ShotVL-7B on 60 of our frames against our shot sheets, then decide whether it joins the review | ~$0.5 GPU | your yes |
| 6 | Hosted Qwen3-VL grounding on our stills | ~$0.10 LLM | your yes |
| 7 | Split the fused review judge into tasks | code only | — |

Each step records what it measured. Adoption goes through the registry, from candidate to approved: an override that can be rolled back, regression-tested, never switched autonomously.

## 6. Economics and infrastructure

**Where to run models:**
- **Hosted endpoints for text and vision-language roles.** At our measured 24% GPU utilisation, self-hosting a 32B model is about 10× the hosted price per token.
- **Self-host only specialist models no provider serves,** batched on a machine that is already warm for the same family. Never load several large intelligence models onto a production GPU.
- **A cost tool by default.** GLM-5.3 on DeepInfra is about a quarter of Sonnet per call. That only matters where it holds quality, and our own blind trials say it doesn't for direction.
- **Pay for judgement where it prevents renders.** A few cents more on direction that avoids one failed render (~$0.02–0.10 of GPU) pays for itself.

**Data terms:** route open models only through providers whose retention terms we have checked: OpenRouter's `zdr`, `data_collection: deny` and `require_parameters` settings, or DeepInfra/Together terms verified. Keep a non-Chinese fallback on every role on a Chinese model.

**Infrastructure finding from today's audit.** Presigned downloads didn't resume. 4–10 GB weights were cut short on Salad and Vast, and 8 jobs failed. It's fixed in worker image v32 (Range resume; a truncated download is now a machine fault).

## 7. Needs your authorisation

- **Each run in §5, steps 1–6.** The costs are estimates; each gets its own yes.
- **A worker image with Z-Image ControlNet weights and the measurement models.** The build is free; the weights download is a one-off.
- **Any new provider account or key:** an OpenRouter account with zero-retention routing, or a DeepInfra key if we don't have one. Creating accounts and keys is yours.
- **Twelve Labs:** only with the training opt-out set, and the terms' clause on identifiable people checked.
- **The unread licences, before adoption:** HY-Motion and SMPL, the Kimi K3.1 Pro text, the Qwen 4 Max terms, and the MoGe-2 and Wan Fun Control weights.
