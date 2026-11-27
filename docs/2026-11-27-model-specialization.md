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

## 8. Evaluation as a standing capability (2026-11-27, later)

**What the first benchmark showed** (creative-v1.1: 7 scenarios on Maren, 2 independent samples each). These are hypotheses per task, not assignments:

| Model | vs Sonnet (judgements) | Strongest | Weakest | $/answer vs Sonnet | Latency |
|---|---|---|---|---|---|
| Kimi K3.1 Pro | 29–7 | monologue 6/6, restrained reaction 6/6, photo direction 6/6 | high-energy reaction 3–3 | 4.4× | 92 s |
| GLM-5.3 | 18–6 | both reactions 8/8, series ideas 4/4 | photo direction 1–3 | 2.5× | 30 s |
| Qwen 4 Max | 15–6 | photo direction 4/4, reactions 7/8 | series ideas 0/4 | 0.77× | 20 s |

The narrative scene is **untested**: the run stopped at its cap before judging it. All the judges were non-Claude, and every result is from one creator.

**Built since:**
- **Rigorous judging.**
  - Every judge family judges, Claude included.
  - One majority vote per independent sample, reported apart from raw judgements, with and without judges who have a stake.
  - A **self-preference measure** per judge family: how much more often it picks its own family than judges with no stake do on the same pairs.
  - How often the longer answer won.
  - 13 creative dimensions scored 1–5: originality, delight, wit, voice, nuance, naturalness, charisma, memorability, aspiration, distinctiveness, taste, surprise, plausibility. They are evidence only, never gates.
  - Untested scenarios are named. `mw llm evidence` pools every trial per model and task from the ledger.
- **The cap holds.**
  - Every call is reserved at its estimate before it is made, using observed token use as the run goes.
  - The run stops when actual plus reserved would pass the cap.
  - Results report estimated, committed and actual cost.
  - The $0.01 overrun happened because a call was checked only after spending.
- **Evidence from normal work: `llm.splits.<task>`.** A share of a task's real calls go to a candidate, whose answer is used, with the incumbent as fallback. The ledger now records each call's **task and route** (role, task, override, split, shadow; migration 057). So every model joins downstream to review acceptance, rerenders, cost per accepted post and, later, audience outcomes, through the spine and attribution we already have. It's off as shipped. Incumbent routes and fallbacks are unchanged, and a promotion is a reversible override.
- **Inference controls per model:** `extra_body` passes host-specific parameters (reasoning budgets, thinking toggles), next to `system_prefix`, sampling and `json_mode`.

**Why GLM's reasoning ran away.** Under Together's `json_schema` mode, the grammar constrains output from the first token. A reasoning model then has nowhere to think except inside the first string field:
- 15,000 characters of "Let me think about this carefully…";
- zero reported reasoning tokens;
- cut off at the token limit.

With the schema in the prompt instead (`json_mode: prompt`), it reasons in its own channel (`reasoning_content`) and answers cleanly. Its cost is now $0.022 per answer, 2.5× Sonnet, because the hidden reasoning is still billed as output. Whether Together honours a reasoning budget or thinking toggle for GLM (and Kimi) is unverified. `extra_body` is ready to test it; that's a few cents.

**Cost in context.** The premium models cost 1–3 cents more per creative answer. A director's shot sheet feeds about $0.05–0.20 of GPU, and a failed render costs $0.02–0.10. So for concept, script and direction tasks, the price is noise next to any effect on acceptance or rerenders. For high-volume routine tasks (reviews, captions, classification) it isn't, and those should stay on the cheapest model that holds up. The 92-second latency is fine for background brain jobs, but not for anything interactive or run hundreds of times a day.

## 9. Highest-value next steps

In order of value. Each needs your yes for its spend.

1. **Camera conditioning for photos** (~$1.5–2 GPU, plus a worker image with Z-Image's Fun ControlNet, Apache-2.0). The range audit's largest gap was camera distance, height and angle (54–60%). A depth or layout map, generated from the director's framing, is the direct fix. Measure with the same audit readers on the same 20 concepts: before and after on finished images.
2. **Direction to finished images: Sonnet vs Qwen 4 Max vs Kimi** (~$1–1.5). The same 8 ideas, each model writes the shot sheets (`as_task director_shots`), the renders are identical otherwise. Judged on the images: readers for intent executed, gates, a blind pairwise image comparison across families, and rerender counts. This answers "does better direction make better pictures".
3. **Dialogue to spoken lines** (~$0.5). Kimi's and Sonnet's monologues spoken by her Qwen3-TTS voice. The existing voice judges measure naturalness, intelligibility and timing, and a cross-family judge reads the transcripts against delivery. Text that reads well can speak badly.
4. **Judge calibration** (~$1.5). Re-run creative-v1.1 with Claude judges included, the narrative scene, and a second persona. The sandbox music artist's bible exists, so no new launch is needed. This measures each family's self-preference directly.
5. **Reasoning controls** (cents). GLM and Kimi with a capped reasoning budget through `extra_body`, on the 2 scenarios where each is strongest. Does the quality hold at lower cost and latency?
6. **Splits in normal production** (no extra cost beyond the work itself, once test or production runs resume). Qwen 4 Max on `director_shots` at 30%; Kimi on series-idea and hook generation at 30%. Promote only on downstream evidence, through a reversible override.

The spatial, director and performance side (measurement pass, ShotVL, image-to-motion with structural control) follows step 1, sharing its worker image.
