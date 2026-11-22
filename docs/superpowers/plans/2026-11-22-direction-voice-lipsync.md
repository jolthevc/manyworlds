# Direction, Voice and Lip Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every scene is directed by a modular shot sheet (set, wardrobe, hair and makeup, framing and camera,
blocking, performance, sound, edit, continuity), each creator speaks in a designed voice of their own with
per-line delivery, and talking video uses one take per scene with better lip-sync settings and a second model to
compare against.

**Architecture:** A shot sheet is structured data on the post (`body.shots`). A prompt compiler turns it into
each model's prompt in a fixed order, so models stay swappable by config. `body.scenes` stays as a one-line
summary of each shot, so existing gates and repair keep working. Voices are designed once per creator with
VoxCPM2 and cloned for every script. The talk job takes the finished audio (`audio_uri`) it already supports.

**Tech Stack:** Python 3.12, pydantic, Postgres, ComfyUI + WanVideoWrapper (pin `a7439db`, v1.4.8), VoxCPM2
(`voxcpm` 2.0.3, Apache-2.0), pytest.

**Spec:**
- roadmap section "Owner direction, 2026-11-22 (direction and modular control)"
- owner messages of 2026-11-22: modular control, own voice, natural lip sync, one take per scene, cut only at
  scene changes
- research: the quality-gap audit, VoxCPM2 / Qwen3-TTS, and the lip-sync options, summarised in the
  "Research decisions" section below

## Global Constraints
- No magic numbers in code: realism defaults, take limits, voice candidate counts and models go in `policy/default.yaml`.
- Shot sheet fields are all optional. A missing field uses the creator's or the format's default. Old posts
  without `shots` behave exactly as today.
- One take per scene. Cut only at a scene change. A take longer than `generation.talk.max_take_seconds` (60) is
  split at the nearest sentence end, as a safety net.
- Generated surfaces never carry legible text. Text is added in post as an overlay.
- Adults only, original people, the disclosure line, nothing in `scope.excluded_topics`.
- Motion and dance references must be ours or licensed. No third-party creator clips.
- Spend: the test run is at most ~$3.50 of GPU inside the $10/day cap. Benchmarks are 3 clips, pinned to one pool.
- The worker image is rebuilt once for this plan (v9), with a smoke test, not per task.

## Research decisions (2026-11-22)
- **Voice: VoxCPM2.** One model designs a voice from a description (put in parentheses at the start of `text`),
  clones it (`prompt_wav_path` + `prompt_text`), and follows a delivery instruction on the clone (a parenthetical
  with `reference_wav_path`). It outputs 48 kHz, needs ~8 GB VRAM, and requires `transformers>=4.36.2`, which fits
  our 4.57.6 pin.
  - Qwen3-TTS can't follow instructions on a cloned voice, and it pins transformers 4.57.3, so it's out.
  - Long text degrades, so speech is generated line by line and joined with pauses.
  - VoxCPM2 varies from run to run: generate `voice.takes` (2) per line, keep the best by WER.
- **Lip sync settings.** `audio_cfg` only acts when the sampler's cfg is above 1. At cfg 1, today's `audio_cfg`
  does nothing, so the s8audiocfg benchmark tested nothing. Kijai's example uses `audio_scale 1.5`. The official
  repo recommends audio cfg 2 with a LoRA.
- **Second model: LongCat-Video-Avatar 1.5,** already supported by our wrapper pin. It is MIT, uses a Whisper-v3
  audio encoder, and has an 832×480 / 93-frame / overlap-13 example. It needs SageAttention 2.2.0 or sdpa (1.0.6
  fails) and a bf16 base.
  - Move to 1.6 when a wrapper node exists.
  - FlashTalk is held back: it needs 64 GB or its own repo.
- **Readiness of influencer actions:**
  - Ready: talking head, b-roll, photos.
  - Partial (short and slow only): walk to camera, phone-down reveal, outfit change via a cut, dance via motion
    transfer.
  - Not yet: hand-object work (makeup, cooking), gym.
  
  The planner offers only moves whose `readiness` is in `direction.offer_readiness`.

## Review Focus
1. A post planned before this change (no `shots`) must still produce exactly as today. Test: `test_post_without_shots_keeps_old_prompts`.
2. The model may return a shot sheet with empty or unknown values (e.g. camera move "drone orbit"). The compiler
   must not crash, and must drop moves not in `direction.camera_moves`. Test: `test_unknown_camera_move_is_dropped`.
3. A talking scene with more than `max_take_seconds` of speech must split into takes, not fail. Test: `test_overlong_take_splits_at_sentence`.
4. A creator without a designed voice (Maren today) must keep working on Kokoro until one is designed. Test: `test_kokoro_until_voice_designed`.
5. A delivery instruction must never be spoken aloud: the parenthetical is stripped from the text that WER is
   checked against. Test: `test_delivery_is_not_in_transcript_check`.

---

### Task 1: Shot sheet model
**Files:** Create `manyworlds/direction.py`; Test `tests/test_direction.py`.
**Produces:**
- `class ShotSheet(BaseModel)` with optional submodels:
  - `set: SetSpec` (location, set_id, time_of_day, season, weather, light_direction, light_quality, color_temp,
    practical_lights, set_dressing, props, background_action, background_people: int, depth_of_field)
  - `wardrobe: WardrobeSpec` (outfit_id, outfit, accessories, state)
  - `look: LookSpec` (hair_style, hair_state, makeup, skin_state)
  - `framing: FramingSpec` (shot_size, angle, lens_mm, aspect, camera_move, handheld: none|light|held, device:
    propped|held|selfie, headroom)
  - `blocking: list[Beat]`, where `Beat` is (at_s: float, action, eyeline: lens|off|prop|screen, hands)
  - `performance: PerformanceSpec` (emotion_arc, energy: low|medium|high, body_language, lines: list[Line]),
    where `Line` is (text, delivery, pause_after_s)
  - `sound: SoundSpec` (mic: phone|lav|room, room_tone, effects: list[str], music)
  - `edit: EditSpec` (hook, on_screen_text, caption_style, transition_in, cover)
  - `continuity: ContinuitySpec` (carry_over, story_time)
- `def summary(shot: ShotSheet) -> str`: the one-line `scenes` entry (location, framing, action, wardrobe, light).
- `def spoken(shot: ShotSheet) -> str`: the lines' text joined, with deliveries excluded.
- Steps: test that `summary()` mentions the location, outfit and first beat; that `spoken()` excludes deliveries
  (Review Focus 5); and that an empty `ShotSheet()` gives `summary() == ""`. Implement, run, commit.

### Task 2: Prompt compiler and realism defaults
**Files:** `manyworlds/direction.py` (add), `manyworlds/creator/lab.py` (photo/clip/talk prompts delegate when a shot
exists), `policy/default.yaml` (new `direction:` block), Test `tests/test_direction.py`.
**Consumes:** Task 1. **Produces:** `def compile_prompt(shot: ShotSheet | None, bible: dict, target: Literal["photo",
"i2v", "talk"], policy) -> str`.
- Section order: camera/lens → subject (the full appearance) → wardrobe and look → action/blocking → set and
  lighting → style → sound cues (talk only). Then `direction.negatives`.
- Policy `direction`:
  - `realism` (photo: "visible pores, fine skin texture, natural imperfections, phone camera HDR, slight grain";
    video: "slight handheld micro-jitter" when handheld != none)
  - `negatives` ("no text, no logos, no mirrors behind the subject, no extra people")
  - `camera_moves` [static, handheld, push_in, pull_out, pan_left, pan_right, tilt_up, tilt_down, follow, orbit_slow]
  - `offer_readiness` [ready, partial]
- Unknown camera moves are dropped (Review Focus 2). A photo prompt ignores blocking after its first beat, and
  ignores camera moves.
- `photo_prompt`, `clip_prompt` and `talk_prompt` gain `shot: ShotSheet | None = None`. When it's None, the output
  is byte-identical to today's (Review Focus 1).
- Tests: `test_post_without_shots_keeps_old_prompts`, `test_unknown_camera_move_is_dropped`,
  `test_talk_prompt_orders_camera_subject_action_set`, and `test_photo_prompt_has_realism_and_negatives`.

### Task 3: Planner and showrunner write shot sheets
**Files:** `manyworlds/brain/planner.py` (PlannedPost.shots), `manyworlds/brain/agents/showrunner.py` (prompt text,
move library), the producer path that writes post bodies (`brain/ideas.py` schedule_ready), `policy/default.yaml`
(`moves:` library), Test `tests/test_direction_planning.py`.
- `PlannedPost.shots: list[ShotSheet] = []`, one per scene, in order.
- When `shots` is present, `scenes[i] = summary(shots[i])` if the planner left `scenes[i]` empty.
- The post body stores `shots` as dicts.
- `moves:` library entries look like
  `{formats, readiness, note, shot: {...defaults}}`: phone_down_reveal (partial), walk_to_camera (partial),
  point_at_caption (ready), storytime_to_camera (ready), outfit_change_cut (partial), b_roll_detail (ready).
- The showrunner offers the moves whose readiness is in `direction.offer_readiness` and whose formats production
  makes. It records `move` as an idea attribute.
- The system prompt adds a paragraph:
  - direct every scene
  - recurring sets and outfits are named (set_id, outfit_id) and reused from memory facts
  - hands are relaxed or out of frame unless the action needs them
  - text belongs in `on_screen_text`, never on surfaces
- Tests: a stub LLM returning shots → the post body has `shots`, and `scenes` are filled from summaries; a move
  whose readiness isn't offered is absent from the prompt.

### Task 4: Production uses shots; one take per scene
**Files:** `manyworlds/production.py` (`_scene_job`, `_clip_job`, `_talk_job`, `talk_segments`, hybrid), `policy`
(`generation.talk.max_take_seconds: 60`, `generation.talk_hybrid.segment_by: scene`), Tests `tests/test_talk_long.py`,
`tests/test_production.py`.
- Each job passes `shot=body["shots"][i]` when present.
- A talking video's takes are its shots. The text of take i is `spoken(shots[i])`, or the script split by
  `split_script` when there are no shots (today).
- A take over `max_take_seconds` (words / words_per_second) splits at a sentence end (Review Focus 3).
- Hybrid: shots whose `framing.device` is set and whose blocking has the creator on camera are `talk`. A shot
  flagged `cutaway: true` (a ShotSheet bool, added in Task 1) is `narrated`.
- Tests: `test_one_take_per_shot`, `test_overlong_take_splits_at_sentence`, and `test_cutaway_shot_is_narrated`.

### Task 5: Designed voices (VoxCPM2)
**Files:**
- Create `manyworlds/worker/handlers/tts_voxcpm.py` (kinds `voice.design`, `voice.speak`)
- Create `manyworlds/creator/voice_lab.py`
- Modify `docker/worker-talkcomfy.Dockerfile` (`voxcpm==2.0.3`, `--no-deps` if it would move transformers)
- Modify `manyworlds/worker/smoke.py` (import check plus a 1 s design run on CPU-safe settings, or a skip if no GPU)
- Modify `policy` (`voice:` block)
- Tests: `tests/test_voice_lab.py`, `tests/test_tts_voxcpm.py`
- Policy `voice`:
  - `engine: voxcpm`
  - `model: openbmb/VoxCPM2`
  - `candidates: 4`
  - `takes: 2`
  - `anchor_text` (a neutral 15 s passage)
  - `pause_s: 0.25`
  - `max_wer: 0.1`
  - `min_distance: 0.35` (speaker-embedding cosine distance from every other creator's anchor)
  - `kinds` routed to the talk family's pools
- `voice.design` payload `{description, text, n, seed}`. It returns n wavs with WER (from the same ASR the speech
  gate uses) and a speaker embedding (the VoxCPM2 / ECAPA embedding the handler can compute; licence checked in
  the smoke test file comment).
- `voice_lab.design(conn, policy, creator_id)` enqueues a design job from `bible.voice_notes` plus the appearance
  (age, gender presentation).
- `voice_lab.pick(conn, policy, creator_id, job)` picks automatically (never by ear): the lowest WER under
  `max_wer`, then distance ≥ `min_distance` from other creators, then the highest pitch variance (less monotone).
  It stores `anchors.voice = {engine: "voxcpm", ref_uri, ref_text, description}` and records a decision.
- `voice.speak` payload `{ref_uri, ref_text, lines: [{text, delivery, pause_after_s}], takes}`:
  - each line is generated `takes` times; the lowest WER against the line's text (delivery stripped) wins
  - lines are joined with `pause_after_s` (default `pause_s`)
  - returns one 48 kHz wav plus per-line timings
- Tests: payload validation; pick ordering with fake candidates; `test_delivery_is_not_in_transcript_check`; and
  `test_kokoro_until_voice_designed` (a creator with `engine: kokoro` still gets Kokoro jobs).

### Task 6: Talk jobs use the designed voice; lip-sync settings
**Files:** `manyworlds/production.py` (enqueue `voice.speak` per take before the talk job, then pass `audio_uri`; the
cutaway uses the same audio), `policy` (`generation.talk.audio_scale: 1.5`; profiles `s10cfg: {steps: 10, cfg: 2.0,
audio_cfg: 2.0}`), Test `tests/test_talk_long.py`.
- A post step `voice` must be done before its `talk` and `cutaway` steps. This is idempotent: one voice job per
  take per attempt.
- When the creator's voice engine is kokoro, the flow is unchanged.
- Test: with a voxcpm creator, the talk payload carries `audio_uri` from the voice job's output, and no `script`
  TTS runs on the talk worker.

### Task 7: LongCat-Video-Avatar 1.5 graph
**Files:** `manyworlds/worker/comfy.py` (`graph_longcat(...)`, weights `WEIGHTS_LONGCAT`),
`manyworlds/worker/handlers/talk_comfy.py` (`params.model: infinitetalk|longcat`), Dockerfile (SageAttention
2.2.0 for longcat, or `attention: sdpa` in params), `smoke.py` (validate the graph against object_info), `policy`
(talk profile `longcat: {model: longcat, steps: 10, cfg: 1.0, audio_cfg: 3.0, frame_window: 93, overlap: 13,
scheduler: longcat_distill_euler, lora_strength: 0.9}`), Test `tests/test_talk_comfy.py`.
- Node names and wiring are copied from Kijai's
  `example_workflows/LongCatAvatar_audio_image_to_video_example_01.json` at our pin (fetched while implementing),
  plus `LongCatAvatarWhisperEmbeds (v1.5)`. The checker (`missing_inputs`) proves it against the real catalogue in
  the smoke test.
- Test: `graph_longcat` uses only the example's node classes, and wires image, audio embeds and sampler.

### Task 8: Gates check the direction
**Files:** `manyworlds/review/video.py`, `manyworlds/review/gates.py` (judge prompts get the shot summary plus beats,
and score `direction` 1–10), `policy/rubrics/` (direction criterion), `policy` (`review.direction.min_score: 6`),
Tests `tests/test_video_review.py`.
- Test: a job payload with `shot` passes the beats to the judge prompt, and a direction score below the minimum
  rejects with stage `direction`.

### Task 9: Build, deploy and the test run (~$3.50)
- Bump the image tag to `worker-talkcomfy-v9`; build and smoke-test on Depot; set policy `tag: v9`; CI; deploy.
- Run, in order, each pinned to `runpod_secure_l40s_vol`:
  1. `voice.design` for Maren (4 candidates), then pick.
  2. 3 directed talk clips with her new voice: InfiniteTalk at 10 steps with audio_scale 1.5.
  3. The same 3 clips with LongCat 1.5.
  4. One 30 s single take with a scripted look-away and a push-in.
  5. One walk-to-camera clip on Wan 2.2 (the story-clip path, now on v9).
- Report pass rates, lip sync, render seconds and cost per accepted clip in `docs/overnight-2026-11-22.md` (a new
  section), and send the owner the best clips.
