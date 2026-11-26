# Directing performance: what we have, what's missing, the simplest way forward (2026-11-26)

> **Status (2026-11-26, branch `performance-controls`):** steps 1 and 2 of §6 are built with stubbed tests: the
> library (`policy/performance/performance-v1.yaml`, `manyworlds/performance.py`), the dossier `performance`
> section, shot-sheet `intent`/`controls` (in the director's edit pass: the shots call's strict grammar has no room
> for another nested list), the compiler step with natural edges, IndexTTS voice mapping, observation in the existing
> image and clip judges (rank only), selection by appeal/naturalness/closeness with a variety penalty,
> `performance_observations`, calibration (off; waits for measured judge agreement) and own-content annotation in the
> iia_v2 vocabulary (off). Owner refinements added: photo selection prefers appeal, charisma, expression and variety
> among acceptable candidates; one intent sentence first and only the salient controls; naturalness scored, level 4
> only with a reason; natural beginnings and endings. Not yet run against a real model or a GPU; step 3 (audio-first
> re-timing) is another branch; the Maren experiments (§5) wait for the owner's go.

The owner's brief asks for directed virtual talent rather than technically correct media: expression, charisma, emotional range, vocal and physical coherence, and context-dependent behaviour. The control is to be expressed in a shared creative language that the existing agents use, and learned from results.

This document covers:
- what the code does today (surveyed with file references; summary below);
- where the gaps are;
- where I disagree;
- the smallest design that gets us there;
- a controlled experiment with Maren.

Nothing in it adds an agent, a model or a rejection gate.

## 1. What exists today

| Area | Exists | Reaches the model? | Evaluated? |
|---|---|---|---|
| **Character performance in the bibles** | Free text only: `personality`, `voice_notes`, dossier `photographic_style` ("poses, gaze, framing, mood"), `appearance_traits` ("bearing"). No fields for expressions, smile, gaze habits, posture, gestures, walk, mannerisms or vocal cadence. | Only as prose in the director's brief. | No |
| **Shot sheet** (`direction.PerformanceSpec`) | `energy` (low/medium/high), `body_language` (free text), `emotion_arc` (free text). Beats with `at_s`, `action`, `eyeline` (lens/off/prop/screen), `hands`. Lines with `delivery` and `pause_after_s`. | Photo: first beat, energy, body_language; the emotion arc is dropped. i2v: all beats with timing. Talk: beats, eyes, idle movement by energy. The delivery never reaches any visual model. | One blended `direction` score from 4 keyframes. It ranks candidates but can never reject (`min_score: 0`). |
| **Intensity** | Only `energy`. Each of its 3 levels maps to one fixed sentence of micro-expressions. | Yes, as words. | No |
| **Pattern-card performance band** | pace_wps, eye_contact share, deliveries, mannerisms. | Prose rules in the brief, talking video only. | No |
| **Voice** | Six-clip delivery bank (excited, confiding, deadpan, amused, storytelling, questioning). IndexTTS-2.5 takes an 8-value emotion vector per line. Qwen3 currently ignores per-line emotion. `*emphasis*` styles captions only. Laughter and non-verbal sounds are stripped. | Bank clip or emotion vector per line. | MOS, WER, pitch movement ("flat"), pace and energy are measured per line. Nothing checks that "excited" sounds excited. |
| **Talking video** | The voice line is made first, then lip sync from a photo, or from a silent motion clip (walk-and-talk, InfiniteTalk V2V). Word timestamps exist (captions, punch-ins, lip-closure check). | The InfiniteTalk prompt carries beats, eyes and idle movement. It carries no delivery, emotion or audio timing. | Defect gates only. |
| **Photo selection** | 4–6 Z-Image candidates per scene. The judge picks the highest identity + realism + scene + direction score. | — | No expression, appeal or variety score. |
| **Editing** | Transitions, punch-ins (random timing within a rhythm), faceless cutaways under the voice, Ken Burns on photos, room tone, effects, music. | — | — |
| **Learning** | The annotation vocabulary codes pose, gaze, expression, gestures, eye contact and mannerisms, but only on other creators' posts. Pattern cards have expression and social-energy dimensions. | — | Our own outputs are never coded in that vocabulary, so we have no record of requested vs achieved. |

**Plain summary.** We write performance down in a few places and pass some of it to the models as adjectives. We never check whether it came through. Voice is the only channel with a real per-line expressive control and a measurement, and even there the measurement isn't compared with the request.

Talking video is not yet a production format (photo, carousel, Story, narrated edit and story clip are). So the performance gap in talking video is a development gap, not a live-content problem.

## 2. Where I disagree, or would narrow the brief

1. **Numbers imply precision we can't deliver.**
   - Image and video models cannot tell "sass 0.62" from "0.70". They respond to a handful of distinguishable behaviours.
   - Agents should work in **5 levels per control** (0 none, 1 subtle, 2 moderate, 3 strong, 4 extreme), stored as integers.
   - "More" and "less" become ±1 level.
   - Calibration (§4) then learns, per model, which level actually produces which observed level.
2. **In a talking video, the voice drives the face more than the prompt does.**
   - InfiniteTalk runs with the 4-step distillation at text guidance 1, so text has weak influence. Audio drives the mouth, the head and much of the expression.
   - Gestures and body movement come mostly from the source motion (V2V), not from words.
   - So the biggest levers for expressive talking video are, in order:
     1. an expressive vocal performance;
     2. good source motion (V2V from a directed Wan clip);
     3. editing;
     4. prompt words, a distant fourth.
   - This favours the audio-first workflow you describe, and the Qwen3 / IndexTTS choice you just made.
3. **Photo expression is under-selected, not under-generated.**
   - Z-Image already makes 4–6 candidates per scene. We pick by identity and realism, and among passing candidates nothing prefers the most charismatic or best-expressed one.
   - Adding an "appeal and intended-expression" score to the call the judge already makes is the cheapest quality win available. It is a few extra output tokens, no new call and no new gate.
   - Two further facts limit what prompts can do:
     - Z-Image Turbo runs at guidance 0, so negative prompts do nothing. "Not stiff" must be phrased as the positive behaviour.
     - Only the first beat reaches a photo prompt, and the emotion arc is dropped. Photos are one moment, so we should compile that moment's expression explicitly.
4. **Lip-sync repair (approach A): not now, and not as shipped.**
   - LatentSync 1.6 has the better sync of the two mouth-only models and keeps identity best, by the InfiniteTalk paper's own numbers. But at inference it uses an InsightFace detector, and InsightFace's pretrained models are **non-commercial**.
   - MuseTalk 1.5's code and weights are MIT, but its face parser was trained on CelebAMask-HQ, which is non-commercial research only. Its mouth is 256 px, soft at 1080p.
   - Both are usable only after swapping those components. More importantly, lip sync is not our failing gate: the two clips you reviewed scored 0.59 and 0.64 (bar 0.3).
   - I'd write down the licence-clean recipe (swap LatentSync's detector) and build it only when a measured share of otherwise-excellent clips fails on sync.
5. **The reusable performance library (approach B): later, and small.**
   - We already have motion transfer (Wan-Animate) and V2V.
   - Once V2V proves out, keep accepted motion clips, tagged by control levels and action, as candidate source footage.
   - Variety rules: never reuse within N posts; mirror, trim and speed variation; only for movement, never for faces.
   - Until there is a corpus of our own good motion, a library is premature.
6. **InfiniteTalk V2V (approach C): the most promising of the three, and already built.**
   - Walk-and-talk exists, but nothing has measured it. Our setting redraws half the steps (start_step 2 of 4), where Kijai's example appears to redraw the whole frame.
   - It deserves the first controlled test: same voice line, photo start vs a directed Wan 2.2 motion clip.
7. **No new gates.** Requested-vs-observed should rank candidates and teach the system, never reject. Defects still reject; a different-but-compelling result is kept and its observed levels recorded.

## 3. The design: extend what exists

### 3.1 A shared control vocabulary (policy, versioned)
`policy/performance/controls-v1.yaml` holds about 14 controls:
- energy, confidence, playfulness, sass, flamboyance, elegance, warmth;
- facial expressiveness, gesture, movement;
- vocal expressiveness, pace;
- candidness, polish.

Each control has a one-line meaning and 5 levels. This is the creative console. Every agent reads and writes the same names, and the vocabulary is versioned like the annotation vocabularies, so learning survives changes.

### 3.2 A performance lexicon: intent into behaviour (policy, per channel)
For each control and level, **observable behaviours** per channel:
- **face:** brow, eyes, mouth, smile type;
- **head and gaze:** tilt, glance pattern, eye contact share;
- **body:** posture, shoulders, hands, weight, turns;
- **walk:** stride, hips, pace, pauses;
- **voice:** pitch range, pace, emphasis, pauses, laughter;
- **camera and edit:** handheld vs locked, candid framing, cut rhythm.

This is where "sass" becomes "a knowing half-smile, one brow lifted, chin tilted down, a slow sideways glance that lands on the lens", rather than the word "sassy". It is written once, reviewed, and refined by calibration. No more adjectives accumulating in prompts.

### 3.3 Each creator's performance profile (dossier section, talent manager)
One more dossier section, `performance`, written by the talent manager in one checked call like the others:
- **baseline level** per control: Maren might be warmth 2, sass 1, elegance 2, energy 1;
- **natural range** per control: Maren's playfulness 0–3, never flamboyance 4;
- **signature behaviours:** her resting expression, her smile (a lopsided one that arrives late), gaze habits, how she stands and walks, hand habits, vocal cadence (dry pause before a punchline, volume falls at sentence ends), laugh, two or three conversational tics;
- **context shifts:** outfit check, GRWM, luxury, photo dump, story time, quiet morning, each as level offsets.

Identity is the signature; behaviour varies within the range. That answers "consistency without monotony" in data rather than in prose.

### 3.4 The shot sheet carries levels, and the compiler translates per model
- `PerformanceSpec` gains `controls: {name: level}` for the shot. `energy` stays as an alias. Each beat may override a few levels, so the exciting part rises and the reflective part falls.
- The director starts from the creator's baseline plus the format's context shift, then moves a few controls on purpose. It names what it changed and why, which is then checkable.
- `direction.compile_prompts` already has per-model targets and token budgets. It gains a lexicon step: for each control away from the creator's baseline, take that level's behaviours for the target's channels, phrased positively where the model has no negative channel (Z-Image).
  - photo: face, gaze, pose;
  - i2v: body, walk, gesture timing;
  - talk: face, head, gesture.
- Voice gets the same levels:
  - IndexTTS: the emotion vector scaled by vocal expressiveness;
  - Qwen3: its style or instruct text where the model accepts it, otherwise the reference clip. Whether Qwen3's clone mode follows an instruction is to be verified in the listening cast.

### 3.5 Observation, at almost no extra cost
- **Image and video judges:** the call we already make returns `observed: {control: level}` for the shot's directed controls, plus `appeal` (1–10) and `expression` (does the face carry the intended emotion, 1–10). Among candidates that pass the defect gates, selection adds appeal and closeness to the request.
- **Voice:** the per-line judge already measures pitch movement, energy variation and pace. A fixed mapping turns those into observed vocal expressiveness, energy and pace levels, at no cost.
- **Stored per asset:** requested levels, the lexicon phrases used, the model and settings, observed levels, defect scores, all-in cost. Later we join outcomes through evidence.py, as you list in §9 of your brief.

### 3.6 Calibration (code, a chief-of-staff refresh)
- For each (model, control) it records requested level against observed level.
- When a model consistently undershoots (asked 3, observed 2), the compiler asks it one level higher, or uses the next lexicon phrasing.
- A simple table per model, refreshed with the capability card. That is "increase the intensity, see what came out, refine the instruction" as a mechanism rather than one-off prompt tuning.

### 3.7 Audio-first talking video (wiring, mostly existing pieces)
1. The showrunner and director mark beats in the script (excited → reflective, emphasis words, laughs, pauses), and these set per-line voice levels.
2. The voice is rendered and judged (existing).
3. **New:** the director's beats are re-timed to the real audio. Word timestamps and the loudness and pitch contour place:
   - gestures on stressed words;
   - reactions in pauses;
   - head movement on rising phrases.

   The talk prompt then carries the timed beats, as it already does for i2v. Punch-ins move from random positions to emphasis words.
4. Then the visual takes.

**Editing gains** cheap J- and L-cuts: audio leading or trailing the picture by a few hundred milliseconds at shot changes. This is an ffmpeg change in the existing joiner.

**Laughter and non-verbal sounds:** currently stripped. They come back as directed events where the engine can render them (IndexTTS and Qwen3 to be tested), otherwise as a short library sound, timed in the pause.

## 4. What it costs and what it changes

| Piece | Cost to build | Running cost | Risk |
|---|---|---|---|
| Controls, lexicon, profile section, shot-sheet levels, compiler | Code and policy only | A few hundred tokens per brief | Low: same calls, better content |
| Observed levels and appeal in existing judges | Code only | ~5% more output tokens on existing calls | Low: rank only, never reject |
| Calibration table | Code only | Nothing | Low |
| Audio-first re-timing, J/L cuts, emphasis punch-ins | Code only | Nothing (word timestamps already exist) | Medium: needs a GPU run to judge |
| V2V test (approach C) | Exists | ~$1–2 test | Answers the biggest open question |
| Lip-sync repair (A), performance library (B) | Deferred | — | Deferred until measured need |

## 5. Validation with Maren (each run needs the owner's go)

1. **Photo controls** (~$0.50):
   - One outfit and set.
   - Sass and candidness at levels 0, 2 and 4, with everything else at her baseline.
   - 4 candidates each.

   Questions:
   - Can a blind judge rank the levels correctly?
   - Does identity hold?
   - Does moving sass also move warmth or polish (leakage)?
2. **Voice** (~$0.50; this is also the listening cast):
   - Qwen3 and IndexTTS.
   - The same three lines at vocal expressiveness 0, 2 and 4.

   Questions:
   - Do the measured pitch and energy change in order?
   - Do you hear it?
3. **Talking video** (~$1.50–2):
   - One line, Qwen3 or IndexTTS (the winner of step 2).
   - Photo start vs V2V from a directed Wan motion clip, at energy 1 and 3.

   Questions:
   - Do face, gestures and voice move together?
   - Which start reads more alive?
   - What does each cost per accepted clip?

**Success:**
- the levels are distinguishable in order, by the judge and by you;
- identity holds;
- changing one control mostly leaves the others alone;
- cost per accepted asset rises by less than the quality gain justifies.

## 6. Order of work
1. Controls vocabulary, lexicon (first draft for the 14 controls across photo, motion and voice), Maren's performance profile, shot-sheet levels, compiler step. Tests with stubs.
2. Observed levels and appeal in the existing judges, selection by appeal, stored per asset; calibration table.
3. Audio-first re-timing, emphasis punch-ins, J/L cuts.
4. The three Maren experiments, with your go for each spend.
5. Only then decide on A (licence-clean LatentSync) and B (motion library), on measured need.
