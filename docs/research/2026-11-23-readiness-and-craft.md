# Before the breadth test: what top creators do, and what we still miss (2026-11-23)

The owner held the full test: "get everything in... latest and greatest... is there anything missing, tech debt,
noticeably bad?" and "do more research, find high performing clips." Three inputs:
- 456 top public TikTok clips across 12 hashtags (Apify, ~$1; 33 watched frame by frame). Links:
  `samples/inspiration/top-clips.md`; contact sheets `samples/inspiration/sheet*.jpg` (gitignored).
- Craft research with graded sources: `research_notes/Creator video craft 2026/craft.md` (kept out of git: it names creators).
- A code audit, every claim checked at file:line.

## What the top clips actually look like (watched, not guessed)
| Format | Median length (top 10) | What is on screen |
|---|---|---|
| ootd / outfit transition | 9 s | Locked-off propped phone, full body, plain room; cut on a spin, hair flip or blur; outfit 1 → 2 → 3 |
| fitcheck / outfitcheck | 12 s | Same: full body, small poses (hand on hip, hair touch, step), numbered looks "1. 2. 3." |
| outfitinspo (32.7M top) | 12 s | Talking to camera full body, numbered tips as on-screen labels ("Tight on baggy ✅") |
| mirror video | 11 s | Phone in hand at a mirror, mid-shot, lip-sync or chat |
| GRWM | ~60 s | Chest-up, propped phone, makeup in hand, question bubbles at the top ("What's your skin type?") |
| day in my life | 33-60 s | Mostly POV B-roll, 2-4 s clips, a timestamp label per clip ("7:30 AM Hygiene"), creator rarely on camera |
| posing ideas | 19 s | Named poses as labels, one per beat |
| walking vlog | 29 s | Handheld or gimbal follow, outdoors |

Common to nearly all of them:
- **Phone-native look:** bright, flat, slightly overexposed, ordinary rooms, no cinematic grade, no vignette. The
  person is often small in a wide room.
- **Original sound dominates outfit content** (fitcheck 10/10, outfitcheck 9/10 of the top clips). It is a voice or
  a quiet sound, not always a trending track.
- **On-screen text is a label, not karaoke:** one short line in the upper third or centre, in TikTok-native white
  text or a white box. Word-by-word captions belong to talking and storytime videos only.
- **Cuts happen on movement** (spin, hair flip, hand over the lens), never mid-sentence.

## What the research adds (grades in the craft notes)
- Length is per format. Breakouts: fashion ~20 s, GRWM ~95 s, day in the life ~73 s, storytime ~63 s, talking
  head ~66 s.
- The first frame should be moving or beautiful. A face starting to talk does about half as well (GRWM 17% vs 38%).
- AI content does better on Instagram (1.12x) than TikTok (0.79x). **An AI presenter talking to camera is the
  weakest AI shape** (0.54x TikTok, 0.87x IG). Faceless and aesthetic formats and genre vlogs do best.
- Character beats prettiness: the AI winners have a voice, an opinion and a familiar human genre.
- Top "reads as AI" tells:
  - a static talking face from frame 1
  - waxy skin
  - unmotivated studio light
  - dead stare or drifting eyeline
  - a gimbal-smooth "phone"
  - hair or clothes changing between shots
  - gibberish signage
  - hands doing fine work
  - no point of view
  - a glossy ad look

## What the audit found (checked in code)
Blockers for a real end-to-end test:
1. Production jobs are batch `post-*`. While paused, the autoscaler and the workers (policy baked into the image)
   take only `test-`/`bench-`, and `produce_tick` does nothing. There is no committed test driver.
2. `talking_video` and `narrated_edit` are not in `production.formats`.
3. A failed video ships quietly as a photo (`reroute_failed_video`).
4. Walk-and-talk crosses three worker families (voice, comfy, talk), so a post pays three cold L40S boots, each
   behind the batch wait.
5. No CLI to design and pick a voice. Maren is still on Kokoro.
6. 60 s takes time out (about 50 s of GPU per second of video, against a 3000 s limit). The 720p profile times out
   past ~25 s.
7. The direction gate enforces min 6 without calibration, and repair only re-seeds. InfiniteTalk cannot hit timed
   beats, so directed talk clips would be dropped wholesale.
8. Dance runs as a benchmark only. LongCat runs only by profile.
9. `autonomy.publish: autonomous` in policy (CLAUDE.md says paused). Harmless while operations are paused and no
   account is connected, but it must be `paused` before anyone goes live for a test.

Quality weak spots:
- Video scene photos are 4:5 and cover-cropped to 9:16 (26% of the width lost). Everything renders at 480p and is
  upscaled to 720p, not 1080x1920.
- Mixed frame rates: story clips 16 fps, talk 25, segments forced to 30 by frame duplication.
- "No mirrors, no text, no extra people" sits in the positive prompt where there is no negative channel, so it can
  summon them.
- No sound at all (library empty): silent story clips, dry voice.
- Captions:
  - timed as one block for Kokoro talk
  - always in the lower placement
  - the shot's caption style is ignored
  - the hook and `on_screen_text` are never drawn on video
- Undirected talking videos with 2+ photos are split every 8 s (jump cuts, intonation resets), against "cut only at
  scene changes".
- V2V lip sync uses 4 steps (8 steps already collapsed lip sync).
- Clause-by-clause speech ships on before its comparison. Lines with a delivery are cloned without the reference
  text, so timbre can drift.
- The finished video is never reviewed. The cover is picked from raw 480p frames by sharpness.

## Build before the breadth test (proposed order)
1. **A committed breadth driver** (`mw test breadth`): posts with handwritten shot sheets in `test-` batches at
   priority (no batch wait), pinned to the L40S volume pool. Records every gate, lip sync, WER, fps, resolution, $,
   seconds. Voice, motion and talk on one machine family for the test, so one boot.
2. **Test policy overlay:**
   - formats on
   - reroute off
   - direction gate report-only
   - `max_take_seconds: 30`
   - `autonomy.publish: paused`
3. **9:16 native scene photos** for video formats (no crop), and a 1080x1920 finish (AI upscale if one passes,
   otherwise lanczos).
4. **One frame rate:** 25 fps, interpolating 16 fps sources rather than duplicating frames.
5. **Negatives out of positive prompts** (into the negative channel where a model has one, else dropped).
6. **Voice:** CLI to design, pick and listen to a voice. Design Maren's. Clone with the reference text on every line.
   Whole-line vs clause-by-clause compared before choosing.
7. **On-screen text system:**
   - labels (TikTok-native white text and white box), numbered looks, timestamps, question bubbles
   - the hook drawn in the first second
   - word timestamps on for talking videos
   - face-safe placement
8. **Phone-native look** as the default finish (bright, flat, no vignette, light grain). Room tone and a quiet bed
   from CC0 sources, so nothing ships dry.
9. **Capabilities, not formats** (owner, 2026-11-23: "we shouldn't overfit to specific formats... have all the
   necessary weapons in our arsenal... exact content will depend on the creator archetype"). The Brain and director
   compose posts from a toolbox:
   - camera devices: propped, mirror, selfie arm, POV, handheld follow, locked-off
   - actions: talk, walk, pose, dance, prop or makeup work, outfit change
   - edits: cut on motion, labels, timestamps, hook text, voiceover over B-roll
   - sound: voice, room tone, bed

   The researched formats (outfit check, GRWM, day in the life, mirror talk, walk-and-talk vlog, posing, silent
   aesthetic) are examples the director may draw on per archetype, not production formats or code paths. The breadth
   test is organised by capability coverage.
10. **Cuts only at scene changes:** hybrid splitting off for directed posts. Transitions on motion (spin, blur, hand
    over the lens) in the joiner.
11. **Review the finished file** (one pass on final.mp4: captions over the face, seams, sync). Cover from the
    finished frames.
12. **Dance and pose as production capabilities** (any shot whose blocking calls for them) with the identity gate. A pose library from the
    Pexels clips.

Then the breadth test: one item per format and feature (the audit's list of 18), within the $10/day cap, likely over
two UTC days.
