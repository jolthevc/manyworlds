# Video quality build-out: autonomous plan (2026-11-22 evening)

Owner (going offline for ~4 hours): "begin building all these items... captions with nice fonts and styles, nothing
crappy, everything professional... ensure that it is sufficient enough to continue working autonomously and improve
recursively on gaps, next steps."

Builds on `2026-11-22-direction-voice-lipsync.md` (done: shot sheets, director, designed voices, one take per
scene, LongCat, direction gate). This plan turns the direction into what is actually on screen.

## Rules for the run
- **Code only.** No new GPU runs beyond the test already queued (`/tmp/dtest.py`, `/tmp/e2e.py` on the trends
  machine). The $10/day cap stays.
- **No `flyctl deploy`** until `/tmp/dtest.log` says "ALL ENQUEUED" and those jobs have finished. A deploy restarts
  the machine and kills the drivers. Pushing to `main` is fine: the GitHub deploy workflow deploys nothing.
- **Builders own disjoint files** and never commit. The coordinator commits each wave, runs CI, fixes failures, and
  pushes to the branch and to `main`.
- **No new paid services and no accepted licences.** Anything that needs a licence or terms read is researched,
  written up and left switched off in policy.
- **Every feature has a policy switch,** off where it is unproven.

## Wave 1
| # | Item | Owner files |
|---|---|---|
| 1 | Walk-and-talk: a silent motion clip (Wan 2.2 I2V) from the shot's blocking, then InfiniteTalk V2V lip sync onto it | worker/comfy.py, worker/handlers/talk_comfy.py, worker/smoke.py, production.py (talk steps), tests/test_talk_comfy.py, tests/test_talk_long.py |
| 2 | Captions: word-timed, styled, professional (OFL fonts, style presets, safe zones), burned in at the join | new manyworlds/captions.py, manyworlds/fonts/, media.py (caption burn only), tests/test_captions.py |
| 3 | Finishing: grade, grain, hook text, cover frame, loudness and true peak | new manyworlds/finishing.py, tests/test_finishing.py |
| 4 | Finer quality checks: hand and teeth crops, lip closure on b/p/m, background warp, set and wardrobe consistency against references; LatentSync SyncNet licence write-up | review/*, worker/lipsync*, tests/test_video_review.py, tests/test_gates*.py |

## Wave 2
| # | Item |
|---|---|
| 5 | Set and wardrobe bibles: canonical descriptions plus reference images, generated once; reference-conditioned scene images if an Apache image-edit model fits (research first); otherwise canonical text plus the consistency gate |
| 6 | Integration: captions and finishing in production's join and final steps, for every video format |
| 7 | Natural speech: breaths and pauses between clauses, a small disfluency option, irregular blinks and glances in the talk prompt |
| 8 | 720p option for close-ups (a talk profile, with the face filling the frame) |

## Wave 3 (research plus switched-off adapters)
- 9: Sound: room tone, effects and music bed from licence-clean sources (research models and licences), mic character and room reverb matched to the set.
- 10: Dance and motion transfer (Wan-Animate-2): confirm the licence; adapter switched off; references must be ours.
- 11: Hosted comparison (Hedra, Kling Avatar, OmniHuman): terms on synthetic people summarised; adapter switched off.
- 12: Two-person scenes (InfiniteTalk multi-person): design note and graph, switched off.

## Recursive loop after each wave
1. Commit, push, and fix CI until it is green.
2. Re-read the roadmap's direction section and this plan, and list what is still missing or weak. Add it as the
   next wave's items here, under "Found gaps".
3. Start the next wave.
4. Stop starting new work 30 minutes before the owner returns, and write the summary in
   `docs/overnight-2026-11-22.md` (section "Evening build").

## Found gaps
(appended as waves finish)
- (wave 1, walk-and-talk) Motion clips (`payload.intermediate: true`) must be excluded from video review
  (`review/video.pending_jobs`) and from routing cost per accepted clip (`routing.kind_costs`).
- (wave 1) Motion clips are capped at 5 s, so a longer line holds the last frame. Extend by chaining motion clips
  (continue from the last frame) up to the line's length, or end the movement into a static hold on purpose.
- (wave 1) Production should check that the talk engine is ComfyUI before sending `video_uri`.
- (wave 1, speech) Irregular blinks and glances in the talk prompt (direction compiler: performance and eyeline
  beats, so the talk target says "blinks naturally, glances away between thoughts").
- (wave 1, speech) Clause-by-clause synthesis may give each clause a sentence-final intonation. Compare it against
  whole-line synthesis on the GPU before relying on it (`voice.split_clauses`).
- (wave 1, checks) Hands and teeth are judged on full frames. Crop them with landmarks on the reviewer once
  MediaPipe is in the control-plane image, or on the worker. Line straightness is not built yet.
- (wave 1) The worker files changed (tts_voxcpm, comfy, talk_comfy, lipsync), so the next GPU run needs image
  talkcomfy v10.
- (wave 2, eyes/720p) Wire `direction.talk_profile_for(shot, policy)` into the talk job in production (merge
  `generation.talk_profiles[prof]` into params before the V2V update). It stays off (`direction.hd_close_ups`)
  until `hd720` is benchmarked.
- (wave 2) The glance fallback says "the object in her hands". Make it follow the creator's gender presentation
  (or "their hands").
- (wave 2, sets) Give `image.qwen_edit` its own worker family (`edit`, with disk and RAM sized for ~58 GB of
  weights), so the no-output guard can't stop all image work.
- (wave 2, sets) The image gate skips identity for set and outfit plates (`payload.plate`), so plates don't drag
  down Z-Image's measured acceptance.
- (wave 2, sets) `mw bench image.qwen_edit` needs an images option. `ideas.direct_posts` should pass `post_id` to
  `director.direct`.
- (wave 2, sets) Plates are picked up lazily. Refresh pending plates in the producer tick, so the first post in a
  new set gets references.
- (wave 2, integration) Narrated edits and cutaways need image worker v26 for `captions: false` (until then they
  would get both drawn and burned captions; no live impact while those formats are off).
- (wave 2, integration) Cut-joined talk segments are 480x832, so captions burn at 480p. Upscale to 720x1280 in
  the finish stage (a finishing scale hook) for every video.
- (wave 2, integration) Instagram publishing should use `body.cover` as `cover_url`. `finishing.py`'s docstring
  and CLAUDE.md need the finish stage.
- (wave 2, integration) Trim the repeated first frame at each motion-chain join.
