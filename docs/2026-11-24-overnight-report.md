# Overnight report, 2026-11-24

Everything below ran on sandbox creators only. Operations stayed paused, no launches were scheduled, and no GPU
machine is running now.

> **Correction, 2026-11-26.** The dance result below is void. Both dance outputs were byte-for-byte the driving
> clips: the ComfyUI output picker returned the loader's preview of its INPUT as the render. Every walk-and-talk
> (V2V) take that night was likewise its silent source clip with the voice muxed on, so the "mouth never redrawn"
> drops and the V2V repairs measured nothing. See `docs/research/2026-11-26-maren-performance-experiments.md` §4.

## The short version

- **Photos, carousels, Stories, narrated edits, story clips and reference edits all passed** (dance: void, see the correction) the breadth test
  on real GPUs.
- **Talking video is not there yet.** Last night's run ended 13 of 17 items ready, 6 of its 10 talking items. I fixed the faults in code and reran the 4 failed
  items today; none passed.
  - That isn't the lip sync: mouth, identity and hands all passed their checks.
  - All three talking drops were audio faults. The fourth item, an outfit change, was dropped by the safety check
    (below).
  - The biggest fault: the voice model was reading its acting directions aloud ("A quiet, sincere, one thing nobody
    tells you..."). It's fixed, but no talking video has passed since the fix.
- **The 90-second single take** (LongCat 1.6) kept her face and colour for the whole 90 s, for about $1.70. Against the
  HeyGen bar, the weak spots are a frozen hand, no gestures and loose lip sync.
- **Music:** the first song (Kettle at Midnight) cost $0.05. The whole music department runs end to end on a sandbox
  artist for about $0.43 of Claude per run. The critic's first real review is specific and useful.
- **One consistent singing voice per artist is not solved yet.** Quick voice conversion didn't make the voice more
  consistent. The next step needs your licence decision (below).
- **Spend since 03:00 UTC: about $9.70** ($8.33 GPU machines, $1.36 Claude). Every run stayed inside its approved cap.

## Breadth test (production pipeline on GPUs)

| Format | Result |
|---|---|
| Photo, carousel, Story, narrated edit, story clip, reference edits | go |
| Dance (Wan-Animate-2) | void: the outputs were the driving clips (2026-11-26 audit); untested |
| Talking video | not yet: 6 of last night's 10 talking items ready (the hybrid with a cutaway finished today, once its join ran on the Fly finisher); 0 of 4 on today's rerun of the failed items |

What sank the talking items, and what I did about each:

1. **The voice read its directions aloud.**
   - This happened with "ultimate" cloning, the mode that keeps one timbre across lines: the model continues on from
     the reference transcript, so the "(quiet, sincere)" instruction gets spoken. It caught 2 of 3 talking items
     today.
   - Fixed: with that mode, the direction no longer goes to the model (policy `voice.deliveries_spoken`).
   - The cost is that lines lose their delivery direction for now. Steering delivery without the leak is listed in
     tech debt.
2. **A segment lost its first words.** "Two am. I take it back" was heard as "and take it back", on all three takes.
   The cause is not found yet; it's listed in tech debt.
3. **An outfit was judged unsafe** ("appears to wear nothing under the cardigan"). That is the safety gate doing its
   job; no change.
4. **Last night's faults**: number words in the speech check, a lip-sync pass that never redrew the mouth, a join that
   timed out on the small machine. All fixed before the rerun.
   - The rerun shows lip sync now passes.
   - Its talk clips use the plain InfiniteTalk path. The walk-and-talk item, which uses the mouth-redraw pass, dropped
     at its voice line before reaching lip sync, so that pass is still unproven on a GPU.

Two operational problems found and fixed today:

- **The 90 s LongCat test filled the 150 GB RunPod weights volume.** Its "container disk" folder was really inside the
  volume.
  - The next three machines on that volume died at their first download. The autoscaler's no-output breaker then
    stopped the pool, as it should, and the rerun stalled about 3 h.
  - Fixed: LongCat now writes to the machine's own disk, and a test guards it.
  - **The stray 47 GB is still on the volume** (decision below).
- **The breadth driver's 4 h deadline ran out during that stall.** I resumed the run with a longer deadline. Nothing
  was lost.

## Talking video against the HeyGen bar

The 90 s LongCat 1.6 take: `out/video/longcat16-90s-maren-small.mp4` (frames: `out/video/longcat16-frames.jpg`).

- **Holds:**
  - one continuous 90 s take;
  - face identity steady at 0/30/60/90 s;
  - no colour drift;
  - about $1.70 per 90 s.
- **Short of HeyGen:**
  - the hand is frozen and there are no gestures;
  - lip sync is loose (correlation 0.325);
  - blink and gaze are not controlled.
- **Next experiments:**
  - a reference video for body motion, or LongCat's own gesture prompting;
  - a sweep of the lip-sync redraw strength (`v2v.start_step`);
  - regenerating only a drifted chunk instead of the whole take.

## Music

- **First song:** Kettle at Midnight (ACE-Step 1.5, 2 min, $0.05), with its stems.
- **Department:** the department is built: talent manager, songwriter, lyricist, melody, producer, vocal director, mix
  master, curator, A&R, head of music, plus the critic.
  - Music features are scored against outcomes per artist, genre and portfolio.
  - Songs also get predictions, experiments and genre gaps.
  - The sandbox runs it end to end ("Kitchen Light On", $0.43 of Claude).
- **Critic on Kettle at Midnight** (reads its measurements; it can't listen yet; $0.03). Verdict: revise.
  - **Songwriter:** the first chorus arrives at 74 s and the title isn't in the hook. Have the chorus land by ~30 s and
    cut to ~90-100 s.
  - **Mix master:** the mix is dark and bass-heavy with no headroom. Cut the lows by 2-3 dB, add air above 4 kHz, aim
    for -1 dBTP and -10 LUFS.
  - **Lyricist:** six lines is too thin. Add a varied chorus and a 2-4 line bridge.
- **Singing voice:** I tried zero-shot conversion (SoulX-Singer-SVC from an 8 s sample). It did not make the voice more
  consistent across songs. Files: `out/songs/voice-test-*.wav`.

## Security

- **Signed links:** rented machines would hold no storage key. This passed on one image job (a Vast 4090, 4 images,
  $0.11). I switched it back off because the talk, ComfyUI and music workers haven't run with it yet.

## Spend (since 03:00 UTC)

| Item | Approved | Spent |
|---|---|---|
| Breadth rerun of failed items | $5 | $1.80 |
| 90 s LongCat test | $3 | $1.97 |
| Voice consistency (conversion test) | $2 | within cap (music machines $1.09 in all, including the song) |
| Signed-links check | $1 | $0.11 |
| Critic trial | $2 | $0.03 |
| Claude for music agents (sandbox) | $5 | under $1.50 |
| **Total bill** | | **about $9.70** |

## Decisions for you

1. **The full weights volume.** Either delete the stray `/models/longcat` folder (one small pod with the volume
   attached, about $0.10) or grow the volume (about $0.07 per GB per month). Until then the volume pool is unusable,
   so talk and story jobs fall back to machines without it, which download their weights at every start.
2. **RVC for per-artist singing voices.** Its assets repository adds a research-use clause, and the licences of the
   pitch and feature weights it needs are unstated. Without it, the next options are a longer, cleaner voice sample
   for SoulX, or ACE-Step's own reference timbre.
3. **A talking-video rerun after today's voice fix** (about $2). Its result decides whether talking video can be
   switched on.
4. **Switching on the formats that passed** (photo, carousel, Story, narrated edit, story clip). They stay off until you
   say so.
5. **Housekeeping:** delete the folders `C:\mwwt` and `C:\mwb`, answer the two old launch approvals, connect an
   Instagram account.
