# Talking-video calibration from real examples (2026-11-24)

84 public short videos, found with Apify (logged-out data, TikTok and Instagram) and measured on the control plane
for internal calibration only. Nothing is reused. The downloads were deleted after measuring, and no handles or links
are kept here. 83 measured cleanly; 43 qualified for their category (speech actually present, the right mode).
Measured per video:
- cuts: hard (scene score > 0.3) and soft (> 0.15, which catches punch-ins);
- shot lengths and the time of the first cut;
- speaking pace from word timestamps;
- face-to-camera and b-roll share, captions, on-screen text and hook type, from a Haiku read of 8 frames.

Cost: Apify $1.17, Haiku about $0.10.

## What real creators do

| Category (qualified) | Hard shot median | Shots < 1 s | First hard cut | Face to camera | Cutaway / b-roll | Burned captions | Hook in first 2 s | Words/s |
|---|---|---|---|---|---|---|---|---|
| UGC talking (16) | 2.4 s (p25 1.0, p75 5.5) | 25% | 7.9 s | 71% | 25% | 7/16 | 16/16 (12 text hooks) | 4.3 (3.9-4.4) |
| Day in the life (12) | 1.1 s (p25 0.7, p75 2.3) | 47% | 2.5 s | 33% | 62% | 6/12 | 8/12 | 3.9 (3.1-4.2) |
| AI creators (13) | 3.3 s (p25 2.1, p75 5.9) | 9% | 5.6 s | 25% | 75% | 7/13 | 12/13 | 3.7 (3.0-3.8) |
| Story time (2) | too few to read | | | | | | | |

The views in the sample are high: UGC median about 200k, day-in-the-life about 460k, AI creators about 640k. These
are what the search surfaced, not a random sample.

## What it means for us

1. **Speaking pace is not cut rhythm.** People speak at 3.7-4.3 words a second, wherever they cut.
   - UGC talkers hold a talking shot for 2-5 s and punch in between.
   - Day-in-the-life cuts every second, and most of that time isn't her face.
   - AI creators cut slower than real ones. Our edge is to look real, so we shouldn't copy them.
2. **Openings:** almost every video has a hook in the first 2 seconds, and in UGC it's mostly a text hook on screen
   from the first frame. The first hard cut comes late in UGC talk (about 8 s): the hook is a line and text, not a
   cut.
3. **Cutaways:** a quarter of a UGC talking video is cutaways, and they are real inserts (her hands, the product, the
   screen), not decorative B-roll.
4. **Captions:** only about half burn the spoken words. Captions are a style choice per creator, not a must.
5. **Polish:** almost none is cinematic. The common devices are jump cuts, punch-ins, a handheld phone and a text
   overlay.

## Set in policy (generation.talk_edit)

- **story_time** (from UGC talking):
  - punch-ins every 2-4 s (unchanged);
  - cutaways about a quarter of the time, one every 4-8 s (was every 6-10 s at most);
  - a text hook from the first frame.
- **commercial** (from day-in-the-life):
  - shots 1-2.5 s, a few up to 4 (was 1.5-3 s);
  - face to camera about a third of the time;
  - punch-ins every 2.5-4 s, with a 1 s minimum shot.

  Shorter shots also mean fewer generated frames per shot, so it is cheaper per finished second.
- **Speaking pace:** `talk_long.words_per_second` (2.5) only estimates take lengths for planning; it doesn't set how
  fast she speaks. Her real pace is the voice model's.
  - Measure it from the comparison's voice lines.
  - Set the estimate from that measurement.
  - Make pace a performance dial per creator (a calm storyteller about 3.3 words/s, an energetic UGC voice about
    4.3), through the voice delivery bank or a light time-stretch. Not done yet.

## Per creator, not one formula

The grammar is picked per idea, else per archetype (`by_archetype`). These numbers are only a starting point. Each
creator's own outcomes, once Instagram is connected, should move her pace, cut rhythm, cutaway share and caption
style within her Performance Bible.

## Limits

- The sample is small, especially story time (2).
- Category labels come from searches and a frame-based read.
- Pace was measured only where speech was clean (n = 5-9 per category).
