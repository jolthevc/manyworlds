# One singing voice per artist (2026-11-23)

Web research only. Owner's requirement: the underlying voice is the same on every song; delivery and style may vary.

## What ACE-Step 1.5 offers (MIT code and weights)
- LoRA / LoKr training is official (`train.py`, Gradio tabs; ~10-20 songs at 800 epochs; 16-20 GB VRAM; "8 songs,
  1 hour on a 3090"). The docs show style LoRAs only: nothing claims a LoRA locks a singer's identity.
- `reference_audio` steers "timbre, mixing style, performance style" globally (a whole-mix timbre, not a speaker
  embedding). In ComfyUI master only `ReferenceTimbreAudio` (experimental) exists; Cover, Repaint and Lego need the
  ACE-Step API outside ComfyUI.
https://github.com/ace-step/ACE-Step-1.5 (docs/en/LoRA_Training_Tutorial.md, docs/en/Tutorial.md)

## Singing voice conversion (keeps melody and phrasing, changes the timbre)
| Model | Licence | Notes |
|---|---|---|
| SoulX-Singer-SVC | Apache 2.0 code and weights (README; HF card returned 401) | zero-shot from a prompt clip, source F0 kept; released 2026-03; VRAM and speed unknown |
| RVC | MIT (base trained on VCTK; auxiliary ContentVec/HuBERT and RMVPE licences unchecked) | most proven for a locked timbre; ~10 min+ sung data per voice; trains in under an hour |
| DDSP-SVC 6.3 | MIT code; vocoder (NSF-HiFiGAN) and ContentVec licences unchecked | ~1000 clips; small GPU |
| Seed-VC | GPL-3.0 code, weights unstated; archived 2025-11 | skip |
| Vevo2, FreeSVC | non-commercial | excluded |

A synthetic target voice is fine (none needs a real singer). Train on SUNG material: speech-only data (e.g. VoxCPM2
output) lacks pitch range and breaks on high notes.

## Measuring consistency
ECAPA-TDNN (SpeechBrain, Apache 2.0) and WavLM-base-plus-sv on the Demucs vocal stem, scored against an artist
centroid (mean embedding over voiced 3-6 s windows), median not mean. Speech encoders degrade on singing (vibrato,
range; FruitsMusic F 0.64) and on stem bleed, so calibrate on our own data: same-artist spread vs between-artist and
between-seed spread, gate in between, report-only until calibrated.

## Recommended pipeline
1. ACE-Step writes the song (per-song style and delivery from tags) → Demucs vocal stem → SVC into the artist's voice
   (source F0 kept) → remix over the instrumental stem. Benchmark SoulX-Singer-SVC (zero-shot, nothing to train)
   against RVC trained per artist, on centroid similarity and lyric line coverage.
2. Creating the voice: generate many ACE-Step vocals, pick a seed timbre; build 10-30 min of sung material across
   registers (many songs converted zero-shot toward the seed, filtered by similarity); train RVC; store the centroid,
   prompt clip and model as the artist's voice asset. Original voices only, never derived from a real singer.
3. Helper, not the lock: an ACE-Step LoRA/LoKr per artist on 10-20 finished songs, plus ReferenceTimbreAudio.
4. Gate: centroid similarity (ECAPA + WavLM, calibrated) and sung-lyric coverage, report-only until calibrated.

Unverified: SoulX-Singer-SVC weights card, RVC/DDSP auxiliary weight licences, VCTK and UniSpeech licences, whether an
ACE-Step LoRA preserves identity, YuE2's existence and licence. YuE v1 is Apache 2.0 with ICL voice prompts (~80 GB).
