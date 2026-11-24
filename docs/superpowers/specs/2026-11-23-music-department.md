# The music department (spec, 2026-11-23)

Owner's brief (2026-11-23): AI musician creators are a big enough opportunity for their own department of agents,
with a talent manager per music star. Song making is modular (songwriter, lyricist, melody, producer...), every role
tunable and learning from what did well. Songs are measured and standardised like posts ("Song A did well, B didn't;
C shares A's high metric"), with thresholds and metric mixes that diagnose and drive trial and error towards hits. A
critic agent reviews and gives feedback. The same brain fundamentals apply: portfolio-level optimisation across
genres (country, pop, rock, rap, indie...), personalities and fit, missing pieces, opportunities to create artists.
The underlying voice of an artist is the same on every song; style and delivery may change per song.

Builds on what exists: `music.song` (ACE-Step 1.5 in ComfyUI, Demucs stems, lyric line coverage, loudness),
`brain/lyrics.py` (lyricist + checks), the `music_video` format, the musician archetype playbooks, and the agency
layer (proposals, rights, tiers, predictions, chief of staff, evidence, analysis, experiments). Research:
`docs/research/2026-11-23-consistent-singing-voice.md`, `docs/research/2026-11-23-avatar-and-music-critic.md`.

All of it ships OFF (`music.enabled: false`, no launches); runs only as tests until the owner says so.

## 1. Song anatomy (the unit of learning)

Every song gets one `songs` row (new table) and a versioned feature record:
- identity: artist (creator), title, version, parent song (a revision), release (single/EP/album) and track number;
- the brief: concept, theme, point of view, mood, genre and subgenre, reference archetype pattern, target listener;
- writing: lyrics, structure (sections with timings), hook line(s), repetition ratio, rhyme density, singability
  (syllables per beat), title-in-hook, language;
- composition and production: tempo, key/mode, time signature, energy curve per section, build into the chorus,
  time to first vocal and to first chorus, length, arrangement tags, instruments;
- sound measurements (worker or control-plane CPU, licences checked): LUFS, true peak, loudness range, crest factor,
  clipping, spectral balance, stereo width and mono compatibility, onset density, self-similarity and chorus repeats
  (librosa recurrence), vocal clarity (stem ratio) and lyric line coverage;
- voice: centroid similarity to the artist's voice (ECAPA + WavLM), report-only until calibrated;
- critic: the critic's structured assessment (below), never a raw 1-10 trusted on its own;
- outcomes (real only, through `brain/evidence.py`; simulated never counts): for posts using the song (reach, watch
  time and completion, saves, shares, comments, sound re-uses) and later streams. Linked to posts via the song id.

Features are typed and listed in policy (`music.features`: name, source, unit, direction-agnostic), so new metrics are
added by config and analysis picks them up.

## 2. Learning loop (the same as posts)

- Analysis (`brain/music_analysis.py`, analyst-like): per artist, per genre and portfolio-wide, which features and
  feature mixes go with outcomes, controlling for format and baseline (reuse `brain/analysis.py` machinery and its
  caveats: marginal effects, warm-up, repeated looks). Output: findings (tentative/confirmed) in `findings` with a
  `domain: music` marker.
- Thresholds: per artist and genre, "what good looks like" bands for key features, derived from that artist's best
  songs once there are enough (policy minimum), else from the genre playbook. Bands are advisory to the roles and the
  critic, never hard gates on media (CLAUDE.md: only media defects kill after media exists).
- Experiments: a tentative finding becomes a two-arm test (e.g. hook in the first 8 s vs 20 s) through
  `brain/experiments.py`, slots assigned per release batch.
- Predictions: the critic and the music talent manager file checkable predictions per song ("top-third saves for this
  artist within 14 days") through `brain/predictions.py`; the chief of staff scores them; `mw brain scorecard` shows
  whether each role's judgement improves.
- Similarity ("Song C may do well because it shares A's strong metric"): nearest songs by feature vector among the
  artist's and the genre's history, shown to the roles with their outcomes.

## 3. Roles (each an agent in brain/agents/, own budget, decisions recorded, swappable model per role)

| Role | Decides | Reads | Writes |
|---|---|---|---|
| head_of_music | the music slate: which genres and artist concepts, budget split, release cadence across artists; reports to the strategist (CEO) | portfolio music snapshot, findings, scorecard | proposals (launch artist, budget), decisions |
| a_and_r | artist concept pitches from public, logged-out trend data (rising sounds, genres, personalities); gaps vs our roster | research findings (evidence.py), roster | concept proposals to head_of_music |
| music_talent_manager (one per artist) | career arc, sonic identity, release plan (singles → EP), what to drop next and why, fan relationship | the artist's dossier, songs, findings, comments themes | the song brief, predictions, requests |
| songwriter | concept, theme, POV, structure, hook idea | brief, artist dossier, nearest songs | song concept |
| lyricist | the words in the artist's voice | concept, voice rules | lyrics (existing checks) |
| melody | candidate takes (N seeds/variations), picks the strongest hooks, regenerates sections (repaint where the engine allows) | lyrics, producer settings, critic notes | chosen take |
| producer | arrangement, tempo, key, instrumentation, engine tags | brief, genre playbook, thresholds | engine settings |
| vocal_director | delivery and style per song over the fixed voice | brief, voice asset | vocal tags, conversion settings |
| mix_master | loudness, balance, polish (DSP on the control plane) | measurements | final mix |
| curator | EP/album selection, order, cohesion | the artist's finished songs, outcomes | release plan |
| critic | the structured review (below) and specific feedback routed to the role that can fix it, before any video | measurements, listener description, thresholds, nearest songs | assessment, feedback, prediction |

Roles are modular: each takes typed inputs and returns a typed output (pydantic), so a role can be replaced, skipped,
or run with a different model, and the pipeline (`music/pipeline.py`) composes them per artist from policy
(`music.pipeline`). Revisions loop (critic → role → critic) up to a policy limit, cheap steps only before the GPU.

## 4. The critic

Describe, measure, calibrate; never trust an LLM's raw score (MuseCritic: off-the-shelf judges don't reproduce expert
ratings). Inputs: the sound measurements; Meta Audiobox Aesthetics (CC-BY-4.0) on mix and vocal stem; a listener
model's factual answers (genre, mood, instruments, vocal timbre, hook entry, audible lyrics, mix problems) from an
Apache-2.0 audio LLM (Qwen3-Omni hosted if verified, else Step-Audio-2-mini or Qwen2.5-Omni-7B on a GPU). Claude
reasons over these against the artist's thresholds and nearest songs and returns: per-dimension findings with
evidence, the top 3 fixes each addressed to a role, a verdict (ship / revise / rethink) and a prediction. Its
calibration is checked against outcomes (predictions) and against the owner's ear on a labelled set.

## 5. Voice per artist

Pipeline (research recommendation): ACE-Step song → Demucs vocal stem → singing voice conversion into the artist's
voice (source pitch kept) → remix over the instrumental. Adapters: SoulX-Singer-SVC (Apache 2.0, zero-shot) and RVC
(MIT, trained per artist), benchmarked on centroid similarity and lyric coverage. Voice creation: pick a seed timbre
from many ACE-Step vocals, build 10-30 min of sung material across registers, train RVC, store centroid + prompt clip
+ model as the artist's `voice_asset`. Original voices only. Licences of auxiliary weights checked before use.

## 6. Portfolio

Music creators are creators (same tables, `creators.vertical: music`), in the strategist's portfolio snapshot with
their own genre and market dimensions. head_of_music owns the music share of the budget within the CEO's allocation
(tiers, rights). Diversification across genres and personalities is measured (feature-space distance between
artists, overlap of audiences), gaps are surfaced, and artist launches go through the existing approval path.

## 7. Build order

1. songs table + feature registry + measurements + critic (describe/measure/reason) + learning hooks (analysis,
   thresholds, predictions, similarity).
2. The roles and the modular pipeline; music talent manager and head of music; a_and_r stub fed by research.
3. Voice asset and conversion adapters (SoulX-Singer-SVC, RVC), similarity score.
4. GPU tests (owner-approved caps): voice consistency (4 songs, ≤$2), critic listener trial (≤$2), department on a
   sandbox artist (Claude ≤$5).
