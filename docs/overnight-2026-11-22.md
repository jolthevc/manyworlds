# Overnight, 2026-11-22: video lock-down

Updated as work lands. Operations stayed paused; nothing was published; no accounts, purchases or new paid
services. GPU budget for the night: at most $20.

## Results so far

### Talking video (ComfyUI InfiniteTalk, L40S with the weights volume)
| Setting | Passed the gate | GPU per second of video (warm) | Notes |
|---|---|---|---|
| 6 steps (old default) | 1 of 3 | ~30-35 s | rejects: realism or motion 5 (hair clip doubles, face softens late); speech WER 0; lip sync 0.61-0.70 |
| 8 steps | 3 of 3 * | ~40 s | one clip's lip sync only 0.28 |
| 10 steps | pending | | |
| 6 steps on a community RTX 4090 ($0.34/h) | pending | | |

\* filled in below as the last clips are reviewed.

### Story clips (ComfyUI Wan 2.2 I2V A14B)
Not yet rendered. Three blockers found and fixed in turn:
1. Two machines downloading to the shared volume at once broke huggingface_hub's records (UnboundLocalError) -> one
   downloader per volume, complete files skipped, broken records cleared and retried (image v6).
2. The 150 GB volume was full with the old research script's ~82 GB of weights ("Disk quota exceeded") -> the
   ComfyUI worker prunes weights it no longer uses before downloading (image v7).

## Built tonight
- Model-trial spend cut: Kimi out of every trial, 20% samples (was 100% for creator and judge), trial cap
  $0.50/day; judge trials on vision models only.
- Ledger: an open-model call that times out is recorded at an estimated cost (hosts bill it; ~$1.80 of Together's
  bill was missing).
- Hybrid talking video (`generation.talk_hybrid`): the hook and sign-off lip-synced, enough of the middle on camera
  to reach 40% of the words, the rest cutaway photos under the same voice; all segments normalised to
  720x1280 / 30 fps / -14 LUFS and joined. Talking-video posts may plan one or two cutaway scenes.
- ComfyUI speed options to benchmark: fp8 fast matmul (`fp8fast`), torch.compile (`compiled`); the smoke test
  validates both graphs.
- Roadmap: video cost levers, trend watcher next steps, sound effects (licence check first), new formats.

## Needs you
- ScrapeCreators API key (trend watcher), `NTFY_TOPIC`, `FLY_API_TOKEN` in GitHub (the deploy workflow passes but
  deploys nothing without it; I deployed by hand), the Meta setup in docs/instagram-setup.md.
