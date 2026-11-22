# LatentSync's SyncNet (`stable_syncnet.pt`): licence and commercial use (2026-11-22)

Question: can we use ByteDance's pretrained SyncNet from LatentSync as a lip-sync scorer in the video gate (in
place of, or beside, our own MediaPipe score in `manyworlds/worker/lipsync.py`)? Nothing was downloaded or added.

## What the sources say (read 2026-11-22)

| Item | Where | Licence | Notes |
|---|---|---|---|
| LatentSync code | https://github.com/bytedance/LatentSync (`LICENSE`) | Apache-2.0 | Standard Apache-2.0 grant: copyright and patent licence, "royalty-free, irrevocable"; keep notices; no trademark rights. |
| LatentSync 1.6 weights, incl. `stable_syncnet.pt` | https://huggingface.co/ByteDance/LatentSync-1.6 | `openrail++` (repo metadata tag only) | The file list includes `stable_syncnet.pt`, `latentsync_unet.pt`, `whisper/tiny.pt` and `auxiliary/*`. The model card has **no licence text**, only the tag. It says nothing about commercial use, research-only use, or acceptable use. |
| `auxiliary/syncnet_v2.model` (the original Oxford SyncNet, used for evaluation) | same HF repo; original at https://github.com/joonson/syncnet_python | repo: MIT | Has no statement of its own about the weights. |

The README says the team released "a pretrained SyncNet with 94% accuracy" on VoxCeleb2 and HDTF. It is downloaded
with `huggingface-cli download ByteDance/LatentSync-1.6 stable_syncnet.pt`.

### What `openrail++` usually means
On Hugging Face, `openrail++` is the tag for the **CreativeML Open RAIL++-M** licence, the one SDXL ships with
(https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md). That text:
- grants a royalty-free copyright licence and allows hosted use (SaaS). It doesn't separate commercial from
  non-commercial use, so commercial use is allowed.
- says the licensor "claims no rights in the Output You generate".
- has **use-based restrictions** (Attachment A). These bar unlawful use, harming minors, harmful false
  information, harmful personal data, defamation and harassment, fully automated decisions that harm legal rights,
  discrimination, exploiting vulnerable groups, medical advice, and law-enforcement uses.
- if the model or a derivative is **redistributed**, requires passing those restrictions on, enforceably, and
  including a copy of the licence.

None of the Attachment A restrictions conflicts with what we do. Our creators are openly AI, adults only and
original people, with a disclosure line and AI flags on every post. A sync *scorer* on our own clips generates no
public output anyway. We would not redistribute the weights: they stay inside a worker image we run ourselves.

## Risks and unknowns
1. **No licence file in the model repo.** The tag is the only licence statement. We can't confirm which RAIL++
   text ByteDance intends, or that they intend the SDXL one. A tag without a text is weaker than a LICENSE file.
2. **Training data.** SyncNet was trained on VoxCeleb2 and HDTF. VoxCeleb's own terms are research-oriented.
   Whether dataset terms bind a released model is unsettled, and the licence doesn't address it. That is a risk, not
   a known restriction.
3. **The code is Apache-2.0, but the weights aren't.** The weights carry their own (unstated-text) terms.
4. **The old note in `worker/lipsync.py`** ("SyncNet's weights are research-only") was about the original
   SyncNet / Wav2Lip weights. It doesn't describe LatentSync's `stable_syncnet.pt`, whose tag is openrail++.

## Recommendation
**Adopt conditionally. Not now, and owner's decision.** The terms as tagged allow commercial use of a scorer we
run ourselves, and our use breaks none of the restrictions. The licence text is missing, and accepting licences is
the owner's call (plan rule: "no accepted licences"). Before adopting:
1. The owner reads the RAIL++-M text and accepts it, or asks ByteDance (an issue on the GitHub repo) to add a
   LICENSE file to the HF repo.
2. If accepted: add an adapter behind a switch, off by default (`review.video.syncnet.enabled: false`). It would
   score `stable_syncnet.pt` confidence and offset on 16-frame mouth crops, next to the MediaPipe score.
3. Calibrate both scores against the same clips judged by eye, and keep whichever separates pass from fail better.

Until then, the MediaPipe correlation (`lip_sync`) and the lip closure on b/p/m (`review.detail.lip_closure`) are
the sync signals. Both use Apache-2.0 code and models only.
