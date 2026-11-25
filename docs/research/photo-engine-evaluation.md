# The photographic production engine: how models are evaluated

Owner, 2026-11-25: photography may be our most important production capability, not a step on the way to video. We want the simplest, cheapest and most reliable architecture that produces exceptional lifestyle, fashion and commercial photography at scale. The aim is not to keep adding better models: it is to improve quality, economics, reliability and commercial capability while making the architecture simpler.

## Criteria (in the order that decides)

1. **Cost per accepted asset, all-in.** Includes retries, QA calls, boot time, model downloads and idle. It is measured from the ledger with machine overhead spread over outputs, not the per-job GPU seconds. A model on dearer hardware wins if acceptance, fewer retries or fewer model switches make the accepted asset cheaper.
2. **Acceptance under production conditions.** The full post path, our review gates, real shot sheets and Maren-class identities. Never a showcase benchmark: LTX passed 3 of 4 in a benchmark and 1 of 9 lines in a full post.
3. **Photorealism, aesthetic quality and attractiveness**, judged against the strategy's editorial standard.
4. **Character identity consistency** across sets, outfits and months.
5. **Natural candid photography:** phone-photo realism where the brief asks for it, editorial polish where it asks for that.
6. **Detail fidelity:** fashion and clothing, grooming, hands and nails, jewelry, eyewear, accessories.
7. **Reference-conditioned generation and editing:** set plates, outfits, follower shots from one place.
8. **Environment and wardrobe consistency** across a carousel or a shared set.
9. **Commercial product integration:** a specific real product rendered accurately, the path to advertiser-grade product photography.
10. **Throughput and end-to-end time.**
11. **Simplification:** fewer model families, weights and switches, and less orchestration.
12. **Commercial licence** that permits our use, checked before any test.

**Evaluation settings.** Use the resolutions and settings we publish at: about 1080x1350 for feed and 1080x1920 for Stories and Reels, with our finish pass. Higher native resolution counts for nothing unless it changes the published result.

## Method

A breadth plan per candidate, run against the current stack (Z-Image Turbo + Maren's LoRA for generation, Qwen-Image-Edit for reference-conditioned edits and set plates) on the same briefs:
- about 12 photos across 4 sets;
- 2 carousels;
- 1 shared-set talking post's photos;
- 1 product-integration brief.

Report per candidate:
- accepted / attempted, first pass;
- $ per accepted photo and per finished carousel (all-in);
- minutes per accepted photo;
- identity and detail scores;
- model families and weights needed;
- licence.

## Candidates held for later (after consolidation, infrastructure fixes and baseline validation)

| Candidate | Why it is interesting | Questions before testing |
|---|---|---|
| MiniMax H3-Image (open weights; INT8 fits one 80 GB GPU) | One checkpoint generates and edits: could replace Z-Image + Qwen-Edit (one family, fewer switches, a consistent look across a set) | Licence; identity (LoRA or reference edits); speed per image at 1080-class output; cost on an 80 GB card vs acceptance; whether its strength at readable text matters to us (slides are drawn in code) |

## Baseline first

Before any candidate, the current stack's own numbers, through rungs 1 and 2 of the validation ladder (`docs/2026-11-25-consolidation-review.md` §9), are the bar to beat.
