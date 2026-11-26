# Annotation v2: a coded section the statistics can use, and a vision benchmark

2026-11-25. Code: `manyworlds/research/annotation.py` (schema `iia_v2`), `manyworlds/brain/agents/intelligence.py`
(`read_system`, `cluster`, `attributes`), `manyworlds/research/vision_bench.py`; policy
`intelligence.annotation`, `intelligence.vision_bench`, `llm.roles.vision_bench_*`; tests
`tests/test_annotation_v2.py`.

## What v1 captured

`iia_v1` had three layers (observed, mechanism, speculation) plus code-filled evidence and measurement. Its observed
layer mixed ten coded fields (situation, nearest shot type, device, gaze, video format, movement, hook, camera
movement, structure, relationship) with free text: pose, hands, framing, lighting, wardrobe, expression, humour,
delivery, tone, product categories. Distill (`cluster`) could only test the ten coded fields for high-vs-low lift.
Everything a fashion and lifestyle business needs to learn sat in the free text, where nothing can be counted.

## Gaps against the business

| What the business decides | v1 | Missing |
|---|---|---|
| Fashion: what she wears, how it fits, colour story | `wardrobe` free text | styling category, garments, fit, palette |
| Grooming | none | hair, makeup, nails |
| Accessories and jewellery (commerce adjacency) | none | accessories, jewellery level |
| Composition and camera | `framing` free text, `device` | composition, subject scale, camera angle |
| Light | `lighting` free text | lighting kind |
| Candid vs staged | none | candid level (candid, semi-candid, posed, editorial) |
| Environment and travel | `situation` (one value) | environments (several), travel context |
| Narrative and tone | `tone` free text | lifestyle narrative, emotional tone, aspiration (1-5) |
| Carousel ordering | none | slide roles in order, opener has face |
| Product integration | `sponsored` yes/no | none / incidental / featured / ad, product categories |
| Video structure | `hook`, `structure` | face in first second, opening, setting changes, movement complexity, sound, production complexity, renderable now |

## What v2 codes

`observed.coded` (class `Coded`): 33 fields, every one a controlled vocabulary or a number.

- Single value (one name or `unknown`): styling_category, fit, palette, hair, makeup, nails, jewelry_level, pose,
  body_language, expression, composition, subject_scale, camera_angle, lighting_kind, candid_level, travel_context,
  product_integration; carousel: opener_has_face; video: face_in_first_second, opening, movement_complexity, sound,
  production_complexity, renderable_now.
- Label lists: garments, accessories, environment, lifestyle_narrative, emotional_tone, product_categories;
  carousel: slide_roles (ordered, one per slide).
- Numbers: aspiration (1 relatable ... 5 aspirational), video setting_changes.

The values live in policy `intelligence.annotation.vocab`, so the lists change without a deploy. The read is given
the lists in its system prompt (the same text for a policy version, so it caches). A value outside its list becomes
`unknown` (`annotation.coerce_coded`, applied before a read is stored), never a failed read; a list keeps its listed
labels. v1 records still load (their coded section is all unknown); distill reads both schemas, the newest per post.
With `intelligence.annotation.reread_older: true` posts read under v1 are read again for the coded section, if their
media is still held (read media is purged after `data_policy.retention.media_after_read_hours`; a v1 post without
media keeps its v1 read). The rules are unchanged: never brands (categories only), never private life, never other
people in frame (`friends` in slide_roles says only that a slide shows others). The read stays blind to performance:
the model sees media, caption and the code's measurements; code attaches the numbers afterwards.

## How statistics use it

`cluster` turns each read into attributes: the v1 coded fields as `field=value`, v2 single values as
`field=value`, every label of a list as its own `field=label`, numbers as `field=<band>` (policy
`intelligence.annotation.bands`). Fields a medium is not asked (video fields on a photo) are never attributes, and
neither is `unknown`. Within a cluster each attribute's share among high-traction posts is compared with its share
among low ones (two-proportion test in `store_card`, q = p x n_tests, Bonferroni).

`n_tests` counts every attribute tested. An attribute carried by fewer than `cluster_min_share` of the cluster's
high+low posts, or fewer than `cluster_min_posts` of them, is skipped before testing. The skip looks only at the pooled
count, never at which side the posts are on, so leaving those attributes out of `n_tests` does not bias q. The number
skipped is reported as `n_skipped_rare`. v2 roughly triples the attributes per cluster, so q gets stricter: the
correction is the price of looking at more things at once.

## The vision benchmark

`mw research vision-bench --assets N --models haiku,sonnet,opus --usd X [--creator H] [--dry-run]`
(`vision_bench.run`).

- **Items: our own media only.** Accepted image assets of sandbox creators (every review passed), or a video's
  frames. Research media is refused: the bench must not become a way round `data_policy.processors`.
- **Same prompt for every model.** The coded vocabularies and the item, the coded section only (`CodedRead`, schema
  in the prompt as in production). Images are scaled to `max_side` so tokens and cost stay predictable.
- **Reference: Opus 5.5** (`intelligence.vision_bench.reference`; otherwise the candidate with the highest price).
  It is an answer key, not ground truth: the first run should include a small hand-checked sample to see where
  Opus itself is unreliable.
- **Metrics per model.** Cost per item and latency (from the ledger, agent `research:vision_bench`, failed calls
  included). Schema validity (share of calls that returned a valid answer) and in-vocabulary rate (share of the values
  it gave that were on the lists). Agreement with the reference per field: exact match for single values (`unknown`
  counts as a value), Jaccard for label lists, position match for slide_roles, band match for numbers. Agreement is
  grouped by family (fashion, grooming, body, composition, place and story, commerce, carousel, video) with an
  overall mean. **Useful agreement per dollar** = overall agreement x validity / $ per item.
- **Spend.** The run stops before an item whose expected cost (each model's dearest measured call, else the policy
  token estimate) would cross `--usd`. Every model reads an item or none does, so comparisons stay paired.
- **Other vendors.** Each sits behind the same small interface (`check`, `read`). `GeminiModel` is an adapter point
  only: it refuses unless `GEMINI_API_KEY` and policy `llm.hosts.gemini` exist, and refuses even then until its call
  is written and verified against the live API.

### What "accuracy sufficient" means

A cheaper model is good enough for a field family when its agreement with the reference is at or above that
family's threshold *and* its validity and in-vocabulary rates are high. The thresholds are not set yet. They come
from the first run:

1. Run 20-30 items (photos and carousels first; video frame sets once a video handler stores frames for our assets).
2. Measure the reference against itself on a few repeated items (its run-to-run agreement is the ceiling no candidate
   can beat), and hand-check a small sample.
3. Set each family's threshold just under the reference's own consistency for that family. Concrete families (fashion,
   grooming, composition) should hold high agreement. Judgement families (place and story, tone, aspiration)
   should hold lower agreement, as the reference is less consistent there too.
4. Route each family to the cheapest model that clears its threshold. One annotation can be split across models if
   the cheap model fails only the judgement families.

Estimated cost per item with the policy assumptions (one image at most 1024 px, about 1,100 image tokens; 3,500 text
tokens for rules, vocabularies and schema; 700 output tokens; standard prices in `llm.usd_per_mtok`, no caching or
batch discount): Haiku 5.5 about $0.0008, Sonnet 5.5 about $0.016, Opus 5.5 about $0.032. All three together cost
about $0.05 per item, so a 20-item run is about $1. The system prompt is cached after the first call, so real input
cost should come in lower.

## First benchmark run (2026-11-26, live, $0.31)

**Setup**
- 12 of Maren's own accepted photos (our assets, so rights are clean).
- One coded read per model.
- Opus 5.5 is the reference, so agreement means agreement with Opus, not with the truth.

**Results**

| Model | $ per photo | Mean latency | Valid schema | In vocabulary | Agreement with Opus |
|---|---|---|---|---|---|
| Haiku 5.5 | $0.00074 | 4.4 s | 12/12 | 99.7% | 0.81 |
| Sonnet 5.5 | $0.0085 | 2.5 s | 12/12 | 100% | 0.85 |
| Opus 5.5 (reference) | $0.017 | — | 12/12 | — | — |

**By field**

| Haiku agreement | Fields |
|---|---|
| Near-perfect (≥ 0.9) | styling category, fit, jewelry level, expression, body language, camera angle, lighting, environment, product integration, aspiration |
| Usable (0.75–0.9) | makeup, composition, garments, accessories, pose, travel context |
| Weak (< 0.7) | nails 0.25, candid level 0.50, emotional tone 0.58, lifestyle narrative 0.63, palette 0.67, hair 0.67, subject scale 0.67 |

Sonnet is also weak on nails (0.25), hair (0.42) and subject scale (0.33). That suggests those vocabularies are ambiguous, not that Haiku is blind. Sonnet beats Haiku on palette, tone and narrative.

**Reading**
- Haiku delivers about 95% of Sonnet's agreement at about 1/11 of the cost. Broad annotation goes to Haiku.
- The weak fields need clearer vocabularies before they get a more expensive model:
  - nails: "not visible" should be its own value;
  - candid level and subject scale: one-line definitions per value.
- Tone and narrative could take a Sonnet second pass on a sample.

**Caveats**
- One creator, 12 photos, and our own generated images, not real creators' photos.
- No carousel or video items yet.
- The next run should cover carousels and video frames, with a 3-way human spot check on the weak fields.
