# Manyworlds: status for outside review (2026-11-25, 20:00 UTC)

**What Manyworlds is.** An autonomous studio that researches, casts, produces and (eventually) publishes openly
AI-generated lifestyle personalities: desirable, aspirational, commercially valuable virtual talent, not a content
farm.

**How it is built.**
- **Brain:** Claude agents in roles. A CEO/strategist, talent managers, a style director, a showrunner and a director,
  plus a casting step and a research agent.
- **Production:** open-weight models on rented, interruptible GPUs: Z-Image for photos, Qwen-Image-Edit, Wan 2.2,
  InfiniteTalk and LTX-2.5 for video, and VoxCPM2 for voices.
- **Control plane:** runs on Fly.io and controls a Postgres job queue.
- **Accounting:** every dollar goes into a ledger.

**Current state.**
- Operations are deliberately **paused**: no publishing, no launches, no Instagram connection.
- The owner's stance: "We are still building the factory."
- The test creator is "Maren", a sandbox bartender persona.

Please critique freely. Open questions are at the end.

---

## 1. Where we stand: the headline

| Area | State |
|---|---|
| End-to-end fully edited video | **Not yet achieved.** Four attempts at one 40 s "story time" talking video. Each attempt found and fixed a different process defect (below). Next attempt starts at 00:13 UTC. |
| Photos | Reliable for face shots in one shared set (a bar). Face-free cutaways (hands, objects) are the weak point and were fixed today, unproven. |
| Voice | **Solved for now.** An automatic bake-off picked VoxCPM2 on quality per dollar: composite 0.923, $0.042/min. Qwen3-TTS was close; IndexTTS-2.5 failed a gender check on a male test voice. Maren was recast automatically (female-fit 0.999, MOS 4.26, 0 word errors). The owner calls it "really solid". |
| Talking video model | **InfiniteTalk (with cuts) leads; LTX-2.5 is the fallback.** LTX looked great in a 4-take benchmark but failed in a full post: lip sync passed on 6 of 15 takes, and only 1 of those 6 also passed realism (face drift). |
| Research / intelligence | Infrastructure built: creator registry, tiered monitoring, transient media analysis, a pattern library with statistical evidence labels, and culture signals. **Almost no real intelligence yet:** 8 annotations, 0 pattern cards. The first real analysis runs tonight ($2.29 over 1,551 stored posts). |
| Data policy | Green/amber/red policy implemented and enforced in code (section 4). |
| Spend today | $28.83 of a one-off $30 cap (normal cap $25/day). About $15.64 was a research bug (since fixed); about $8 was GPU. |

## 2. What broke today and how the process changed

Each failure was treated as a process defect, never a one-off shot.

1. **Follower shots copied the lead photo.**
   - What happened: shots in a shared set were image-edited from the first accepted photo, which contains the creator. The model kept her pose: four hands in one photo, and her face in "hands only" cutaways.
   - Fix: the lead photo is first edited into an empty plate (person removed), and followers are drawn into the plate.
   - Status: verified. The plate came out clean.
2. **Reviewers saw problems but nothing rejected them.**
   - What happened: the vision judge wrote "a phone in her other hand" (four hands) and scored a cutaway 3/10 for fit, yet both passed. Shot fit only ranks candidates after media exists, by owner rule: only media defects kill.
   - Fix: two new always-enforced media-defect checks:
     - anatomy: extra hands or arms;
     - framing: a face in a shot meant to have none, or anyone in a plate.
3. **The talking-video fallback was dead code.**
   - What happened: a per-take repair cap of 2 dropped the post before the route fallback (LTX → InfiniteTalk) could run. Tests missed it because they bypassed the cap.
   - Fix: each route now gets its own repair budget.
4. **The prompt worked against itself on face-free shots.**
   - What happened: the camera line said "phone held by someone else" (third person), and a repair put "…, no face" first. Diffusion models tend to draw negated things.
   - Fix: hands shots are always framed "point of view from her eyes, cropped at the forearms", and negations are moved to an avoid list.
   - Status: deployed, unproven.
5. **The budget design changed.**
   - Before: the daily cap stranded nearly finished validation runs.
   - Now:
     - the cap only blocks new work;
     - posts already in production may finish within a $5 completion headroom;
     - one-day owner grants exist (`mw budget grant`);
     - the policy number never changes.
6. **Research analysis failed at once.**
   - What happened: the annotation schema was too large for Claude's structured-output grammar, even in the compact strict form.
   - Fix: a third LLM mode puts the schema in the prompt and validates the answer, with one retry. Verified with a live call.

## 3. Learning loops (owner's framing)

1. **External intelligence** (pre-launch). Pieces:
   - Tiered monitoring of a creator universe: cheap metadata, dedupe by content hash, never re-analysing unchanged content.
   - Annotation in three layers: measured by code, observed, and mechanism vs speculation.
   - Distillation into pattern cards, comparing high- vs low-traction posts against each creator's own median.
   - Culture signals from news (GDELT) across fashion, beauty, grooming, interiors, architecture, travel and brand campaigns.

   Weaknesses:
   - Unproven in real data yet.
   - Selection bias (creators found via hashtags) and survivorship.
   - Instagram's cheap scraper doesn't give reel views.
2. **Production intelligence** (pre-launch).
   - Pieces: capability card (acceptance and $ per accepted post per format), review rows on every gate, cost per machine and throughput.
   - Strongest loop. Today shows its gates still miss defects; each miss becomes a new enforced check.
3. **First-party audience intelligence** (post-launch).
   - Pieces: Instagram insights, card outcomes, a predictions scorecard, randomised experiment slots.
   - Built and dormant. When live, "validated on our own posts" outranks all external evidence.

**Epistemic labels, set in code, never by an LLM:**
- Pattern cards are *judgment* (craft knowledge), *observed* (researched, not significant), *correlation* (two-proportion test, p ≤ 0.1, n ≥ 5 per side), or *validated* (our own outcomes). Each kind is weighted differently, and agents see the label, effect, p and n.
- Culture signals are *corroborated* (2+ publications) or *single source* (confidence capped).

**Consumption:** cards and culture signals are injected into the showrunner, director, style director and casting prompts.

## 4. Data policy (owner directive, today)

"Maximize permissible data and intelligence. Minimize unnecessary retention of third-party content."

| Class | What it covers | Rule |
|---|---|---|
| **Green** | Facts, metrics, growth, our observations, cards, first-party data | Persists. |
| **Amber** | Third-party media and caption text | Deleted 24 h after analysis, 14 days at most (was 180). Caption features stay: hashtags, @mentions, calls to action, ad disclosure. |
| **Red** | Face or voice templates of real people, private content, comments, follower lists, contact details, training on third-party media | Never collected. |

**How it is enforced:**
- A source registry with status and legal basis is checked before any provider call.
- Bios are cleaned of contact details. The audit found 35 of 52 bios with e-mails or phones; all are cleaned.
- Open-weight hosts are blocked from research media until their terms are checked.
- The face-similarity "impersonation" check was removed. Prevention is upstream instead: real media never reaches generation, and faces are designed from scratch.

**Paused:** direct publisher RSS feeds. Their terms are often personal and non-commercial only. The owner's rule: if rights are uncertain, pause and find an alternative.

## 5. Live issues right now

1. **ScrapeCreators** (validated as about 7× cheaper than Apify on TikTok, with better data) used its 100 free credits within a day. Research falls back to Apify. Needs an owner decision: the $47 pack buys 25k credits.
2. **GDELT** returns HTTP 429 from Fly's shared outgoing IP, even on a first request. Retry with backoff is deployed. If GDELT keeps refusing, the alternative is its free daily bulk files.
3. **The existing weights volume is unused.** It filled up on a test, so plans were routed off it. Since then about 30 GPU boots ($20.85) downloaded 40–90 GB each, cold. This is the biggest steady-state cost lever and is not yet fixed.
4. **RunPod L40S availability** dropped to zero for about 30 minutes today. Edits may now use A40s (also 48 GB). This argues for weights in a second data centre, but only after #3.

## 6. Economics so far (development, not steady state)

- **GPU waste audit (72 h, $17.87 billed):** 12% went to boot, 13% to idle, 12% to machines that produced nothing. Cold first jobs take 2–12× longer than warm ones. Warm reuse, a warm hold, model affinity and a cap that drains instead of killing jobs are built.
- **Unit costs:**
  - a photo: cents;
  - a 2–4 s LTX take: about 40–70 s of L40S time;
  - InfiniteTalk: about 50 GPU-s per output second;
  - voice: about $0.04 per minute.
- **Not yet measured:** cost per accepted finished video second, because no full video has passed yet.

## 7. Questions we'd value feedback on

1. **The 40 s single-take story video as the end-to-end test.** Is it the right first proof? Or should we prove a cheaper composite first: photos plus voice plus captions (narrated edit), which already passes. Then add lip-sync.
2. **Media-defects-only rejection.** After media exists we reject only for defects in the media (identity, realism, safety, anatomy, framing), never for idea or brief fit. Is that the right line? It saved 74% waste, but it let wrong shots through until today.
3. **Cutaways.** Image-edit conditioning on an empty plate vs generating cutaways text-only and checking set consistency after. Any better pattern for keeping one coherent location across 9 shots with open models?
4. **Research validity.** With 50–5,000 public creators, high- vs low-traction comparisons normalised to each creator's median, and two-proportion tests: what would make this more trustworthy? We're thinking of matched comparisons, time controls, and correcting for many tests at once.
5. **Data policy.** Is the green/amber/red split right for a US-based company studying public creators, with EU exposure? Particularly:
   - keeping identifiable creator records for 730 days;
   - holding third-party media for up to 14 days for analysis;
   - not honouring text-and-data-mining opt-outs yet.
6. **Autonomy vs safety.** The owner wants no human selection gates, yet nothing should run without the owner's go. Today that's handled with a master switch, per-run caps, daily caps, one-day grants and a completion headroom. Where would you expect this to fail?
7. **Anything we're over-building** for a pre-launch, single-creator system?
