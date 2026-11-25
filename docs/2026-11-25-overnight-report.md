# Overnight report, 2026-11-25

Everything below ran on sandbox creators. Operations stayed paused and launches stayed at 0.

## The short version

- **The fully edited Maren video is not finished.**
  - Two attempts dropped on one faceless shot (her hands filling a kettle at the bar sink). The cause was found and fixed: her cardigan description was flooding hands-only prompts, and faceless shots weren't using the set photo.
  - On the third attempt all 9 scene photos passed first time.
  - It then stopped at the voice stage, because the **$25 daily total budget** was reached.
  - It resumes by itself at 00:00 UTC, unless you raise the cap.
- **Why the budget ran out:** the research pilot spent $15.64 on Apify collecting 54 creators and 1,551 posts. A bug re-pulled creators it already had, so nothing was analysed and no pattern cards were made. A fix is on a branch; analysing the stored posts later costs about $1–2.
- **Voice:**
  - The automatic bake-off picked **VoxCPM2** on quality per dollar. Qwen3-TTS was a close second; IndexTTS-2.5 was excluded for failing the male test voice's gender check.
  - Maren was recast automatically: the Qwen-designed voice you liked, every casting line passing, female fit 0.999.
- **Video model:** LTX-2.5 is the talking-shot route, with InfiniteTalk as fallback.
- **Shipped to production:**
  - talent-agency strategy (casting rubric, brand bibles with presentation registers and personal style, three-way QA, commercial categories, portfolio map, Instagram account metrics);
  - GPU factory fixes (warm reuse, no-capacity fallback, timing report, warm hold, cap drains instead of killing);
  - provider-agnostic research with tiered monitoring (off);
  - the fixed run driver.

## Spend today (UTC day so far)

| Item | $ |
|---|---|
| Research pilot, Apify collection | 15.64 |
| GPU machines billed (voice bake-off, Maren runs, casts) | 8.05 |
| Provider benchmark | 0.54 |
| Claude | 0.73 |
| Fixed | 0.35 |
| **Total bill** | **25.31** |

The research overspend came from a bug: the scout step re-pulled creators on each retry. Fix: `fix/scout-repull`.

## What failed and what was fixed

1. **Maren drop 1:** faceless cutaways were drawn from text only, so they came out as a daylight home kitchen. Fixed: they are now conditioned on the shared set photo alone.
2. **Maren drop 2:** the hands-shot prompt pasted in her whole signature paragraph, including the cardigan. Fixed: it now keeps only hand-level details (rings, nails, watch, scars).
3. **The run driver stalled:** it advanced the post before its own review. Fixed: it advances again after review and drives for 5 minutes per turn.
4. **The voice cast wasn't stored while paused.** Fixed.
5. **The research pilot re-pulled creators** and spent its budget before analysis. Fixed and deployed:
   - each creator's pull is recorded as it completes;
   - creators fetched in the last 14 days are reused;
   - 25% of the cap is reserved for analysis;
   - hashtag pools are capped at $1;
   - Instagram posts come through the cheaper profile scraper.

   The stored 1,551 posts can now be analysed without collecting again. Dry run: **$2.29** (cap $4) for 54 creators. Command: `mw research pilot --resume-annotate test-iia-95904e18 --usd 4`. It is held until the daily budget allows it.

## Decisions for you

1. **Daily total budget.** Raise today's $25 to about $30 to finish the Maren video now, or let it resume at 00:00 UTC.
2. **ScrapeCreators.** Sign up (100 free credits) and set `SCRAPECREATORS_API_KEY`. That lets us validate it; it looks about 12× cheaper on TikTok.
3. **Instagram terms §4.2.** Automated collection without permission is disallowed, logged in or not. This applies to Apify today; accept or restrict.
4. **Connect an Instagram account.** It is the only way to get real audience data, and it also unlocks free Business Discovery research.
5. **Research monitoring.** Turn it on at a daily budget (shipped off; $1/day suggested).
6. **The range photo test** (12 photos, $2). It needs a male test creator, which is a launch decision.
7. **A second weights volume** in another data centre. This is a new provider resource.
