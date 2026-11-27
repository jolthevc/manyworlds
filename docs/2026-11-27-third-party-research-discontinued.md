# Third-party creator research discontinued: what was stopped, deleted and retained (2026-11-27)

The owner decided to discontinue third-party creator scraping, automated creator discovery and collection, and multimodal analysis of third-party creators' photographs and videos. This was a deliberate strategic decision, not a problem to work around. Intelligence now comes from web research, our own content and, once publishing is authorised, our own first-party analytics.

## Stopped and sealed (commit 95e5cbb)

**One switch, `intelligence.external.enabled: false`, is checked at every entry point:**

| Entry point | Behaviour while off |
|---|---|
| The Apify call itself (`research.apify.run_actor`) | Refuses every actor, so the trend watcher and any direct pull fail closed |
| The intelligence agent's external modes: scout, pilot, annotate, distill, monitor, resume, commerce | Fail at once (non-retryable) |
| The monitor's daily schedule; `research_monitor` | Removed from `operations.while_paused`; the monitor is off |
| Discovery's schedule (`brain.scouts`) | Never queued |
| Creator proposition reads and external gold sets | Refused |
| Agent requests of kind `intelligence` | Declined in code, even though they are auto-accepted |
| `mw research pilot`, `monitor`, `graph-check`, `propositions`, `gold` | Refuse with the reason |

**Data policy:** the `apify`, `scrapecreators` and `graph` (Business Discovery) sources are now blocked, so the provider router never calls them. Platform APIs are for our own accounts only.

**Unaffected:** the research desk (Claude web search and fetch), research questions and the trial ledger, and everything about our own content (spine, gold sets, Pegasus, recipes, inquiry, hypotheses, format fit).

No third-party job was queued or running when this was done. A test (`tests/test_external_research_off.py`) checks the shipped policy: every path refuses, and no schedule queues anything.

## Deleted (2026-11-27, verified after deletion)

| Where | What | Result |
|---|---|---|
| Postgres | 54 reference creators; 1,596 reference posts (captions, bios, counts, snapshots); 284 annotations and video measures of them | Deleted: 0 remain |
| Postgres (spine) | 1,650 external units with their attribute values and 9,781 outcomes | Deleted: 0 external units remain |
| Postgres | 360 trend items, 12 research run records | Deleted |
| Postgres | 48 job records of research jobs | Payloads, results and errors scrubbed; rows kept only for ledger links |
| Postgres | 78 journaled LLM calls of those jobs; 2 shadow rows of the research roles | Deleted |
| Postgres | 7 decision records of the research agents | Inputs and outputs scrubbed |
| Object storage (`research-refs/`) | 1,941 objects (images, video covers, 33 videos, frames, grids; about 533 MB) | Deleted: a listing of the prefix now returns 0 objects |
| Apify | 253 run datasets, 254 key-value stores, 253 runs | Deleted: listings return 0 datasets and 0 runs |
| This machine | Checked the working copy and scratch folders | No third-party media found; the 2026-11-24 calibration downloads had already been deleted, with no handles kept |

## Retained, and why
- **Ledger cost entries** for past research spend: amounts, providers and agents, with no content. The accounts must reconcile.
- **The research desk's web findings** (`research_findings`): our synthesis of published web sources, with links. These are web research, not creator data.
- **Our own content's reads** (`reference_annotations` with source own; spine units with source own).
- **The code and its tests.** The machinery is dormant, not deleted. Turning it back on would take your decision, a policy change and a source review.

## Outside our control
- **Anthropic API retention.** Images and captions sent in earlier vision reads are retained under Anthropic's commercial API terms for their standard period. We have no zero-retention agreement, so we can't delete them; they age out.
- **Neon point-in-time history.** Deleted rows stay recoverable within Neon's history window, then age out.
- **ScrapeCreators** keeps whatever it keeps of past requests (a small number of free-credit calls). It offers no deletion API we use.

## Not affected, for the record
- Licensed or authorised calibration footage (stock or CC-licensed talking heads) is still allowed for lip-sync calibration. It isn't creator data.
- Published industry research and official trend resources found through web search stay inside the desk. Any new paid provider needs the owner's approval.
