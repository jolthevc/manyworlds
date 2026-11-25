# Data acquisition and retention policy

Owner, 2026-11-25: maximise permissible data and intelligence, minimise unnecessary retention of third-party content.
Learn broadly, synthesise intelligently, create originally. This is enforced in code wherever practical, not left to
prompts.

## The three classes

| Class | What it covers | How long |
|---|---|---|
| **Green**: persistent intelligence | Creator registry: handles, public professional names, profile links, sanitised bios, follower history, roles and tiers, scores. Per-post facts: id, link, type, date, metrics and their snapshots, `caption_features`. Annotations written in our own words. Pattern cards with their evidence. Culture signals. Findings, predictions and experiments. All first-party data. | Registry: 730 days after a creator was last seen, unless something still cites them. Our own conclusions: indefinitely. |
| **Amber**: transient observation | Third-party photos, videos, frames and caption text. Publication headlines are kept only as source references. | Media: deleted 24 h after it is read, and at most 14 days if never read (previously 180). Captions: deleted 30 days after the last fetch once read, 60 days at most. Feed and GDELT items (link, title, date): 180 days. |
| **Red**: never | Face embeddings or other biometric templates of real people, voiceprints, private content, comment text, follower lists, contact details, sensitive personal data, location histories, and any training on third-party media. | Not collected. |

## What enforces it

| Layer | Mechanism |
|---|---|
| **1. Source registry** | `policy data_policy.sources`. Each source has a status (approved, paused or blocked), its legal basis, its approved tasks, and whether it may supply transient media. `data_policy.refused` is checked by the provider router, the direct Apify pulls, the trend watcher and the culture reader. A source that isn't listed is refused. Current state: Apify and ScrapeCreators approved under a bounded risk the owner accepted; Graph approved (official API); HikerAPI blocked; publisher RSS feeds **paused** (terms unconfirmed, see below); GDELT approved. |
| **2. Structured ingestion** | Raw API responses were already normalised to named fields before storage; nothing raw is persisted. New: bios lose e-mail addresses and phone numbers (`clean_bio`). Captions also produce `caption_features`: size, hashtags, @mentions (commercial intelligence), calls to action, disclosure and emoji, so the caption text itself can expire. |
| **3. Transient visual analysis** | Media is downloaded to the research store with a purge date of at most 14 days. Analysis reads it (annotation in three layers, video measurement), then `references.purge` deletes it 24 h later. Annotations are written in our own words, with notes of at most 20 words and never quotes. The research store is never a production input (the `assets_not_research` guard). |
| **4. Creator Intelligence Registry** | `reference_creators` and `reference_posts`, plus tiered monitoring (cheap metadata checks, dedupe by content hash, no re-reads of unchanged content). Green, persistent. |
| **5. Pattern library and institutional memory** | Pattern cards carry a basis computed by code, never by a model: **validated** (our own posts' real outcomes), **correlation** (two-proportion test on high- vs low-traction attributes, p ≤ 0.1 with at least 5 posts per side), **observed** (researched, not significant) or **judgment** (craft knowledge). Each kind is weighted differently. Every agent's card line states the kind, effect, p and n. Culture signals carry a kind from code too: corroborated by 2+ publications, or single-source with capped confidence. |
| **6. Retention, caches, logs, backups, AI providers** | Housekeeping purges media, captions and culture items. The LLM journal drops request bodies once submitted. Research media goes only to processors whose terms have been checked: Claude models by default; hosted open-weight models are refused until listed. `mw research data-audit [--apply]` reports and cleans. Not yet controlled (owner/legal, below): how long Neon backups and point-in-time recovery keep deleted rows; how long R2 keeps deleted objects (versioning is off); Fly log retention; Anthropic's 30-day abuse-monitoring retention, unless zero-data-retention is agreed. |
| **7. Provider selection** | Routing order is cost and quality, but a provider runs only while approved for the task. Permissions come before price. |
| **8. Synthesis across sources** | Cards come from clusters across creators, each post compared with its own creator's median, never from one creator's post. Culture signals need several publications to count as corroborated. Briefs carry no source links and tell agents never to copy. |
| **9. Audit** | Run 2026-11-25 on production: 54 creators, 1,551 posts. 35 bios held contact details; cleaned. 1,551 captions backfilled with features. 1,691 media items (448 MB) now expire by 2026-12-09 at the latest, and sooner once tonight's analysis reads them. No face or voice data anywhere. The face-comparison code path was removed. |
| **10. First-party vs external** | External research feeds pattern-card evidence and culture context. Real outcomes of our own posts (`patterns.feedback`, `evidence.py`, predictions, experiments) **outrank** it: "validated" carries the highest basis weight. First-party data is green and unrestricted. |

## The three learning loops

1. **External intelligence.** Pieces:
   - the monitor (growth and engagement against each creator's own median, snapshots, tiers);
   - pilot annotate and distill (pattern cards with significance tests);
   - culture signals (GDELT);
   - commerce (sponsored share per category).

   Before launch it is strong in design but **unproven in data**: only 8 annotations exist so far. Tonight's analysis ($2.29) is the first real test. Known biases: selection (creators found through hashtags), survivorship, and one platform's metrics. The `low_comparator` role and the within-creator normalisation reduce them but don't remove them.
2. **Production intelligence.** Pieces:
   - capability card (acceptance and $ per accepted post, per format);
   - card acceptance;
   - review rows on every gate;
   - throughput and cost per machine.

   It is the most mature loop. Today's fixes show it still misses defects (extra hands, a face in a cutaway, the dead fallback). Each fix becomes a gate, so the loop gets stricter over time.
3. **First-party audience intelligence.** Pieces:
   - Instagram insights;
   - card outcomes;
   - predictions scorecard;
   - randomised experiment slots.

   Built and dormant until accounts connect. When it starts, its "validated" evidence outranks everything external.

## What the new rules give up, and the alternatives

| Lost | Why it mattered | Alternative |
|---|---|---|
| Re-reading old media with a better model later | Re-annotation after a schema or model upgrade | Read richly the first time (three layers plus measurements). Re-fetch from the stored post link when it's needed (~$0.002 a post), within the retention window. |
| Face look-alike check against named references | Catches a candidate who resembles a real creator | Prevent it upstream instead: real media never reaches generation, and faces are designed from scratch. The name and handle checks remain. A transient visual comparison by a vision model would need legal review (biometric laws) before use. |
| Caption wording over time | Hook phrasing trends | `caption_features` plus the annotation's own-words hook type and structure. |
| Publisher feeds (Vogue, Elle and others) directly | Rich summaries | GDELT gives headlines across the same publications. Re-approve a publisher once its terms or a licence allow it. |
| Near-duplicate check needs the cited images | Stops copying a specific post | Compute it at audition time from a transient fetch. Storing only the 64-bit hash (not reconstructable) would make it free; that needs a decision. |

## Needs legal review or an owner decision

1. **Instagram Terms §4.2 and TikTok's terms on automated collection through third parties.** The owner has accepted a bounded risk. Get counsel's view before scaling the monitored universe.
2. **GDPR / UK GDPR / CCPA for the creator registry.** Public creators are still identifiable people. We need:
   - a lawful basis (legitimate interests) with a written assessment;
   - a decision on the Article 14 notice (exemption for disproportionate effort, documented);
   - the retention schedule (730 days);
   - deletion on request (built: `mw research forget`);
   - possibly a DPIA.
3. **Biometric laws (BIPA, Texas, GDPR Art. 9).** We create no templates. Confirm that a vision model reading third-party photos for aesthetics is not "biometric processing".
4. **Copyright of transient copies** (EU text-and-data-mining exception with opt-outs; US fair use). We keep media up to 14 days and only for analysis. Confirm, and decide whether to honour TDM opt-out signals (robots and TDM-reservation headers) at fetch time.
5. **Publisher feed terms**, per publisher, before re-enabling any direct RSS. GDELT's own terms are fine, but it links third-party articles.
6. **ScrapeCreators' collection method** (logged-in or not) is not stated. Ask in writing.
7. **AI processors**: consider zero-data-retention with Anthropic for research calls. Open-weight hosts stay blocked from research media until their terms are reviewed.
8. **Backups and logs**: set and record the retention of Neon point-in-time recovery, R2 deleted objects and Fly logs, so deletion requests are complete within a known window.
