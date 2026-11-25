# Social-intelligence data acquisition: providers, cost and routing (2026-11-25)

Owner request (2026-11-25): revisit how we acquire public creator data, especially Apify's cost; assess ScrapeCreators
(primary multi-platform candidate), HikerAPI (low-cost Instagram specialist), Apify (fallback), the Instagram Graph API
Business Discovery (our own account, once connected) and free sources; make acquisition provider-agnostic and the
intelligence loop cost-aware.

Sources were read on 2026-11-24/25 (vendor pages, docs, terms; quotes are short and exact as fetched; recheck any clause
before relying on it legally). Benchmark numbers are ours, measured on 2026-11-25 on the Fly `trends` machine through
`manyworlds.research.providers`.

## 1. Summary

- **The pilot's baseline** (job `8061d0d1`, batch `test-iia-95904e18`, still running at the time of reading): $12.23 of
  Apify for 39 creators and 1,101 posts, **$0.31 a creator, $0.011 a post**, before any vision. Half of it ($6.19) went
  to the hashtag candidate pools (924 IG + 1,000 TikTok hashtag items), not to the creators we kept.
- **Apify is cheap when used right.** One Instagram profile-scraper item ($0.0023 on our plan) carries the profile and
  its latest 12 posts with captions, likes, comments, timestamps, image and video URLs and carousel children. That is
  the same price per creator as ScrapeCreators. What it lacks is reel **view counts**; the post scraper has them at
  $0.0023 **a post** (it bills `post` + `post-details` on every post: our policy assumed $0.0017).
- **ScrapeCreators' benchmark is half confirmed.** Its Apify figure ($1.38 for 600 posts, $0.0023 a post) matches our
  bill exactly for the post scraper. Its own figure ($0.094 = 50 requests x $0.00188) is consistent with its price list
  but is **not independently validated**: we hold no ScrapeCreators key. The real gap is on **TikTok** (Apify bills each
  result: $0.024 for 12 posts vs one $0.0019 credit) and on **Instagram with view counts** (12x).
- **HikerAPI conflicts with our rules**: it states it uses Instagram's private mobile API through managed sessions, and
  its terms require written permission for commercial use and forbid building a database from the content. Not built,
  not routed.
- **Business Discovery** (official, free, ~200 calls/hour) covers Business/Creator accounts only and needs a connected
  professional account (pending). It is the first route for Instagram metadata once connected.
- **Built**: `research/providers.py` (adapters + router + ledger per call), migration 045 and `research/monitor.py`
  (tiers, cadences, cheap checks, dedupe, deterministic selection, media and vision only for selected posts,
  content-hash keyed annotations), the agent's `monitor` mode and `mw research monitor`. Off until the owner turns it on.
- **Expected cost** for a 5,000-creator universe (tier mix below): ~$25/month on ScrapeCreators, ~$30/month on Apify
  for Instagram only, ~$140/month on Apify with half the universe on TikTok; vision on the selected posts ~$6/month.
  The pilot's way (a full pull per creator) would be ~$1,550/month for the same universe monthly.

## 2. Our needs

Creator discovery; profile (followers, bio, business flag); recent posts with pagination and a date floor; carousel
children; media URLs (and their expiry); reels with view counts; captions and hashtags; engagement (likes, comments,
shares, saves); freshness; commercial-use terms; reliability. Transcripts are a nice-to-have (we transcribe selected
videos ourselves on the finisher).

## 3. Providers

### 3.1 Apify (what the pilot uses)

Pricing: pay-per-event per actor, no compute charge; plans Free ($5 credit), Starter $19/month (Bronze store prices),
Scale $199 (Silver), Business $999 (Gold); unused credit does not roll over (https://apify.com/pricing). Our bills show
we are on **Bronze** prices. Per-event prices from the store API (`https://api.apify.com/v2/acts/<actor>`,
`pricingInfos`), Free / Starter / Scale:

| Actor | Event | Price | 30-day runs, success (store API) |
|---|---|---|---|
| apify~instagram-profile-scraper | profile (embeds latest 12 posts) | 0.0026 / 0.0023 / 0.0020 | 9.8M, 99.55% |
| apify~instagram-post-scraper | post + post-details (both billed) | 0.0027 / 0.0023 / 0.0020 | 6.4M, 99.90% |
| apify~instagram-hashtag-scraper | result | 0.0026 / 0.0023 / 0.0021 | 824k, 99.43%; 3.49 stars |
| clockworks~tiktok-scraper | result (+0.001 a run; transcript $0.034-0.048 a minute) | 0.0037 / 0.003 / 0.0023 | 8.5M, 98.15% |
| clockworks~tiktok-profile-scraper | result (+ video download 0.0013 / 0.001) | 0.003 / 0.002 / 0.0015 | 3.6M, **94.9%** |

Effective cost (Starter, measured where marked): profile $0.0023 (measured); 12 recent IG posts $0.0023 via the profile
(measured, no views) or $0.0276 via the post scraper (measured); 30 posts $0.069; a reel media URL comes with the post
(signed CDN link, see expiry below); a hashtag page of 30-50 posts $0.069-0.115; 12 TikTok posts $0.024 (measured).
An error item (no account, no posts in the window) is billed like a result (measured: `katiefanggg` and
`realraywilliam` returned one billed item each on Instagram).

Fields: profile followers, bio, business flag, related profiles; posts `caption`, `hashtags`, `likesCount` (-1 when
hidden), `commentsCount`, `timestamp`, `type`/`productType`, `childPosts`/`images`, `displayUrl`, `videoUrl`,
`videoViewCount`/`videoPlayCount` (post scraper only); `onlyPostsNewerThan`, `resultsLimit`, `skipPinnedPosts`. No
Instagram transcripts. TikTok: `playCount`, `diggCount`, `shareCount`, `commentCount`, `collectCount`,
`slideshowImageLinks`, `videoMeta`; video files only with the paid download.

Terms: the actors scrape logged-out public data ("The Actor scrapes public data, so it gets the same counts Instagram
shows a visitor who isn't signed in" - post scraper README; "No Business or Creator account, no OAuth" - instagram-scraper).
Apify General Terms (https://docs.apify.com/legal/general-terms-and-conditions) §5.8: "You are solely responsible for
the legality, accuracy, quality, appropriateness, and use of all Customer Data"; §3.1: Actors "are not part of the
Services". Retention of run data: not stated on the pricing page (unverified).

### 3.2 ScrapeCreators

Pricing (https://scrapecreators.com, terms dated 2026-11-15): free 100 credits at sign-up; Freelance $47 = 25,000 credits
($0.00188); Business $497 = 500,000 ($0.000994); Enterprise custom. "Purchased credits do not expire." One request = one
credit except `download_media` (+10), long transcripts (10 a minute), comments with replies (15). Cache hits
(`cache_max_age`), empty pages, missing transcripts and confirmed 403/404 are free.

Effective cost (Freelance / Business): profile $0.0019 / $0.0010; 12 posts (one `/v2/instagram/user/posts` page)
$0.0019 / $0.0010; 30 posts (3 pages) $0.0056 / $0.0030; reel media URL with the post, or a permanent copy for 11
credits ($0.021); hashtag or reel-search page $0.0019 (Google-indexed, "best-effort, not a complete hashtag feed";
a few items a page); TikTok 12 posts one credit.

Endpoints (https://docs.scrapecreators.com): `/v1/instagram/profile` (user, counts, some recent posts, related
profiles), `/v2/instagram/user/posts` (`next_max_id` paging, 12 a page: `caption.text`, `like_count`,
`comment_count`, `play_count`/`ig_play_count`, `image_versions2`, `video_versions`, `carousel_media`, `taken_at`,
`is_paid_partnership`), `/v1/instagram/user/reels`, `/v1/instagram/post`, `/v2/instagram/media/transcript`,
`/v2/instagram/post/comments`, `/v1/instagram/search/hashtag`, `/v2/instagram/reels/search`; TikTok profile, profile
videos (`/v3/tiktok/profile/videos`), transcript, keyword and hashtag search; ~38 platforms. Unverified: carousel
structure in the feed answer, how many posts the profile answer embeds.

Rate limits: "Scrape Creators does not enforce API rate limits" (under 500 concurrent advised). Freshness: live by
default, optional cache. Reliability: homepage 98.2% success, 3.12 s mean "last 30 days"; status page lists 36
incidents (Instagram: 2026-10-14, 08-25, 08-17 play counts missing, 07-14, 06-28); since 2026-10-18 private or
restricted profiles answer `success: true`, so the body must be checked.

Terms (https://scrapecreators.com/terms, /privacy; Web Scraping Guy LLC): "retrieve publicly available information
from third-party websites" (whether logged-in accounts are used is **not stated**); "You may use the data returned by
the Service however you want, including outreach, commercial use"; "You are solely responsible for ensuring your use of
the Service complies with the terms" of third-party sites; "We may store API response data, request metadata, usage
logs ... Usage logs may be retained indefinitely."

Its benchmark (https://scrapecreators.com/blog/apify-instagram-posts-scraper-benchmark, 2026-08-19, by its founder):
the 50 most-followed accounts x 12 posts; 50 feed requests ($0.094) vs one `apify/instagram-scraper` run of 600 results
at $0.0023 ($1.38); 10.2 s vs 121.4 s; 98.8% of shortcodes overlapped; fields not compared. Its companion test (1,000
post URLs) found Apify "had a better first-attempt success rate" (100% vs 99.6%).

### 3.3 HikerAPI

Pricing (https://hiker-doc.readthedocs.io/en/latest/guides/rate-limits/): 100 free requests; $0.02 a request until
~$100 of top-ups (Start), then $0.001 (Standard), $0.00069 (Business), $0.0006 (Ultra); 200/403/404 are billed. Profile
$0.001; 12 posts $0.002 (needs the user id first); reel media $0.001; hashtag page $0.001 (only `/g2/hashtag/medias`
remains; the recent/top/clips tabs are "Gone"). 15 requests a second on paid plans. Cursors are short-lived.

How it collects: "HikerAPI routes the request through managed Instagram sessions" on "the unofficial endpoints that
Instagram's mobile app uses internally" (https://hikerapi.com/instagram-private-api); the homepage calls it an
"Unofficial Instagram REST API" wrapping "the same Instagram private endpoints as the instagrapi" library.

Terms (https://hikerapi.com/help/terms-of-service): Section F lists, as needing "HikerAPI written permission", using
the Content "for commercial purposes" and "Create a database by downloading and storing the Content"; based in
Kazakhstan; media links are "temporary, signed links hosted on Instagram's content-delivery network".

Sign-up would take: an account on hikerapi.com (email; a third-party review mentions Telegram verification,
unverified), 100 free requests, then a top-up (payment methods not stated). **We did not sign up.**

### 3.4 Instagram Graph API: Business Discovery

https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/business_discovery. Free.
Needs our own Instagram professional account connected (Unified Login's system user: `instagram_basic`,
`instagram_manage_insights`, `pages_read_engagement`). Reaches **Business and Creator accounts only**; age-gated
accounts are not returned. Fields: `followers_count`, `media_count`, media `caption`, `like_count` (omitted when
hidden), `comments_count`, `media_type`, `media_product_type`, `media_url`, `permalink`, `shortcode`, `timestamp`,
`view_count` ("Available for Business Discovery only"), `children`. Not available: saves, shares, comment text,
transcripts. "The media_url field is omitted for video media that contains copyrighted or licensed audio" (and when the
owner turned off reel downloads). Cursor paging on `media`. Rate limit: "Calls within one hour = 200 * Number of Users"
(Platform Rate Limits). Hashtag search: 30 unique hashtags per 7 days, `recent_media` only the last 24 hours, needs
Instagram Public Content Access (App Review): not useful for discovery.

Terms (https://developers.facebook.com/terms/, 2026-02-03): no "Selling, licensing, or purchasing Platform Data"
(§3.a.iv); no "Processing Platform Data without valid User consent in order to build or augment user profiles for any
purpose" (§3.a.v); delete when "no longer necessary for a legitimate business purpose" (§3.d.i.2.a). §3.a.v needs an
owner reading: our reference_creators rows are research records about public professional accounts.

### 3.5 Free and official sources

- **Our own trend watcher** (`research_items`, ScrapeCreators search pages already paid for): a free candidate pool
  (anonymised authors: handles are hashed, so it gives formats and outliers, not names).
- **Apify profile scraper `relatedProfiles`**: neighbours of creators we already study, at no extra cost.
- **TikTok Research API**: academic and non-profit only ("independent of commercial interests"): not eligible.
- **TikTok Creative Center** (trending hashtags, songs, creators; free web pages): no public API confirmed.
- **YouTube Data API v3**: free, 10,000 units a day; `search.list` now has its own bucket of 100 calls a day.
- **Meta Ad Library API**: identity confirmation; covers political ads worldwide and all ads delivered in the EU/UK;
  branded-content access criteria unverified.

### 3.6 Platform terms (all scraping providers)

Instagram Terms of Use §4.2 (https://www.facebook.com/help/instagram/termsofuse): "You can't attempt to ... collect
information in an automated way without our express permission, regardless of whether such automated access or
collection is undertaken while logged-in to an Instagram account." This applies to Apify and ScrapeCreators alike
(both put the legal responsibility on the customer); only Business Discovery is permitted collection. The owner's
2026-11-25 decision (migration 041) allows logged-out public research data; CLAUDE.md's research privacy rule asks for
logged-out public data within platform and provider terms. **This tension predates this work and needs the owner's
explicit acceptance** (see section 8).

## 4. Terms verdicts

| Provider | Collection | Commercial use | Retention on their side | Verdict |
|---|---|---|---|---|
| Graph Business Discovery | official API, our account | allowed for our app's purpose; no selling, no user-profile building without consent (§3.a.v) | n/a | **Preferred** for Business/Creator accounts once connected; owner to read §3.a.v |
| Apify actors | logged-out public (stated) | customer's responsibility | unstated | Acceptable (current practice); fallback |
| ScrapeCreators | public data; logged-in use **not stated** | explicitly allowed | response data and logs, possibly indefinitely | Acceptable as primary **if** the owner accepts it on the same footing as Apify; ask them in writing whether logged-in accounts are used |
| HikerAPI | private mobile API via managed sessions | needs written permission; no database building | unstated | **Conflicts** with CLAUDE.md (logged-out only) and its own terms for our use: not recommended, not built |

## 5. Independent validation (2026-11-25, Fly `trends`, through `manyworlds.research.providers`)

Keys in our environment (names only): `APIFY_TOKEN` (Fly secret and local `.env`). No `SCRAPECREATORS_API_KEY`, no
HikerAPI key, no `META_SYSTEM_USER_TOKEN` (no connected account). So only Apify could be run. Script: 10 Instagram
handles (the pilot's 3 Instagram creators plus 7 of its TikTok creators tried under the same handle) and 10 of the
pilot's TikTok creators, 12 recent posts each. Billed amounts are Apify's own (`usageTotalUsd`,
`chargedEventCounts` per run, matched to our inputs); the pilot's concurrent runs were excluded.

| Method | Found | Billed | Per creator | Latency | Captions | Likes/comments | Image URL | Video URL | Reel views | Carousel children | URL expiry (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IG profile scraper (one run, 10 handles), latest posts | 9/10, 87 posts | $0.023 | **$0.0023** | 18 s for all 10 | 99% | 100% | 100% | 100% (48) | **0%** | 100% (34) | 105 h |
| IG post scraper, 12 per creator | 8/10 (2 billed error items), 85 posts | $0.2047 | $0.0276 ($0.0023/post) | 17 s median each | 99% | 100% | 100% | 100% (46) | 100% | 100% (34) | 105 h |
| TikTok profile scraper, 12 per creator, no download | 10/10, 120 posts | $0.216 | $0.024 ($0.002/result) | 8 s median each | 98% | 100% | 100% | 0% (download not bought) | 100% | 0% (10 slideshows) | 47 h |

Total billed $0.44 (cap $1). Our ledger estimate for the same runs was $0.54: policy prices were Free-tier list prices
and missed the post-details event; corrected in policy (`intelligence.agent.usd_per_item`, `usd_per_hashtag_item`).

Comparison with vendor claims:

| Claim | Vendor | Validated |
|---|---|---|
| Apify, 600 IG posts / 50 creators | $1.38 ($0.0023 a post) | **Confirmed** for the post scraper: $0.0023 a post billed. Via the profile scraper the same posts cost 50 x $0.0023 = **$0.115** (no reel views) |
| ScrapeCreators, same job | $0.094 (50 x $0.00188) | **Not validated** (no key). Consistent with its price list |
| ScrapeCreators vs Apify on TikTok, 12 posts | 1 credit vs per-result | Apify measured $0.024; ScrapeCreators $0.0019 by price list (12.6x) |
| HikerAPI | $0.001 a request (Standard) | Not validated; not recommended (terms) |

Media URL expiry: Instagram CDN links carry `oe=` (hex expiry), median ~4.4 days from fetch; TikTok ~2 days. Media for
selected posts is copied in the same check that found it, never later.

Pilot findings worth the owner's eye (read-only, not touched):
- 50% of its spend is the hashtag candidate pool (items the selection then mostly discards).
- Its ledger overstates TikTok: it records $0.0043 an item (Free-tier result + download); our plan bills ~$0.003.
- `tiktok-profile-scraper` has 45 ledger entries for 36 creators, and during the benchmark Apify showed the pilot
  re-running 30-post pulls with video downloads for creators it already stored (reba, katiefanggg, katiepettine): the
  scout step is remembered only when it completes, so a lost lease or retry mid-scout re-pulls (and re-pays).
- Instagram: 182 post items billed, 46 posts stored (10 profiles pulled, 3 creators kept).

## 6. Recommended routing (policy `research.providers.routes`)

| Task | Instagram | TikTok |
|---|---|---|
| profiles (and their latest posts: the monitor's cheap check) | graph -> scrapecreators -> apify (profile scraper, 25 handles a run) | scrapecreators -> apify |
| posts (metadata) | graph -> scrapecreators -> apify | scrapecreators -> apify |
| posts_media (selected posts: media URLs) | scrapecreators -> apify | scrapecreators -> apify |
| search (discovery pages) | scrapecreators -> apify | scrapecreators -> apify |

The router skips a provider that is unavailable (no key, no connected account, over 180 Graph calls an hour), fails,
or answers without the task's required fields for 80% of items (`research.providers.required`), and writes every call
to the ledger as `<provider>:<task>` under the caller's agent and job (a failed per-request call is charged one credit).
Until the owner buys ScrapeCreators credits or connects an account, every route resolves to Apify.

## 7. What was built

- `manyworlds/research/providers.py`: `ApifyProvider` (reuses the pilot's actors and normalisers; 12 or fewer IG posts
  come from the profile scraper's embedded posts), `ScrapeCreatorsProvider` (reuses `scrapecreators.get`; cursor paging,
  one credit a page), `GraphProvider` (Business Discovery through `publish.instagram.graph`), the `Router`, and
  `shortcode_to_id` so a post fetched by any provider is one row. HikerAPI: catalogued as `private_api`, disabled.
- Migration `045_research_monitoring.sql`: `reference_creators.tier, tier_scores, next_check_at, last_checked_at,
  last_post_id, content_hash, check_failures`; `reference_posts.content_hash, selected`; `reference_annotations.content_hash`.
- `manyworlds/research/monitor.py`: tiers from relevance, growth, performance and intelligence value
  (`intelligence.monitor.weights`, `bands`; dormant after 120 days without a post); cadences hot 1 day, warm 7, cold 30,
  dormant 90 (spread by +-10%); each check is one cheap call (Instagram: batched 25 handles a run); dedupe by post id;
  deterministic selection (outperformer >= 2x own median, new format, >= 2 new hashtags, commercial marker or
  paid-partnership flag; at most 3 a check, last 45 days); research copies (images and video) only for selected posts,
  each kind once; failures back off (1, 2, 4 ... days); a daily budget (`daily_usd`, ledger agent `intelligence:monitor`).
- Annotations remember the content hash (media type + caption) they read: unchanged content is never read again; an
  edited caption is read once more.
- The intelligence agent: `monitor` mode (checks, then a vision read of selected posts not yet read, within the cap);
  scout enrolls every creator it pulls into the monitored universe; scout copies videos only for selected posts
  (`intelligence.agent.videos_for: selected`, Apify's paid TikTok download is no longer bought for every video).
- `mw research monitor [--dry-run | --estimate N | --usd X]`; `monitor.ensure_due` on the Brain cycle queues one run a
  day while `intelligence.monitor.enabled` and operations are live (shipped off).

## 8. Expected cost

`mw research monitor --estimate N` uses the policy mix (hot 2%, warm 18%, cold 60%, dormant 20%), select rate 15% of
checks and $0.004 a vision read. For **5,000 creators**: ~10,190 checks a month (3,000 hot, 3,857 warm, 3,000 cold,
333 dormant), ~1,530 selected posts.

| Universe of 5,000 | Checks | Vision | Total / month |
|---|---|---|---|
| Instagram only, Apify profile scraper ($0.0023) | $23 | $6 | **~$30** |
| Instagram only, ScrapeCreators ($0.00188) | $19 | $6 | ~$25 |
| Instagram, Business Discovery for professional accounts, the rest ScrapeCreators | ~$5-10 | $6 | ~$12-16 |
| Half TikTok, Apify ($0.024 a TikTok check) | $134 | $6 | ~$140 |
| Half TikTok, ScrapeCreators everywhere | $19 | $6 | ~$25 |
| For comparison: the pilot's full pull, monthly | | | ~$1,550 |

Discovery (finding new creators) is separate: it should prefer the trend watcher's existing pages and related profiles
over fresh hashtag pulls, which were half of the pilot's bill.

## 9. Needs the owner

1. **ScrapeCreators**: sign up (100 free credits) so we can run the same benchmark on it (`bench` script in this work,
   ~50 credits), then decide whether to buy the $47 pack; set `SCRAPECREATORS_API_KEY` as a Fly secret. Ask them in
   writing whether any data is collected with logged-in accounts.
2. **Terms decision**: Instagram's §4.2 forbids automated collection without permission, logged in or not; it applies
   to Apify today. Confirm the research may continue on that footing (or restrict it to Business Discovery).
3. **HikerAPI**: rejected under CLAUDE.md's logged-out rule and its own terms; only revisit with an explicit owner
   override and HikerAPI's written commercial permission.
4. **Connect an Instagram professional account** (free official route; read Meta Platform Terms §3.a.v).
5. **Turn monitoring on**: `intelligence.monitor.enabled: true` and a `daily_usd` (shipped $1/day).
6. **Pilot**: its scout re-pulls creators after a retry, and half its spend is hashtag pools; consider capping
   `items_per_hashtag` for the next pilot.
