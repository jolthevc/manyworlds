# Social distribution at scale: Instagram, TikTok, YouTube Shorts (research, 2026-11-22)

Question: what is the legitimate way to run hundreds to thousands of openly AI-generated, labeled
creator accounts, with as much automation as the platforms allow? Out of scope: evading anti-abuse
systems. Account farms, phone-verification bypass, bought or aged accounts, anti-detect browsers and
proxy rotation all break platform terms (Instagram Terms: "You can't attempt to create accounts or
access or collect information in unauthorized ways"; TikTok bans "automation tools, scripts, or other
tricks designed to bypass TikTok's systems"). This note does not cover them further.

Confidence markers: **[official]** means a platform or vendor's own page. **[3rd]** means a third-party or
competitor page. **[unverified]** means sources disagree or I could not find the official page.

---

## TL;DR

1. **No platform has an account-creation API.** A person creates every Instagram, TikTok and YouTube
   account in the app or on the web, with their own email or phone. That human step is the real limit on
   scale, so plan staff time for it. After creation, automation covers publishing, comments, DMs
   (Instagram) and insights.
2. **Instagram: go direct to Meta's API.** We already do this (M4). Meta's App Review table says an app
   "only for a business I own or manage" needs **Standard Access, no App Review**. Advanced Access, with
   App Review and Business Verification, is only for serving accounts you don't own. Every creator is
   ours, so a Meta app claimed by our business portfolio, with the accounts as portfolio assets, can
   publish to all of them, set `is_ai_generated` on every post, and moderate comments and DMs. Limit: 50
   posts per account per 24 h (`content_publishing_limit` reports `quota_total` 50; one docs page says
   100). We pay no per-account fee.
3. **Every account that features an AI person must turn on Instagram's "AI-generated profile" label**
   (new 2026-08-31). Unlabeled accounts that Instagram detects get reduced reach and can lose
   recommendation eligibility. The label is set in profile settings; I found no API for it.
4. **TikTok: we can't get our own app audited for our own accounts.** TikTok's sharing guidelines list "a
   utility tool for uploading content to accounts that you or your team manage" as an unacceptable app,
   and unaudited apps can only post privately, for at most 5 users per day. The practical route is an
   aggregator whose TikTok app is already audited. Post with `is_aigc=true`. Limits: about 15 posts per
   creator per day, and 6 init requests per minute per token.
5. **YouTube: go direct.** `videos.insert` now has its own quota bucket: 100 calls per day per project at
   1 unit each [official]. `status.containsSyntheticMedia` is settable on upload. A Google account can
   manage up to 100 channels. Above 100 uploads per day, request a quota extension, which comes with a
   compliance audit.
6. **Best value at 1,000 profiles: Zernio (formerly Late), $1,218/mo** [official], with comments and DMs
   included. **bundle.social costs $400/mo flat** for unlimited accounts but has quotas. Neither one
   documents the AI-label flags I could find. **Upload-Post documents all three flags explicitly**
   (`is_aigc`, `is_ai_generated`, `containsSyntheticMedia`), but its largest plan is 225 profiles.
   **Ayrshare** documents `isAIGenerated` and has every feature, but costs about 4x Zernio at scale.
   Recommendation: Instagram and YouTube direct, TikTok (optionally all three) through an aggregator
   behind our publisher adapter. Details below.

---

## 1. Unified social media APIs

| Vendor | IG publish (Reels/photo/carousel/Stories) | IG comments | IG DMs | TikTok | YT Shorts | AI-label flag | Account connection | 100 profiles | 1,000 profiles |
|---|---|---|---|---|---|---|---|---|---|
| **Zernio (ex-Late, getlate.dev now redirects to zernio.com)** | Yes, incl. Stories; 100 posts/24 h stated | Yes (inbox) | Yes (inbox; 10k sent/mo free, then $1 per 10k) | Yes (+comments for Business API accounts) | Yes | Not documented | OAuth per account; Instagram Login default or Facebook Login | **$318/mo** | **$1,218/mo** |
| **bundle.social** | Yes (Reels/Stories not stated on pricing page) | Yes (50k/mo on Business) | Yes, IG + FB only, 1,000 sent per account/mo on Business, enabled by email | Yes (publish cap 15/day, comments) | Yes | TikTok flag "abstracted, but you must set it"; IG/YT not confirmed | OAuth per account, under bundle.social's reviewed Meta app; white-label on Enterprise | Pro **$100/mo** (10k posts/mo) | Business **$400/mo** (100k posts/mo, unlimited accounts) |
| **Ayrshare** | Yes: Stories, Reels (`audioName`), carousels (10), trial reels, collaborators | Yes | Yes | Yes | Yes | **Yes: `instagramOptions.isAIGenerated`** | OAuth per profile (Instagram Login default; Facebook Login needed for Reel music) | **$1,228/mo** (official example) | ~$5,400/mo list price [estimate from tier text: $8.99 profiles 31-100, $7.99 to 500, $1.99 above 500 billed annually]; Enterprise "about $1/profile" |
| **Upload-Post** | Yes (Reels, photos, carousels) | Not documented | Not documented | Yes (paid plans) | Yes | **Yes: generic `is_ai_generated` maps to TikTok `is_aigc`, IG `is_ai_generated`, YT `containsSyntheticMedia`** | OAuth per profile; white-label on Professional and above | Advanced (75) + 25 pack = **$212/mo** | No plan; Business tops out at 225 profiles ($438/mo). Custom |
| **Outstand** | Yes (11 platforms) | ? | ? | Yes | Yes | ? | BYO app by default; managed OAuth $69/mo [3rd] | ~$19/mo + per-post fees | Pay per post: $19 includes 3,000 posts, then ~$0.005-0.007 each [3rd, unverified]; unlimited accounts |
| **Postiz** (AGPL, self-host) | Yes | Partial | ? | Yes | Yes | ? | **Self-host means your own developer app on every platform** | Cloud Ultimate $99/mo (100 channels) | Self-host: free software, unlimited channels, but you need your own audited TikTok app (not possible for our case, see §3) |
| **Buffer API** (launched 2026-05-27) | Yes | Limited | No | Yes | Yes | Has an AI flag in the UI ("Flagging posts as AI-generated") | Personal API keys; third-party OAuth not yet enabled [3rd] | ~$5-6/channel/mo | ~$5-6k/mo; request caps (free plan: 3,000 req/mo) |
| **Publer** | Yes | Yes (UI) | No | Yes | Yes | ? | API only for "eligible" Business/Enterprise accounts | Business $10 + ~$7/account [3rd] | ~$6-7k/mo |
| **Metricool** | Yes | Yes (UI) | Inbox (UI) | Yes | Yes | Has EU AI Act labeling guide | API only on Advanced/Custom; Advanced covers 15/25/50 brands | Advanced "from $53/mo" (50 brands max) | Custom only |
| **Hootsuite** | Yes | Yes | Yes (UI) | Yes | Yes | ? | Per-user seats: $99 / $199 / $399, Enterprise custom | Seat-priced, not API-first | Enterprise quote |
| **Sprout Social** | Yes | Yes | Yes (UI) | Yes | Yes | ? | API only on Advanced, $399/seat/mo annually | Seat-priced | Enterprise quote |
| **Data365** | **Read-only public-data extraction; no publishing** | n/a | n/a | read | read | n/a | n/a | ~€300-850/mo | Not a distribution tool. It conflicts with our research-privacy rule unless we restrict it to logged-out public data |

Notes:
- **Official partner status:** none of these vendors claims Meta Business Partner or TikTok Marketing
  Partner status on the pages I checked. bundle.social says it publishes "under bundle.social's reviewed
  Meta app". All of them use the official APIs, so the platform limits below apply whatever the vendor.
- **Rate limits:** bundle.social allows 100 req/s, 500 per 10 s and 2,000 per minute, with per-account
  daily caps (IG 100/day publish and comment on Business; TikTok 15; YouTube 15). Ayrshare enforces IG's
  50 per 24 h and returns `usedQuota`. Zernio states IG at 100 per 24 h. Upload-Post's per-day caps are
  reported as TikTok 15 and IG 50 [3rd].
- **Instagram Reels with licensed music** needs Facebook Login (Ayrshare returns error 514 on Instagram
  Login). Meta's media reference says music tagging is only for original audio.

Sources: [Ayrshare pricing](https://www.ayrshare.com/pricing.md), [Ayrshare IG docs](https://www.ayrshare.com/docs/apis/post/social-networks/instagram),
[Zernio pricing](https://zernio.com/pricing.md), [Zernio docs pricing](https://docs.zernio.com/pricing), [Zernio IG docs](https://docs.zernio.com/platforms/instagram),
[bundle.social pricing](https://bundle.social/pricing.md), [bundle.social](https://bundle.social/), [bundle.social AI regs blog](https://info.bundle.social/blog/ai-regulations),
[Upload-Post pricing](https://www.upload-post.com/pricing.md), [Upload-Post AI labeling](https://docs.upload-post.com/guides/ai-content-labeling),
[Postiz cloud](https://docs.postiz.com/cloud/overview), [Postiz plans](https://docs.postiz.com/cloud/plans.md),
[Outstand via That Marketing Buddy](https://thatmarketingbuddy.com/pricing/outstand), [PostPeer on Outstand](https://www.postpeer.dev/blog/outstand-api-vs-ayrshare-vs-postpeer),
[Buffer API launch](https://buffer.com/resources/buffer-api-is-here/), [Buffer pricing](https://buffer.com/pricing), [Buffer AI flag](https://support.buffer.com/article/958-flagging-posts-as-ai-generated-in-buffer),
[Publer plans](https://publer.com/help/en/article/what-are-publers-plans-and-pricing-15h4yqh/), [Metricool API](https://help.metricool.com/plans-add-ons-and-api-access-explained-xux1u),
[Hootsuite plans](https://hootsuite.com/pricing), [Sprout pricing guide](https://sproutsocial.com/insights/which-sprout-social-plan-is-right-for-you/), [Data365 pricing](https://data365.co/pricing).

---

## 2. Meta: rules and limits

**Account creation:** there is no public API for it. Meta's business asset docs say you add an Instagram
account "must use the Business Manager UI"; the API only lists and assigns access to accounts that
already exist (`owned_instagram_assets`, `assigned_users`). Instagram Terms forbid creating accounts "in
unauthorized ways". Each account is made by hand in the app.
([Meta IG business assets](https://developers.secure.facebook.com/docs/business-management-apis/business-asset-management/guides/instagram-accounts),
[Instagram Terms](https://help.instagram.com/581066165581870))

**How many accounts:**
- Instagram has no published cap on how many accounts one person may own. Up to 5 can be logged in per
  device (some sources say 10) [3rd].
- Accounts Center holds up to 25 accounts (1 Facebook + 24 Instagram, or 25 Instagram) [Meta help, via
  search summary; page not fetched].
- Business portfolios: one person can create up to 2 [3rd, consistent across sources]. A cap of 25 IG
  accounts and 200 Pages per portfolio is reported [3rd, unverified]. Plan for several portfolios and
  check the limit in Business Settings.
- The Accounts Center and logged-in limits are app-side. API access uses tokens, so they don't limit API
  publishing.

**API configurations** ([overview](https://developers.facebook.com/documentation/instagram-platform/overview.md)):

| | Instagram Login | Facebook Login | Unified Login (recommended for new integrations) |
|---|---|---|---|
| Facebook Page needed | No | Yes | No |
| Business portfolio | No | Depends | **Required** |
| Token | IG User token (60-day long-lived) | Page token | **Business Integration System User token** |
| Messaging | Native | Via Messenger Platform | Yes |
| Hashtag search, product tags, partnership ads | No | Yes | Yes |

**Permissions:** with Instagram Login, `instagram_business_basic`, `instagram_business_content_publish(ing)`,
`instagram_business_manage_comments` and `instagram_business_manage_messages`; with Facebook Login,
`instagram_basic`, `instagram_content_publish`, `instagram_manage_comments`, `instagram_manage_messages`
and `pages_read_engagement`.

**App Review** ([table](https://developers.facebook.com/documentation/instagram-platform/app-review)):
"My app is only for a business I own or manage" needs Standard Access and no App Review. "I am a Tech
Provider and my app serves multiple businesses" needs Advanced Access, which means App Review plus
Business Verification. Standard Access works for app users "who have a role on the app or on a business
portfolio that has claimed it". Tester roles are capped at 50, or 500 once the business is verified
([app roles](https://developers.facebook.com/documentation/development/build-and-test/app-roles)).
**For us:** claim the app with our business portfolio, make every creator account a portfolio asset,
and use a system-user token (Unified Login). That should avoid per-account OAuth and App Review. **Verify
this on 2-3 accounts before scaling**, because Standard Access "limits the data an app can retrieve".
Business Verification is cheap insurance either way.

**Publishing limits** ([content publishing](https://developers.facebook.com/documentation/instagram-platform/content-publishing.md),
[limit endpoint](https://developers.facebook.com/documentation/instagram-platform/instagram-graph-api/reference/ig-user/content_publishing_limit.md)):
- `quota_total` is "currently 50" per 24 h. The guide page says 100 in one place and 50 in another.
  Read `GET /<IG_ID>/content_publishing_limit` and don't hard-code a number.
- A carousel counts as 1 post, holds up to 20 items, and its AI flag goes on the parent container only.
- Stories (`media_type=STORIES`) and Reels are supported. Meta's 2023 post says all media types count
  toward the cap.
- Trial reels (`trial_params.graduation_strategy` MANUAL or SS_PERFORMANCE) are shown to non-followers
  only.
- **`is_ai_generated=true` is set on container creation** and works with both logins. It can't be
  changed afterwards.
- Business Use Case rate limit: calls per 24 h = 4,800 x impressions. Private replies to comments: 750 per
  hour.

**Messaging** ([policy](https://developers.facebook.com/documentation/business-messaging/messenger-platform/policy)):
- Standard 24 h reply window. Up to 7 days after a click-to-Direct ad.
- The Human Agent tag allows 7 days for manual, non-promotional support only.
- Apps must have an escalation path to a human.
- Bots may only reply to people who wrote first. Cold DMs aren't possible.

**AI disclosure:**
- **Per post:** Meta requires an AI label on photorealistic video or realistic audio (still images are
  excluded from the requirement), and may penalize missing labels. The API flag is `is_ai_generated`.
  ([Meta help](https://www.meta.com/help/artificial-intelligence/1783222608822690/), [Instagram help](https://help.instagram.com/761121959519495))
- **Per profile (new 2026-08-31): the "AI-generated profile" label.** Profiles that feature an
  AI-generated person should turn it on. Turning it on does not reduce reach. Unlabeled profiles that
  Meta detects get reduced reach and may be ineligible for recommendations, recoverable by adding the
  label or appealing via Account Status. The label is set in Edit profile, and I found no API.
  ([Instagram creators blog](https://creators.instagram.com/blog/ai-generated-profile-label), [TechCrunch](https://techcrunch.com/2026/08/31/instagram-puts-new-limits-on-undisclosed-ai-profiles/))

**Inauthentic behavior** ([Meta policy](https://transparency.meta.com/policies/community-standards/inauthentic-behavior), version 2025-12-12):
- The current text has no standalone ban on running multiple accounts.
- It prohibits using networks of assets to mislead about identity, origin, or "a Meta asset's ownership or
  control network", to "deliver substantial quantities of fake engagement", or to make content look more
  popular than it is.
- Coordinated Inauthentic Behavior (CIB) is networks that "primarily seek to manipulate public debate".
  Foreign operators falsely posing as a "domestic or local voice" are called out specifically.

**What a legitimate operator of many disclosed accounts must do (derived from the above):**
1. Turn on the AI-generated profile label on every account, put the policy `disclosure_line` in the bio,
   and set `is_ai_generated` on every post.
2. Be open about ownership: name the studio in the bio or link a public page listing our creators. That
   addresses the "ownership or control network" test.
3. No cross-account engagement. No likes, follows, comments or shares between our accounts, and no
   follow-for-follow or engagement pods. This is already a project rule.
4. Stay away from politics, elections and civic topics. Never present a creator as a local voice in
   another country to influence opinion.
5. Use only official APIs, and create accounts by hand with real business contact details. Never buy
   accounts.
6. Use partnership-ad and branded-content labels for sponsored posts (`is_paid_partnership`, Facebook
   Login only). Never imply first-hand experience.

---

## 3. TikTok and YouTube Shorts

### TikTok
- **Content Posting API:**
  - Direct Post: `/v2/post/publish/video/init/` for video and `/v2/post/publish/content/init/` for photos,
    scope `video.publish`. Call `creator_info/query` first.
  - Upload (inbox/draft): the creator finishes the post in the app, so it doesn't fully automate.
  - Rate limit: 6 init requests per minute per user token.
  - Errors: `spam_risk_too_many_posts` (daily cap), `reached_active_user_cap` (client's daily
    active-creator cap from its audit).
  ([get started](https://developers.tiktok.com/doc/content-posting-api-get-started), [direct post ref](https://developers.tiktok.com/doc/content-posting-api-reference-direct-post))
- **Audit:**
  - Unaudited clients are limited to 5 posting users per 24 h, accounts must be private, and posts are
    `SELF_ONLY`.
  - An audit unlocks public posting and sets an active-creator cap per client.
  - Per-creator cap is about 15 posts per day, shared across all Direct Post clients.
  - **Not acceptable for audit:** "a utility tool for uploading content to accounts that you or your team
    manage", content copied from other platforms, and brand or watermark overlays.
  ([sharing guidelines](https://developers.tiktok.com/doc/content-sharing-guidelines))
  So we can't get our own TikTok app audited for our own creators. Audited third-party tools (the
  aggregators) are the route. This is a gray zone, because the guideline governs which apps TikTok will
  audit, not which tools a creator may use. TikTok's terms apply to the accounts either way.
- **AIGC:**
  - `post_info.is_aigc=true` on video, and top-level `is_aigc` on photo posts. The post is tagged "Creator
    labeled as AI-generated".
  - Community Guidelines: creators "must label AI-generated or significantly edited content that shows
    realistic-looking scenes or people". Unlabeled content may be removed or labeled by TikTok.
  ([guidelines](https://www.tiktok.com/community-guidelines/en/integrity-authenticity), [help](https://support.tiktok.com/en/using-tiktok/creating-videos/ai-generated-content))
- **Comments and DMs:**
  - Comment reply, hide and pin are available for TikTok Business API accounts via aggregators (Zernio
    since 2026-09-10; bundle.social automations) [3rd].
  - DMs: Business Messaging is US-registered Business accounts only, with a 48 h reply window and no DM
    to commenters [3rd]. Treat TikTok DMs as unavailable.
- **Multiple accounts:**
  - TikTok allows multiple accounts "but not to deceive others or break the rules". Fake engagement,
    covert influence operations, spam and ban evasion are prohibited.
  - The per-device account limit isn't published (reported as 3 to unlimited) [3rd].

### YouTube Shorts
- **Quota:**
  - `videos.insert` costs 1 unit in its own Video Uploads bucket, limited to 100 calls per day per project.
    Other calls use the 10,000 units per day pool; `commentThreads.insert` costs 50 units.
  - The revision history describes the move from about 1,600 units per upload to the granular buckets
    (2025-2026).
  - A quota extension form has been available since 2026-06-01; extensions require the API compliance
    audit.
  ([quota costs](https://developers.google.com/youtube/v3/determine_quota_cost), [videos.insert](https://developers.google.com/youtube/v3/docs/videos/insert), [revision history](https://developers.google.com/youtube/v3/revision_history))
- **Unverified projects:** the current `videos.insert` page says uploads "are not restricted to private
  viewing mode". The older rule (projects created after 2020-07-28 lock uploads to private until audited)
  is still quoted by third parties [conflict]. Pass the audit anyway, because we need it for quota.
- **Synthetic disclosure:** `status.containsSyntheticMedia` (boolean) is settable on insert and update.
  Policy requires disclosing realistic altered or synthetic content. Disclosure does not limit reach or
  monetization. Repeatedly failing to disclose can lead to removal or suspension from the YouTube Partner
  Program (YPP). ([YouTube help](https://support.google.com/youtube/answer/14328491))
- **Channels:** up to 100 channels per Google account, as Brand Accounts. Channel creation has no API.
  ([help](https://support.google.com/youtube/answer/3046356))
- **Monetization:** the "inauthentic content" YPP policy (formerly "repetitious") demonetizes generic,
  repetitive or mass-produced content, and **AI personas on sensitive topics (finance, medical, legal,
  health)**. Using AI is not disqualifying in itself. ([YPP policy](https://support.google.com/youtube/answer/1311392), [Tubefilter 2026-07](https://www.tubefilter.com/2026/07/13/youtube-inauthentic-content-monetization-policy-update/))

---

## 4. How established virtual-influencer studios operate, and official programs

- **Studios:**
  - Brud/Lil Miquela ("trans-media studio"), Aww Inc./Imma (Tokyo; manages several "virtual humans" and
    is selective about brand deals) and The Clueless/Aitana López (bio says "virtual soul").
  - All three run a **small number of deeply produced characters**, each a normal, hand-created
    professional account with open disclosure, a human talent manager and brand deals negotiated
    directly.
  - None of the public reporting describes a portfolio of hundreds of accounts. Scaling to 1,000 goes
    beyond what any public studio has shown, which raises scrutiny under the "network" tests.
  ([Hollywood Reporter](https://www.hollywoodreporter.com/business/digital/virtual-influencers-digital-world-1235228125/), [Fortune](https://fortune.com/europe/2023/11/23/spanish-influencer-agency-earned-11000-ai-model-posers/), [TechCrunch](https://techcrunch.com/?p=1768652))
- **Meta Business Partners:**
  - Member and Badged tiers. Criteria are ad spend, performance, certification and compliance; the
    review takes about 4-12 weeks [3rd].
  - Tech Partner status (an upgrade path in the WhatsApp docs) is for platforms. This is not a route to
    running our own accounts.
  ([Meta Tech Partner](https://developers.facebook.com/documentation/business-messaging/whatsapp/solution-providers/upgrade-to-tech-partner/))
- **Instagram Creator Marketplace API (GA 2025-10-01):**
  - Brands use it to find creators for partnership ads. It needs Facebook Login with
    `instagram_creator_marketplace_discovery`.
  - Whether AI-generated profiles can be listed is not documented.
  ([docs](https://developers.facebook.com/documentation/instagram-platform/instagram-api-with-facebook-login/creator-marketplace))
- **TikTok Marketing Partners:**
  - Categories are Agency, Creative, Marketing Technology and Measurement. The Creator Marketing
    specialty covers agencies, talent agencies, SaaS and UGC partners. A Premier Creative Partner tier
    was added 2026-11-10.
  - About $50k of ad spend in 90 days is a reported bar [3rd]. Apply at partners.tiktok.com.
  ([TikTok](https://ads.tiktok.com/business/en-US/blog/channel-sales-partners), [PPC Land](https://ppc.land/tiktok-ties-its-new-premier-creator-partner-badge-to-tiktok-one/))

---

## Recommendation

1. **Instagram: direct Graph API (keep M4).**
   - Move to Unified Login: a business portfolio that claims our Meta app, creator accounts as portfolio
     assets, and a system-user token.
   - Standard Access, no App Review. Do Business Verification.
   - Read `content_publishing_limit` per account, and set `is_ai_generated` on every container
     (carousels: parent only).
   - Comments, DMs and insights come from the same app.
   - Cost: $0 per account. Verify on 2-3 accounts first.
2. **TikTok: aggregator.** We can't get our own app audited for our own accounts.
   - Upload-Post is the only vendor that documents `is_aigc` explicitly: $212/mo at 100 profiles, custom
     quote beyond 225.
   - Zernio is the cheapest at 1,000 profiles ($1,218/mo, comments included). Confirm with Zernio that
     it passes `is_aigc` before committing.
   - bundle.social ($400/mo flat) only if its TikTok flag and IG/YT flags are confirmed.
   - Budget about 15 posts per creator per day at most (we need far fewer).
3. **YouTube Shorts: direct Data API.**
   - One GCP project, a channel per creator under Brand Accounts (up to 100 channels per Google account),
     `containsSyntheticMedia=true` on every upload.
   - Request quota extension and audit before passing 100 uploads per day.
4. **Value at 1,000 profiles if everything goes through one vendor:** Zernio ($1,218/mo, comments and DMs
   included) beats Ayrshare (~$5.4k list) and the seat-priced suites by a wide margin. bundle.social
   ($400/mo) is cheapest but has quotas (1,000 sent DMs per account per month, 50k comments per month).
   Put the vendor behind `publish/` as another adapter (project rule) so we can switch on measured
   reliability.
5. **Account creation is a manual onboarding step.** For each launch, a person creates the IG, TikTok and
   YT accounts with real business contact details, switches IG to professional, turns on the
   **AI-generated profile** label, adds the disclosure line and studio attribution to the bio, then runs
   `mw account connect`. Launches should move from "commissioned" to "awaiting owner account setup". At
   ~10-15 minutes per creator per platform, 1,000 creators is weeks of staff work. Ramp gradually.
6. **Compliance guardrails to encode in policy:**
   - Profile label, post flags and disclosure line on every account.
   - Zero cross-account engagement.
   - No political, civic or sensitive-advice personas (this also protects YPP eligibility).
   - Studio ownership stated publicly.
   - Sponsored posts use partnership labels.
   - Apply to Meta and TikTok partner programs only once there is ad spend. They are not a requirement
     for operating.

**Open items to verify by hand:**
- The per-portfolio IG account cap (Business Settings).
- Whether Standard Access with a system user covers publish, comments and messages on portfolio-owned
  accounts at scale (try 3 accounts).
- Which aggregators pass `is_aigc` and `is_ai_generated` (ask Zernio and bundle.social).
- Outstand's current pricing (outstand.so).
