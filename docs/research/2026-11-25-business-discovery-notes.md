# Instagram Business Discovery: what the official docs say (read 2026-11-25)

Scope: our own Instagram professional account reading other public Business/Creator accounts
through `business_discovery`. Primary sources only (developers.facebook.com). No app created, no
API called. Docs current version: v27.0 ("The latest version is: v27.0", IG Media reference).
Platform Terms and Developer Policies both "Last updated February 3, 2026".

Quotes are short excerpts; the fetch tool summarised long pages, so treat paraphrases (no quote
marks) as paraphrase, and re-read the page before relying on a clause in a contract or policy.

Sources (abbreviations used below):
- [BD-ref] https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/business_discovery
- [BD-guide] https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/business-discovery
- [IGU] https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user
- [IGU-media] https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/media
- [IGM] https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-media
- [OVR] https://developers.facebook.com/docs/instagram-platform/overview (and /documentation/instagram-platform/overview)
- [IGL] https://developers.facebook.com/docs/instagram-platform/instagram-api-with-instagram-login
- [RL] https://developers.facebook.com/docs/graph-api/overview/rate-limiting
- [AL] https://developers.facebook.com/docs/graph-api/overview/access-levels
- [AR] https://developers.facebook.com/docs/instagram-platform/app-review
- [PERM] https://developers.facebook.com/docs/permissions
- [IPCA] https://developers.facebook.com/docs/features-reference/instagram-public-content-access
- [FLB] https://developers.facebook.com/docs/facebook-login/facebook-login-for-business
- [SU] https://developers.facebook.com/docs/marketing-api/system-users/overview
- [CL] https://developers.facebook.com/docs/instagram-platform/changelog
- [PT] https://developers.facebook.com/terms/ (Platform Terms)
- [DP] https://developers.facebook.com/devpolicy/ (Developer Policies)

## 0. Shape of the call

`GET /<OUR_IG_USER_ID>?fields=business_discovery.username(<handle>){followers_count,media_count,media{...}}`
- Read only; we query on our own IG user and name the target by `username` [BD-ref].
- "Returns data about another Instagram Business or Creator IG User." [BD-ref]
- Media can only be read nested: a direct GET on a returned media id "will fail due to insufficient
  permissions." [BD-guide]. So no per-media follow-up calls, no media insights edge.
- Target must be a professional account: "This API returns only data for media owned by Instagram
  professional accounts." [IGM]. Personal accounts are unreachable.

## 1. Fields

"You can use field expansion get public fields on the targeted IG User." [BD-ref]. Public = the
fields marked Public in [IGU]/[IGM]; neither page defines "Public" beyond "Public fields can be read
via field expansion" [IGM].

### User (target account), Public in [IGU]
`id`, `username`, `biography`, `followers_count`, `media_count`, `website`, `alt_text` (listed Public
on the IG User page; odd for a user, likely a doc artefact), plus edge `media`.
Not Public (own account only): `follows_count`, `name`, `profile_picture_url`, `has_profile_pic`,
`is_published`, `legacy_instagram_user_id`, `shopping_product_tag_eligibility`, `collaborative_media_search`.
Note: `name` and `profile_picture_url` are NOT marked Public in the current [IGU] table, although
older docs and many integrations read them via business discovery. Unverified; test before depending.

### Media (target's posts), Public in [IGM]
`id`, `caption` ("Caption. Excludes album children."), `media_type` ("Can be `CAROUSEL_ALBUM`, `IMAGE`,
or `VIDEO`."), `media_product_type` ("Can be `AD`, `FEED`, `STORY` or `REELS`."), `media_url`,
`thumbnail_url` ("Only available on `VIDEO` media."), `permalink`, `shortcode`, `timestamp`, `username`,
`is_shared_to_feed` (Reels only), `media_audio_type` (video only; new 2026-06-01), `alt_text`
(images; new 2025-03-24), `view_count`, `reposts_count`, `total_comments_count`, `total_like_count`,
edge `children`.
- `view_count`: "View count for Instagram Reels, which includes both paid and organic metrics";
  changelog 2025-06-16: "only available using the Business Discovery API for Instagram Reels." [CL]
- `owner`: "Only returned if the app user making the query also created the media." -> not for others.
- `total_like_count` / `total_comments_count`: marked "Available for Instagram API with Facebook
  Login only" [IGM]; whether business discovery returns them for others' media is not stated.

### Counts used in every BD example but NOT marked Public in [IGM]
`like_count`, `comments_count`. The BD reference sample requests
`media{comments_count,like_count,view_count}` [BD-ref], so they are reachable in practice per the docs'
own examples; the Public markers on [IGM] are inconsistent with that.
- Hidden likes: "the `like_count` field is omitted if the media owner has hidden like counts" when
  read through another endpoint/field expansion [IGM]. The BD pages themselves do not mention it.
- `comments_count` excludes album children and the caption [IGM].

### Not reachable for others' media
`insights` edge, `collaborators` edge (not Public), `saved_count` ("media owner or an accepted
collaborator only"), `shares_count`, `total_views_count` (Facebook Login, video only; no BD
statement), `is_ai_generated`, `is_comment_enabled`, `boost_*`, `copyright_check_information`,
`legacy_instagram_media_id`. Stories: no BD path (IG User media edge excludes stories [IGU-media]).
Comment text is not a BD field (and our data policy marks it red anyway).

### Children (carousels)
- `children` is a Public edge: "a collection of Instagram Media objects on an album" [IGM].
- "Some fields, such as `permalink`, cannot be used on photos within albums (children)." [IGM]
- reposts/saved/shares/total_* "are not available for carousel child media." [IGM]
- Per-child engagement does not exist (counts on the parent exclude child likes/comments).
- Nested syntax `media{children{media_type,media_url}}` is standard Graph field expansion; no BD
  page shows it, so treat as expected-but-untested.

### Conditional / omitted
- Age-gated: "Data about age-gated Instagram Business IG Users will not be returned." [BD-ref]
- `media_url` omitted when "The video contains copyrighted or licensed audio" (all video, incl. reels,
  incl. our own) and when "The reel owner has turned off downloads for the reel" (endpoints reading
  others' media: business discovery, tags, mentions, hashtag search, collaborative media) [IGM, CL
  2026-07-30]. Docs: "Treat `media_url` as an optional field ... fall back to `permalink` or
  `thumbnail_url`" [IGM]. Practical effect: a large share of trending Reels (library audio) come
  with no video URL; `thumbnail_url` survives (Public, video only) - docs do not say it is ever omitted.
- `caption`: "The `@` symbol is excluded, unless the app user can perform admin-equivalent" tasks [IGM].
- Engagement fields cover only the organic media, not ads that include it [IGM].

### Pagination and depth
- "The /media edge supports cursor-based pagination"; response "will not include previous or next
  fields" - you build the next request from the `after` cursor yourself, e.g.
  `media.after(<cursor>).limit(N){...}` [BD-ref].
- Max page size: not documented for BD. Max depth: not documented for BD. The owner-side
  `/media` edge says "Returns a maximum of 10K of the most recently created media" [IGU-media];
  assuming the same cap for BD is reasonable but unverified. 50 posts per creator is far inside it.

## 2. Rate limits

- "Business Discovery and Hashtag Search API are subject to Platform Rate Limits." [RL, OVR]
  i.e. NOT the Instagram BUC formula (`Calls within 24 hours = 4800 * Number of Impressions`,
  error 80002, header `X-Business-Use-Case-Usage`), which covers other Instagram endpoints.
- Platform limit: `Calls within one hour = 200 * Number of Users`, "a rolling one hour window";
  Number of Users is "based on the number of unique daily active users an app has" [RL].
- Header `X-App-Usage` {`call_count`, `total_cputime`, `total_time`} as percentages [RL].
- Errors: code 4 (app reached its limit), code 17 ("the User whose token is being used in the request
  has reached their rate limit") [RL]. User-level counts are hidden: docs do "not reveal actual call
  count values for users."
- "each ID counts as one API call"; how nested field expansion is counted is not documented [RL].
- Not BD: changelog 2026-03-30 raised the Creator Marketplace "Discovery API" from 240 to 1,000
  queries per user per hour [CL]. That is a different product (instagram_creator_marketplace_discovery);
  do not apply it to business_discovery.

Throughput estimate (500 creators x 50 posts per week), our own app with 1 daily active user:
- Ceiling 200 calls/h = 4,800/day = 33,600/week (if "users" counts our single operator/system user;
  not documented how a system user counts).
- One BD call returns profile + one media page. If `limit(50)` is honoured: ~500 calls/week; if pages
  cap at 25: ~1,000 calls/week; with carousel `children` nested in the same call, no extra calls.
  That is 1.5-3% of the weekly ceiling, ~6-12 calls/hour if spread. Daily refreshes of counts for all
  500 would be ~500-1,000 calls/day (10-20% of the day). Room is ample; the unknowns are the CPU-time
  share (`total_cputime`) of heavy nested calls, and the hidden per-user limit (code 17).
- Spread calls, read `X-App-Usage`, back off on 4/17/32/613.

## 3. Permissions, access level, tokens

- [BD-ref] Requirements: availability "Instagram API with Facebook Login"; "A Facebook User access token
  with the following permissions": `instagram_basic`, `instagram_manage_insights`,
  `pages_read_engagement`; "If the token is from a User whose Page role was granted via the Business
  Manager," also `ads_management` or `ads_read`.
- [BD-guide]: "This API is available with both the Instagram API with Facebook Login and the Instagram
  API with Unified Login." Unified Login's token is a "Business Integration System User Access Token",
  "Facebook Page: Not required", "Business portfolio: Required" [OVR]. So the guide implies a system
  user token works; the reference lists only a Facebook User token. Unresolved conflict - must be
  tested (our publisher already uses `META_SYSTEM_USER_TOKEN`).
- Instagram API with Instagram Login (graph.instagram.com, `instagram_business_*` scopes, no Page):
  no doc says BD is supported there; [BD-ref] names Facebook Login only, [BD-guide] names Facebook
  and Unified Login only; [IGL] limitation: "This API setup cannot access ads or tagging." Conclusion:
  treat BD as NOT available with Instagram Login (inferred from absence; no explicit "not supported").
- `business_management`: not listed for BD; needed only to manage Business Manager assets (system user
  setup), not for the call itself [PERM].
- Instagram Public Content Access feature: covers "Hashtag Search endpoints" only, not BD [IPCA].
- App Review: [AR] table: "My app is only for a business I own or manage." -> "Standard Access" ->
  App Review "Not required" (both Facebook Login and Instagram Login rows). Standard Access:
  "Permissions with Standard Access can only be requested from app users who have a role on the
  requesting app." [AL]. Advanced Access needs App Review + Business Verification [AL, OVR] and only
  matters if other businesses' users log in. Our own-account use fits Standard Access.
- Caveat: Platform Terms 3.a.vii bans materially changing data processing "without re-submitting for
  App Review" - irrelevant while we never submit, but [PERM] reviewer guidance for
  `instagram_manage_insights` is framed as insights "for other businesses"; nothing says BD-for-research
  is a disallowed use under Standard Access, nor that it is allowed. Silent.
- Tokens: short-lived user token "valid for one hour"; long-lived "valid for 60 days and can be
  refreshed before they expire" [OVR]. System user token "Defaults to never expire"; optional
  `set_token_expires_in_60_days` [FLB]. [SU]: "To create a system user access token, this app must
  have Standard Access"; Standard Access allows 1 system user + 1 admin system user per business.

## 4. Usage restrictions and retention (Platform Terms, Developer Policies)

Platform Data = "any information, data, or other content you obtain from us, through Platform or
through your App" [PT 12], and includes data "anonymized, aggregated, or derived from" Platform Data.
BD output (counts, captions, media URLs, and anything we compute from them) is Platform Data.

Prohibited Practices [PT 3.a] (relevant ones):
- iii "Processing Platform Data to perform, facilitate, or provide tools for surveillance."
- iv "Selling, licensing, or purchasing Platform Data."
- v "Processing Platform Data without valid User consent in order to build or augment user profiles"
  - "User" = the end user of our App (person or entity) [PT 12]. The target creators are not our
  Users and have not consented. Whether a per-creator research record (followers, cadence, format mix,
  pattern annotations) is a "user profile" is not defined. This is the main ambiguity for us; the
  conservative reading is that persistent per-account dossiers on third parties built from BD are at
  risk; aggregated pattern cards (not keyed to a person) are safer.
- vi no re-identification of anonymised data.
- viii "Processing Platform Data for purposes other than the applicable permitted purposes" (the
  Developer Docs). The BD guide's stated purpose: "get basic metadata and metrics" about other
  professional accounts; no list of allowed or disallowed purposes for BD exists.
Sharing [PT 3.c]: only as the Terms allow - with a Service Provider "so they can provide the services"
we requested; Service Provider must first agree "in writing" to "Process Platform Data solely for you
and at your direction", flow down to sub-processors, and "promptly delete all Platform Data" when the
relationship ends [PT 5.a]. Sending a `media_url` image or caption to an LLM vendor = Service Provider
processing: allowed only under a written agreement with those terms (Anthropic's commercial terms /
DPA would need to cover it). Docs are silent on LLMs or AI specifically.
Retention and deletion [PT 3.d]:
- No fixed period, no cache TTL anywhere. Delete "as soon as reasonably possible" when "no longer
  necessary for a legitimate business purpose", when we stop the product, when Meta asks, when a User
  asks, when law requires; update/delete "promptly" on request from Meta or the User; provide an easy
  deletion-request route. Aggregated/de-identified data that cannot be linked to a User is exempt from
  the user-request trigger only (3.d.i.2.d), not from the rest.
- No "cache" limit clause found in [PT] or [DP] (older Platform Policy's 24-hour-style caching rules
  are not in the current text).
Security [PT 6]: "Meet or exceed industry standards"; incident notice; audits "no more than once a
calendar year" with "at least 10 business days' written notice" [PT 7.c].
Developer Policies [DP]:
- 6.1 "Comply with any requirements or restrictions imposed on usage of Instagram user photos and
  videos" by their owners; "solely responsible for making use of User Content" in line with them.
- 6.2 "Don't use the Instagram Platform to simply display User Content, import or backup content"
  ... "without our prior permission." Storing other creators' media as a library would look like
  import/backup; transient analysis then deletion is the safer pattern. Not explicitly addressed.
Machine learning / AI training on Platform Data: neither [PT] nor [DP] mentions it (checked). Silent.
Combined with 6.1 (owners' restrictions) and 3.a.viii (permitted purposes), training any model on
BD media is not supported by the docs; analysis by a processor under 5.a is the defensible reading.
IP: [PT 2.c] - don't provide content that infringes rights; notice-and-takedown if the App shows
third-party content. Reproducing others' media in our product is out.

Fit with our data policy (green/amber/red): BD counts and profile facts = green-ish metrics but still
Platform Data (delete when no longer needed); captions and media = amber (hold until read, then
delete); red items (comment text, follower lists, face templates) are not offered by BD anyway.

## 5. Changes 2025-2026 [CL]

- 2025-01-21: `impressions` deprecated (media and user insights); media metrics `plays` and
  `clips_replays_count` removed; `views` metric added; Instagram v1.0 API deprecated. (Insights, not BD,
  but means `plays`/`video_views` style fields are gone; `view_count` is the BD Reel count.)
- 2025-01-27: old `business_*` scope values deprecated in favour of `instagram_business_*` (Instagram Login).
- 2025-03-24: `alt_text` on image media.
- 2025-06-16: `view_count` - "only available using the Business Discovery API for Instagram Reels."
- 2025-12-03: `crossposted_views`/`facebook_views` insights; "This change does not affect the Business
  Discovery API" (view_count stayed Instagram-only); `instagram_manage_contents` for deletion.
- 2026-03-30: Creator Marketplace Discovery API limit 240 -> 1,000 queries/user/hour (not BD).
- 2026-04-22: BD "`view_count` field now returns combined Instagram and Facebook views for crossposted
  video content" (reverses the Dec 2025 note); `reposts_count`, `saved_count`, `shares_count`,
  `total_views_count`, collaborative media API.
- 2026-06-01: `media_audio_type` (licensed music vs original sound) - Public, useful for our analysis.
- 2026-06-22: `is_ai_generated` on IG Media (not Public).
- 2026-07-30: `media_url` omitted for copyrighted/licensed-audio video and download-disabled reels, on
  BD and other cross-account reads.
- Platform Terms and Developer Policies updated 2026-02-03 (content of the change not diffed).
- Instagram Basic Display API ended (announced 2024-12-04; outside range).
- No `media_product_type` change: values still `AD`, `FEED`, `STORY`, `REELS`; Reels are
  `media_type=VIDEO` + `media_product_type=REELS`.

## What is verified in docs vs. unclear

Verified (stated on a developers.facebook.com page):
- BD returns public fields of professional accounts only; age-gated accounts return nothing.
- User Public fields: id, username, biography, followers_count, media_count, website.
- Media Public fields listed in section 1, incl. view_count (Reels, BD-only), media_audio_type, children.
- like_count omitted when owner hides likes; media_url omitted for licensed audio and download-off reels.
- Cursor pagination with `after`/`before`, no next/previous links.
- BD falls under Platform Rate Limits (200 x users per rolling hour), not the Instagram BUC formula.
- BD reference requires instagram_basic + instagram_manage_insights + pages_read_engagement (+ ads_read
  or ads_management with a Business-Manager Page role).
- Own-business apps: Standard Access, App Review "Not required".
- Token lifetimes: 1 h short, 60 d long-lived user; system user "Defaults to never expire".
- PT: no selling, no surveillance, no profiles without User consent, purpose limitation, delete when
  no longer needed, written Service Provider terms, no fixed retention period.
- DP 6.1/6.2: respect owners' restrictions on photos/videos; no import/backup without permission.

Unclear or silent:
- Whether a system user (Unified Login) token works for BD: guide says Unified Login yes; reference says
  Facebook User token. Test.
- BD with Instagram Login: not documented as supported; assume no.
- Whether name / profile_picture_url are still returned (not marked Public now).
- like_count / comments_count are used in BD examples but not marked Public; total_like_count etc. for
  others' media unstated.
- Max media page size and how far back BD reaches (10K cap documented only for own /media).
- How nested field expansion counts against rate limits; how a system user counts as a "user".
- Whether per-creator research records count as "build or augment user profiles" (PT 3.a.v).
- Any rule on sending Platform Data to an LLM vendor beyond the generic Service Provider clause.
- ML/AI training on Platform Data: not mentioned anywhere.
- Any caching TTL: none in current terms.
- Not checked (outside developers.facebook.com): Instagram Terms of Use and Meta's automated data
  collection terms.
