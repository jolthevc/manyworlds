# Business Discovery: where research ends and "user profiles" begin (read 2026-11-25)

Question: under Meta's Platform Terms, Developer Policies and the Instagram Platform docs, what may we do
with Business Discovery data about public professional (Business/Creator) accounts, studied from our own
professional account? Primary sources only; no app created, no API called. Companion to
`docs/research/2026-11-25-business-discovery-notes.md` (fields, limits, rate limits).

This is a reading of the texts, not legal advice. Every "Interpretation" line is ours, not Meta's.

## Sources

- [PT] Meta Platform Terms, "Last updated February 3, 2026": https://developers.facebook.com/terms
- [DP] Developer Policies, "Last updated February 3, 2026" (incorporated into the Platform Terms by
  reference): https://developers.facebook.com/devpolicy
- [BD] Business Discovery guide: https://developers.facebook.com/docs/instagram-platform/instagram-api-with-facebook-login/business-discovery
- [BD-ref] `business_discovery` reference: https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/business_discovery
- [IGU] IG User reference: https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user
- [IGM] IG Media reference: https://developers.facebook.com/docs/instagram-platform/reference/instagram-media
- [PERM] Permissions reference: https://developers.facebook.com/docs/permissions/
- [DDC] Data deletion callback: https://developers.facebook.com/docs/development/create-an-app/app-dashboard/data-deletion-callback
- [IGT] Instagram Terms of Use: https://help.instagram.com/581066165581870 (instagram.com/legal/terms redirects
  here; the page renders by script and we could not read the current text directly, so the quote below comes from
  Meta's own Help Center copy found by search. Verify it against the live page.)
- [ADC] Automated Data Collection Terms, "Effective October 7, 2024": https://www.facebook.com/apps/site_scraping_tos_terms.php
- [GDPR] Regulation (EU) 2016/679: https://eur-lex.europa.eu/eli/reg/2016/679/oj

## 1. The definitions that decide most of this

- **Platform Data** [PT 12.l]: "any information, data, or other content you obtain from us", explicitly
  "including data anonymized, aggregated, or derived from such data." So our annotations, scores, rankings
  and findings are Platform Data too. Nothing we compute escapes the Terms.
- **User** [PT 12.u]: "the end user of an App (whether a person or an entity)." The creators we look up
  are not end users of our App. Our own account holder is.
- **Process** [PT 12.m]: "any operation or set of operations performed on data". Viewing, storing,
  annotating and ranking all count.
- **Service Provider** [PT 12.r]: "an entity you use to provide you services in connection with Platform".
  Anthropic and Google are Service Providers when they process Platform Data for us.
- **Restricted Platform Data** [PT 12.p]: data that "reasonably can be used to identify a particular User or
  device", plus Meta's designated permissions list, with stated exclusions. The exclusions are a linked list
  we did not resolve. Interpretation: because the subjects are not our Users, their public handle may fall
  outside "Restricted" on a literal reading. Ask counsel, don't rely on it.
- **"user profiles"** (lower case) is **not defined** anywhere in the Platform Terms or the glossary. Meta
  gives no example of what counts.

## 2. The clauses in play

- **Profiles** [PT 3.a.v]: prohibited is "Processing Platform Data without valid User consent in order to
  build or augment user profiles for any purpose."
  [DP 10.7e] "Don't use Meta's data to build or augment any user profiles". Section 10.7 sits under
  "policies apply to all ad buying solutions", so it binds us only if we buy ads through the platform.
- **Surveillance** [PT 3.a.iii]: prohibited is "Processing Platform Data to perform, facilitate, or provide tools for
  surveillance." The definition says it "includes the Processing of Platform Data about people, places,
  groups, or events", followed by law-enforcement and national-security examples.
- **Permitted purposes** [PT 3.a.viii]: prohibited is processing "for purposes other than the applicable permitted purposes"
  in the Developer Docs. [PT 3.a.vii]: a change in core function or processing needs App Review again.
  [PT 4.c]: "You may only Process Platform Data as clearly described in your privacy policy."
- **Selling** [PT 3.a.iv]: prohibited is "Selling, licensing, or purchasing Platform Data." Because derived data is
  Platform Data, this reaches sold or licensed findings.
- **Re-identification** [PT 3.a.vi]: prohibited is "Attempting to decode, circumvent, re-identify, de-anonymize".
- **Eligibility, discrimination** [PT 3.a.i-ii]. These are not relevant to creative research unless the
  rankings are used to decide on people, for example whether to hire a real creator.
- **Deletion** [PT 3.d.i]: delete "promptly after receiving a request from us or the User". You must also
  delete when retention "is no longer necessary for a legitimate business purpose", when you stop operating the
  product, when Meta requests it, and when "a User requests" it, the last "unless the Platform Data has been aggregated,
  obscured, or de-identified." Each trigger is a separate sub-item. The aggregate exception is attached to the
  User-request item, not to the "no longer necessary" item.
- **Service Providers** [PT 5.a.i]: "You will not use a Service Provider in connection with your use of
  Platform unless such Service Provider first agrees in writing" to process "solely for you and at your
  direction", "for no other purpose, including for the Service Provider's own purposes", and to flow this
  down in writing to sub-service providers. Sharing with them is allowed by [PT 3.c.ii.2] "in order for them
  to provide the services you requested."
- **Security** [PT 6]: controls that "Meet or exceed industry standards given the sensitivity of the Platform Data".
- **Instagram Platform** [DP 6.1]: "Comply with any requirements or restrictions imposed on usage of
  Instagram user photos and videos". [DP 6.2]: don't use the platform "to simply display User Content"
  without prior permission.
- **Permissions note** [PERM, top of page, applies to every permission]: you may use any permission "to request analytics insights", "to
  improve your app and for marketing or advertising purposes," "through the use of aggregated and
  de-identified or anonymized information" "(provided such data cannot be re-identified)."

## 3. What Business Discovery is for, in Meta's words

- [BD]: "get basic metadata and metrics about other Instagram professional accounts".
- [BD-ref]: "Allows you to get data about other Instagram Business or Creator IG Users."
- [PERM] `instagram_manage_insights`: "Your app can also discover and read the profile info and media of
  other business profiles." Its allowed usage is "get metadata, data insights and story insights of an Instagram
  Business account." [PERM] `instagram_basic`: "Get basic metadata of an Instagram Business account
  profile, for example username and ID."
- Fields marked Public [IGU]: `alt_text`, `biography`, `followers_count`, `id`, `media_count`, `username`,
  `website`. **`name` is not marked Public** [IGU]. Our spec (`influencer-intelligence.md`) requests `name`, so
  check what the endpoint actually returns before storing it. Media Public fields [IGM] include `caption`,
  `comments_count`, `media_url`, `permalink`, `timestamp`, `view_count`, `thumbnail_url`. `like_count` is
  shown in the [BD] examples but is not marked Public in [IGM].
- Limits [BD]: no data about age-gated accounts. `media_url` is omitted for licensed-audio video, flagged
  content, and Reels whose owner disabled downloads. [IGM]: "Treat `media_url` as an optional field".
- **Silent:** none of [BD], [BD-ref] or [PERM] says anything about storing, caching, retention, competitive
  analysis, benchmarking or tracking over time. Meta does not name a use case for "studying successful
  creators". Interpretation: looking up named accounts and reading their metrics is the product's stated
  function, so doing that is inside its purpose. What we *build* from many lookups over time is not addressed.

## 4. Do the scraping terms apply?

- [IGT] (Help Center copy): "You can't attempt to create accounts or access or collect information in
  unauthorized ways", elsewhere glossed as collecting automatically "without our express permission".
- [ADC] covers "collection of data from Meta Company Products via the use of automated or programmatic
  tools" and requires "Meta's express written permission". The page never mentions APIs; its tools are
  scrapers, bots and crawlers reading "directly from web pages or applications".
- Interpretation: an approved app using Business Discovery under the Platform Terms is the authorised route.
  The Platform Terms are the "express permission", so [ADC] targets scraping, not the Graph API. Meta does not
  say so in terms. This matters only for the Apify/ScrapeCreators routes, which [ADC] and [IGT] plainly cover,
  not for Business Discovery.

## 5. Item by item

**(a) Storing public professional usernames and names.**
The texts allow obtaining these; that is the endpoint's function [BD-ref]. They say nothing about storage.
Deletion triggers apply [PT 3.d.i]. [PT 6] says "protect and not transfer, share, or solicit Meta user IDs".
Interpretation: a handle is needed to look an account up again, so storing it as a lookup key, with a purpose,
is the lowest-risk storage. Joining it with data from other sources (Apify pulls, research-desk findings, face
vectors in `impersonation.py`) is "augmenting". The more a row reads as a dossier on a person, the closer it is
to 3.a.v.

**(b) Storing public creator and post metrics.**
This is the stated purpose: "metadata and metrics" [BD]. Storage is not addressed. "No longer necessary" deletion
applies. Interpretation: metrics attached to a handle are the core of any "profile" in ordinary usage, but
they are also exactly what Meta markets the endpoint for. Lower risk when held per post for pattern
analysis. Higher risk when assembled into a per-creator record that grows.

**(c) Historical performance tracking (snapshots over time).**
The texts are silent on snapshots. Two clauses bite on a plain reading: "augment user profiles" [PT 3.a.v],
since each snapshot adds to a record about one account, and surveillance "about people" [PT 3.a.iii].
Interpretation: this is the item most likely to be read as profile-building, especially for creators who are
natural persons. Aggregate time series (a cohort's median growth) carry far less risk than per-handle histories.

**(d) Transient visual/video analysis of `media_url`.**
Our own code: processing is permitted if it serves a stated purpose and the privacy policy describes it
[PT 3.a.viii, 4.c]. [DP 6.1] requires honouring owners' restrictions, so respect omitted `media_url` and
never work around download-disabled Reels with another source. LLM providers: Anthropic and Google are
Service Providers [PT 12.r]. They may be used only if each "first agrees in writing" to the 5.a.i terms
(solely for us, no own purposes, flow-down to sub-processors). Interpretation: standard API terms with
no-training defaults and a DPA may satisfy this, but [PT 5.a.i] asks for specific written undertakings. We did
not read the vendors' terms for this note. Counsel should compare them clause by clause. The texts don't
mention face analysis; our data policy already forbids face and voice templates of real people (red).

**(e) Stored generalized creative annotations without the media.**
These are still Platform Data ("derived from such data", [PT 12.l]). There is no special permission and no ban.
Interpretation: an annotation such as "candid travel photo, golden hour" keyed to a post id is a fact about
content, not a person. The risk is low *if* it is not rolled up into a per-creator description of the person
(body, ethnicity, lifestyle inferences). Those become profile attributes, and some would be sensitive data under
GDPR Art. 9.

**(f) Creator benchmarking (ranking creators against each other).**
The texts are silent on ranking. Interpretation: a ranking of named people is a derived attribute of each person,
which is profile-like, and it creates [PT 3.a.ii] eligibility risk if it ever drives decisions about real people.
Ranking pattern cards, formats or cohorts instead of people avoids most of this. Selling or licensing a
ranking is [PT 3.a.iv].

**(g) Aggregated research findings across many accounts.**
This is the best-supported use. [PERM] expressly allows analytics via "aggregated and de-identified or anonymized
information" that cannot be re-identified. The User-request deletion trigger exempts aggregated data [PT 3.d.i.2].
Conditions: the findings remain Platform Data [PT 12.l], so no selling or licensing [PT 3.a.iv], deletion when no
longer needed, and no small-n findings that point to one identifiable creator ([PT 3.a.vi], "cannot be re-identified").

**(h) Deletion, retention, Data Deletion requests, derived data.**
- Delete on Meta's request or the User's [PT 3.d.i.1]. The deletion request route must be available "to all Users
  who can access your App" [PT 4.b]. The callback [DDC] fires when "an app user removes your app", so it
  covers our own account only, not the creators we look up.
- Delete when "no longer necessary for a legitimate business purpose" [PT 3.d.i.2]. This reaches derived data,
  and the aggregate exception is not attached to it.
- Delete when we stop operating the product, and as Section 7 (enforcement) requires.
- Silent: no fixed retention period, and no mechanism for a looked-up creator to ask *us*. Interpretation:
  GDPR fills that gap, see below, and `mw research forget` is the right shape for it.

## 6. GDPR angle: a legal question, not resolved here

A creator account run by a natural person in the EU or UK is a data subject even when it is a "business"
account. Handles, names, metrics and per-person annotations are personal data. Points for counsel:
lawful basis (Art. 6(1)(f) legitimate interests and its balancing test); Art. 14 notice for data not obtained
from the subject, and whether Art. 14(5)(b) "disproportionate effort" applies to large-scale research;
Art. 21 right to object; data minimisation and storage limitation (Art. 5(1)(c),(e)); Art. 9 if annotations infer
sensitive traits; Art. 22 if rankings feed decisions about people; a DPIA for systematic monitoring (Art. 35(3)(c)
covers "systematic monitoring of a publicly accessible area on a large scale", which per-creator tracking may
resemble); processor terms with Anthropic and Google (Art. 28) and transfers (Chapter V). The Platform Terms add
their own layer: [PT 10A] and [PT 10B] reference Regulation (EU) 2016/679 and the UK GDPR.

## 7. Recommendation (risk-ranked, lowest risk first)

| Item | Verdict | Conditions / why |
|---|---|---|
| (g) Aggregated findings across many accounts | **Allowed with conditions** | Minimum cohort size (no finding traceable to one creator); never sold or licensed; listed in our privacy policy; deleted when no longer needed. Best textual support ([PERM] note). |
| (e) Generalized creative annotations, no media | **Allowed with conditions** | Keyed to post id, about the content not the person; no physical, ethnic, health or lifestyle inferences about the person; not rolled up into per-creator descriptions; a retention period. |
| (b) Public creator and post metrics | **Allowed with conditions** | Per-post storage for pattern analysis; per-creator current values only as needed; retention period; no joining with non-Meta sources on the same row. |
| (a) Usernames (and names) | **Allowed with conditions** | Store handles as lookup keys for a stated purpose; drop `name` unless the endpoint really returns it and we need it; honour `mw research forget`; no cross-source joins. |
| (d) Transient media analysis, own code | **Allowed with conditions** | In memory or under 24 h; honour omitted `media_url`; no face or voice templates; described in the privacy policy. |
| (d) Transient media analysis by Anthropic / Google | **Uncertain: keep disabled pending counsel** | Needs written Service Provider undertakings matching [PT 5.a.i], including flow-down; confirm no training or retention on the vendor side. |
| (f) Ranking named creators | **Uncertain: keep disabled pending counsel** | Rank patterns, formats and cohorts instead (allowed under (g)). Never rank real people for hiring or partnership decisions without counsel. |
| (c) Per-creator historical snapshots | **Uncertain: keep disabled pending counsel** | The reading closest to "augment user profiles" and "surveillance about people". Cohort-level time series only, until cleared. |
| Persistent per-creator dossier joining Business Discovery with other sources (scrapers, web research, face vectors) | **Not allowed** (on our reading) | This is "build or augment user profiles" without User consent [PT 3.a.v]; no consent route exists for non-Users. |
| Selling or licensing any Business Discovery-derived dataset, ranking or finding | **Not allowed** | [PT 3.a.iv] together with [PT 12.l] (derived data is Platform Data). |

Code implications, if the owner adopts this: a source tag `business_discovery` in `data_policy.sources` with its
own retention; a minimum cohort size for findings in `intelligence.retrieval`; per-handle history tables and
creator ranking off by default; LLM vision on Business Discovery media gated by a flag that stays off until
counsel confirms the Service Provider terms.

## 8. Questions for counsel

1. Does "user profiles" in [PT 3.a.v] cover a record about a professional account that is *not* a User of our
   App? If it does, since such a person can never give "valid User consent" through our App, is any persistent
   per-account record barred?
2. Where is the line between storing what Business Discovery is built to return ([BD]: "metadata and metrics
   about other ... accounts") and "augmenting" a profile? Is the trigger time (snapshots), joining sources, or
   inference?
3. Does per-creator tracking over time fall under the surveillance definition [PT 3.a.iii] ("about people")
   when the purpose is commercial creative research?
4. Do Anthropic's and Google's commercial API terms and DPAs meet [PT 5.a.i] (solely for us, no own purposes,
   written flow-down)? If not, what addendum is needed before media goes to either?
5. Is ranking named creators against each other a "profile" or an eligibility risk [PT 3.a.ii], and does it change
   if rankings are only internal?
6. Does the aggregate and de-identified exception reach only User-request deletion [PT 3.d.i.2], or also
   "no longer necessary"? What retention period for derived findings is defensible?
7. Is `name` (not marked Public in [IGU]) obtainable through Business Discovery, and should we store it at all?
8. Does [PT 4.c] require our public privacy policy to describe research on non-user accounts, and in how
   much detail?
9. GDPR and UK GDPR: legitimate-interest assessment; whether Art. 14 notice is required or Art. 14(5)(b) applies;
   whether a DPIA is required (Art. 35); handling objections and erasure from creators who never used our App;
   transfers to US model providers.
10. Do the Instagram Terms "unauthorized ways" clause [IGT] and the Automated Data Collection Terms [ADC] have
    any application to an approved app's Graph API calls? Separately, do they bar the Apify/ScrapeCreators routes,
    and does mixing those sources with Business Discovery data worsen the 3.a.v position?
11. Does the commercial model (owned brands, possible data or insight products) risk [PT 3.a.iv] if any
    deliverable embeds Business Discovery-derived findings?
