# Hosted talking avatars: terms on synthetic people (2026-11-22)

This note is for the owner, to read before setting `hosted_avatar.terms_reviewed: true` (manyworlds/hosted_avatar.py).
It is not legal advice.

How the quotes were checked:
- **Re-read 2026-11-22** means we fetched the page that day and the quoted text was there.
- **From the research note** means the quote comes from `docs/research/2026-11-22-video-quality-wave2-3.md` §4 and we did not re-read it today.

Our position: every creator is an original synthetic adult, never based on a real person. Every post carries the platform AI label and the disclosure line. We never imply real first-hand experience in sponsored content.

## Summary

| Provider | Commercial use of outputs | Synthetic people | Labelling | Verdict |
|---|---|---|---|---|
| **Kling AI Avatar v2** (via fal, the built adapter) | API terms: "not restricted" (§6.4) | not addressed; we carry likeness risk (§8.3–8.4) | API terms: none found; consumer terms require Kling branding (§4.5) | **Use the API terms only.** Lowest-friction ceiling benchmark |
| **Hedra Character-3** | Hedra claims no ownership (§3.2) but takes a broad licence that includes the right to "derive revenue" (§3.3) | AUP bars "fake personas" (see quote) | none found | **Ask Hedra in writing first** |
| **OmniHuman 1.5** (ByteDance, via fal or BytePlus) | not checked | BytePlus video terms bar portrait-right infringement (from the research note) | other BytePlus services require an AI label (OmniHuman text unverified) | Benchmark only after reading the BytePlus terms in full |
| **HeyGen** | paid plans may use outputs commercially; free plan may not | consent flow assumes a real person; synthetic avatars not addressed | must disclose where the law requires; never "entirely human generated" | Poor fit (built for real-person clones) |
| **Synthesia** | not checked | consent from "the individual" for custom avatars | must not strip provenance marks | Poor fit (corporate video; stock avatars barred from paid social ads without written consent) |

## Kling AI (API paid-service terms): re-read 2026-11-22
Source: <https://kling.ai/document-api/guides/protocols/paid-service.md>
- §6.4 permits commercial use: "use of the AI-generated content for commercial purposes is not restricted". It also says: "Please carefully evaluate before using it for external commercial purposes."
- §6.3 says the IP rights in outputs "still belong to you", subject to the law at the time.
- §8.3 covers rights in uploaded and generated content.
  - Protected rights include "portrait right, personality right, copyright…".
  - Content must "not infringe the legal rights of any third party".
- §8.4 puts disputes on us: "you shall be solely responsible", and you "bear full and sole liability".
- We found no AI-labelling clause and nothing on synthetic or fictional people.

**What this means for us.** Our creators are original, so no third party's portrait right is involved. The risk the terms
put on us is the risk our creator guardrails (`creator_min_age`, never based on a real person) already manage.

Kling's **consumer** user policy is stricter. Use it only through the API.
Source: <https://kling.ai/docs/user-policy>, re-read 2026-11-22.
- §4.5 requires the output to carry Kling's brand: "you shall label the Output interface with the brand and logo involved".
- §4.6 restricts commercial use: "without our written permission, you may not use, reproduce, distribute".

The **fal** endpoints are `fal-ai/kling-video/ai-avatar/v2/standard` at "$0.0562 per second" and `.../pro` at "$0.115 per
second". The output duration "Matches audio length". Read from the fal model pages, 2026-11-22. fal's own platform
terms sit on top of Kling's. They are **not reviewed** in this note: read them before turning the switch on.

## Hedra: re-read 2026-11-22
Sources: Terms <https://www.hedra.com/terms>; AUP <https://www.hedra.com/acceptable-use>
- **§3.2:** "Hedra does not claim ownership of any Inputs or Outputs."
- **§3.3:** we grant Hedra a licence that is "non-exclusive, transferable, perpetual, irrevocable, worldwide", including the right to
  "derive revenue or other remuneration from Your Content".
- **AUP:** bars using Hedra "to impersonate real entities or create fake personas (also known as "deepfakes") to falsely
  attribute content or mislead others about its origin without consent or legal right".
- **Reading.**
  - Disclosed AI creators arguably fall outside "falsely attribute… or mislead". The clause is about deception.
  - But "create fake personas" is broad, and we will not rely on a reading. Get a written answer from Hedra before any production use.
  - Pricing (from the research note): $0.025/s at 540p, $0.05/s at 720p, $0.0625/s at 1080p, up to 10 min.

## OmniHuman 1.5 (ByteDance)
- **fal: re-read 2026-11-22.** "Your request will cost $0.16 per second". Audio formats are listed; no maximum length is stated. Source: <https://fal.ai/models/fal-ai/bytedance/omnihuman/v1.5>
- **BytePlus Video Generation terms: NOT re-read.** The page did not render its body text for us.
  - The research note says they bar content that infringes portrait rights.
  - Other BytePlus services require a prominent AI label. The OmniHuman-specific text is unverified.
  - The same docs site links "Copyright and Portrait Feature Usage Rules" and a C2PA guide: read both. Source: <https://docs.byteplus.com/en/docs/ModelArk/Specific_Terms_for_the_BytePlus_Video_Generation_Model_Services>

## HeyGen: re-read 2026-11-22
Source: <https://www.heygen.com/terms>
- **Paid plans:** you own outputs, and the terms do not restrict use of User Output "for your own purposes (including for commercial purposes)".
- **Free plan:** outputs "may not be sold, sublicensed, redistributed, monetized".
- **Disclosure:** you must not present output as "entirely human generated". You "must proactively disclose" AI creation "to the extent required by applicable law".
- **Likeness:** reproducing someone's likeness "without the required permission" is objectionable. The research note quoted "explicit consent of the individual being represented". We did not find that phrase on the terms page today: it may be in the moderation policy or the avatar supplemental terms. **Unverified today.**
- **Synthetic avatars:** not addressed.

## Synthesia: re-read 2026-11-22
Source: <https://www.synthesia.io/legal/acceptable-use-policy>
- **Custom avatars in promoted social ads:** allowed "only with appropriate consent from the individual".
- **Stock avatars in promoted or paid social ads:** not allowed "absent written express consent from Synthesia".
- **Provenance:** you may not disable mechanisms that "help validate provenance or differentiate between human-generated and AI-generated content".

## What turning it on needs
1. **Read the terms.** The owner reads Kling's API paid-service terms and fal's platform terms (current versions), then sets `hosted_avatar.terms_reviewed: true`.
2. **Switch it on.**
   - Set `hosted_avatar.enabled: true`.
   - Set `FAL_KEY` as a Fly secret, piped and never typed in a terminal.
   - Add `hosted.avatar: {pools: [hosted], outputs_per_job: 1}` to `job_kinds`.
   - Add the kind to the hosted worker's kinds in `fly.toml`.
3. **Run as a benchmark only.**
   - 20 clips under a `bench-` batch.
   - `max_usd_per_job` caps each clip: $2 by default, so Standard tops out at 30 s, about $1.69.
   - 20 clips at about 10 s on Standard ≈ $11. That is over one day's $10 cap, so split the run across two days or use 8 s lines.
4. **Hedra stays out** until Hedra answers in writing.
