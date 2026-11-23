# LLMs and VLMs for content planning, creative writing and automated judging (Nov 2026)

Research date: 2026-11-21. All prices are USD per 1M tokens (input / output) unless noted. Pages fetched on 2026-11-21. Most provider pages carry no "last updated" date, so "as of 2026-11-21 fetch" is the best date available. Many model names (DeepSeek V4.x, Kimi K3.x, GLM-5.3, Qwen3.8/Qwen4, GPT-6.x, Gemini 4.x) came out after the researcher's training data. They are reported exactly as the provider pages list them.

## Q1. Anthropic lineup and prices (Opus 5.5, Sonnet 5.5, Haiku 5.5, Fable 5.2), batch and caching

### Takeaway
Opus 5.5 costs $4/$20, Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50 (prompts up to 100K tokens) and Fable 5.2 $8/$40. Batch takes 50% off. Cache hits cost 0.05x input on Opus/Sonnet 5.5 and 0.1x on Haiku 5.5, and the batch and cache discounts stack. Haiku 5.5 is 20x cheaper than Sonnet 5.5 and as cheap as the cheapest open-weight models. It is the lowest-friction way to cut cost (same SDK, native structured outputs, vision), but its intelligence score is much lower.

### Cited Findings
- Full price table (base input / 5-minute cache write / 1-hour cache write / cache hit / output) — [Anthropic pricing docs](https://platform.claude.com/docs/en/about-claude/pricing):
  - Fable 5.2: $8 / $10 / $16 / $0.20 / $40
  - Opus 5.5: $4 / $5 / $8 / $0.20 / $20
  - Sonnet 5.5: $2 / $2.50 / $4 / $0.10 / $10
  - Haiku 5.5 (prompt ≤100K tokens): $0.10 / $0.125 / $0.20 / $0.01 / $0.50
  - Haiku 5.5 (prompt >100K tokens): $0.50 / $0.625 / $1 / $0.05 / $2.50
  - Haiku 4.5: $1 / $1.25 / $2 / $0.10 / $5
- Cache-hit multipliers: Opus 5.5 and Sonnet 5.5 pay 0.05x base input, Fable 5.2 and 5.1 pay 0.025x, all other models 0.1x. The caching multipliers stack with the Batch discount — [Anthropic pricing docs](https://platform.claude.com/docs/en/about-claude/pricing)
- Batch API: 50% off input and output. Batch prices: Opus 5.5 $2/$10, Sonnet 5.5 $1/$5, Haiku 5.5 $0.05/$0.25 (≤100K), Fable 5.2 $4/$20 — [Anthropic pricing docs](https://platform.claude.com/docs/en/about-claude/pricing)
- Haiku 5.5's long-context tier is decided per request, by total prompt length including cache reads and writes — [Anthropic pricing docs](https://platform.claude.com/docs/en/about-claude/pricing)
- Claude 4.7 and later use a tokenizer that produces about 30% more tokens for the same text, so compare per-task cost, not per-token cost — [Anthropic pricing docs](https://platform.claude.com/docs/en/about-claude/pricing)
- US-only inference (`inference_geo: "us"`) costs 1.1x. Fast mode for Opus 5.5 costs $8/$40 and can't be combined with Batch — [Anthropic pricing docs](https://platform.claude.com/docs/en/about-claude/pricing)
- API behaviour that matters for the router: on Opus 5.5 and Sonnet 5.5, forced `tool_choice` returns a 400, so use `output_config.format` structured outputs. Opus 5.5 can't disable thinking and its effort defaults to `medium`. Haiku 5.5's effort also defaults to `medium` and it has no server-side refusal fallback. Source: the Claude API skill reference bundled with Claude Code (model table cached 2026-11-03), consistent with the [Anthropic pricing docs](https://platform.claude.com/docs/en/about-claude/pricing)
- Artificial Analysis Intelligence Index (cost per index task in parentheses): Fable 5.2 Max 65 ($26.54), Opus 5.5 Max 59 ($10.04), Sonnet 5.5 Max 55 ($10.02), Haiku 5.5 Max 36 ($0.41) — [Artificial Analysis leaderboard](https://artificialanalysis.ai/leaderboards/models)
- DeepInfra resells Claude: claude-opus-5-5 at $4/$20 (cache $0.20) and sonnet-5-5 at $2/$10, the same as first-party. It lists no Haiku 5.5 — [DeepInfra pricing](https://deepinfra.com/pricing)

### Inferences
- In the Artificial Analysis index, Sonnet 5.5 costs about as much per task as Opus 5.5 ($10.02 vs $10.04), even though its token prices are half. It uses more tokens per task at max effort. For the strategist and creator lab (low volume, high stakes), Opus 5.5 at a set effort level may cost no more per task than Sonnet.
- The showrunner and evaluator account for most of the spend. Two free wins before changing any model: (1) cache the stable persona bible, rubrics and system prompt (hits at 0.05x on Sonnet/Opus 5.5); (2) send non-urgent evaluator and idea-scorer work through Batch (50% off, stackable). Together these can cut those roles' Claude bill by roughly half or more without any quality risk.
- Haiku 5.5 is the natural candidate for the idea scorer and analyst summaries, and as a first-pass vision filter. Its index score (36) sits around DeepSeek V4.1 Flash and Tencent Hy4, so it should be shadow-tested before it takes judge roles.

### Gaps
- No public evidence was found on how Haiku 5.5 performs on vision judging or persona voice.
- The fetch didn't extract image-token pricing for Claude vision, which is billed as input tokens.

## Q2. Open-weight leaders on Together AI and DeepInfra: prices, structured outputs, vision, licenses

### Takeaway
The open-weight frontier in late 2026 is Kimi K3/K3.1 Pro, GLM-5.2/5.3, DeepSeek V4/V4.1 and Qwen3.7–4. Gemma 4, gpt-oss-120b and Nemotron 3.5 cover the cheap tier. Together lists structured outputs as supported for every serverless chat model except Tev1 4B Experimental. DeepInfra supports strict `json_schema`. On price per token, DeepInfra is usually 30–60% cheaper than Together for the same model. On the Artificial Analysis index the best open model (Kimi K3.1 Pro, 46) still sits well below Sonnet 5.5 (55) and Opus 5.5 (59), but the cheap open models (DeepSeek V4.1 Flash, MiniMax M3.1 Flash, Qwen3.8 Flash) are near Haiku 5.5's level.

### Cited Findings
**Together AI serverless prices (input / cached / output)** — [Together pricing](https://www.together.ai/pricing), fetched 2026-11-21, no date on the page:
- Kimi K3.1 Pro $2.00 / $0.20 / $10.00; Kimi K3 $3.00 / $0.30 / $15.00
- GLM-5.3 and GLM-5.2: $1.40 / $0.26 / $4.40 each; GLM-5.3-Flash $0.15 / $0.03 / $0.50
- DeepSeek V4 Pro 0813 $1.32 / $0.13 / $3.96; DeepSeek V4.1 Flash (and the -1017 snapshot) $0.30 / $0.006 / $1.20
- Qwen4 Max $1.50 / $0.075 / $4.50 (promotional $1.05 / $3.15 "valid until 11/23/2026"); Qwen3.7-Max $2.50 / $0.50 / $7.50; Qwen3.7-Plus $0.32 / $1.28; Qwen3.6-Plus $0.50 / $3.00; Qwen3.8 Flash $0.15 / $0.47; Qwen3.5 9B $0.17 / $0.25
- MiniMax M3.1 Flash $0.15 / $0.015 / $0.60; gpt-oss-120B $0.15 / $0.60; NVIDIA Nemotron 3.5 Super VL $0.15 / $0.02 / $0.50; Tencent Hy4 $0.83 / $0.042 / $2.50; Muse Spark Open 816B A42B (Meta) $1.25 / $0.15 / $4.25
- The page has a Batch API heading but shows no batch values for chat models, so Together's batch discount could not be verified. Llama, Mistral and Gemma are not in Together's serverless chat table.

**Together capabilities** — [Together serverless models](https://docs.together.ai/docs/serverless/models):
- Every serverless chat model is marked Yes for function calling and structured outputs, except Tev1 4B Experimental (No/No).
- Models that take image input: Kimi K3 (`moonshotai/Kimi-K3`), Kimi K3.1 Pro, NVIDIA Nemotron 3.5 Super VL, Qwen3.5 9B, Qwen4 Max, Tencent Hy4, Inkling 2 Small (Thinking Machines).
- Together's JSON docs don't describe strict-mode guarantees. They warn about truncated JSON at `max_tokens` and recommend validating output in code — [Together JSON mode docs](https://docs.together.ai/docs/json-mode)

**DeepInfra prices (input / cached / output; discounted price where shown)** — [DeepInfra pricing](https://deepinfra.com/pricing), fetched 2026-11-21:
- DeepSeek V4.1 Flash-1017 $0.098 / $0.0029 / $0.294 (51% off list $0.20/$0.60); V4-Flash-0731 $0.06 / $0.18; V4-Pro(-0813) $1.30 / $0.10 / $2.60; V3.2 $0.26 / $0.38
- DeepSeek-V4-Flash-Vision-Exp (VLM) $0.2156 / $0.6468 (discounted)
- Kimi K3.1 Pro $1.90 / $9.50; Kimi K3 $2.85 / $14.25; Kimi K2.6 $0.75 / $3.50
- GLM-5.3 $0.5625 / $2.50 (38% off); GLM-5.3-Flash $0.1005 / $0.335; GLM-5.2 $0.5625 / $1.80; GLM-4.7 $0.40 / $1.75; GLM-4.6 $0.50 / $2.00
- Qwen4-Max $1.413 / $4.24; Qwen3.8-Flash $0.113 / $0.382; Qwen3.8-27B $0.15 / $1.875; Qwen3.5-397B-A17B $0.45 / $3.00; Qwen3-235B-A22B-Instruct-2507 $0.09 / $0.55
- VLMs: Qwen3-VL-235B-A22B-Instruct $0.20 / $0.88; Qwen3-VL-30B-A3B-Instruct $0.15 / $0.60; Nemotron-3.5-Super-VL $0.15 / $0.50
- Gemma: gemma-4-31B-it $0.20 / $0.40 (turbo variant $0.09 / $0.34); gemma-4-26B-A4B-it $0.07 / $0.34; gemma-3-27b-it $0.08 / $0.16
- Llama-4-Scout $0.10 / $0.30; Llama-3.3-70B-Turbo $0.10 / $0.32; Mistral-Small-3.2-24B $0.075 / $0.20; Nemotron-3-Super-120B $0.075 / $0.40
- Not on the page: gpt-oss and MiniMax.
- DeepInfra also resells Gemini (3.7-flash $0.75/$3.75, 3.1-flash-lite $0.25/$1.50).

**DeepInfra structured outputs** — [DeepInfra structured outputs docs](https://docs.deepinfra.com/chat/structured-outputs):
- Supports both `json_object` and `json_schema`. `json_schema` "Enforces a strict output schema", and `"strict": true` is recommended for production.
- "Many" models are supported; no list is given.
- Caveats: forced structure can make a model fabricate values; output can truncate at `max_tokens`; temperature below 0.7 is recommended.
- The page doesn't say whether `json_schema` works with vision models.

**Intelligence (Artificial Analysis index, with cost per task)** — [Artificial Analysis](https://artificialanalysis.ai/leaderboards/models), no snapshot date:
- Kimi K3.1 Pro 46 ($3.68); Qwen4 Max 43 ($3.27); Tencent Hy4 35; DeepSeek V4.1 Flash 1017 33 ($0.60); GLM-5.3 Max 32 ($5.01); MiMo-V2.6-Pro 30 ($0.26)
- MiniMax-M3.1-Flash 29 ($0.49); Qwen3.8-Flash-Next 29; DeepSeek V4 Pro 0813 23; Nemotron 3.5 Super VL 14; gpt-oss-120b High 4; Gemma 4 31B Reasoning 4
- For comparison: Sonnet 5.5 55, Haiku 5.5 36.
- The page names only Kimi K3.1 Pro, Tencent Hy4 and DeepSeek V4.1 Flash as open-weight leaders. Other open-weight labels are the researcher's.

**Licenses:**
- DeepSeek V4 (Pro and Flash): plain MIT. GLM-5.3-Flash: MIT — [The New Stack](https://thenewstack.io/zai-glm-weights-license/); [Forkast](https://forkast.news/chinese-open-weight-frontier-compresses-five-labs-thirty-days-two-licensing-models/)
- GLM-5.3 flagship: a bespoke licence, not MIT. Hosting companies above $10B revenue need a security review — [The New Stack](https://thenewstack.io/zai-glm-weights-license/)
- Kimi K3: modified-MIT "Kimi K3 License", weights released 2026-07-27. Products with more than 100M users or $20M monthly revenue must display "Kimi K3", and sources disagree on the revenue-gate wording — [Forkast](https://forkast.news/chinese-open-weight-frontier-compresses-five-labs-thirty-days-two-licensing-models/); [Scientific American](https://scientificamerican.com/article/china-kimi-k3-and-the-rise-of-open-weight-ai-models)
- These are secondary sources. Check the LICENSE file in each Hugging Face repo.

### Inferences
- **Using these models through a hosted API** (Together or DeepInfra) means the hoster's terms govern use. The weight-license gates (Kimi's display clause, GLM's security review) only bite at scales far beyond Manyworlds. All the listed models look usable commercially here, but each LICENSE should still be checked.
- **Best open candidates by role**, on price per token and quality signals:
  - Showrunner and creator writing: Kimi K3.1 Pro or GLM-5.3. GLM-5.3 on DeepInfra at $0.56/$2.50 is about 4x cheaper on output than Sonnet 5.5, and it ranks highly on creative writing (Q4).
  - Cheap structured JSON (idea scorer, analyst): DeepSeek V4.1 Flash on DeepInfra ($0.10/$0.29), MiniMax M3.1 Flash or Qwen3.8 Flash on Together ($0.15/$0.47–0.60).
  - Vision: Qwen3-VL-235B on DeepInfra ($0.20/$0.88), Kimi K3 or Qwen4 Max on Together, Nemotron 3.5 Super VL ($0.15/$0.50).
- **Haiku 5.5 ($0.10/$0.50) undercuts or matches most cheap open models on price** and has a higher index (36) than most of them. So open models win on cost mainly at the high-quality tier, as GLM-5.3 or Kimi in place of Sonnet/Opus, not at the bottom tier.
- **JSON reliability**: DeepInfra advertises strict schema enforcement; Together advertises support without a stated strict guarantee. Keep validation and the Claude fallback in place either way.

### Gaps
- No independent measurement of JSON-schema adherence rates per model and host was found.
- Together's batch discount for chat models was not shown.
- Not found on DeepInfra's pricing page: gpt-oss and MiniMax.
- Not confirmed: whether vision plus `json_schema` works together on DeepInfra VLMs, and Qwen3.x/4 and Gemma 4 licenses (historically Qwen was Apache-2.0 and Gemma had its own Gemma terms).
- Not found: Llama 4.x successors (Meta now appears as "Muse"), and Mistral Large and Medium on these hosts.

## Q3. Vision judges: open VLMs for image quality and identity; VLM-as-judge agreement with humans; face-embedding models and licenses

### Takeaway
VLM judges agree with humans reasonably well on overall preference, but they are unreliable on fine-grained aesthetics, anatomy and identity. Holistic pointwise scores suffer from "self-consistency collapse". Pairwise comparisons and decomposed, feature-level criteria work better. Identity should be measured with a face-embedding similarity score. InsightFace's pretrained models are non-commercial without a paid licence; AuraFace (Apache 2.0) and AdaFace (MIT code, training-data caveat) are the permissive alternatives.

### Cited Findings
- AUTORATER (WACV 2026 workshop) feeds specialist metrics (text–image alignment, quality, detail preservation) into a VLM judge and reports 2–8% better human alignment than the specialists alone — [WACVW 2026 paper](https://openaccess.thecvf.com/content/WACV2026W/WVAQ/papers/V._Vision_Language_Models_Learn_to_Assess_Images_with_Specialists_WACVW_2026_paper.pdf)
- AAAI 2026 (Parthasarathy et al.): multimodal LLMs struggle to generalise on anatomical correctness, style and aesthetics, which humans find easy, even when they match humans on overall rating — [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39666); [arXiv 2509.12750](https://arxiv.org/html/2509.12750v1)
- A February 2026 paper on judging visual generation finds that absolute pointwise scoring suffers "self-consistency collapse" and proposes pairwise, Elo-style comparison instead — [arXiv 2602.06013](https://arxiv.org/pdf/2602.06013v1)
- A March 2026 study finds the most self-consistent VLMs are not necessarily the most human-aligned, and VLMs are more reliable when stimulus differences are clear — [arXiv 2603.24578](https://arxiv.org/abs/2603.24578)
- CHARIS ("Beyond the Pixels") breaks identity assessment into feature-level checks and reports strong human alignment. It finds that a broad prompt such as "does this preserve identity" gets superficial answers — [arXiv 2511.08087](https://arxiv.org/html/2511.08087v1)
- FG 2026: VLM face-quality assessment depends on architecture more than parameter count, and larger models can hallucinate more on clean images — [arXiv 2605.17489](https://arxiv.org/html/2605.17489)
- InsightFace: the code is MIT, but the pretrained models (e.g. buffalo_l, antelopev2) are "for non-commercial research purposes only". Commercial use needs a licence from insightface.ai — [InsightFace README](https://raw.githubusercontent.com/deepinsight/insightface/HEAD/README.md); [insightface.ai](https://insightface.ai)
- AuraFace: Apache 2.0, "trained on commercially and publicly available data sources", built on ArcFace principles — [AuraFace blog](https://huggingface.co/blog/isidentical/auraface); [AuraFace model card](https://huggingface.co/fal/AuraFace-v1/blob/main/README.md)
- AdaFace: the repository is MIT-tagged, but the CVLFace model cards say to follow the training dataset's licence (e.g. WebFace12M, MS1MV2), which may restrict use — [AdaFace GitHub](https://github.com/mk-minchul/AdaFace); [CVLFace model card](https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m/blob/ef3f286d354933596c8b7e8937700ed4f11c6afa/README.md?code=true)
- One secondary guide claims no open face recognizer has both a commercial licence and ArcFace-class accuracy. This is opinion, not a benchmark — [Hailo community](https://community.hailo.ai/t/face-recognition-without-using-insightface-models/2504); [fullstackcv playbook](https://fullstackcv.github.io/playbook/face/)
- Open VLMs available as of the 2026-11-21 fetch: Qwen3-VL-235B-A22B and Qwen3-VL-30B-A3B, DeepSeek-V4-Flash-Vision-Exp and Nemotron-3.5-Super-VL on DeepInfra; Kimi K3/K3.1 Pro, Qwen4 Max, Tencent Hy4 and Inkling 2 Small on Together — [DeepInfra pricing](https://deepinfra.com/pricing); [Together models](https://docs.together.ai/docs/serverless/models)

### Inferences
- **Review gates (identity)**: replace or supplement the VLM identity check with AuraFace cosine similarity against the creator's anchor and reference embeddings, with the threshold set in policy. It is cheap (runs on CPU or the GPU worker), deterministic and Apache-licensed. If InsightFace models are in use today, that is a licence risk for a commercial product.
- **Evaluator (rubric scoring)**: switch to pairwise or decomposed per-criterion questions rather than one holistic 1–10 score. This helps any model. Candidate: Qwen3-VL-235B on DeepInfra ($0.20/$0.88), about 10x cheaper than Sonnet 5.5, shadow-tested against Claude with rubric-drift and calibration tooling. Keep Claude for final gates where human calibration shows it matters.
- **Anatomy and hands**: these are a known VLM weakness. Keep a stronger model (Sonnet/Opus 5.5) or a specialist detector for those rubric items.

### Gaps
- No 2026 benchmark directly compares Qwen3-VL, GLM-4.5V, InternVL or Gemma 4 as aesthetic or identity judges against Claude or Gemini on human agreement.
- No price for GLM-4.5V/5V or InternVL on these hosts was found, nor any listed GLM vision model.
- No published accuracy (e.g. IJB-C) comparison of AuraFace vs ArcFace was retrieved.

## Q4. Creative-writing benchmarks (EQ-Bench Creative Writing v3, LMArena creative), late 2026

### Takeaway
Claude leads both major creative-writing boards. Opus 5 is first on EQ-Bench Creative Writing v3 (Elo 2116) and Fable 5 is first on the arena.ai human-vote creative board (Aug 27, 2026 snapshot). The best open-weight models are close behind on EQ-Bench (Kimi K3 #2, GLM-5.3 #3, about 45 Elo behind) but further back on human votes (GLM-5.3-max #13, Kimi K3-max #23). The two boards largely invert each other, so neither is decisive for persona voice.

### Cited Findings
- EQ-Bench Creative Writing v3: claude-opus-5 #1 (Elo 2116.1, rubric 85.35), kimi-k3 #2 (Elo 2070.8), GLM-5.3 #3 (Elo 2062.4), claude-fable-5 #6. No as-of date; the last dated changelog entry is 2026-03-01 — [Digital Applied, 2026-08-31](https://www.digitalapplied.com/blog/which-ai-model-writes-best-leaderboards-disagree)
- arena.ai Creative Writing (human blind votes, snapshot 2026-08-27, 1.21M votes, 393 models): claude-fable-5 #1 (1505±9), claude-opus-5 #10, glm-5.3-max #13 (1467±18), kimi-k3-max #23 (1458±11) — [Digital Applied](https://www.digitalapplied.com/blog/which-ai-model-writes-best-leaderboards-disagree)
- The live EQ-Bench table renders client-side and couldn't be read. The aggregators disagree (llm-stats shows Grok-4.1 Thinking on top on a different scale), so the leaderboard data is inconsistent — [eqbench.com](https://eqbench.com/creative_writing.html); [llm-stats](https://llm-stats.com/benchmarks/creative-writing-v3)
- EQ-Bench uses a Claude model as judge (the repo's run command uses claude-sonnet-4-6), which may bias it towards Claude-like prose — [EQ-Bench GitHub](https://github.com/EQ-bench/creative-writing-bench)

### Inferences
- Opus 5.5 and Sonnet 5.5 have no published scores in this data; the boards rank Opus 5 and Fable 5. The Claude family's lead on human votes supports keeping Claude as the primary for persona voice in captions and scripts.
- GLM-5.3 and Kimi K3 are the right open candidates to shadow-test for the showrunner. Score them with your own human or engagement signals, not EQ-Bench alone, since EQ-Bench is judged by Claude.

### Gaps
- No November 2026 snapshot of either board was found.
- Not found: Sonnet 5.5, Opus 5.5 and Haiku 5.5 creative-writing scores, and short-form social-caption benchmarks.

## Q5. Other cheap hosted options: Gemini Flash / Flash-Lite, GPT-5 mini/nano

### Takeaway
Cheap proprietary options with vision and JSON support exist at Haiku 5.5's price level or below: GPT-6 Luna ($0.10/$0.50), GPT-5-nano ($0.05/$0.40), GPT-5-mini ($0.25/$2.00) and Gemini 3.1 Flash-Lite ($0.25/$1.50). Gemini prices are rising: Gemini 3.8 Flash and 3.6 Flash double on 2027-01-01, and Gemini 4 Carbon and Argon double on 2026-12-01. Both providers' batch tiers take 50% off. The router supports only OpenAI-compatible hosts (Together, DeepInfra), so using these needs a new host entry. DeepInfra resells Gemini, which is one route in.

### Cited Findings
- OpenAI, Standard tier (input / cached / output) — [OpenAI API pricing](https://developers.openai.com/api/docs/pricing), fetched 2026-11-21:
  - gpt-6-luna $0.10 / $0.01 / $0.50
  - gpt-5-nano $0.05 / $0.005 / $0.40; gpt-5.4-nano $0.20 / $1.25
  - gpt-5-mini $0.25 / $0.025 / $2.00; gpt-5.4-mini $0.75 / $4.50
  - gpt-6.1-terra $1.00 / $0.05 / $5.00; gpt-6.2-sol $2.00 / $0.10 / $10.00; gpt-6.2-astra $8 / $40
  - Batch and Flex take 50% off. GPT-6.x models get an extra 30–40% off for background requests.
- Artificial Analysis index: GPT-6.2 Sol Max 57 ($1.01 per task), GPT-6.1 Terra Max 51 ($0.96), GPT-6 Luna Max 28 ($0.15), Gemini 4 Neon Xhigh 61, Gemini 3.5 Flash-Lite 12 — [Artificial Analysis](https://artificialanalysis.ai/leaderboards/models)
- Gemini, per 1M tokens — [Google AI Studio pricing](https://aistudio.google.com/docs/pricing), fetched 2026-11-21:
  - 3.5 Flash-Lite $0.30 / $2.50 (batch $0.15 / $1.25)
  - 3.1 Flash-Lite $0.25 / $1.50 (batch $0.125 / $0.75)
  - 2.5 Flash-Lite $0.10 / $0.40
  - 3.8 Flash and 3.6 Flash $0.75 / $3.75, rising to $1.50 / $7.50 from 2027-01-01
  - Gemini 4 Hydrogen $1.50 / $7.50; Gemini 4 Neon $3 / $15
  - Gemini 4 Carbon and Argon $2 / $10, rising to $4 / $20 from 2026-12-01
  - 2.5 Flash and 2.5 Pro are deprecated; they route to newer models on 2026-12-01
- DeepInfra resells Gemini 3.1-flash-lite at $0.25 / $1.50 and gemini-3.7-flash at $0.75 / $3.75 — [DeepInfra pricing](https://deepinfra.com/pricing)

### Inferences
- **GPT-6.1 Terra** ($1/$5, index 51, $0.96 per index task vs Sonnet 5.5's $10.02) is the most cost-efficient high-quality proprietary option found. It is a strong candidate for the strategist, analyst and evaluator if an OpenAI host is added. Its creative-writing standing wasn't found.
- **For cheap vision and structured tasks**, GPT-6 Luna and Haiku 5.5 have identical list prices. Haiku keeps everything on one SDK with existing fallbacks.
- **Avoid building on Gemini 3.6 and 3.8 Flash and Gemini 4 Carbon and Argon** at their current prices; scheduled increases double them.

### Gaps
- No confirmation that the OpenAI pricing page's gpt-6-luna is vision-capable.
- No creative-writing or vision-judge comparisons were found for GPT-6.x or Gemini 4.x.
- gpt-oss pricing on OpenAI wasn't found; it is $0.15/$0.60 on Together.
