# Voice: TTS, voice design and cloning landscape (November 2026)

Research date: 2026-11-21. Conversion used throughout: about 900 characters per spoken minute (about 150 wpm × 6 chars incl. spaces). This is my assumption, not a sourced figure. So $X per 1M chars ≈ $X × 0.0009 per minute (e.g. $15/1M ≈ $0.0135/min; $100/1M ≈ $0.09/min).

Overall picture: the open-weight field moved a lot since Kokoro (Jan 2025). The engine's needs are (a) a unique persistent voice per creator, created from text, (b) emotion and direction control, (c) commercial license, (d) cheap self-hosting. The **Apache-2.0 voice-design trio of Qwen3-TTS VoiceDesign, VoxCPM2 and MOSS-VoiceGenerator** covers this, plus **Ming-3.0-tts (MIT)** if its license checks out. The usual workflow is to design a voice once, save a reference clip, then clone that synthetic clip zero-shot for every script. Several top-ranked open-weight models are NOT commercially usable: Fish S2 Pro, Higgs Audio v3, Breeze TTS 2, F5-TTS, Voxtral TTS, XTTS v2. Fish Drama 3 is commercial only under a revenue cap.

## Open-weight leaders and their commercial license status

### Takeaway
Best commercially usable open options in late 2026 are Qwen3-TTS (Apache-2.0; VoiceDesign + 3 s clone + instruction emotion control), VoxCPM2 (Apache-2.0; design + clone + style guidance, 30 languages, ~8 GB VRAM), the MOSS-TTS family (Apache-2.0; includes VoiceGenerator and the long-dialogue TTSD), and Ming-3.0-tts (reportedly MIT; voice design plus a "director mode"). Kokoro is still the cheapest permissive baseline, but it cannot design or clone voices. Fish Audio Drama 3 is the top-ranked open-weight model by Elo, but it carries a custom license (free commercial use only under $500k revenue, plus attribution).

### Cited Findings
**Leaderboard context (Artificial Analysis Speech Arena, open-weights view, fetched 2026-11-21; no data date shown on page):** Fish Audio Drama 3 is at 1277 Elo (Oct 2026), Drama 3 Preview 1237, Breeze TTS 2 1228, Ming-3.0-tts 1179 (Oct 2026), Gradium Phonon 31M 1108, Fish S2 Pro 1101, Step Audio EditX 1100, Voxtral TTS 1094, NVIDIA Magpie-Multilingual 357M 1064, Kokoro 82M 1060, Maya1 1047, OpenAudio S1 Mini 1042, Higgs Audio V3 1035, Chatterbox 1019, Zonos v0.1 1000, VibeVoice 1.5B 954, XTTS v2 914, StyleTTS 2 887 — [AA open-weights leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights). Qwen3-TTS, VoxCPM2, IndexTTS-2, CosyVoice, MOSS and Orpheus are not in the AA open-weights list. AA lists "Qwen3 TTS" and "Qwen3 TTS Flash" only as Alibaba's hosted API (928 and 940 Elo) — [AA leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard).

Per model:
- **Kokoro-82M**: Apache-2.0 weights, no voice cloning; hosted resale under $1/1M chars — [Cartesia license roundup](https://www.cartesia.ai/learn/open-source-tts). AA price $0.65/1M, Elo 1060 — [AA](https://artificialanalysis.ai/text-to-speech/leaderboard).
- **Qwen3-TTS** (Alibaba, Jan 2026): tokenizers and models released under Apache 2.0 — [Qwen3-TTS tech report](https://arxiv.org/html/2601.15621). The release covers 1.7B-VoiceDesign, 1.7B-CustomVoice, 1.7B-Base, 0.6B-CustomVoice, 0.6B-Base and Tokenizer-12Hz — [Gigazine](https://gigazine.net/gsc_news/en/20260123-qwen3-tts-family-opensource). The model card gives Apache-2.0, 10 languages, natural-language control of "timbre, emotion, and prosody", streaming latency "as low as 97ms", and Seed-TTS zero-shot WER of 1.24 (en) — [HF model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign). The Base model clones from about 3 seconds of audio — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts).
- **VoxCPM2** (OpenBMB): Apache-2.0, "free for commercial use". 2B params. Voice design from a text description (gender, age, tone, emotion, pace) with no reference audio. Clones from a short clip, and "ultimate cloning" also takes the transcript. Style guidance steers emotion while keeping timbre. 30 languages. RTF about 0.30 on an RTX 4090 (about 0.13 with Nano-vLLM). About 8 GB VRAM. Max 8192 tokens — [HF model card](https://huggingface.co/openbmb/VoxCPM2).
- **MOSS-TTS family** (OpenMOSS): "Models in MOSS-TTS Family are licensed under the Apache License 2.0". The family includes MOSS-TTS-v1.5 8B (2026-05-26), MOSS-TTSD-V1.0 8B (multi-speaker, "ultra-long dialogues"), MOSS-VoiceGenerator 1.7B (voice design from text prompts, no reference speech), MOSS-TTS-Realtime 1.7B, MOSS-TTS-Nano ~0.1B (CPU) and MOSS-TTS-Local-Transformer-v1.5 4B (48 kHz). The page states no maximum duration in minutes — [GitHub OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS).
- **Ming-3.0-tts** (Ant Group / inclusionAI, Oct 2026): reported MIT-licensed. Ming-3.0-tts-0.5B supports voice design via text description or reference audio, with a "director mode" for emotion, stress and pacing. Reported weaker than Qwen3-TTS at open-ended voice design. These claims come from secondary sources — [kie.ai](https://kie.ai/blog/what-is-ming-3-0), [AlphaSignal](https://alphasignal.ai/news/ant-group-s-ming-3-0-tops-open-source-voice-charts-with-five-speech-models). Elo 1179 — [AA](https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights).
- **Fish Audio Drama 3** (Oct 13, 2026): 35B MoE (3B active), weights at fishaudio/drama-3 — [MarkTechPost via search](https://www.marktechpost.com/2026/10/14/fish-audio-drama-3-open-weight-tts-model/).
  - License: Fish Audio Community License v1.0. Free for research and non-commercial use, and free commercial use under US$500k annual revenue. Above that, or to offer it as a hosted TTS or cloning service, you need a separate license. "Built with Fish Audio" attribution is required, and cloning a real person without consent is forbidden — [magica.com summary](https://magica.com/news/fish-audio-drama-3-open-weights-commercial-limits), [oflight guide](https://www.oflight.co.jp/en/columns/fish-audio-drama-3-requirements-license-guide-2026).
  - Conflict: the HF license tag reportedly reads fish-audio-research-license — [search summary of explainx/others](https://explainx.ai/blog/fish-audio-drama-3-open-weights-35b-a3b-word-level-editing-how-to-run-2026).
  - Control: inline point tags ([sigh], [laughs]), paired tags (`<whisper>`), free-text direction mid-line, and 22 preset tags. Cloning from a 3.9 s or 15 s reference. Long text is auto-chunked. No voice design from text is documented — [kie.ai deep dive](https://kie.ai/blog/fish-audio-drama-3-deep-dive-35b-moe-tts).
- **Fish S2 Pro / OpenAudio S1**: Fish Audio Research License, not commercial — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts). The Fish README reportedly says the S2-Pro weights stay under the Research License — [search result, speech.fish.audio](https://speech.fish.audio).
- **Higgs Audio v3 TTS 4B** (Boson AI, Jun 2026): "Boson Higgs TTS 3 Research and Non-Commercial License". Production, hosted, or embedded use needs a commercial license. A "Creator Use Grant" reportedly permits monetized creator content with attribution to "Boson AI's Higgs Audio" — [HF model card](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b). The Higgs v2 tokenizer license needs an expanded license above 100k annual active users — [search summary](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/discussions/8).
- **Chatterbox (Resemble)**: MIT code and weights, with watermarked output — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts). Resemble states that as of August 2026 Chatterbox and DramaBox are "retired commercially" and remain as an unmaintained archive, and that Resemble no longer sells TTS. The same site also promotes Chatterbox Nano/Flash, which is a conflict — [Resemble](https://www.resemble.ai/resources/why-a-former-voice-ai-company-went-all-in-on-deepfake-detection), [EveryDev](https://www.everydev.ai/tools/chatterbox-tts).
- **VibeVoice (Microsoft)**: MIT weights but a research-only model card, and Microsoft does not recommend commercial use. The TTS code was removed from the repo in September 2025 — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts). The tech report claims up to 90 min with 4 speakers in a 64K context — [arXiv 2508.19205](https://arxiv.org/html/2508.19205v1). The project page says the repo was disabled until out-of-scope use can be prevented — [VibeVoice page](https://microsoft.github.io/VibeVoice/).
- **F5-TTS**: MIT code, CC BY-NC 4.0 weights, so not commercial — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts).
- **Voxtral TTS (Mistral, open)**: CC BY-NC 4.0 weights, not commercial — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts). The hosted Voxtral 3 TTS is at 1211 Elo, $13/1M — [AA](https://artificialanalysis.ai/text-to-speech/leaderboard).
- **Breeze TTS 2**: research and non-commercial, despite a high Elo (1228) — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts), [AA](https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights).
- **XTTS v2**: Coqui Public Model License, not commercial, and Coqui has shut down — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts).
- **Orpheus 3B**: Apache-2.0, but on a Llama backbone so Meta's Llama license also applies — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts).
- **Dia 1.6B**: Apache-2.0, uses speaker tags. Without a reference clip the voice changes on every run — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts).
- **Zonos v0.1**: Apache-2.0 — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts). Elo 1000 — [AA](https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights).
- **CosyVoice 2**: Apache-2.0 weights — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts). Alibaba's newer hosted Fun-Realtime-TTS took #1 on AA in June 2026 — [AA article](https://artificialanalysis.ai/articles/fun-realtime-tts-new-text-to-speech-model-topping-artificial-analysis-leaderboard).
- **Maya1**: Apache-2.0, "expressive voice design" — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts). Elo 1047.
- **NVIDIA Magpie TTS Multilingual 357M**: NVIDIA Open Model License (commercial OK), five English speaker voices — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts).
- **IndexTTS-2 / 2.5 (bilibili)**: the paper covers emotionally expressive, duration-controlled AR zero-shot TTS — [arXiv 2506.21619](https://arxiv.org/html/2506.21619v1). IndexTTS 2.5 has a tech report — [arXiv 2601.03888](https://arxiv.org/html/2601.03888). The official HF page showed no license text. A third-party re-upload says Apache-2.0, which is unreliable — [HF indexteam](https://huggingface.co/indexteam/indextts-2).

### Inferences
- Best fit for "unique persistent voice, commercial, self-hosted": **Qwen3-TTS** (VoiceDesign → save clip → Base clone, or CustomVoice + instruct) and **VoxCPM2** (design and clone in one 2B model at ~8 GB VRAM, which fits the existing RunPod 3090 pool). Both are Apache-2.0. MOSS-VoiceGenerator + MOSS-TTS is a third Apache option. Ming-3.0 is a fourth, but only after the MIT license is confirmed on the HF repo.
- Drama 3 is likely the most expressive open model (tags plus free-text direction) and is commercially free for us while we are under $500k revenue. The license is custom, needs attribution, and its revenue cliff matters for a scaling business, so treat it as a candidate with a legal-review flag. The "Built with Fish Audio" display requirement on Instagram posts needs checking.
- Higgs v3's "Creator Use Grant" might actually cover monetized social content with attribution. That is unusual and worth reading in full, but the base license is non-commercial, so the default is not OK.
- Chatterbox is now an unmaintained archive, so it is a poor choice for new adoption even though MIT.
- None of the open models has published, comparable 60–90 s benchmarks on our use case. A bench (`mw bench`) of 3–4 candidates through the video gate is the real decision tool.

### Gaps
- Not verified from LICENSE files: IndexTTS-2 (I recall from prior knowledge a custom "bilibili Model Use License" with commercial restrictions; UNVERIFIED), Spark-TTS (I recall CC BY-NC-SA weights, which would mean not commercial; UNVERIFIED), Sesame CSM-1B (Apache-2.0 per my prior knowledge, Llama-based; UNVERIFIED), Kani-TTS, Dia2 and CosyVoice 3 (no sources found in this pass), and MOSS-TTSD specifically (covered by the family-level Apache statement).
- Ming-3.0 license is only from secondary sources. The Drama 3 license name conflicts between README and HF tag.
- No VRAM or throughput figures found for Drama 3, Ming-3.0 or Qwen3-TTS 1.7B on a 3090.

## Voice design from a text prompt (unique persistent voices)

### Takeaway
Text-prompt voice design is now mainstream in both camps. Hosted options: ElevenLabs Voice Design v3, Hume Octave 2, Inworld Realtime TTS-2 (Advanced Voice Design), MiniMax Voice Design ($3/voice), and Qwen3-TTS-VoiceDesign on DeepInfra ($20/1M). Open options: Qwen3-TTS-VoiceDesign, VoxCPM2, MOSS-VoiceGenerator, Ming-3.0-tts, Maya1. Persistence works by saving the designed voice (hosted) or saving a reference clip and cloning it (open). Cloning a synthetic, self-generated clip avoids the real-person-consent problem.

### Cited Findings
- ElevenLabs Voice Design creates "a unique voice from a text prompt". The API has a preview endpoint and a save-to-library endpoint, and the Spanish page labels v3 as alpha — [ElevenLabs Voice Design v3 blog](https://elevenlabs.io/blog/voice-design-v3), [help center](https://elevenlabs.io/docs/help-center/product/voice-customization/voice-design/what-is-voice-design). All paid plans include a commercial license and you "retain rights to the audio you generate", while the free tier is non-commercial. This is a general audio statement, not one specific to Voice Design — [search summary of ElevenLabs voice generator page](https://start.elevenlabs.io/voicegen/v5). Also on fal.ai — [fal](https://fal.ai/models/fal-ai/elevenlabs/text-to-voice/design/eleven-v3/api).
- Hume Octave 2 does synthesis, voice design, cloning and editing in one model across 11 languages. I found no separate voice-design fee — [search summary; aiwiki](https://www.aiwiki.ai/wiki/hume_octave_2), [Hume blog](https://hume.ai/blog/octave-2-launch).
- Inworld Realtime TTS-2 supports text-based voice design via an API endpoint ("Advanced Voice Design" builds a saved voice from prose) — [Inworld docs](https://docs.inworld.ai/tts), [Inworld TTS-2 blog](https://inworld.ai/blog/realtime-tts-2).
- MiniMax Voice Design costs $3 per voice, charged at first synthesis, with preview audio at $60/1M chars (one blog says $30) — [MiniMax pay-as-you-go](https://platform.minimax.io/docs/guides/pricing-paygo).
- Qwen3-TTS-VoiceDesign on DeepInfra: free-text description (e.g. "A deep male voice with a calm, authoritative tone"), 10 languages, streaming, $20/1M chars, Apache-2.0, "zero retention" — [DeepInfra](https://deepinfra.com/Qwen/Qwen3-TTS-VoiceDesign).
- Cartesia: about 1 credit per char. Pro Voice Clone costs about 1.5 credits per char per the docs (conflicts with an FAQ saying 1). No voice-design pricing found — [Cartesia docs pricing](https://docs.cartesia.ai/pricing.md), [Cartesia pricing](https://cartesia.ai/pricing).
- Open models with voice design: VoxCPM2 — [HF](https://huggingface.co/openbmb/VoxCPM2); MOSS-VoiceGenerator — [GitHub](https://github.com/OpenMOSS/MOSS-TTS); Ming-3.0-tts — [kie.ai](https://kie.ai/blog/what-is-ming-3-0); Maya1 — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts); Qwen3-TTS VoiceDesign — [HF](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign).
- Dia without a reference clip changes voice every run, so persistence requires a reference clip — [Cartesia roundup](https://www.cartesia.ai/learn/open-source-tts).

### Inferences
- Recommended pattern for Manyworlds:
  1. Design the voice from the creator bible's description (Qwen3 VoiceDesign or VoxCPM2).
  2. The judge picks the best of N candidates.
  3. Store a 10–20 s "voice anchor" clip plus transcript in the reference pack, alongside the image anchors.
  4. Voice every script by zero-shot cloning that anchor (Qwen3 Base, VoxCPM2 ultimate clone, Drama 3, IndexTTS-2).
  This mirrors the image-anchor/LoRA flow and keeps voices swappable by config.
- Uniqueness: designed voices from the same model may cluster. Add a speaker-embedding distance check between creators' anchors (e.g. an ECAPA/WavLM speaker-verification model) as a review-gate step. I found no source on this; it is a design suggestion.

### Gaps
- No published voice-design quality benchmarks comparing hosted and open designers.
- ElevenLabs and Hume voice-design-specific pricing and terms not found on the pages fetched.

## Hosted APIs: price and terms

### Takeaway
Arena quality leaders are ElevenLabs v4 / v4 Turbo (1340 Elo), StepAudio 3, Qwen-Audio-3.1, Cartesia Sonic 3.6, Speechify Simba 3.5, Gemini 3.8 Flash TTS and Fish Drama 3. Cheapest per minute are the token-billed LLM TTS models: Gemini 3.8 Flash TTS (≈$0.0135/min preview pricing, doubling Jan 2027) and gpt-4o-mini-tts (≈$0.015/min). Next come Speechify Simba ($6.58/1M ≈ $0.006/min), Fish ($15/1M) and Inworld TTS-2 Flash. ElevenLabs v4 is about $0.036–0.072/min, and MiniMax HD and Hume are at the top of the range.

### Cited Findings
AA Speech Arena main leaderboard, fetched 2026-11-21. Elo and price are per 1M chars as listed by AA — [AA leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard):

| Model | Elo | $/1M chars (AA) | ≈$/min (my calc) |
|---|---|---|---|
| ElevenLabs Eleven v4 Turbo | 1340 | 40 | 0.036 |
| ElevenLabs Eleven v4 | 1340 | 80 | 0.072 |
| StepFun StepAudio 3 TTS (Nov 2026) | 1307 | 14 | 0.013 |
| Alibaba Qwen-Audio-3.1-TTS-Plus | 1298 | 19.31 | 0.017 |
| Cartesia Sonic 3.6 | 1294 | 49 | 0.044 |
| Speechify Simba 3.5 | 1293 | 6.58 | 0.006 |
| Google Gemini 3.8 Flash TTS | 1282 | 16.49 | 0.015 |
| Fish Audio Drama 3 (open weights) | 1277 | 15 | 0.0135 |
| Inworld Realtime TTS-2 | 1269 | 20.83 | 0.019 |
| Gemini 3.8 Flash-Lite TTS | 1261 | 11.03 | 0.010 |
| ElevenLabs v3 Conversational | 1215 | 50 | 0.045 |
| Inworld Realtime TTS-2 Flash | 1213 | 10.42 | 0.009 |
| Mistral Voxtral 3 TTS | 1211 | 13 | 0.012 |
| ElevenLabs Eleven v3 | 1191 | 100 | 0.09 |
| MiniMax Speech 2.8 HD | 1173 | 100 | 0.09 |
| Azure HD 2.5 | 1149 | 22 | 0.020 |
| OpenAI TTS-1 HD | 1109 | 30 | 0.027 |
| Chatterbox HD (Resemble) | 1109 | 40 | — |
| Hume Octave 2 | 1058 | 87.50 | 0.079 |
| Kokoro 82M | 1060 | 0.65 | 0.0006 |

Price sources by provider:
- **ElevenLabs** (official page, undated): v4 $0.08/1K, v4 Turbo $0.04/1K, v3 $0.08/1K, v3 Conversational $0.04/1K, Flash/Turbo $0.04/1K. The page mislabels these "per minute" in plan tables — [ElevenLabs API pricing](https://elevenlabs.io/pricing/api). A June 2026 guide gave Flash/Turbo at $0.05/1K, so prices conflict over time — [Puter guide](https://developer.puter.com/tutorials/elevenlabs-api-pricing/). An AA-listed v3 price of $100/1M also conflicts with the official $80.
- **Google Gemini 3.8 Flash TTS (Preview)**: $0.50/1M text input tokens, $9.00/1M audio output tokens through 2026-12-31, then $1.00 and $18.00 from 2027-01-01. Audio is 25 tokens/s, so about $0.81/hour — [eesel summary of Google pricing](https://www.eesel.ai/blog/gemini-3-8-flash-tts), [Google Cloud TTS pricing](https://cloud.google.com/text-to-speech/pricing).
- **OpenAI gpt-4o-mini-tts**: $0.60/1M text input, $12/1M audio output tokens. Community estimate about $0.015/min — [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-4o-mini-tts), [OpenAI forum](https://community.openai.com/t/new-tts-api-pricing-and-gotchas/1150616). It is not in AA's top list (TTS-1 HD sits at 1109).
- **Inworld**: Realtime TTS-2 is $25/1M on demand, falling to $15 at $300/mo and $12.50 at $1,500/mo, with Enterprise reported at $5–10. TTS-2 Flash is $15/1M on demand. Older 1.5 Mini/Max were $5/$10. Inworld's own pages are inconsistent — [Inworld pricing](https://inworld.ai/pricing), [Inworld billing docs](https://docs.inworld.ai/tts/resources/billing). AA lists $20.83 and $10.42.
- **Cartesia**: about 1 credit/char. Plans are Pro $5/mo for 100K credits, Startup $49/mo for 1.25M, and Scale $299/mo for 8M (≈$37/1M at Scale) — [Cartesia pricing](https://cartesia.ai/pricing), [eesel](https://www.eesel.ai/blog/cartesia-sonic-3-pricing).
- **Hume Octave 2**: aggregator figures conflict. $0.063–0.078/1K is commonly cited, plan rates run $0.075 (Creator) to $0.05 (Scale) per 1K, and one outlier cites $7.60/1M — [AA Octave 2](https://artificialanalysis.ai/text-to-speech/models/ocateve-2), [activepieces](https://www.activepieces.com/blog/hume-ai-pricing-free-tier-plans-and-costs-2026.md). Not verified on hume.ai.
- **MiniMax**: Speech 2.8 HD $100/1M, Turbo $60/1M, Voice Design $3/voice. Resellers list lower (e.g. $30/1M turbo) — [MiniMax PAYG](https://platform.minimax.io/docs/guides/pricing-paygo), [OpenRouter](https://openrouter.ai/minimax/speech-2.8-hd/api).
- **Fish Audio API**: $15/1M per AA. One outlet reports $15 per 1M UTF-8 bytes for drama-3-preview, which another could not confirm — [AA](https://artificialanalysis.ai/text-to-speech/leaderboard), [kie.ai](https://kie.ai/blog/fish-audio-drama-3-deep-dive-35b-moe-tts).
- **Hosted open TTS**: DeepInfra Qwen3-TTS VoiceDesign $20/1M, with "flex" (cheaper, may time out) and "priority" tiers — [DeepInfra](https://deepinfra.com/Qwen/Qwen3-TTS-VoiceDesign). Kokoro hosted at $0.65/1M — [AA](https://artificialanalysis.ai/text-to-speech/leaderboard).
- **Azure**: HD 2.5 $22/1M, Neural $15/1M (AA-listed prices) — [AA](https://artificialanalysis.ai/text-to-speech/leaderboard).

### Inferences
- At our volumes (e.g. 1,000 talking videos × 60 s ≈ 1,000 min ≈ 0.9M chars/month), even ElevenLabs v4 costs about $72/month. TTS cost is small next to the GPU lip-sync cost (InfiniteTalk 14B). The decision should hinge on quality, voice uniqueness and terms, not TTS price, unless volume grows by 100×.
- Self-hosting a 2B model on an RTX 3090 at RTF about 0.3 (VoxCPM2 4090 figure, with the 3090 probably slower) means about 18–25 s of GPU per minute of audio. At about $0.2–0.3/h that is around $0.002/min plus boot and download overhead. It only beats Gemini/OpenAI (about $0.014/min) if co-located on the same machine as the lip-sync job. My estimate is unverified on a 3090.
- Hosted APIs add an external dependency and terms risk (synthetic-persona rules, ownership of designed voices). Open Apache models keep the "every capability behind an adapter" rule cheap to honour.

### Gaps
- Synthetic-persona and AI-influencer use clauses in ToS were not read for ElevenLabs, Hume, Cartesia, Inworld, MiniMax, Google or OpenAI. A legal pass is needed. OpenAI and Google typically require disclosure that voices are AI-generated (from my prior knowledge; UNVERIFIED in this pass), which we already do.
- Official Hume pricing page not fetched. Together and fal per-minute prices for hosted open TTS not collected.
- AA leaderboard data date is not shown. Model release dates up to Nov 2026 suggest it is current.

## Long-form handling (60–90 s in one pass, prosody continuity)

### Takeaway
Few sources give explicit single-pass limits. VibeVoice claims 90 min and 4 speakers, but its availability is restricted and its card is research-only. MOSS-TTSD targets "ultra-long dialogues" under Apache-2.0. VoxCPM2 allows 8192 tokens per pass. Drama 3 auto-chunks long text with unclear context carry-over. The practical fix for intonation resets is to generate the full 60–90 s script in one TTS pass, or chunk with the previous chunk's audio as the prompt, and then split audio (not text) into lip-sync segments.

### Cited Findings
- VibeVoice: long-form up to 90 min with up to 4 speakers in a 64K context — [VibeVoice tech report](https://arxiv.org/html/2508.19205v1). TTS code removed or repo disabled, research-only card — [Cartesia](https://www.cartesia.ai/learn/open-source-tts), [VibeVoice page](https://microsoft.github.io/VibeVoice/).
- MOSS-TTSD-V1.0 8B for "ultra-long dialogues", podcasts and dubbing. MOSS-TTS supports "long-speech generation" with token-based duration control. Sample max_new_tokens is 4096. No minute limit stated — [GitHub MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS).
- VoxCPM2 max length is 8192 tokens — [HF](https://huggingface.co/openbmb/VoxCPM2).
- Drama 3 auto-chunks long text and streams per chunk, and it is unclear how context carries across chunks — [kie.ai](https://kie.ai/blog/fish-audio-drama-3-deep-dive-35b-moe-tts).
- IndexTTS-2 offers explicit duration control, useful for fitting audio to video segment lengths — [arXiv 2506.21619](https://arxiv.org/html/2506.21619v1).

### Inferences
- Qwen3-TTS uses a 12 Hz tokenizer, so 90 s of speech is about 1,080 audio tokens. That should fit one pass easily. This is my inference from the tokenizer rate; the model card does not state a max length.
- For Manyworlds, change `talk_segments.py` flow so TTS runs once per full script on the control plane or one GPU job, with segment boundaries chosen at pauses in the audio. InfiniteTalk then gets audio slices, and intonation no longer resets at cuts.

### Gaps
- No measured long-form drift or consistency data for Qwen3-TTS, VoxCPM2, Ming or Drama 3 at 60–90 s.
- Dia2 and Sesame CSM context lengths not found.

## Benchmarks and leaderboards (late 2026)

### Takeaway
Artificial Analysis Speech Arena is the live reference (blind-vote Elo, with provider-voice, selected-voice and controlled-voice views, plus pronunciation robustness). As of the Nov 2026 fetch, ElevenLabs v4 leads overall at 1340, and Fish Drama 3 leads open weights at 1277. Leadership changes fast: Fun-Realtime-TTS was #1 in June 2026 and Inworld 1.5 Max in April 2026.

### Cited Findings
- Current top: Eleven v4 / v4 Turbo 1340, StepAudio 3 1307, Qwen-Audio-3.1-TTS-Plus 1298, Sonic 3.6 1294, Simba 3.5 1293 — [AA](https://artificialanalysis.ai/text-to-speech/leaderboard).
- Alibaba Fun-Realtime-TTS topped the AA Speech Arena on 2026-06-03 — [AA article](https://artificialanalysis.ai/articles/fun-realtime-tts-new-text-to-speech-model-topping-artificial-analysis-leaderboard). An April 2026 snapshot had Inworld TTS 1.5 Max at #1 (1217) — [magichour snapshot](https://magichour.ai/model-leaderboard/text-to-speech).
- AA also runs a Controlled Voice arena and a Pronunciation Robustness benchmark — [AA controlled voice](https://artificialanalysis.ai/text-to-speech/leaderboard/controlled-voice), [AA methodology](https://artificialanalysis.ai/text-to-speech/methodology).
- Hume Octave 2 sits at only 1058 Elo despite its emotion focus — [AA](https://artificialanalysis.ai/text-to-speech/leaderboard).

### Inferences
- Arena Elo measures short, neutral-ish clips. Expressive performance in our genres (press conferences, confessionals, comedy) and lip-sync compatibility are not measured. Our own rubric-based evaluator plus the video gate is the deciding benchmark.

### Gaps
- TTS Arena (HF) late-2026 standings not fetched.
- No benchmark specifically for voice-design quality or emotional range found.
