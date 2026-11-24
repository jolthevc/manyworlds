# Expressive voice: engines beyond VoxCPM2 (2026-11-24)

Problem: VoxCPM2 under ultimate cloning (`voice.clone: ultimate`) sounds flat, and a delivery note
in the text is read aloud (breadth rerun 2026-11-24), so `voice.deliveries_spoken.ultimate: false`
removed our only delivery control. Bar: HeyGen-level naturalness. Constraints: commercial use of code
and weights, 24-48 GB GPU, one timbre across lines, delivery control that never becomes words, English.

Research only; nothing run, nothing spent. "Est." marks my estimates; "unverified" marks things from
prior knowledge I did not re-check today.

## 1. Ranked shortlist

| # | Engine | Licence (code / weights) | How you control emotion | Why it ranks here |
|---|---|---|---|---|
| 1 | **IndexTTS2 / 2.5** (Bilibili; 2.5 released 2026-08-10) | Bilibili Model Use License covers both. Royalty-free and commercial use is allowed. A separate licence is needed above 100M MAU or RMB 1bn revenue. You may not use it to improve other AI models. | Timbre and emotion are separate inputs: a speaker clip sets timbre, and a **separate emotion reference clip** (`emo_alpha` 0-1), an 8-value emotion vector, or a text description (`emo_text`, via a Qwen emotion model) sets delivery. | Separating speaker from emotion is the fix we need, and the delivery never enters the spoken text. Speed: RTF 0.21 (2.5, bf16) on a 4090. Failure modes: random emotion sampling (and, est., a high `emo_alpha`) lowers speaker similarity; paper results are the authors' own. |
| 2 | **Step-Audio-EditX** (StepFun, 3B; release 2026-01-29) | Code Apache-2.0. **Weights licence not stated** on the repo; check the HF card before use. | Clones from clip + transcript, then **edits** the audio in passes: 14 emotions, 32 styles (incl. whisper, chat, warm), plus paralinguistic tags (laugh, chuckle, sigh, breath, uhm...). | 12-16 GB. It can also edit the output of any other engine, including VoxCPM2. Authors' data: emotion accuracy rises from 55.6 to 74.3 after 3 passes, and editing ElevenLabs v2 output raises it from 55.7 to 72.2. Risk: each pass may drift timbre (check ECAPA every pass). |
| 3 | **Higgs Audio v2 / Higgs TTS 2** (Boson, 3B LLM + audio expert, ~6B in BF16) | Boson Higgs Audio 2 Community License (Llama 3-based). You must ask for a licence above **100k annual active users**, must show a "Built with Higgs" notice, and may not use it to improve other LLMs. | Expression emerges from context (scene description, prior turns); clones from a reference clip. v2.5 adds expressiveness tags (Azure listing). | EmergentTTS-Eval win rate vs gpt-4o-mini-tts: **75.7% on emotions**, 55.7% on questions (Gemini judge). Licence risk: our audience could pass 100k users. **v3 is non-commercial**, so don't use it. |
| 4 | **Chatterbox / Chatterbox Turbo** (Resemble, 350M, English) | MIT / MIT. Every output carries a PerTh watermark (fine for us, since we disclose AI use). | Turbo: inline tags `[laugh] [chuckle] [sigh] [gasp] [cough]...` (exact list in `added_tokens.json`). Original Chatterbox: an `exaggeration` knob. | Cheapest, cleanest licence, and it laughs on cue. Emotion control is coarse. No public blind test against ElevenLabs. |
| 5 | **Qwen3-TTS** (Alibaba, 0.6B/1.7B; released 2026-01-22) | Apache-2.0 / Apache-2.0. | Instructions in plain language work only on **VoiceDesign/CustomVoice**. The cloning (**Base**) model has no `instruct` parameter. | Strong cloning (EN WER 1.24, SIM 0.775) and a clean licence. Delivery has to come from the reference clip (see §2: emotion bank), which makes it a like-for-like swap for VoxCPM2. |
| 6 | **Fun-CosyVoice3-0.5B-2512** | Code Apache-2.0. Weights licence not shown on the repo page; check it. | Instructions set emotion, speed and volume. The repo does not say whether instructions work together with zero-shot cloning. | Small and accurate (EN WER 1.68-2.24, SIM ~0.70). Expressiveness unproven for us. |

**Excluded on licence:**
- Fish Audio S2 Pro: research licence; it has free-form `[whisper in small voice]` tags.
- Higgs TTS 3: non-commercial, despite the best EmergentTTS win rate (53.7%).
- Voxtral TTS: CC BY-NC, and the open weights have preset voices only, no cloning.
- Breeze TTS 2: non-commercial (secondary source).
- F5-TTS weights: CC-BY-NC.
- MaskGCT and Spark-TTS weights: non-commercial (unverified).

**Lower ranked (unverified):**
- Dia: dialogue and non-verbal sounds, but a weak hold on a cloned voice.
- Sesame CSM-1B: conversational, but plain when cloning.
- Orpheus: tags `<laugh>`; built on Llama.
- Zonos: emotion vector, often unstable.
- Kyutai TTS: built for streaming.

Sources: [IndexTTS repo](https://github.com/index-tts/index-tts), [licence](https://raw.githubusercontent.com/index-tts/index-tts/main/LICENSE), [paper](https://arxiv.org/abs/2506.21619v2);
[Step-Audio-EditX](https://github.com/stepfun-ai/Step-Audio-EditX), [paper](https://arxiv.org/abs/2511.03601v1);
[Higgs TTS 2 card](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base), [licence](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/blob/main/LICENSE), [Higgs v3](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b), [v2.5 on Azure](https://ai.azure.com/catalog/models/Higgs-Audio-v2.5);
[Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo), [tag list discussion](https://huggingface.co/ResembleAI/chatterbox-turbo/discussions/21);
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), [report](https://arxiv.org/abs/2601.15621); [CosyVoice](https://github.com/FunAudioLLM/CosyVoice);
[Fish S2 Pro](https://huggingface.co/fishaudio/s2-pro); [Voxtral TTS](https://mistral.ai/news/voxtral-tts/); [2026 open-model roundup (secondary)](https://presenc.ai/research/best-open-weight-text-to-speech-models-2026).
Public leaderboards are unreliable for open weights right now. One roundup [withdrew its numbers](https://www.offlinetts.com/blog/tts-arena-leaderboard-2026/) in September 2026. Treat our own bake-off as the evidence.

## 2. Techniques: social video, not audiobook

1. **Write for the ear.** Use contractions, fragments, a filler where a person would really hesitate ("okay so", "like", "honestly"), restarts, and short sentences with one idea each. Our `anchor_text` already does this; scripts from the showrunner/director should too. Read flatness is mostly a script problem, est. 30-50% of it.
2. **Emotion bank per creator.** Ultimate cloning copies the *reference's* prosody, so one calm anchor gives calm lines. Record (synthesise) 5-6 delivery anchors in the creator's voice: excited, confiding, deadpan, amused, storytelling, questioning. Gate each on ECAPA similarity to the base anchor (`voice.min_distance` logic reversed). For each line, choose the anchor whose delivery matches its `delivery`. This works **with VoxCPM2 today**: clip + transcript, no text delivery, so nothing is read aloud. In IndexTTS2 it maps onto the separate emotion clip.
3. **Prosody transfer / editing.** Make a plain take and then apply emotion to it (Step-Audio-EditX passes), or use IndexTTS2's emotion clip with `emo_alpha` ~0.6-0.8 (est.) to keep the timbre.
4. **Non-verbal sounds as tokens, never as words.** Use only engines that have dedicated laugh/breath/sigh tokens (Chatterbox Turbo, Step-Audio-EditX, Higgs). Our own breaths, a soft inhale before a line start, can come from a small clip library mixed into the gaps (cheap, deterministic).
5. **Pacing.** Real creators speed up on lists and slow down on the reveal. Keep `synthesis: whole_line` (intonation across the whole sentence). Vary `clause_gap_s` by delivery: tight for excited, longer before a punchline.
6. **What the hosted tools do.** ElevenLabs v3 drives delivery with inline audio tags (`[whispers]`, `[laughs]`, `[sarcastic]`) and a low "stability" setting (Creative/Natural). It relies on punctuation and capital letters for emphasis ([v3 prompting guide](https://elevenlabs.io/docs/best-practices/prompting/eleven-v3), [audio tags](https://elevenlabs.io/blog/v3-audiotags)). The AI-UGC avatar tools mostly sit on such hosted TTS. Their naturalness is script + tags + several takes picked by ear (est., from their own docs and marketing).

## 3. Blind listening bake-off

**Lines** (one creator, Maren; ~5-8 s each):
1. Story-time: "Okay so. I'm at the airport, right, and the guy next to me just... opens a whole rotisserie chicken."
2. Excited product: "No, wait, this actually works? I've used it every day this week and my skin is — look at it!"
3. Deadpan aside: "Great. Love that for me."
4. Confiding: "Can I tell you something I've never said on here? I almost quit last spring."
5. Question to camera: "So what would you have done? Seriously, tell me."
6. Laugh: "I tripped. In front of everyone. [laugh] I'm fine, I'm totally fine."

**Engines** (4): VoxCPM2 ultimate + emotion bank (the baseline, and also the cheapest fix), IndexTTS2.5 (speaker clip + emotion clip), Step-Audio-EditX (clone, then 1-2 edit passes), Chatterbox Turbo (tags). Optional hosted anchor: ElevenLabs v3 with an Instant Voice Clone of Maren's reference (~1k characters, ~$0.10), to calibrate the HeyGen-level bar.

**Protocol:**
- Each engine gets the same reference clip and transcript, and the same per-line delivery (as an emotion clip, vector or tag, by engine).
- Make 3 seeds per line and auto-pick by WER, which mirrors production.
- Rename the files to random ids. The owner rates each clip 1-5 on *naturalness* and *delivery matches the note*, plus a forced "which would you post" per line.
- Automatic scores, all on CPU or the same GPU: Whisper WER (fail above `voice.max_wer` 0.1), ECAPA similarity to the base anchor (one creator's timbre), and a leak check (any delivery word showing up in the transcript fails).
- Winner rule: the best median owner score among engines with WER ≤ 0.1 and similarity within 0.05 of VoxCPM2's.

**Cost (est.):**
- 4 engines × 6 lines × 3 seeds = 72 clips, about 6 min of audio, under 10 min of generation.
- Boot, environments and weights downloads (~5-15 GB per engine) dominate: ~45-60 min on one L40S/4090 at ~$0.5-1/h, so **~$0.50-1.00 GPU**, plus ElevenLabs ~$0.10.
- Run as a `test-` batch with the owner's go, the same as every GPU run.

## 4. Hosted reference prices (comparison only)

- ElevenLabs Eleven v3 API: ~$0.10/1k characters, ~$0.09 per spoken minute (900 characters/min assumed). v3 Conversational: ~$0.05/1k ([ModelsLab, Sept 2026](https://modelslab.com/elevenlabs-api-pricing), [UsagePricing](https://www.usagepricing.com/blueprint/activity/elevenlabs-2026-09-28-v3-conversational-dubbing-v2), [official API page](https://elevenlabs.io/pricing/api)). An Eleven v4 and a 20% price cut in October 2026 are reported but unverified.
- MiniMax speech-2.8-hd: $100/M characters (~$0.09/min); turbo: $60/M ([invideo, Aug 2026](https://invideo.io/blog/ai-voice-model-pricing/)).
- Gemini 3.1 Flash TTS: $1/M input tokens + $20/M audio output tokens (same source).
- Cartesia Sonic: ~$0.06/min, enterprise (same source).
- Self-hosted, for comparison (est.): a 30 s talking line with 3 seeds at RTF 0.2-0.5 takes 20-45 GPU-seconds on a ~$0.8/h card, about **$0.005-0.01 per line**, before idle time.

## Recommendation

1. Build the **emotion bank** for VoxCPM2 first. It costs nothing new and fixes the leak while restoring delivery.
2. Run the bake-off against IndexTTS2.5 and Step-Audio-EditX.
3. Confirm the Step-Audio-EditX and CosyVoice3 weights licences, and get legal sign-off on the Bilibili and Higgs terms, before adopting either.
