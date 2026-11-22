# Video quality, waves 2–3: references, sound, motion, avatars, two-shots, speech (2026-11-22)

Web research for the next production waves. Every fact has a link. **[unverified]** means the claim
comes from a secondary or mirrored source, or we did not read the primary text. This is not legal
advice. Clear licence questions with counsel before we depend on a model commercially.

## 1. Reference-conditioned images (same person, same room, same outfit)

| Model | Weights licence | Notes |
|---|---|---|
| **Z-Image-Edit** | n/a | Not released. The official README lists it as "To be released" on HF and ModelScope, with no date ([README](https://raw.githubusercontent.com/Tongyi-MAI/Z-Image/main/README.md)). Turbo is Apache-2.0 ([HF](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/commit/eeaf506a8f400e5184f9b05b342aa7c469a69ecf)). |
| **Qwen-Image-Edit-2511** | **Apache-2.0** ([HF card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)) | 20B parameters. Multi-image input (the example fuses 2 people). Claims better character consistency and less drift. Native ComfyUI template ([Comfy docs](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511.md)). |
| Qwen-Image-2.1 (unified gen+edit, up to 10 refs) | Qwen Research Licence, no commercial use without a deal ([alternativeto](https://alternativeto.net/news/2026/9/alibaba-launches-qwen-image-2-1-a-7b-ai-model-with-native-transparency-and-a-license-change/), [eesel](https://www.eesel.ai/blog/qwen-image-2-1)) [unverified, secondary] | Stays on the avoid list. 2511 is a separate, earlier Apache release. |
| Step1X-Edit (v1/v2) | Apache-2.0 [unverified: only a [Gitee mirror](https://gitee.com/ppandaer/Step1X-Edit) and [RunComfy](https://www.runcomfy.com/models/stepfun-ai/stepx-edit2) say so] | Single-image edit model. Not a multi-reference compositor. |
| OmniGen2 | Apache-2.0 ([HF](https://huggingface.co/OmniGen2/OmniGen2)) | Multi-reference "in-context generation". One unofficial page claims non-commercial weights ([page](https://omnigen2-pro.pages.dev/license-agreement)), which conflicts with the HF card. |
| FLUX.1 Kontext dev | FLUX Non-Commercial for the model. Outputs are not "Derivatives" ([BFL licence](https://bfl.ai/legal/non-commercial-license-terms)) | Running it to make revenue is not a non-commercial purpose, so a paid licence is needed. |
| FLUX.2 dev / klein-9B | Non-commercial. **klein-4B is Apache-2.0**, with multi-reference editing ([BFL](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence?lid=ECSSqRZRPUuf3RTN), [VentureBeat](https://venturebeat.com/technology/black-forest-labs-launches-open-source-flux-2-klein-to-generate-ai-images-in)) | klein-4B is a cheap fallback candidate. |
| UNO | CC BY-NC 4.0 weights ([HF](https://huggingface.co/bytedance-research/UNO)) | Avoid. |
| USO, InstantCharacter, ACE++ | Built on FLUX.1-dev / Fill-dev, so they inherit its NC terms ([USO](https://huggingface.co/bytedance-research/USO), [InstantCharacter paper](https://arxiv.org/html/2504.12395v1), [ACE++](https://huggingface.co/ali-vilab/ACE_Plus)) | Avoid. |
| DreamO | Code is Apache-2.0 ([Gitee](https://gitee.com/ByteDance/DreamO)). Built on FLUX [unverified for weights] | Avoid until the weights licence is clear. |

**Speed and VRAM.** The Lightning LoRA for 2511 cuts sampling to 4 steps, about 10× faster than 40 steps
([lightx2v](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)). Reported numbers:

- BF16 runs with offload on 10 GB VRAM plus 64 GB RAM ([Civitai](https://civitai.com/models/2247803/qwen-image-edit-2511)).
- About 10 s per 4-step image on a 4090 for the older 2509 model with INT4 [unverified].
- FP8 base plus Lightning LoRA has produced grid artifacts and "fp8 broken" reports
  ([ModelTC](https://github.com/ModelTC/Qwen-Image-Lightning), [discussion](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/discussions)).
  Use bf16 on the L40S, or the lightx2v-matched fp8 files only after a test.

**Recommendation: Qwen-Image-Edit-2511 (Apache-2.0) in native ComfyUI.**

- **Inputs:** image 1 is the creator anchor (face/outfit), image 2 is the set plate (room). The prompt describes
  the new pose and action.
- **Nodes:** the template's `TextEncodeQwenImageEditPlus` with multiple image inputs [node name unverified; read it from the template], then KSampler at 4 steps with the
  Lightning LoRA.
- **Hardware:** L40S at bf16. The 3090/4090 need offload.
- **Gate:** run the result through the existing image gate plus a face-similarity check against the anchors.
- **Fallbacks:** diffusers `QwenImageEditPlusPipeline` with the HF card example, or klein-4B.
- **Note:** this is a second image family alongside Z-Image Turbo. Our creator LoRAs will not transfer to it.

## 2. Sound: room tone, Foley, music beds

| Source | Licence / commercial | Verdict |
|---|---|---|
| MMAudio (video→audio) | Code MIT. **Checkpoints CC-BY-NC 4.0** ([GitHub](https://github.com/hkchengrex/MMAudio)) | Avoid. |
| ThinkSound | Labelled Apache-2.0, but the README says "Commercial use is NOT permitted". Bundles the Stable Audio Open VAE ([GitHub](https://github.com/FunAudioLLM/ThinkSound)) | Avoid. |
| AudioX | CC-BY-NC 4.0 ([discussion](https://hug-proxy.pages.dev/HKUSTAudio/AudioX/discussions/3)) [mirror] | Avoid. |
| HunyuanVideo-Foley | Tencent Hunyuan Community Licence ([NOTICE](https://huggingface.co/tencent/HunyuanVideo-Foley/blob/main/NOTICE)). The family excludes the EU, UK and South Korea, outputs included ([HunyuanImage-3 licence](https://scancode-licensedb.aboutcode.org/tencent-hunyuan-image-3.0-cla.html)) [Foley text unverified] | Avoid. Our audience reaches the EU/UK. |
| Stable Audio Open 1.0 / 3.0 Small, Small-SFX, Medium | Stability Community Licence: commercial use free under **$1M annual revenue**, Enterprise above that. 3.0 is trained on licensed data ([Stability](https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models), [comparison](https://stability.ai/explainers/stable-audio-vs-competitors-licensing-export-rights-and-self-hosting-compared)) | **Revenue-capped.** Usable now. Record the cliff in policy. Best licensed-data option for SFX and beds. |
| ACE-Step 1.5 (music) | **MIT**. The card says outputs may be used commercially ([HF](https://huggingface.co/ACE-Step/acestep-v15-base/blob/main/README.md)). Training-data sourcing is only partly documented ([arXiv](https://arxiv.org/abs/2602.00744)) | Recommended for music beds. |
| YuE (songs with vocals) | Moved from CC-BY-NC to Apache-2.0 ([HF](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot/blob/main/README.md)) | OK, but heavy, and we rarely need vocals. |
| MusicGen | **CC-BY-NC 4.0 weights** ([HF](https://huggingface.co/facebook/musicgen-large)) | Avoid. |
| Freesound API | **API free only for non-commercial use.** Commercial use needs a deal with MTG/UPF ([API terms](https://freesound.org/docs/api/terms_of_use.html)) | No API use. A human could hand-download CC0 files into a curated pack [verify each file]. |
| Pixabay | Pixabay Licence: music and SFX free for commercial use, no attribution needed ([FAQ](https://pixabay.com/service/faq/)). **The API covers images and video only** ([API docs](https://pixabay.com/api/docs/)) | Curated, hand-picked pack only. Pixabay music can draw Content ID claims [unverified]. |
| OpenAIR impulse responses | "Mostly" Creative Commons. The audb catalogue lists CC-BY-4.0, commercial allowed ([audb](https://audeering.github.io/audbcards/audb-public/openair.html), [sonicfield](https://sonicfield.org/openair-a-collective-library-of-impulse-responses)) | OK per file. Check each IR's licence and credit it. The official site's TLS cert had expired when we checked. |

**Mic and room simulation.** ffmpeg `afir` convolves with an IR file: `ffmpeg -i voice.wav -i room_ir.wav -lavfi afir=dry=…:wet=…`.
Use `irnorm` / `irgain` to set the level ([ffmpeg filters](https://ffmpeg.org/ffmpeg-filters.html)).

**Recommended stack.**

1. Voice → `afir` with an OpenAIR small-room IR at a low wet mix, then a band-limit EQ (`highpass` / `lowpass`) for phone-mic character.
2. Room tone and SFX from Stable Audio 3.0 Small-SFX, prompted per scene, plus a curated CC0/Pixabay pack held in R2 with
   a licence column per file.
3. Music beds from ACE-Step 1.5, instrumental, ducked under voice with `sidechaincompress`.
4. No video-to-audio model for now. Every strong open model is non-commercial or territory-restricted.

## 3. Dance and motion transfer

- **Wan-Animate-2-14B** (`Wan-AI/Wan2.2-Animate-2-14B`):
  - **Apache-2.0** on the official HF card. Base and distilled weights released 2026-08-07 ([HF](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B)).
  - The DiT consumes the driving video directly, with no pose extractor, and adds text control of the viewpoint.
  - The distilled variant runs 10 steps at CFG 1.0. Default 720p.
  - The defaults are tuned for 8×A800, and 480p was tested on 2×A800. No single-GPU VRAM figures are published.
  - The card claims "strong identity preservation" but gives no numbers.
  - NVIDIA's hosted copy adds the NVIDIA Open Model Licence for its endpoint only ([NVIDIA](https://build.nvidia.com/wan-ai/wan2.2-animate-2-14b/modelcard)).
- **ComfyUI support is native, not in Kijai's wrapper:**
  - Kijai himself wrote the core PR #15362, merged 2026-08-07. It adds the `WanAnimate2ToVideo` node, with pose/reference strength, pose start/end windowing and `continue_motion` extension.
  - It also adds the optional `WanAnimate2Cache` node, which "roughly halves generation time" and can store in RAM or VRAM as int8/int4 ([PR](https://github.com/Comfy-Org/ComfyUI/pull/15362)).
  - The template is `video_wan_animate2`, and the weights are at `Comfy-Org/Wan2.2-Animate-2` ([Comfy blog](https://blog.comfy.org/p/wan-animate-2-is-now-available-in), [HF](https://huggingface.co/Comfy-Org/Wan2.2-Animate-2)).
  - We found **no Animate-2 support in WanVideoWrapper's README** ([repo](https://github.com/kijai/ComfyUI-WanVideoWrapper)).
  - So a dance job needs a ComfyUI core recent enough for PR #15362, not a newer wrapper pin.
  - On a 48 GB L40S with the fp8 or int8 checkpoint it should fit [unverified, needs a benchmark].
- **Original Wan2.2-Animate-14B** is supported in the wrapper (`WanVideoAnimateEmbeds`, example `wanvideo_WanAnimate_example_01.json`)
  ([node ref](https://www.runcomfy.com/comfyui-nodes/ComfyUI-WanVideoWrapper/wan-video-animate-embeds), [HF discussion](https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF/discussions/2)). We did not confirm whether it is in pin a7439db.
- **Identity drift:** we found no measured data. Gate on face similarity per keyframe against the anchors.
- **Rights in reference dances:**
  - Choreography can be copyrighted by its creator, separately from the music ([MSU law](https://mclellan.law.msu.edu/questions/tiktok-dances-are-not-infringing), [Harper Macleod](https://harpermacleod.co.uk/?p=16499)).
  - TikTok business accounts are limited to the Commercial Music Library for commercial use ([TikTok](https://ads.tiktok.com/help/article/commercial-music-library), [support](https://support.tiktok.com/sk/business-and-creator/creator-and-business-accounts/commercial-use-of-music-on-tiktok)).
  - Policy should be: drive only from clips we own or that are licensed (self-shot stock, our own renders, CC0 mocap), never a downloaded trend video. Add trend audio only in-app via the platform's licensed library at post time, never baked into our file.

## 4. Hosted talking avatars (quality ceiling)

| Service | Price | Max length | Terms that matter |
|---|---|---|---|
| **Hedra Character-3** | API $0.025/s at 540p, $0.05/s at 720p, $0.0625/s at 1080p | 10 min ([model page](https://www.hedra.com/models/video/hedra/character-3)) | Hedra claims no ownership of outputs but takes a broad licence, including to "derive revenue" (§3.2–3.3, [Terms](https://www.hedra.com/terms)). The AUP bars using it "to impersonate real entities or create fake personas (also known as "deepfakes") to falsely attribute content" ([AUP](https://www.hedra.com/acceptable-use)). Disclosed AI creators arguably fall outside "falsely attribute", but this is a risk to clear with Hedra. |
| **Kling AI Avatar 2.0** | $0.056/s Std, $0.115/s Pro on fal ([fal](https://fal.ai/models/fal-ai/kling-video/ai-avatar/v2/pro)) | Up to 5 min per Runware ([Runware](https://runware.ai/models/klingai-avatar-2.0-pro)) [unverified] | The consumer terms §4.5 require Kling brand/logo labelling and §4.6 require written permission for commercial use ([user policy](https://kling.ai/docs/user-policy)). The API paid-service terms say commercial use "is not restricted" (§6.4) and you carry the portrait-rights risk (§8.3) ([API terms](https://kling.ai/document-api/guides/protocols/paid-service.md)). Use the API terms only. |
| **OmniHuman 1.5** (ByteDance) | $0.16/s on fal ([fal](https://fal.ai/models/fal-ai/bytedance/omnihuman/v1.5)); $0.12/s on Atlas ([Atlas](https://www.atlascloud.ai/zh/models/bytedance/avatar-omni-human-v1.5)); BytePlus first-party ([BytePlus](https://www.byteplus.com/en/product/OmniHuman)) | v1 capped audio under 30 s ([fal docs](https://fal.ai/docs/model-api-reference/video-generation-api/bytedance-omnihuman.md)). v1.5 cap unknown | BytePlus video terms bar content that infringes portrait rights ([terms](https://docs.byteplus.com/en/docs/ModelArk/Specific_Terms_for_the_BytePlus_Video_Generation_Model_Services)). Other BytePlus services require a prominent AI label [OmniHuman-specific text unverified]. |
| **HeyGen** | not checked | not checked | Paid plans: you may use outputs for your own purposes. Free-plan outputs "may not be... monetized". You "must proactively disclose" AI creation where the law requires, and must not represent output as "entirely human generated". Custom avatars need "explicit consent of the individual being represented" ([Terms](https://www.heygen.com/terms), [moderation](https://www.heygen.com/moderation-policy)). Nothing says whether an avatar of a synthetic person is allowed. HeyGen's consent flow assumes a real person [unverified fit]. |
| **Synthesia** | not checked | not checked | Custom avatars need consent from "the individual". Paid social ads with stock avatars need written consent. You must not strip provenance marks ([AUP](https://www.synthesia.io/legal/acceptable-use-policy)). Built for corporate video, so it fits us poorly. |

**Takeaway.** The Kling API (via fal) and Hedra are the realistic ceiling benchmarks at about $0.05–0.12/s. We
already label every post as AI, which satisfies the disclosure clauses. Hedra's "fake personas" clause needs a
written answer from Hedra before any production use.

## 5. Two-person scenes

- **InfiniteTalk/MultiTalk in WanVideoWrapper.** These names come from a mirror of `multitalk/nodes.py`; check them against pin a7439db.
  - Loaders: `Wav2VecModelLoader` and `MultiTalkModelLoader`.
  - **`MultiTalkWav2VecEmbeds`** takes `audio_1` (required) and optional `audio_2`–`audio_4`, plus `ref_target_masks`, which the tooltip describes as
    "one mask per speaker (plus optional background) to guide mouth assignment".
  - `multi_audio_type`: `para` overlays the speakers and zero-pads to the longest; `add` concatenates them on one timeline.
  - Other inputs: `normalize_loudness` (−23 LUFS), `audio_scale`, `audio_cfg_scale`.
  - `WanVideoImageToVideoMultiTalk` handles long windowed I2V ([mirror](https://huggingface.co/aliensmn/ComfyUI-WanVideoWrapper/blob/main/multitalk/nodes.py), [node ref](https://comfy.icu/node/MultiTalkWav2VecEmbeds)).
  - **Binding:** audio_N goes to mask N, in order. The code does not check that the mask count matches the speaker count.
  - ComfyUI's native node is different: `WanInfiniteTalkToVideo` takes `two_speakers` / `mask_1` / `mask_2` ([Comfy docs](https://docs.comfy.org/built-in-nodes/WanInfiniteTalkToVideo)).
- **Failure modes:**
  - Both mouths move, or the audio binds to the wrong person. The MultiTalk paper calls this the "incorrect binding problem" ([arXiv](https://arxiv.org/html/2505.22647v1)).
  - Masks that touch or overlap, or a missing second mask, make it degrade to single-speaker behaviour ([RunComfy](https://www.runcomfy.com/comfyui-workflows/comfyui-multitalk-workflow-multi-speaker-lip-synced-video-generator)) [third-party].
  - Swapped track order swaps the speakers.
  - Official MultiTalk inference is 480p only ([GitHub](https://github.com/MeiGen-AI/MultiTalk)).
  - Mitigation: generate the masks from face bboxes (dilated, non-touching). For turn-taking use `add`. Gate with a per-face
    mouth-motion vs audio-energy check.
- **LongCat-Video-Avatar 1.5:**
  - **MIT** licence. Has a multi-person mode with the same `para` / `add` audio types. The default order puts person 1 first. Whisper-large replaces wav2vec2.
  - Use audio CFG 3–5 for lip sync. Int8 DiT available. The page does not say how a track binds to a face ([HF](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5)).
  - fal's multi endpoint maps person 1 to the left and person 2 to the right ([fal](https://fal.ai/models/fal-ai/longcat-multi-avatar/image-audio-to-video/api)).
  - ComfyUI routes: WanVideoWrapper's `longcat_avatar` branch ([Patreon](https://www.patreon.com/posts/146489049)) [unverified] or rookiestar28's
    ComfyUI-LongCat-Avatar, where multi-audio needs `ai2v` ([RunComfy](https://www.runcomfy.com/comfyui-nodes/ComfyUI-LongCat-Avatar)).

## 6. Natural speech with VoxCPM2

- **Control:** VoxCPM2 takes natural-language control in a leading parenthesis, e.g. "(Speaking slowly with a whispering, mysterious tone)".
  Voice design covers gender, age, tone, emotion and pace. Results vary between runs, and the card advises generating 1–3 times
  ([docs](https://voxcpm.readthedocs.io/en/latest/models/voxcpm2.html), [HF](https://huggingface.co/openbmb/VoxCPM2)).
- **Tags:** we found **no documented breath, laughter or filler tokens** (`[laugh]`, `[breath]`). Writing "um" or "haha" into the text, or
  "(with a light laugh)" as an instruction, may work but is undocumented [unverified, test it].
- **Pause practice.** Do not rely on the model for pauses. Synthesise one take per clause or sentence and join them with explicit silence:
  - In spontaneous English, pauses between sentences are mostly over 500 ms and pauses between clauses are mostly under 500 ms
    ([Pfitzinger & Reichel](https://www.phonetik.uni-muenchen.de/~reichelu/publications/PfitzingerReichelSP2006.pdf), [de Gruyter](https://www.degruyterbrill.com/document/doi/10.1515/lingvan-2022-0156/html)).
  - Pause length rises with boundary strength ([Elmers et al.](https://www.coli.uni-saarland.de/~trouvain/docs/Elmers_et_al_2021a.pdf)).
  - Listeners are sensitive to pause length between sentences ([Google](https://research.google/pubs/a-study-of-raters-sensitivity-to-inter-sentence-pause-durations-in-american-english-speech/)).
  - Starting values, as policy: comma or clause 180–300 ms, sentence 450–700 ms, topic shift 800–1000 ms, ±15% jitter.
  - Before a long sentence, insert a soft breath sample (−30 to −24 dB, 250–400 ms) drawn from a CC0 breath pack, or cut from the
    creator's own design-voice takes.

## Next-wave build list

1. **`image.qwen_edit` handler**: Qwen-Image-Edit-2511 bf16 plus the 4-step Lightning LoRA in native ComfyUI. The inputs are an anchor and a set plate. Gate on face similarity.
2. **Set plates**: generate one empty-room plate per creator location and reuse it as the reference for every scene there.
3. **`audio.sfx` handler**: Stable Audio 3.0 Small-SFX for room tone and Foley. Record the $1M revenue cap in policy.
4. **`audio.music` handler**: ACE-Step 1.5 instrumental beds, ducked under voice.
5. **Mix step in `media.py`**: `afir` with a per-file-licensed OpenAIR IR, mic EQ, loudness to −14 LUFS for social.
6. **Curated SFX/IR pack in R2** with a licence and source column per file. Freesound API stays out.
7. **`video.wan_animate2` handler**: update ComfyUI core to PR #15362 or later, use the distilled 10-step weights with `WanAnimate2Cache`, benchmark on L40S. Drive only from clips we own.
8. **Two-shot `talk.multitalk`**: masks from face bboxes, `add` mode for turn-taking, and a per-face lip-activity gate. Bake off against LongCat 1.5 multi.
9. **Speech assembler**: one VoxCPM2 take per clause, then policy pause lengths with jitter and an optional breath insert. A/B test the "(with a light laugh)" instruction.
10. **Ceiling benchmark**: 20 talking clips through the Kling Avatar 2.0 API (fal) vs our InfiniteTalk/LongCat output. Get a written answer from Hedra on its "fake personas" clause before trying it.
11. **Policy `licences` block**: add MMAudio, ThinkSound, AudioX, MusicGen, HunyuanVideo-Foley, UNO, USO, InstantCharacter, ACE++, FLUX Kontext dev and Qwen-Image-2.1 to the avoid list. Mark Stable Audio as revenue-capped.
