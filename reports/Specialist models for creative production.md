# Control signals beat bigger writers for creative lift

The cheapest creative gains for the studio come from structure, not from a smarter language model. Each generator in the stack already has an official prompt-rewrite format that our director does not follow yet, and each has a free, Apache-licensed structural control (pose, depth, line and camera conditioning) that turns "the seed decides the pose" into "the shot sheet decides the pose". Both cost nothing in licences and little in GPU time, so they rank first. Next comes a cheap deterministic measurement pass on every still (metric depth, masks, body and hand keypoints), which lets the director plan motion from measured facts that code can check. After that come specialist judges: ShotVL for cinematography, HPSv3 and UnifiedReward for preference, and Nemotron 3 Nano Omni for whole-video performance. These run on one warm GPU as rank-only signals, never as gates. Creative writing is where spending buys the least. The best open models (GLM-5.3, Kimi K3.1 Pro) reach the same band as Claude Opus 5.5 on creative-writing boards but do not beat it, and Kimi on DeepInfra costs about the same per call as Sonnet. The models that beat Opus 5.5 there are GPT-6.2 Astra and Anthropic's own Fable 5.x. The licence picture is the main hazard. At least eight tempting tools are non-commercial or research-only: Hemmingway-1, OpenFace 3.0, LibreFace, Sapiens, Depth Anything 3's giant checkpoint, Audio Flamingo 3, InsightFace's models and the Molmo pointing family. Several others carry revenue gates or data terms that must be recorded in policy before use: LTX-2.5, Kimi, GLM, HY-Motion, Twelve Labs and the SMPL body models.

## Twelve candidates ranked by creative impact per dollar

The ranking below is a judgement built from the evidence in the sections that follow. "Impact" means expected improvement in accepted, on-brief output for a lifestyle creator. "Cost" is marginal: licence plus GPU or API spend once the studio is already running the generator. None of these candidates has been measured on our own content. Every GPU trial still needs the owner's go for that run, and every hosted model needs a shadow test through `llm.py` before it takes a role.

| Rank | Candidate | Problem it fixes | Hosting | Marginal cost | Licence |
|---|---|---|---|---|---|
| 1 | Model-specific prompt compilers (Z-Image `pe.py`, Wan I2V rewrite rules, LTX paragraph format, Wan-Animate caption rules) | Prompts in the wrong shape for each generator | Code in `direction.py` | ~0 | Source repos Apache-2.0 |
| 2 | Z-Image-Turbo Fun ControlNet Union 2601/2602 | "Seed fixes pose"; composition drift | Image worker | Some loss of Turbo speed | Apache-2.0 |
| 3 | Wan 2.2 Fun Control + Fun Camera | Unplanned blocking and camera in clips | talkcomfy pool, ComfyUI | One graph per clip | Apache-2.0 (Fun Camera verified) |
| 4 | Scene measurement pass: MoGe-2, SAM 3 / SAM 3D Body, RTMPose or ViTPose, DA3-BASE | Director plans motion blind | Image worker | ms to ~1 s per image | MIT code / SAM License / Apache |
| 5 | Grounding VLM: Qwen3-VL (hosted) or GLM-4.6V-Flash | Points and boxes for hands, props, free space | DeepInfra or self-host | $0.15-0.20/M input | Apache (unverified) / MIT |
| 6 | Image preference judges: HPSv3, UnifiedReward-2.0 4B/8B, VisualQuality-R1 | Cheap ranking of candidate takes | A `judge` worker family | Cents per thousand images (estimate) | Apache / MIT / Apache |
| 7 | ShotVL-7B in direct-answer mode | Checks that shot size, angle and lens match the sheet | Judge worker | Same | Apache-2.0 |
| 8 | Deterministic AV stack: LatentSync SyncNet, emotion2vec+, Whisper timestamps, MediaPipe | Time-coded checks on lip sync and delivery | Finisher or talk worker | Negligible | OpenRAIL++ / FunASR licence / MIT / Apache |
| 9 | Nemotron 3 Nano Omni FP8 | Whole-video "intended vs actual performance" | One L40S 48 GB | Under a cent per clip if warm (estimate) | NVIDIA Open Model Agreement |
| 10 | Writing shadows: GLM-5.3, Kimi K3.1 Pro, Qwen 4 Max, Grok 4.7 humour probe | Voice variety, humour, cost | DeepInfra / Together | $0.002-0.009 per call | MIT-style with revenue gates / proprietary |
| 11 | HY-Motion 1.0 as a pose-video source | Synthetic choreography and gestures | One 24-26 GB card | Seconds per motion | Tencent Hunyuan Community + SMPL terms |
| 12 | Qwen3.5-Omni-Flash API, Twelve Labs Pegasus, Wan-Move | Hosted AV timestamps; point-trajectory drags | Alibaba / Twelve Labs / ComfyUI | Fractions of a cent to $0.029/min | Proprietary API / training opt-out / unverified |

The order reflects a simple asymmetry. Ranks 1-4 change what the generators make. Ranks 5-9 change how we pick among what they made. Ranks 10-12 are experiments whose payoff is speculative or whose terms still need work.

## Controls and model-native prompts fix what the director cannot say in words

**Each generator wants a differently shaped prompt, and the official rewriters spell it out.**

*Z-Image Turbo* runs at guidance 0 with no negative prompt and works best with long, detailed prompts. Its official enhancer, `pe.py`, keeps subject, count, action and any quoted text unchanged. It then adds composition, lighting, materials, colour and spatial layering in objective language, and bans metaphors and meta tags such as "8K" or "masterpiece" ([Z-Image GitHub](https://github.com/Tongyi-MAI/Z-Image); [pe.py](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/raw/main/pe.py)).

*Wan 2.2 I2V* wants the opposite. Its English I2V rewrite rules cap the prompt at **100 words**, keep the subject's actions and camera moves in the user's order, and **strip static descriptions of what the image already shows** ([Wan2.2 system_prompt.py](https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/wan/utils/system_prompt.py)). Describing her face, outfit or set again in a Wan prompt spends the budget on the wrong signal.

*LTX-2.5* wants one present-tense paragraph in chronological order, with dialogue in quotation marks and the language or accent stated ([LTX prompting guide](https://docs.ltx.video/api-documentation/prompting-guide)). Lightricks' 2.5 guide reportedly asks for under about 200 words, with camera as one element among six ([LTX-2.5 prompt guide](https://ltx.io/blog/ltx-2-5-prompt-guide)).

*Wan-Animate-2* wants a caption that describes appearance and background only, never actions, because the driving video supplies the motion ([Wan-Animate-2 GitHub](https://github.com/Wan-Video/Wan-Animate-2)).

Building these four shapes into the prompt compiler per model costs only code, needs no licence and touches every render. That is why it ranks first.

**Prompts alone cannot fix pose, and the control layer that does exists for every stage.**

*Stills.* Alibaba PAI's Z-Image-Turbo Fun ControlNet Union takes Canny, HED, depth, pose, MLSD, scribble and gray inputs and also inpaints, under **Apache-2.0**. The 2601 release fixed the blur and bright-spot artifacts of 2.0 and 2.1. The card recommends a control context scale of 0.65-1.00 and admits that control reduces Turbo's acceleration ([HF card](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1)). That it works alongside a per-creator LoRA is shown only by a third-party Replicate endpoint ([Replicate](https://replicate.com/benjyazoulay/z-image-turbo-lora-controlnet/readme)), not by any official statement. Our own LoRA plus ControlNet test is the first thing to run.

*Clips.* Wan 2.2 Fun Control takes a start image plus a control video of pose, depth, Canny, MLSD or trajectories, with in-graph preprocessors ([ComfyUI docs](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control)). Wan 2.2 Fun Camera takes a camera path from the `WanCameraEmbedding` node and allows commercial use under Apache-2.0 ([ComfyUI Fun Camera](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-camera)). ComfyUI also supports Wan 2.2 Fun InP with the LightX2V 4-step LoRA ([Comfy blog](https://blog.comfy.org/p/comfyui-wan22-fun-inp-support)). Whether lightx2v keeps its speed and adherence with Control or Camera is unverified.

*Point trajectories.* Wan-Move handles simple drags such as a hand to a cup, but it is built on Wan **2.1**, so it means a second checkpoint family ([ComfyUI Wan-Move](https://docs.comfy.org/tutorials/video/wan/wan-move)).

*LTX-2.5.* LTX-2.5 conditions on images at chosen frames and does native multi-shot. Its card names only a detail-upscaler IC-LoRA ([Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)). A pose, depth and Canny union control exists only as a community LoRA for LTX-2 19B ([HF yrrhall](https://huggingface.co/yrrhall/LTX-2-19b-IC-LoRA-Union-Control)). LTX structural control is therefore the weakest link.

**Synthetic motion can feed these controls without touching anyone's footage.** Tencent's HY-Motion 1.0 is a 1.0B text-to-motion DiT (0.46B Lite) that outputs skeleton animation on an SMPL-H skeleton. It takes English prompts under 60 words, needs **24-26 GB of VRAM**, and cannot do multi-person interaction or loops ([HY-Motion GitHub](https://github.com/Tencent-Hunyuan/HY-Motion-1.0)). Research such as CAMEO already chains text-to-motion into rendered SMPL guidance video and then video generation ([arXiv 2510.03909](https://arxiv.org/pdf/2510.03909v1)). No public workflow yet feeds HY-Motion into Wan Fun Control or Wan-Animate-2, which takes raw driving video with no skeleton step ([ComfyUI Wan-Animate-2](https://docs.comfy.org/tutorials/video/wan/wan-animate-2)). The licence is the catch. Hugging Face tags it "tencent-hunyuan-community" ([HF](https://huggingface.co/tencent/HY-Motion-1.0)), the full text was not readable, and SMPL-family body models carry their own Max Planck terms. HY-Motion is a promising rank-11 experiment, not a dependency.

## Measured scene facts make the director's motion plans checkable

**No study shows that an LLM director given pose and depth plans better human motion than one working from text alone.** The closest evidence comes from physics and object motion. In VLIPP, a VLM plans coarse trajectories that become structured conditioning for the I2V model, and this beats text-only prompting on physical plausibility ([VLIPP](https://arxiv.org/html/2503.23368v3)). LAMP maps LLM-written motion programs to 3D trajectories ([LAMP](https://arxiv.org/pdf/2512.03619v2)). Both papers point the same way. Most of the benefit comes from passing structure to the generator, and showing it to the planner matters less. For the studio, that means measured facts in the shot sheet ("0.9 m free to her left", "right wrist 0.3 m from the cup", camera height and FOV) feeding a rendered control video. Code can check those facts in `direction.feasibility` before any GPU spend.

**A commercially clean measurement stack exists, but each component's licence needs care.**

- *MoGe-2* returns a metric point map, depth, intrinsics and FOV from one image, at about 60 ms on an A100 or 3090. Its code is MIT, but no weight licence is stated ([microsoft/MoGe](https://github.com/microsoft/MoGe)).
- *Depth Anything 3:* the DA3-BASE checkpoint is Apache-2.0, but **DA3NESTED-GIANT-LARGE is CC BY-NC** ([DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE/blob/main/README.md); [DA3NESTED](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE)).
- *SAM 3 and SAM 3D Body* (full-body and hand mesh) use Meta's SAM License. It permits commercial use with use-based exclusions and lets Meta change the terms ([sam-3d](https://github.com/facebookresearch/sam-3d); [Roboflow summary](https://roboflow.com/model-licenses/segment-anything-3)).
- *RTMPose and ViTPose* are reported as Apache-2.0, though from a secondary source ([Fora Soft](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/openpose-mediapipe-rtmpose-pose-tracking)).
- *Sapiens* pose weights are **CC BY-NC** and must be avoided ([Sapiens card](https://huggingface.co/facebook/sapiens-pose-0.6b/blob/main/README.md)).

All of these are single-image models that can ride on the image worker between renders.

**For points and boxes, the commercially usable VLMs are Qwen3-VL and GLM-4.6V-Flash. The best pointer is research-only.**

- *Qwen3-VL* comes in sizes from 2B to 235B-A22B and supports 2D boxes and points plus 3D boxes ([Qwen3-VL README](https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/README.md)). An independent RefCOCO test put the 4B at 92.09 and the 8B at 91.11 F1@.50 ([GroundingPI](https://arxiv.org/pdf/2609.39601)). DeepInfra serves Qwen3-VL-30B-A3B at **$0.15/$0.60 per million tokens** ([DeepInfra pricing](https://deepinfra.com/pricing)).
- *GLM-4.6V-Flash* (9B) is MIT-licensed and returns boxes on a 0-1000 scale ([zai-org/GLM-V](https://github.com/zai-org/glm-v); [HF GLM-4.6V](https://huggingface.co/zai-org/GLM-4.6V)).
- *Mage-VL-4B* (Apache-2.0) claims VSI-Bench 64.3 against 53.3 for Qwen3-VL-4B. These are the authors' own figures ([microsoft/Mage-VL](https://huggingface.co/microsoft/Mage-VL)).
- *MolmoPoint-8B* reports the best open pointing scores (PointBench 70.7%). Its weights are Apache-2.0, but the card frames it as **"intended for research and educational use"** and says it was trained on third-party data restricted to non-commercial research ([MolmoPoint card](https://huggingface.co/allenai/MolmoPoint-8B/blob/main/README.md); [Ai2 blog](https://allenai.org/blog/molmopoint)). Molmo2 carries the same caveat ([Molmo2-8B card](https://huggingface.co/allenai/Molmo2-8B)). The studio already refused RVC and IndexTTS training on similar grounds, and the same discipline should keep Molmo out of production.

The plan then flows into a control video. Analysis yields her keypoints and depth. The director outputs a few waypoints and a root path. Code interpolates them into an OpenPose skeleton video. Wan 2.2 Fun Control or Fun VACE renders it from the still. Fun VACE's official weights need about 60 GB ([RunComfy](https://www.runcomfy.com/comfyui-workflows/wan-2-2-vace-in-comfyui-pose-driven-motion-video-workflow)). Because these graphs load video inputs, the `comfy.verified` / `NotRendered` guards added after the 2026-11-26 picker bug are mandatory on this path.

## Specialist judges are cheap rank signals, not authorities

**ShotVL is the one cinematography specialist worth running, and only as a multiple-choice classifier.** ShotVL-3B and 7B are Apache-2.0 fine-tunes of Qwen2.5-VL, trained on about 70k QA pairs from award-nominated films. On ShotBench, ShotVL-3B averages **65.1% against 59.3% for GPT-4o** and 59.1% for Qwen2.5-VL-72B. Its lead is largest on shot size, camera angle and lens size ([ShotBench arXiv](https://arxiv.org/html/2506.21356v2)). Those are the fields our shot sheets direct.

The independent RefineShot study confirmed the accuracy lead (ShotVL-7B 70.2 against 51.7 for its base) but found the model's reasoning is unfaithful. Asked to think step by step, ShotVL-7B **collapses to 34.0%**, and on camera movement it falls from 59.7 to 15.1 ([RefineShot](https://arxiv.org/html/2510.02423v1)). So ShotVL should get one direct-answer question per directed field, with no explanation requested. Claude keeps the free-text critique.

Two cautions apply. ShotVL was trained on 16:9 cinema stills, not 9:16 phone-native photos, so a small calibration set of our own labelled frames must come before any use. No 2026 successor was found. For camera movement in clips, CameraBench shows geometric estimators (MegaSAM, CUT3R) beat VLMs on pan, tilt and dolly, while VLMs beat them on semantic moves like "follow" ([CameraBench](https://linzhiqiu.github.io/papers/camerabench)). Optical flow on our own frames plus ShotVL is the sensible hybrid. CineTechBench's data is **CC BY-NC-ND**, so it is usable only as an internal reference ([CineTechBench](https://arxiv.org/html/2505.15145v1)).

**The open preference and quality judges are good enough to rank takes, and they are biased in the wrong direction for "real phone footage."**

- *HPSv3* (Apache-2.0, Qwen2-VL-7B, pip-installable) leads its own comparison with 76.9% pairwise accuracy on HPDv3, against 65.6% for PickScore and 59.9% for the LAION aesthetic predictor ([HPSv3 arXiv](https://arxiv.org/html/2508.03789v2); [HF](https://huggingface.co/MizzenAI/HPSv3/blob/main/README.md)).
- *UnifiedReward-2.0* (MIT tags, 2B-32B, image and video, pairwise and pointwise) ([HF CodeGoat24](https://huggingface.co/CodeGoat24/UnifiedReward-Think-qwen35-4b/blob/main/README.md)) and *VisualQuality-R1* (Apache-2.0) ([GitHub](https://github.com/TianheWu/VisualQuality-R1)) complete a stack that fits one 24 GB card.

The biases are documented. HPSv2 favours bright, vivid, detailed images, and PickScore favours dim, muted ones ([Follow-Your-Preference](https://arxiv.org/pdf/2509.23082)). Under RL, HPS scores keep rising while images collapse into over-saturation, and UnifiedReward drifts toward an unnatural dark style ([Pref-GRPO](https://arxiv.org/pdf/2508.20751)). Multimodal RewardBench 2 finds reward models still trail frontier VLM judges, by up to **12% on same-model pairs** ([MMRB2](https://openaccess.thecvf.com/content/CVPR2026/papers/Hu_Multimodal_RewardBench_2_Evaluating_Omni_Reward_Models_for_Interleaved_Text_CVPR_2026_paper.pdf)). Same-model pairs are exactly our case when we pick among several Z-Image takes of one scene. Best-of-N on HPS is therefore a mild push toward glossy stock imagery.

The safeguards follow from this. Normalise scores per creator and scene, use them as one feature among several, and check that selected takes do not drift from the creator's grade. That fits `review.reject_only`: these judges rank and never kill. "Q-Judger" and "Qwen-Image-Bench" could not be found under those names.

**Identity needs a face embedding, and the default one is a licence trap.** InsightFace's pretrained models (buffalo_l, antelopev2) are **non-commercial research only**. AuraFace is Apache-2.0 and trained on commercially available data ([InsightFace README](https://raw.githubusercontent.com/deepinsight/insightface/HEAD/README.md); [AuraFace card](https://huggingface.co/fal/AuraFace-v1/blob/main/README.md)). Pairwise or decomposed criteria also beat holistic VLM scores, which suffer "self-consistency collapse" ([arXiv 2602.06013](https://arxiv.org/pdf/2602.06013v1)).

## Audiovisual judging: deterministic signals carry the timeline, an omni model reads intent

**Nemotron 3 Nano Omni is the best self-hostable whole-video judge.** NVIDIA's 31B-total, 3B-active hybrid MoE takes video up to 2 minutes, plus audio and text. It outputs text or JSON. Its **FP8 checkpoint runs on one L40S 48 GB**, it serves through vLLM, and it is licensed for commercial use under the NVIDIA Open Model Agreement ([HF card](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8)). NVIDIA reports WorldSense 55.4 and DailyOmni 74.1, slightly above Qwen3-Omni ([HF blog](https://huggingface.co/blog/nvidia/nemotron-3-nano-omni-multimodal-intelligence)).

It has three limits. It is English-only. It samples at most 1-2 FPS (and only about 32 frames by default in vLLM), which is too coarse for lip closure or micro-expressions. And it documents no video timestamp output. The card also asks users to hold the rights to input videos showing people, which our synthetic creators satisfy.

**The alternatives are bulkier or hosted.**

- *Qwen3-Omni-30B-A3B* is Apache-2.0 but needs **69-108 GB in BF16** for 15-60 s of video ([Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni)).
- *Qwen3.5-Omni-Flash* is API-first. It inserts explicit timestamps between audio and video tokens and produces scene-segmented script captions ([arXiv 2604.15804](https://arxiv.org/abs/2604.15804)). It accepts over 400 s of 720p input at CNY 3 per million video tokens and CNY 22.48 per million audio tokens ([Alibaba Model Studio](https://help.aliyun.com/en/model-studio/qwen3-5-omni-flash)). That is likely the best route to real timestamps, at fractions of a cent per clip, though tokens per second were not published.
- *Twelve Labs Pegasus 1.6* costs **$1.75 per video-hour (about $0.029/min)** plus output tokens ([list price](https://www.twelvelabs.io/list-price)). Its self-serve terms license customer data, including outputs, for training **unless you opt out in account settings**, and the licence survives termination ([Terms of Use](https://twelvelabs.io/terms-of-use)). It would also be a new processor under `data_policy`.

All the omni models sit within 1-2 benchmark points of each other on multiple-choice AV QA. No one has measured performance description or timestamp accuracy, so choose on licence, VRAM and duration, then calibrate on our own takes. The known-bad clips from the 2026-11-26 audit are the ideal negative set.

**The timeline should come from specialists, and their licences split cleanly.**

- *emotion2vec+ large* gives 50 Hz frame-level vocal emotion features across nine classes ([HF](https://huggingface.co/emotion2vec/emotion2vec_plus_large)). It uses the FunASR model licence, which requires attribution and is silent on commercial use ([FunASR MODEL_LICENSE](https://raw.githubusercontent.com/modelscope/FunASR/main/MODEL_LICENSE)).
- *Lip-sync scoring is less clean than it looks.* `syncnet_python`'s code is MIT ([LICENSE](https://raw.githubusercontent.com/joonson/syncnet_python/master/LICENSE.md)), but the Oxford VGG weights are offered "for research purposes", and the code licence does not cover them ([syncnet-python PyPI](https://pypi.org/project/syncnet-python/)). LatentSync's `stable_syncnet.pt` sits in an OpenRAIL++-tagged repo and is the more licensable scorer ([HF LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6)). Its text must be read before relying on it.
- *OpenFace 3.0* (landmarks, action units, gaze) is **"Noncommercial Research Use Only"** ([LICENSE](https://raw.githubusercontent.com/CMU-MultiComp-Lab/OpenFace-3.0/main/LICENSE)). *LibreFace* is under a USC non-commercial research licence ([GitHub](https://github.com/ihp-lab/LibreFace)). *Audio Flamingo 3*'s checkpoints are non-commercial ([HF](https://huggingface.co/nvidia/audio-flamingo-3)).

Facial action units therefore have no verified commercial tool yet. MediaPipe and py-feat are the candidates to check. The working design anchors every omni observation to Whisper word timestamps and `body.talk_beats`: the model labels pre-cut segments instead of inventing times, and its agreement with SyncNet offset, the emotion2vec trajectory and pitch/energy serves as a reliability score.

## Open writers reach Claude's band but not beyond it

**On creative writing, GLM-5.3 and Kimi K3.1 Pro are peers of Opus 5.5, and only GPT-6.2 Astra and Anthropic's Fable 5.x clearly beat it.**

| Board | Leaders | Where Claude and the open models sit |
|---|---|---|
| Lech Mazur story benchmark (67 models, updated 2026-11-17) | GPT-6.2 Astra 4.302; Claude Fable 5.2 3.564; Qwen 4 Max 3.548 | Opus 5.5 #6 (3.355); GLM-5.3 #9 (2.613), top open-weight; Kimi K3.1 Pro #11 (2.366) ([lechmazur/writing](https://github.com/lechmazur/writing)) |
| EQ-Bench Creative Writing v3, later snapshot | GPT-6 Astra; Kimi K3.1 Pro #2 | Opus 5.5 #8, below both Kimi K3.1 Pro and GLM-5.3 ([Sophon](https://sophon.at/evals/eqbench-creative-writing)) |
| arena.ai Creative Writing (human votes, Aug 2026) | Claude Fable 5 #1 | Kimi falls to #23 ([Digital Applied](https://www.digitalapplied.com/blog/which-ai-model-writes-best-leaderboards-disagree)) |

The boards disagree with each other. EQ-Bench uses Claude judges with no self-preference control ([Digital Applied](https://www.digitalapplied.com/blog/which-ai-model-writes-best-leaderboards-disagree)), so no board settles the question for 20-second spoken monologues. Two findings matter for baselines. Opus 5 scores above Opus 5.5 on both LLM-judged boards, so it belongs in any bake-off. And the ceiling inside the vendor we already use is Fable 5.2 ($8/$40 per million tokens per our earlier pricing notes: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)), worth trying on low-volume, high-stakes text such as dossier voice sections.

**Price separates the open candidates more than quality does.**

- *GLM-5.3* on DeepInfra costs **$0.5625 in / $2.50 out** per million tokens, about 40% of Z.ai's own price ([DeepInfra](https://deepinfra.com/pricing); [Z.ai](https://docs.z.ai/guides/overview/pricing.md)). A typical 2,000-in / 500-out showrunner call costs about $0.0024, against roughly $0.009 on Sonnet 5.5 at $2/$10.
- *Kimi K3.1 Pro* on DeepInfra ($1.90/$9.50) costs about $0.0086 for the same call. That is Sonnet's price, so Kimi buys voice diversity and humour, not savings.
- *Qwen 4 Max* is the surprise. It ranks #3 on Mazur and costs $1.50/$4.50 on Together ([Together pricing](https://www.together.ai/pricing)). It also won the studio's own small style-director routing trial. Its licence and open-weight status are unverified.
- *Grok 4.7* ($2/$6, proprietary) tops no prose board, but the Grok line leads one joke benchmark ([comedeez](https://comedeez.ai/joke-bench)). Kimi sits #2 on two humour boards ([memelord](https://www.memelord.com/humorbench); [HumorRank](https://arxiv.org/abs/2604.19786)). A humour probe of these two is cheap.

All of these fit `llm.hosts` as OpenAI-compatible endpoints.

**Character embodiment depends more on how the model is prompted than on which model it is.** Practitioner guidance for Kimi K3 recommends:

- a short system prompt built as a character card (character, personality, speaking style, scenario, instructions);
- low reasoning effort for short replies;
- temperature 0.6-0.7 ([MegaNova](https://blog.meganova.ai/how-to-get-kimi-k3-to-write-whatever-you-want/)).

Below about 0.7, models "converge on their median style", and sampling several continuations near 0.9 restores variety ([Featherless](https://featherless.ai/blog/best-ai-for-creative-writing-2026)). Thinking mode is costly here. Kimi K3.1 Pro took 105 s for one short piece ([Overchat](https://overchat.ai/ai-hub/kimi-k3-1-pro)).

That points to one approach for every model. The dossier's voice and performance sections serve as the card. Generate N samples at about 0.9 with reasoning off, and let a non-Claude judge plus the existing evaluator choose, as `mw llm judge` already does. No benchmark measures in-character consistency across a series, which is our real risk.

The local fine-tunes are mostly traps:

- *Hemmingway-1* scores well (EQ-Bench #17) but is **CC BY-NC** ([VDF AI](https://vdf.ai/blog/best-local-llm-for-writing/)).
- *PaintedFantasy v4* is MIT-tagged but built for "uncensored... RP / ERP" on Reddit NSFW prompts, WildChat and GLM-4.7 rewrites with no dataset licences stated ([HF card](https://huggingface.co/zerofata/MS3.2-PaintedFantasy-v4-24B)). That conflicts with our rules on simulated intimacy and gratification-first content, on top of the provenance risk.
- *Muse Glimmer 30B* (Apache-2.0 plus a Meta usage policy, about 17 GB at Q4) and *Qwen3.8-27B* (Apache-2.0) are the clean self-host floors ([Muse Glimmer card](https://huggingface.co/meta-models/Muse-Glimmer-30B); [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B)). Both rank well below the hosted leaders.

## Hosting defaults to serverless, and the licence ledger needs fourteen entries

**Self-hosting a language or vision model only pays when the model rides on a machine that is already warm.** A single L40S running Qwen3-32B AWQ reaches about 333 tokens/s at high concurrency ([Wojdylak benchmarks](https://michalwojdylak.com/blog/llm-inference-benchmarks-vllm-sglang-tensorrt-llm)). At an assumed $0.80/h, that is about $0.67 per million output tokens at full use, against $0.28 on DeepInfra ([DeepInfra](https://deepinfra.com/pricing)). At the 24% GPU utilisation the studio has measured, the effective self-host cost is roughly ten times hosted.

The rule follows. Use hosted serverless for text and VL roles: DeepInfra or Together, and Alibaba for Omni. Self-host only models no host serves (ShotVL, HPSv3, UnifiedReward, Nemotron Omni FP8, the measurement stack), and run them as a `judge` family that batches a post's candidates on a warm card, or that co-resides with generation. The capability card's all-in $ per accepted output should make each call, not hourly price.

**Data terms differ by route, and OpenRouter can enforce them.** DeepInfra and Fireworks are described as no-retention, no-training by default. Together's status is disputed ([Spheron](https://www.spheron.network/blog/zero-data-retention-for-llm-inference-what-a-fixed-plan-guar/); [Eden AI](https://www.edenai.co/docs/v3/data-governance/provider-data-policies)). OpenRouter's `provider` settings for strict-JSON roles should be `zdr: true`, `data_collection: "deny"` and `require_parameters: true`. Without the last one, OpenRouter treats `response_format` as a soft preference ([OpenRouter provider selection](https://openrouter.ai/docs/guides/routing/provider-selection); [OpenRouter ZDR](https://openrouter.ai/docs/guides/privacy/zdr)).

No US rule bars commercial use of Chinese open models. Zhipu has been on the Entity List since January 2025, which matters for export-controlled technical data, not creative briefs. Restrictions on Kimi and Qwen have been floated but not found enacted ([VentureBeat](https://venturebeat.com/security/deepseek-v4-1-flash-1017-arrives-with-improved-benchmarks-just-as-the-u-s-government-bans-it-from-agencies-what-enterprises-should-do)). That argues for US-hosted GLM and a non-Chinese fallback on every role. Moonshot's K2 line closed to new users on 31 Oct 2026 and retires by end of December ([Kimi pricing](https://platform.kimi.ai/docs/pricing/chat)).

The licence ledger below covers every trap found. Entries marked unverified need the LICENSE text read and its hash recorded at image build, as is already done for the voice models.

| Model or tool | Trap | Action |
|---|---|---|
| Hemmingway-1 / 1.5 | CC BY-NC | Do not use |
| PaintedFantasy v4, Cydonia and other RP merges | Undeclared scraped and distilled data; tuned for ERP | Do not use |
| MolmoPoint-8B, Molmo2 | Apache weights, but research-use framing and non-commercial training data | Research only, or counsel first |
| OpenFace 3.0, LibreFace | Non-commercial research licences | Do not use, even for internal benchmarking |
| Sapiens pose; DA3 nested giant | CC BY-NC | Use RTMPose/ViTPose and DA3-BASE |
| Audio Flamingo 3 | Non-commercial checkpoints | Use Qwen3-Omni-Captioner (Apache) |
| InsightFace pretrained models | Non-commercial without a paid licence | Use AuraFace |
| SyncNet (VGG weights) | MIT code, research-only weights | Prefer LatentSync's OpenRAIL++ scorer after reading it |
| CineTechBench data | CC BY-NC-ND | Internal evaluation reference only |
| LTX-2.5 | Free under $10M revenue; must not remove watermark or provenance; no training competitors | Record the threshold; check re-encoding against the provenance clause |
| Kimi K3 / K3.1 Pro | Display duty above 100M MAU or $20M monthly revenue; MaaS above $20M per 12 months ([Kimi-K3 LICENSE](https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/LICENSE)) | Not binding now; K3.1 Pro text unread |
| GLM-5.3; Qwen3.8-Max | Security review above $10B revenue; display duty and inference-service licence at scale ([SQ Magazine](https://sqmagazine.co.uk/qwen3-8-open-weights-two-licenses/)) | Not binding; read LICENSE |
| HY-Motion 1.0 + SMPL-H | Tencent Hunyuan Community terms unread; SMPL commercial terms | Hold until both are read |
| Twelve Labs (self-serve) | Trains on inputs and outputs unless you opt out; licence survives termination | Opt out first; add as a processor |

Nemotron's NVIDIA Open Model Agreement and the emotion2vec FunASR licence permit commercial use or are silent about it. Attribution duties apply to both.

## Conclusion

The research changes where the studio should look for creative quality. The instinct is to buy a better writer or a bigger judge, but every frontier writer is within noise of what Claude already gives us, and every open judge is biased toward the glossy look our creators avoid. The compounding gain lies in the boundary between the director and the generator. A shot sheet should leave the director as measured facts and control signals in each model's native format, not as prose the generator has to guess at. That also makes the judging cheaper, because a clip rendered from an explicit pose and camera path can be checked against that same plan by code and by ShotVL, instead of being judged afresh by an expensive VLM.

The second takeaway is about licences. The strongest specialist tools in vision and face analysis come out of academic labs and are disproportionately non-commercial, and the permissive alternatives are often a generation older or less accurate. The studio's habit of refusing a model until its licence is read is therefore a creative constraint, not just legal hygiene, and it should be encoded. Each worker image should record the licence it ships under, and policy should hold the revenue thresholds (LTX at $10M, Kimi and Qwen display duties) so that crossing one becomes an owner event rather than a surprise.
