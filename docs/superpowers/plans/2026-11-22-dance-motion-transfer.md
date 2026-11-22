# Dance and motion transfer with Wan-Animate-2 (2026-11-22, wave 3 item 10)

Status: **off**. The policy block `dance` (`enabled: false`, `references: []`), the reference and identity rules
(`manyworlds/dance.py`) and a graph builder (`manyworlds/worker/comfy_animate.py`) are written and tested in CI.
Nothing else is built yet: no handler, job kind, worker image change or GPU run. All of these wait for the owner's go.

## Licence

- **Wan2.2-Animate-2-14B: Apache-2.0** on the official HF card (`Wan-AI/Wan2.2-Animate-2-14B`). Base and distilled
  weights were released on 2026-08-07. See the research note (`docs/research/2026-11-22-video-quality-wave2-3.md` §3).
- The ComfyUI repackage `Comfy-Org/Wan-Animate-2` holds the same weights in ComfyUI layout. Its files are listed below.
- NVIDIA's hosted copy adds the NVIDIA Open Model Licence for its endpoint only. We do not use that endpoint.

## ComfyUI version: our pin already has it

| Check | Result (read 2026-11-22) |
|---|---|
| PR | Comfy-Org/ComfyUI #15362, "feat: Support Wan-Animate2 (CORE-358)", merged 2026-08-07, merge commit `a464ac3` |
| Files it touched | `comfy/ldm/wan/model.py`, `comfy/ldm/wan/model_animate2.py` (new), `comfy/model_base.py`, `comfy/supported_models.py`, `comfy_extras/nodes_wan.py` |
| Our pin | `COMFYUI_REF=v0.45.0` (`docker/worker-talkcomfy.Dockerfile`). Release published 2026-11-10 |
| Contains the PR? | **Yes.** The GitHub compare `a464ac3...v0.45.0` returns `ahead`: 598 commits ahead, 0 behind |
| Nodes at v0.45.0 | `WanAnimate2ToVideo` and `WanAnimate2Cache`, both registered in `comfy_extras/nodes_wan.py` and marked `is_experimental` |

**No core upgrade is needed.** Any talkcomfy image built from the current Dockerfile pin has the nodes. We have not
checked which core earlier tags were built from: confirm with `/object_info` in the smoke test. A dance job needs
only the weights and a handler. Kijai's WanVideoWrapper has no Animate-2 support and is not involved.

### Node inputs at v0.45.0 (from the source)
- **`WanAnimate2ToVideo`**
  - Required: `positive`, `negative`, `vae`, `width` and `height` (step 16), `length` (step 4 from 1, so 1, 5, …, 81), `batch_size`, `video_frame_offset`.
  - Optional: `reference_image`, `pose_video`, `clip_vision_output`, `positive_pose` (a prompt for the pose branch that describes the motion), `clip_vision_output_pose`, `continue_motion` (to extend a clip), `pose_strength`, `pose_start_percent`, `pose_end_percent`, `reference_image_strength`.
  - Outputs: `positive`, `negative`, `latent`, `trim_latent`, `trim_image`, `video_frame_offset`.
- **`WanAnimate2Cache`**
  - Inputs: `model`, `device` (cpu or gpu), `dtype` (default, int8 or int4). The cache "roughly halves generation time".
  - Cost: about 12.5 GB of system RAM at 480×832 and 81 frames in bf16, less with int8. With context windows, each window is cached separately.

## Graph (`comfy_animate.graph`, one window of up to 81 frames)

This follows the official template `video_wan_animate2` (Comfy-Org/workflow_templates), flattened to API format.

```
UNETLoader(wan_animate_2_int8_convrot) → LoraLoaderModelOnly(lightx2v 4-step distill, 1.0)
  → WanAnimate2Cache(cpu, int8) → ModelSamplingSD3(shift 5) ─────────────┐
                                └→ BasicScheduler(simple, 6 steps) ─ sigmas │
CLIPLoader(umt5 fp8, wan) → CLIPTextEncode ×3 (prompt, negative, empty pose prompt)
LoadImage(creator still) → ImageScale(w,h, centre) → CLIPVisionEncode ─┐
LoadVideo(our driving clip) → GetVideoComponents → ImageScale → ImageFromBatch(first) → CLIPVisionEncode
WanAnimate2ToVideo(reference, pose video, both CLIP-vision outputs, pose prompt, strengths)
  → SamplerCustom(lcm, cfg 1) → TrimVideoLatent(trim_latent) → VAEDecode
  → CreateVideo(fps = driving clip's fps, NO audio) → SaveVideo(mp4/h264)
```

What we changed from the template:
- We dropped its loop, its context windows (`ContextWindowsManual`, 21 frames with overlap 8), its side-by-side preview stitch and its resize helper (`ResizeImageMaskNode`). Core `ImageScale` with a centre crop does the resizing instead.
- We never take the driving clip's audio. The template passes the driving video's audio into `CreateVideo`.
- We kept one template quirk as found: `BasicScheduler` reads the cached model before `ModelSamplingSD3`, so the sigmas come from the unshifted model.

### Weights (`Comfy-Org/Wan-Animate-2`, sizes from the HF API)

| Folder | File | Size |
|---|---|---|
| diffusion_models | `wan_animate_2_int8_convrot.safetensors` (template) | 16.7 GB |
| diffusion_models | `wan_animate_2_distill_int8_convrot.safetensors` (benchmark arm B) | 16.7 GB |
| loras | `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` | 0.74 GB |
| text_encoders | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.7 GB |
| clip_vision | `clip_vision_h.safetensors` | ~1.3 GB |
| vae | `Wan2_1_VAE_bf16.safetensors` | ~0.25 GB |

Arm A downloads about 26 GB. The bf16 checkpoints are 32.8 GB each and are not used.

## Rule: reference clips are ours (`manyworlds.dance.reference`)

- **Where references come from.** A driving clip must be an entry in `dance.references`. Each entry needs an `owner` in `allowed_owners` (`ours` or `licensed`), a licence string, a source, and a URI.
- **Blocked sources.** A source on a platform in `dance.blocked_sources` is refused, even if someone labels it "licensed". The list covers TikTok, Instagram, YouTube, Facebook, X, Snapchat, Triller and Likee.
- **Why.** Choreography can be protected separately from its music. A downloaded trend video is never a reference.
- **Allowed sources.**
  - Clips we shoot ourselves.
  - Our own renders.
  - CC0 motion capture rendered to video.
  - Paid stock, with the licence number in `licence`.
- **Audio.** Trend audio, if a post uses it at all, is added in the app from the platform's licensed library at post time. It is never baked into our file. The graph drops the reference's sound.
- **Disclosure.** Unchanged: every post carries the AI label and the disclosure line.

## Identity checks

Animate-2's card claims "strong identity preservation" but publishes no numbers, and we found no measured drift
data. Every dance clip therefore goes through the existing video gate (keyframes, judged against the creator's
reference photos), plus `dance.identity_check`:

- **Per keyframe:** the worst keyframe's identity score must be at least `dance.identity.min_score` (6, matching `review.video.min_identity`).
- **Mean:** the mean across keyframes must be at least `min_mean` (7). A whole-body dance shows the face small and moving, so the gate needs both the worst frame and the mean.
- **More keyframes:** use 8 keyframes rather than 4. Dance drift shows mid-clip, at spins and turns away from camera.
- **When it fails:** the first lever is `reference_image_strength` above 1.0. The node's tooltip says it "tightens it against drift". Next, try a closer framing of the reference image. A clip that still fails is dropped. There is no photo reroute, because a dance post without motion is not the same post.

## What a handler needs (not built)

1. **Handler.**
   - Write `handlers/comfy_dance.py`, kind `comfy.wan_animate2`, family `comfy`. It runs on the talkcomfy image, which already routes the `comfy` family to `comfy_video`, so add a kind branch in `runner.handlers_for`.
   - It uses `comfy.ensure_weights(models_dir, comfy_animate.WEIGHTS)`, `comfy.missing_inputs(graph, object_info)` and `comfy.run(graph)`.
   - It downloads the driving clip and runs `dance.reference(ref_id)` again on the worker.
2. **Frame rate.**
   - The output takes the driving clip's frame rate, so 81 frames at 30 fps last 2.7 s.
   - Decide by benchmark: resample driving clips to 16 fps (Wan's native rate, about 5 s per window) or keep 30 fps.
   - Clips longer than one window chain through `continue_motion` and `video_frame_offset`. Those are the node's own outputs; the template loops over them.
3. **Job kind.** Add `comfy.wan_animate2` to `job_kinds` with 48 GB pools: `runpod_secure_l40s_vol` and `runpod_secure_l40s`, then `runpod_secure_h100`. 24 GB cards need offload; they are unmeasured.
4. **Smoke test.** Add to `worker/smoke.py`: `missing_inputs(comfy_animate.graph(...), object_info)` must be empty on the built image. That catches any input-name drift for free.
5. **Production.**
   - Add a `dance` format, kept out of `production.formats` until the benchmark passes.
   - Driving clips get a `references` catalogue in R2.

## Benchmark plan (owner's go only; nothing is queued)

- **Arms (20 clips each, same 10 driving clips × 2 creators' stills):**
  - **A (template):** int8 base, lightx2v LoRA, 6 steps lcm, cache cpu/int8.
  - **B (card):** distilled int8, no LoRA, 10 steps, cfg 1.0.
  - **C:** A with `reference_image_strength` 1.3.
- **Size:** 480×832, 81 frames.
- **Machine:** one L40S (`runpod_secure_l40s_vol`), so weights are cached after the first boot.
- **Measures:**
  - Render seconds per clip.
  - Peak VRAM and RAM (the cache is in RAM).
  - Video-gate pass rate.
  - `identity_check` pass rate.
  - Hand and teeth checks.
  - Cost per accepted clip.
- **Pass bar:**
  - At least 60% of clips pass the gate, with no identity failure on the creator's face at the gate's keyframes.
  - Cost per accepted clip within 2× a story clip.

## Cost estimate (to check against the benchmark)

| Item | Estimate |
|---|---|
| L40S (RunPod secure) | $1.09/h ≈ $0.0003/s (policy `gpu_pools.runpod_secure_l40s`) |
| Boot + first download (~26 GB) | ~10–15 min ≈ $0.20–0.27 once per machine (the volume pool caches it) |
| Render, one 81-frame window at 480×832, 6 steps + cache | ~2–4 min [unverified: no published single-GPU figure] ≈ $0.04–0.07 |
| Benchmark (60 clips + boot + idle tail) | ~4 GPU-hours ≈ **$4–5**, within the $10/day cap |

## Open questions
- Single-GPU VRAM. The card's defaults target 8×A800, and 480p was tested on 2×A800. int8 plus the RAM cache should fit 48 GB, but this is unverified.
- The 16 fps versus 30 fps decision (see "Frame rate" above).
- Whether the experimental nodes change between core releases. Pin the image tag, and the smoke test catches any change.
