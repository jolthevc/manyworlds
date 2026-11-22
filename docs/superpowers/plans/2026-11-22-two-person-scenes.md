# Two-person scenes with InfiniteTalk multi-person (2026-11-22, wave 3 item 12)

Status: **design only, off.** The only artefact is the policy block `two_person` (`enabled: false`). There are no
production, direction or worker changes. A build starts only on the owner's go, after the single-speaker talk
benchmark has passed.

## Rules that constrain it
- **Both people are ours.** The second face is either another of our creators or a one-off original person made the same way: adult (`creator_min_age`) and never based on a real person.
- **No cross-account engagement.**
  - If both faces are portfolio creators, the post goes on one account only.
  - The other creator never comments on it, tags it or reposts it. Platforms ban coordinated networks.
  - The simplest rule: the second person is a recurring side character of the posting creator, not another account's creator.
- **Disclosure.** Unchanged: the AI label and the disclosure line.

## What the wrapper offers (read at our pin `a7439db`, `multitalk/nodes.py`, 2026-11-22)

### `MultiTalkWav2VecEmbeds`
- **Required inputs:**
  - `wav2vec_model`
  - `audio_1`
  - `normalize_loudness`
  - `num_frames`
  - `fps`
  - `audio_scale`
  - `audio_cfg_scale`
  - `multi_audio_type`
- **Optional inputs:**
  - `audio_2`, `audio_3`, `audio_4`.
  - `ref_target_masks`, with the tooltip: *"Per-speaker semantic mask(s) in pixel space. Supply one mask per speaker (plus optional background) to guide mouth assignment"*.
  - `add_noise_floor` and `smooth_transients`.
- **`multi_audio_type`:**
  - `para`: "overlay speakers in parallel". Each track is zero-padded to the longest.
  - `add`: "concatenate sequentially". Speaker k's embeddings sit at their own offset on one timeline, with silence for the others. For turn-taking, `add` is the natural fit.
- **How masks are used.**
  - Masks travel inside the embeds (`ref_target_masks`).
  - The model resizes them to the latent token grid (`interpolate(..., size=(H//2, W//2), mode='nearest')`, then `> 0`, in `wanvideo/modules/model.py`).
  - A mask is binary after resizing. Soft edges do nothing.
- **Binding.**
  - `human_num` is the number of audio tracks (`len(audio_embedding)` in `nodes_sampler.py`).
  - The attention code has a dedicated branch for exactly 2 speakers (`if human_num == 2`).
  - audio_N binds to mask N, in order. Nothing checks that the mask count matches the speaker count.
- **Adapter.** Multi-person needs the **Multi** adapter: `InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors` in `Kijai/WanVideo_comfy_fp8_scaled` (listed from the HF API). Our talk graph loads the *Single* adapter (`comfy.WEIGHTS`), so a two-shot needs one more file, about the size of the Single adapter.
- **Core alternative.** ComfyUI core v0.45.0 has its own `WanInfiniteTalkToVideo` (`two_speakers`, `mask_1`, `mask_2`). We stay on the wrapper, which our talk graph and benchmarks already use.

## Graph outline (API format, a diff on `comfy._talk_graph`)

```
adapter:   MultiTalkModelLoader(model = InfiniteTalk-Multi fp8)        # was Single
audio_1:   LoadAudio(speaker1.wav)    audio_2: LoadAudio(speaker2.wav)
mask_1:    LoadImage(mask_1.png)      mask_2:  LoadImage(mask_2.png)   # white = that speaker's face region
masks_img: ImageBatch(image1=mask_1, image2=mask_2)                     # core; [2, H, W, 3]
masks:     ImageToMask(image=masks_img, channel=red)                    # core; MASK [2, H, W]
embeds:    MultiTalkWav2VecEmbeds(audio_1, audio_2, ref_target_masks=masks,
                                  multi_audio_type=add, num_frames=sum of turns, ...)
video:     CreateVideo(images, fps 25, audio = embeds output 1)       # the combined track the node returns
```

- `ImageBatch` and `ImageToMask` exist in core v0.45.0 (`nodes.py`, `comfy_extras/nodes_mask.py`). `ImageBatch` is marked deprecated but still registered.
- With `add`, the node returns the concatenated waveform as its `audio` output (RETURN_NAMES `multitalk_embeds, audio, num_frames`). Wire that into `CreateVideo`, so picture and sound share one timeline.
- 480×832 only. Official MultiTalk inference is 480p, and our talk profile is 480p.

## Mask generation from face boxes
1. **Source image.** The two-shot still is made by the image pipeline with both people in frame, faces apart, both facing roughly toward the camera.
2. **Face boxes.** Detect faces with MediaPipe's face detector or landmarker (`num_faces=2`) on the worker. MediaPipe is already in the talkcomfy image for the lip checks.
   - Refuse the still unless exactly 2 faces are found.
   - Each face must be at least N% of the frame width.
3. **Speaker order.** Speaker 1 is the leftmost face (`two_person.speaker_order: left_to_right`, matching fal's LongCat multi convention, "person 1 left").
4. **Masks.**
   - Each face box grows by `masks.dilate` (25% of the box size) to cover the jaw and chin.
   - Clip each mask to the frame and to the midline between the two faces.
   - **Never touching:** shrink both masks until the gap is at least `masks.min_gap_px` (16 px at 480 wide). Touching or overlapping masks degrade to single-speaker behaviour.
   - Write each mask as a white-on-black PNG at the output size.
5. **Background mask.** None by default (`masks.background: false`). The tooltip allows "plus optional background"; try it only if both mouths move.

## Speaker turns from the shot sheet
- **The gap.** `performance.lines` has no speaker today; every line is the creator's.
- **The change (not built):**
  - Add `Line.speaker: Literal["a", "b"] = "a"`, with `a` as the creator.
  - Add `ShotSheet.cast: list[CastMember]`: `{id: "b", name, description, voice}`.
  - The director may write `b` lines only when `two_person.enabled`. The schema stays backward compatible, because `a` is the default.
- **Mapping to audio_N:**
  - "a" and "b" are mapped to left and right by the still's composition. The director's `blocking` says who stands where. The mask step's face order is the truth, and the still prompt places "a" on the left.
- **Building the tracks:**
  - Voice each line in its speaker's designed voice (`voice.speak`, one take per line, as today).
  - For `add`, concatenate speaker k's lines into `audio_k` in turn order.
  - The node lays them on one timeline in **track order, not turn order**. All of audio_1 comes before audio_2.
  - So with `add`, a dialogue that alternates (a, b, a) must be split into **one segment per turn**: each segment is a single-speaker `add` call with the other track silent. Or use `para`, with each track padded with silence to its turn times.
  - **Recommendation:** use `para`, with each speaker's track built on the full timeline: their turns at their times, silence elsewhere, the same length for both. That keeps arbitrary turn order in one render. Use `add` only for exactly two turns (a then b).
  - The policy default is `add`. Change it to `para` once the track builder exists. A benchmark arm compares them.
- **Pauses.** Keep the existing line pauses (`pause_after_s`) between turns. A turn-taking gap of 200–400 ms reads natural.

## Failure modes and gates

| Failure | Cause | Gate / mitigation |
|---|---|---|
| Wrong mouth: B moves to A's line | the "incorrect binding problem" (MultiTalk paper); swapped track order; masks swapped left/right | **Per-face lip correlation:** see below. Fail the clip; never ship |
| Both mouths move | masks touching or overlapping; a missing second mask; background bleed | `max_both_moving_share`; mask gap rule; try a background mask |
| Neither mouth moves on a turn | audio_scale too low for the quieter track; `normalize_loudness` levels them to -23 LUFS, so keep it on | own-correlation floor per turn |
| Identity swap or blend between the two faces | shared attention | video gate identity row per face (crop by face box), both people against their references |
| Drift over long clips | windowed I2V | the same segmenting as long talks (`talk_segments.py`), one turn per segment when possible |

**Wrong-mouth check (per-face lip correlation).** Run this on the worker, where the lip checks already run.

1. For each frame, find both faces' inner-lip opening: MediaPipe landmarks 13/14 over mouth width 78/308, the same ratio as `finishing.mediapipe_face`.
2. For each speaker k, compute the audio energy envelope of track k (RMS per video frame).
3. During speaker k's turns:
   - `own = corr(mouth_k, energy_k)` must be at least `gate.min_own_correlation` (0.3).
   - The silent face's motion over the speaking face's motion (std of mouth opening) must be at most `gate.max_cross_ratio` (0.6).
4. Across all single-speaker frames, the share of frames where both mouths open past a threshold must be at most `gate.max_both_moving_share` (0.2).
5. Record the scores on the review row. A clip that fails is never shipped. Its post reroutes to a single-speaker version (speaker a's lines only, as a normal talking video) or drops.

The thresholds are starting values, to be calibrated on the benchmark. A swapped-binding clip is made on purpose
(tracks swapped), so the test is known to fail it.

## Build steps (after the owner's go)
1. Pure mask builder (`two_person.masks_from_boxes(boxes, w, h, policy)`) and track builder (`para` timeline from turns), with CI tests.
2. Graph `comfy.graph_two_person(...)` as a diff of `_talk_graph`, with a CI test, plus `missing_inputs` in the smoke test.
3. The lip-correlation gate in `review/video.py`, worker side. Test it with a synthetic swapped clip.
4. Shot sheet `speaker` and `cast`, behind `two_person.enabled`.
5. Benchmark: 10 two-shots × {`para`, `add`} on an L40S. Measure gate pass rate, wrong-mouth rate and cost per accepted clip.
6. Optional: a bake-off against LongCat-Video-Avatar 1.5 multi (MIT, `para`/`add`; the page does not say how a track binds to a face).
