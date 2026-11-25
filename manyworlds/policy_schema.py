"""The policy file's schema for its hot sections (queue, gpu_runs, autoscale, review, production, finishing,
generation): the sections whose numbers decide what machines run, what is judged and what is made.

`Policy.get` falls back to a code default when a key is missing, so a misspelt key in the YAML (or one dropped from
it) silently runs on whatever the code says. Each model here forbids unknown keys at the levels it defines, so a
typo fails validation instead. Free-form maps (per-family numbers, per-kind settings, looks, talk profiles, job
parameters handed to workers as they are) stay `dict`s.

Adding a key to one of these sections means adding its field here too (tests/test_policy_schema.py fails until
then). Sections not named here are not validated.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

Num = float                     # ints are accepted where a float is declared
NumMap = dict[str, float]       # {family|kind|provider: number}
Free = dict[str, Any]           # free-form block (passed through to a worker, or keyed by names)

SECTIONS = ("queue", "gpu_runs", "autoscale", "review", "production", "finishing", "generation")


class PolicyInvalid(ValueError):
    """The policy body does not match the schema of its hot sections."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- queue -----------------------------------------------------------------------------------------------------------

class Queue(_Strict):
    lease_seconds: Num
    max_attempts: int
    worker_max_consecutive_faults: int
    job_max_worker_faults: int
    connect_timeout_seconds: Num
    startup_connect_attempts: int
    retry_backoff_seconds: Num


# --- gpu_runs --------------------------------------------------------------------------------------------------------

class GpuRuns(_Strict):
    default_max_hours: Num
    default_max_usd: Num
    max_hours_by_family: NumMap
    max_usd_by_family: NumMap
    max_hours_limit: Num
    idle_stop_minutes: Num
    idle_stop_minutes_by_provider: NumMap
    idle_stop_minutes_no_demand: Num | None = None
    boot_grace_minutes: Num
    boot_grace_minutes_by_provider: NumMap
    stuck_minutes: Num
    max_attempt_minutes_by_family: NumMap
    max_attempt_minutes_by_kind: NumMap
    daily_gpu_usd: Num
    daily_drain_grace_minutes: Num | None = None
    daily_drain_max_overshoot_usd: Num | None = None
    serverless_idle_exit_seconds: Num


# --- autoscale -------------------------------------------------------------------------------------------------------

class AutoscaleScale(_Strict):
    drain_target_minutes: NumMap
    history_days: Num
    default_job_seconds: NumMap


class AutoscaleBatch(_Strict):
    min_work_minutes: NumMap
    max_wait_minutes: NumMap
    urgent_priority: int


class NoOutputGuard(_Strict):
    machines: int
    hours: Num


class AutoscaleVast(_Strict):
    min_reliability: Num
    verified_only: bool
    offers_to_try: int
    min_inet_down_mbps: Num
    exclude_countries: list[str]
    max_inet_down_usd_per_gb: Num
    expected_hours: Num
    candidates: int


class AutoscaleScoring(_Strict):
    enabled: bool
    window_days: Num
    min_outputs: Num | NumMap      # a number or {family: number, default: n}
    trial_usd: Num | NumMap
    pause_after_failed_machines: int
    pause_hours: Num


class AutoscaleWarmReuse(_Strict):
    enabled: bool
    min_remaining_minutes: Num
    max_spend_share: Num


class AutoscaleCapacity(_Strict):
    report_after_minutes: Num
    fallback_after_minutes: Num


class AutoscaleWarmHold(_Strict):
    enabled: bool
    max_hold_minutes: NumMap
    format_families: dict[str, list[str]]


class Autoscale(_Strict):
    tick_seconds: Num
    tick_lease_seconds: Num
    max_groups_per_kind: dict[str, int]
    scale: AutoscaleScale
    batch: AutoscaleBatch
    bench_machines: int
    hard_pin_batches: list[str]
    warm_reuse: AutoscaleWarmReuse
    capacity: AutoscaleCapacity
    warm_hold: AutoscaleWarmHold | None = None
    relaunch_cooldown_minutes: Num
    no_output_guard: NoOutputGuard
    runpod_cuda_versions: list[str]
    vast: AutoscaleVast
    reconcile_days: Num
    reconcile_every_minutes: Num
    reconcile_settle_hours: Num
    worker_self_stop: bool
    scoring: AutoscaleScoring
    # absent in the shipped file (every family on its own machines); the breadth test's overlay sets it
    merge_families: dict[str, Any] | None = None


# --- review ----------------------------------------------------------------------------------------------------------

class ReviewImage(_Strict):
    role: str
    effort: str
    confirm_role: str
    confirm_effort: str
    confirm_top_per_job: int
    clear_pass_margin: Num
    min_identity: Num
    min_realism: Num
    min_scene: Num
    min_brand_alignment: Num
    reference_images: int
    max_jobs_per_run: int
    loop_seconds: Num
    busy_loop_seconds: Num
    max_review_errors: int
    review_error_spacing_seconds: Num


class ReviewVideo(_Strict):
    role: str
    effort: str
    confirm_role: str
    confirm_effort: str
    keyframes: int
    min_identity: Num
    min_motion: Num
    min_realism: Num
    clear_pass_margin: Num
    min_scene: Num
    reference_images: int
    max_jobs_per_run: int
    max_repairs_per_post: int
    kinds: list[str]
    max_review_errors: int
    review_error_spacing_seconds: Num
    lip_sync: Free
    talk_repair: Free = {}
    intelligibility: Free


class ReviewDirection(_Strict):
    min_score: Num
    rubric: str
    enforce: bool | None = None     # the breadth test's overlay sets it (report-only there)


class ReviewDetail(_Strict):
    hands: Free
    teeth: Free
    lip_closure: Free
    background: Free
    consistency: Free
    crops: Free


class ReviewFinal(_Strict):
    enabled: bool
    enforce: bool
    role: str
    effort: str
    keyframes: int
    join_offset_seconds: Num
    max_joins: int
    max_text_frames: int
    max_spans_listed: int
    end_window_seconds: Num
    end_margin_seconds: Num
    min_gap_seconds: Num
    max_frames: int
    frame_height: int
    spoken_kinds: list[str]
    min_seams: Num
    min_end_sync: Num
    min_phone_real: Num
    max_end_drift_seconds: Num
    timeout_seconds: Num


class ReviewSync(_Strict):
    enabled: bool
    enforce: bool
    formats: list[str]
    sample_fps: Num
    max_frames: int
    frame_height: int
    audio_rate: int
    max_lag_ms: Num
    min_frames: int
    min_corr: Num
    min_face_share: Num
    speech_db_below_p95: Num
    open_min: Num
    open_relative_to_p90: Num
    closed_max: Num
    closed_relative_to_p90: Num
    max_silent_mouth_moving: Num
    max_talking_mouth_closed: Num
    landmarker_path: str
    timeout_seconds: Num


class ReviewRepair(_Strict):
    lora_scale_step: Num
    lora_scale_max: Num
    drop_on_safety: bool
    hints: Free
    stand_ins: dict[str, str]


class Review(_Strict):
    image: ReviewImage
    video: ReviewVideo
    direction: ReviewDirection
    detail: ReviewDetail
    final: ReviewFinal
    sync: ReviewSync
    max_repairs_per_post: int
    repair: ReviewRepair


# --- production ------------------------------------------------------------------------------------------------------

class RerouteCaption(_Strict):
    enabled: bool
    effort: Literal["low", "medium", "high"]


class PullForward(_Strict):
    enabled: bool
    horizon_hours: Num
    low_water_jobs: int
    max_posts_per_tick: int
    min_remaining_minutes: Num
    max_spend_share: Num
    budget_share: Num


class Production(_Strict):
    image_kind: str
    formats: list[str]
    video_kind: str
    routing: dict[str, Free]
    talk_kind: str
    dance_kind: str
    candidates_per_scene: int
    lead_hours: Num
    loop_seconds: Num
    busy_loop_seconds: Num
    max_advance_errors: int
    post_lease_seconds: Num | None = None
    max_job_failures_per_post: int
    max_ready_unpublished: int
    reroute_failed_video: dict[str, str]
    pull_forward: PullForward | None = None   # warm machine about to idle: start the next planned posts early
    reroute_caption: RerouteCaption | None = None   # the rerouted post's caption rewritten for a still (core)
    edit: Free
    finish_mode: Literal["queue", "inline"]   # where the finish renders (production.finish)


# --- finishing -------------------------------------------------------------------------------------------------------

class FinishingGrain(_Strict):
    enabled: bool
    strength: Num
    max_strength: Num


class FinishingSharpen(_Strict):
    enabled: bool
    amount: Num
    size: int


class FinishingVignette(_Strict):
    enabled: bool
    angle: Num


class FinishingEncode(_Strict):
    preset: str
    crf: int
    audio_bitrate: str
    timeout_seconds: Num


class FinishingOutput(_Strict):
    width: int
    height: int
    flags: str


class FinishingUpscaler(_Strict):
    enabled: bool
    kind: str | None
    model: str | None
    scale: Num


class FinishingCover(_Strict):
    mediapipe: bool
    analysis_width: int
    eyes_open_min: Num
    mouth_closed_max: Num
    sample_every_s: Num
    edge_s: Num
    max_samples: int
    avoid_margin_s: Num


class Finishing(_Strict):
    enabled: bool
    formats: list[str]
    trim_head_s: Num
    max_failures: int
    on_give_up: Literal["hold", "ship"] = "hold"   # a finish given up: hold the post (fails closed) or ship it unfinished
    default_look: str
    look_by_creator: dict[str, str]
    look_by_set: dict[str, str]
    looks: dict[str, Free]
    limits: Free
    grain: FinishingGrain
    sharpen: FinishingSharpen
    vignette: FinishingVignette
    hook: Free
    loudness: Free
    encode: FinishingEncode
    output: FinishingOutput
    upscaler: FinishingUpscaler
    fps: Num
    interpolate: Free
    transitions: Free
    cover: FinishingCover


# --- generation ------------------------------------------------------------------------------------------------------

class ClipEdit(_Strict):
    enabled: bool
    max_clips: int


class Talk(_Strict):
    """Base talk job parameters (handed to the talk worker; talk_profiles and talk_motion.v2v merge over them)."""
    width: int
    height: int
    fit_stretch_tolerance: Num
    motion_interpolate: Free
    steps: int
    cfg: Num
    shift: Num
    scheduler: str
    audio_scale: Num
    audio_cfg: Num
    frame_window: int
    motion_frame: int
    swap_below_gb: Num
    blocks_to_swap: int
    timeout_s: Num
    max_take_seconds: Num
    voice: str
    model: str | None = None        # read by production (walk-and-talk); absent = infinitetalk


class TalkLong(_Strict):
    max_segment_seconds: Num
    words_per_second: Num
    join: str
    join_timeout_s: Num
    join_seconds_per_input_second: Num | None = None


class TalkHybrid(_Strict):
    enabled: bool
    segment_seconds: Num
    on_camera_share: Num
    output: Free


class TalkMotion(_Strict):
    enabled: bool
    trigger: Free
    max_motion_seconds: Num
    max_motion_chain: int
    chain_min_seconds: Num
    v2v_images: list[str]
    kind: str | None
    keyframes: int
    v2v: Free
    v2v_repair: Free | None = None


class PunchIn(_Strict):
    every_s: list[Num]
    scales: list[Num]
    min_shot_s: Num
    center: list[Num]


class TalkGrammar(_Strict):
    punch_in: PunchIn | None = None
    lip_sync: str                      # on_camera | all (manyworlds.cuts.narrates)
    narrated_angles: list[str] = []
    director: list[str] = []


class TalkEdit(_Strict):
    default: str
    by_archetype: dict[str, str]
    grammars: dict[str, TalkGrammar]


class TalkRoute(_Strict):
    """One way of rendering a talking take (production.talk.talk_routes)."""
    kind: str | None = None              # none: production.talk_kind
    graph: str | None = None             # a comfy_graphs entry (kind comfy.graph)
    params: Free = {}
    pools: list[str] | None = None
    formats: list[str] | None = None
    needs_voice_line: bool = False
    v2v: bool = True
    max_seconds: Num | None = None
    repair: Free = {}                    # over review.video.talk_repair (gates per gate, audio_guidance whole)


class Generation(_Strict):
    image: Free
    clip_edit: ClipEdit
    image_by_format: dict[str, Free]
    video: Free
    talk: Talk
    talk_profiles: dict[str, Free]
    comfy_video: Free
    hosted_video: Free
    talk_long: TalkLong
    talk_hybrid: TalkHybrid
    talk_motion: TalkMotion
    talk_edit: TalkEdit
    talk_route: list[str] = []
    talk_routes: dict[str, TalkRoute] = {}
    borrowed_motion: bool = False


# --- the policy ------------------------------------------------------------------------------------------------------

class PolicySchema(BaseModel):
    """The whole file: the hot sections are checked, every other section passes as it is."""
    model_config = ConfigDict(extra="allow")

    queue: Queue
    gpu_runs: GpuRuns
    autoscale: Autoscale
    review: Review
    production: Production
    finishing: Finishing
    generation: Generation


def errors(body: Any) -> list[str]:
    """What is wrong with `body`'s hot sections, one line per problem ("review.image.max_jobs_per_rn: Extra inputs
    are not permitted"); empty when it validates."""
    if not isinstance(body, dict):
        return [f"<root>: the policy must be a mapping, not {type(body).__name__}"]
    try:
        PolicySchema.model_validate(body)
    except ValidationError as e:
        return [f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors()]
    return []


def validate(body: Any) -> None:
    """Raise PolicyInvalid when `body`'s hot sections don't match the schema."""
    problems = errors(body)
    if problems:
        raise PolicyInvalid("policy does not match its schema (manyworlds.policy_schema):\n  " + "\n  ".join(problems))


def covers(path: str) -> bool:
    """Whether a dotted policy path falls in a section this schema checks."""
    return path.split(".")[0] in SECTIONS
