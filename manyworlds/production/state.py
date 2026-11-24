"""Production state in posts.body. A post's body holds its creative brief (what the planner, showrunner and director
wrote) and, beside it, what production keeps while making it: job ids per piece, repair and failure counters, the
chosen media and the finish's record. Live rows hold these keys as they are, so they stay where they are; this module
only declares them, one line each, and tests/test_production_state.py fails when production's code reads or writes a
body key declared nowhere here (a misspelt key would otherwise read as "never set" and silently reset a counter)."""

from __future__ import annotations

# Keys production itself writes (and reads back on the next tick).
STATE: dict[str, str] = {
    # scenes
    "scene_jobs": "per scene, the image job ids tried so far, oldest first",
    "repairs": "image repairs used (rejected scene images regenerated), across the post's scenes",
    "job_failures": "scene image jobs that failed on their machine (counted apart from repairs)",
    "repair_log": "each targeted image repair: scene, attempt, cause, action, defects",
    "images": "the chosen media, in order: {asset_id, uri} (the slides when a carousel has text slides)",
    "photos": "a carousel's chosen photos when text slides were placed around them (images then holds the slides)",
    "slides_error": "why the carousel's text slides could not be drawn (the photos ship without them)",
    # the follow-up (one video from the chosen photos: followups.FOLLOW_UPS)
    "edit_jobs": "the narrated edit's job ids",
    "photo_spans": "the chosen narrated edit's real per-photo spans [{start_s, end_s}] (the video's meta photo_spans)",
    "clip_jobs": "the story clip's job ids",
    "talk_jobs": "the one-take talking video's job ids",
    "clip_repairs": "follow-up video repairs used (rejected clips or takes remade)",
    "follow_up_failures": "follow-up, voice, motion and upscale jobs that failed on their machine",
    "clip_edit_jobs": "a clip edit's job ids, one list per clip",
    # talking video takes
    "talk_segments": "a talking video's takes' words ([] = one take, in talk_script or script)",
    "talk_script": "a directed one-take talking video's words",
    "talk_shots": "the shot each take belongs to (a directed post)",
    "talk_modes": "per take, 'talk' (lip-synced) or 'narrated' (a cutaway over a photo)",
    "talk_segment_jobs": "per take, its talk or cutaway job ids",
    # the designed voice
    "voice_jobs": "the one-take follow-up's voice.speak job ids",
    "voice_segment_jobs": "per take, its voice.speak job ids",
    "voice_repairs": "voice lines made again because the worker transcript shows the intelligibility gate would reject them (voice.gate_repairs)",
    # the song of a music_video post (production.song)
    "song_jobs": "the music.song job ids (the mix, vocal and instrumental stems)",
    "song_repairs": "songs remade because their objective check failed (music.check.enforce)",
    "song_media": "the chosen song by role: {mix, vocals, instrumental: {asset_id, uri, seconds}}; with the artist's voice (music.voice), mix and vocals are the converted ones and source_vocals the song's own stem",
    "song_voice_jobs": "the music.convert_voice job ids (the song's vocal stem in the artist's voice, remixed)",
    "song_voice_repairs": "conversions remade because they fell under the enforced voice gate (music.voice.gate)",
    "song_voice_skipped": "why the song kept its own voice (a conversion the policy refused)",
    "song_id": "the chosen song's songs row (manyworlds.songs.for_post; posts.song_id too)",
    # walk-and-talk motion clips
    "motion_jobs": "the one-take talking video's first motion clip job ids",
    "motion_segment_jobs": "per take, its first motion clip's job ids",
    "motion_chain_jobs": "the one-take video's later motion clips (chain links 1..n), job ids per link",
    "motion_segment_chain_jobs": "per take (keyed by its index as text), its later motion clips' job ids per link",
    "motion_repairs": "never incremented (a motion clip is not reviewed): the motion Step's repair counter",
    # the join of a long talking video's takes (production.talk.advance_join)
    "join_jobs": "the finish.join job ids (production.finish_mode queue: the takes are joined on the finisher)",
    "join_repairs": "never incremented (a join is not reviewed): the join Step's repair counter",
    "join_failures": "finish.join jobs that failed (production.max_job_failures_per_post of them drop the post)",
    "join_error": "the latest join job's error",
    # the finish
    "upscale_jobs": "the AI upscaler's job ids",
    "upscale_repairs": "never incremented (an upscale is not reviewed): the upscale Step's repair counter",
    "upscale_skipped": "why the upscale was given up (the finish's lanczos scales instead)",
    "video": "the post's finished video: {asset_id, uri}",
    "video_raw": "the made video the finish started from: {asset_id, uri}",
    "finish": "what the finish did: finished, look, captions, placement, passes",
    "cover": "the cover frame the finish picked: {asset_id, uri}",
    "sound": "what was mixed under the video, with licences and credits",
    "sound_error": "why the sound mix failed (the video shipped unmixed)",
    "final_review": "the finished file's final review (review.final), or the judge's error",
    "sync_review": "the finished file's CPU lip-sync score (review.sync, report-only), or why it failed",
    "finish_jobs": "the finish.video job ids (production.finish_mode queue: the finish runs on the finisher)",
    "finish_repairs": "never incremented (a finish is not reviewed): the finish Step's repair counter",
    "finish_failures": "finish attempts that raised, or finish.video jobs that failed",
    "finish_error": "the latest finish error",
    "finish_skipped": "the error after which the unfinished video shipped (finishing.on_give_up: ship)",
    "finish_held": "the error after which the finish was given up and the post held (finishing.on_give_up: hold)",
    # the outcome
    "drop_reason": "why the post was dropped (or, before a reroute, why its video failed)",
    "rerouted_from": "the video format a rerouted post was planned as (it shipped as a photo)",
    "reroute_reason": "the drop reason that sent the post to its reroute",
    "caption_before_reroute": "the caption written for the video, before its rewrite for the photo (reroute_caption)",
    "pulled_forward": "started early on a warm, nearly idle machine (production.pull_forward): when and on which machine",
    "reroute_caption_error": "why the rerouted post's caption rewrite failed (the video's caption shipped)",
    "advance_errors": "ticks in which advancing the post raised",
    "advance_error": "the latest of those errors",
}

# Per-index counters: a prefix plus the clip's or take's index (f"clip_repairs_{i}").
STATE_PREFIXES: dict[str, str] = {
    "clip_repairs_": "a clip edit's clip i: repairs used",
    "clip_failures_": "a clip edit's clip i: failed jobs",
    "segment_repairs_": "a long talking video's take i: repairs used",
}

# Brief fields production reads but never writes (the planner, showrunner, director or breadth driver wrote them).
BRIEF: dict[str, str] = {
    "scenes": "one text per scene photo",
    "shots": "the director's shot sheets, aligned with scenes",
    "script": "the post's spoken words (a narrated edit's script, a talking video's lines, a story clip's motion)",
    "on_screen_text": "a narrated edit's on-screen text",
    "slides": "a carousel's text slides (manyworlds.slides)",
    "song": "a music_video post's song (manyworlds.music spec: title, lyrics, tags, bpm, key, duration_s)",
    "template": "the shot-sheet archetype the caption style is looked up by",
    "breadth_run": "marks a breadth-test post, advanced only by its own driver",
    "shared_key": "posts with the same key share scene photos, voice lines and seeds (a route comparison)",
}

STATE_KEYS = frozenset(STATE)
BRIEF_KEYS = frozenset(BRIEF)


def declared(key: str) -> bool:
    """Whether a body key is production state or a brief field production reads."""
    return key in STATE_KEYS or key in BRIEF_KEYS or any(key.startswith(p) for p in STATE_PREFIXES)
