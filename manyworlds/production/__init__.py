"""Production: turn planned posts into finished, reviewed images. No Claude calls here
(the producer loop also runs brain.planner.ensure_plans, which does).

Each tick (`advance`):
  planned   -> producing  when the post is due within production.lead_hours: one image job
                          per scene (the creator's full appearance + LoRA, see lab.photo_prompt)
  producing -> ready      when every scene has an image that passed both review stages
                          (manyworlds.review.gates); the best one per scene is chosen
  producing -> dropped    when a scene used up review.max_repairs_per_post regenerations

A scene whose job failed, or whose images were all rejected, gets a new job with new seeds.
Narrated edits and story clips are made from the chosen photos once all are chosen; a story clip
must also pass the video review gate (manyworlds.review.video), with review.video.max_repairs_per_post
new seeds before the post is dropped.
A creator with a designed voice (voices.engine: voxcpm) has every spoken piece (a talk take or segment, a cutaway,
a narrated edit's script) voiced first by a voice.speak job; the piece's job starts once that is done, with its wav
as `audio_uri`. A Kokoro creator's pieces are voiced on their own workers, as before.
Walk-and-talk (generation.talk_motion): a talking take whose shot moves (a walking or standing-up beat, a camera
move) gets a silent motion clip first, from the take's photo on the story-clip path, alongside its voice line; the
talk job then lip-syncs that clip (`video_uri`, InfiniteTalk video-to-video, only on the ComfyUI talk engine). A take
longer than one motion clip (max_motion_seconds) gets a chain of them, each from the last frame of the one before,
joined here before the lip sync, up to max_motion_chain; past that the take ends on a hold. Production waits for no
review of the motion clips; the take is judged.
Dance and motion transfer (policy `dance`): a shot with `motion_reference` (a dance.references id) is rendered by
Wan-Animate-2 (production.dance_kind) from that shot's photo while dance.enabled: a story clip's shot becomes the
animate clip (reviewed like any clip), and a talking take's shot becomes the silent motion clip the talk job lip-syncs
onto (the walk-and-talk path, V2V).
Joins: a directed post (body.shots) is cut only where its shots change, one take per shot that speaks (a take split
only past generation.talk.max_take_seconds); each join between shots is the next shot's `edit.transition_in` mapped
by direction.transition_kind (cut, match, whip, blur, flash) and rendered by media.concat. Undirected posts keep the
hybrid split (generation.talk_hybrid) behind its switch.
Finish (`finish_video`, policy `finishing` and `burned_captions`, each with its `formats`): once a video post's video
is made (and joined), it is finished once (production.finish_mode queue: a finish.video job on the finisher machine; inline:
in the producer): the sound mix (policy `sound`: procedural room tone,
voice polish, library cues, a bed for silent formats), then grade, grain and loudness (manyworlds.finishing) with the
spoken words (manyworlds.captions, timed from the voice lines, in the style captions.caption_style_for picks, placed
off the face) and the shots' on-screen text (manyworlds.overlays: labels, the hook, timestamps, question bubbles) in
ONE ASS script, burned in the same ffmpeg pass where this ffmpeg has libass. The cover frame is picked from the
finished file. An AI upscaler (finishing.upscaler, off) runs as a GPU job first when it is on. The post's `video` is
the finished file (post-<id>/final.mp4, reused on a rerun), `video_raw` the one it was made from.
`ready` posts are published by manyworlds.publish.scheduler per policy autonomy.publish (autonomous, or held).
Only formats listed in production.formats are made; video posts wait as 'planned' until their
format is added there.

The package (one module per stage; this module re-exports every name, so `from manyworlds import production` and
`production.<name>` work as they did when this was one file):
  common      shared helpers: the policy block, seeds, spoken text, shots on payloads, the status write, a video's pieces
  steps       Step, advance_step and the job state readers (waiting | chosen | rejected | failed)
  scenes      the identity LoRA, scene image jobs, start, the scene Steps (repairs)
  takes       how a talking video's script is cut into takes and which shot, photo and words each has
  voice       the designed voice: voice lines and their voice.speak Steps
  motion      story clips, clip edits, dance / motion transfer, walk-and-talk motion clips and their joins
  talk        talk jobs, the hybrid's cutaways, long talk segments and their join
  followups   the one video made from the chosen photos (FOLLOW_UPS) and its voice and motion first
  captioning  the captions plan: spoken words timed in the finished video, style, placement
  song        a music_video post's song (music.song: the mix and its stems; policy music, off)
  mixing      the sound under a finished video (a music video's song in place of its vocal stem)
  finish      the upscale Step, the finish Step (a finish.video job) and finish_video (mix, grade, captions,
              on-screen text, cover): finish_plan, render_finish, record_finish
  core        advance_post, advance (the tick) and counting posts whose advance raises
  locks       one producer per post (a lease row per post; a held post is skipped this tick)
  state       every production state key kept in posts.body, beside the brief's own fields

Patching: a name is looked up in the module that defines or imports it. To stand in for one in a test, patch it
there (e.g. monkeypatch.setattr(production.core, "advance_post", fake)); patching the package attribute only changes
what callers outside the package see (e.g. cli's `production.advance`).
"""

from __future__ import annotations

from psycopg.types.json import Jsonb  # noqa: F401 - kept as a package attribute, as in the one-file module

from .. import queue  # noqa: F401
from ..creator.lab import photo_prompt, prompts  # noqa: F401
from ..policy import Policy  # noqa: F401
from ..review import repair  # noqa: F401
from .captioning import (  # noqa: F401
    _audio_meta, _caption_style, _drop_unscripted, _fillers, _piece_timing, _style_keys, _transcribed, _voice_audio,
    caption_lines, caption_placement, caption_plan, caption_style, caption_words, captions_from_words, voice_words,
)
from .common import (  # noqa: F401
    FOLLOW_UP_FORMATS, SPOKEN, SUNG, _cfg, _pieces, _seed, _set, _shot, _with_crops, _with_prompts, _with_shot, log,
    candidates_per_scene, retract_dropped, tier_production, unmarked,
)
from .core import (  # noqa: F401
    RerouteCaptionAnswer, _advance_failed, _slide_formats, advance, advance_post, reroute_caption,
)
from .finish import (  # noqa: F401
    FINISH_KIND, FINISH_SECTIONS, _advance_finish, _advance_upscale, _burn_text, _cover_row, _cover_upload,
    _existing_final, _fetch_frames, _final_key, _punch_in, _talk_edit, _finish_failed, _finish_job, _finish_state, _give_up_finish, give_up_action, _jsonable,
    _keyframe_cover, _overlay_items_from, _spans_from, _switched_on, _text_fonts, _upscale_job, captions_burned,
    face_box, finish_mode, finish_plan, finish_video, finishes, join_times, narrated_spans, overlay_items,
    record_finish,
    render_finish, shot_spans, text_spans, text_windows,
)
from .followups import (  # noqa: F401
    FOLLOW_UPS, _advance_follow_up, _advance_sung, _edit_job, _follow_up, _keep_photo_spans,
)
from .song import (_register_song, _song_advance_voice, _song_job, _song_state, _song_voice_asset,  # noqa: F401
                   _song_voice_job, _song_voice_state, advance_song, sung_text)
from .locks import lease_name, post_lease, release_post_lease, take_post_lease  # noqa: F401
from .mixing import (  # noqa: F401
    _mix_sound, _piece_loader, _song_ref, _sound_cues_from, _sound_plan_from, _sound_ref, sound_cues, sound_plan,
)
from .motion import (  # noqa: F401
    _advance_clip_edit, _advance_motion, _chain_jobs, _clip_job, _clip_params, _dance_kind, _last_frame, _motion_cfg,
    _motion_job, _motion_pattern, _needs_motion, _story_clip_kind, _take_seconds, _wan_frames, clip_edit,
    dance_payload, dance_reference, join_clips, join_motion, motion_lengths, moves, talk_v2v,
)
from .scenes import _scene_job, _scene_steps, ensure_lora, is_video_format, scene_size, start  # noqa: F401
from .state import BRIEF, BRIEF_KEYS, STATE, STATE_KEYS, STATE_PREFIXES, declared  # noqa: F401
from .steps import (  # noqa: F401
    DropPost, Step, _best_confirmed, _clip_state, _edit_state, _scene_state, _status, advance_step,
)
from .takes import (  # noqa: F401
    _directed, _hybrid_cfg, _photo_per_shot, _take_index, _take_photo, _take_script, _take_shot, _talk_long_cfg,
    hybrid_modes, script_takes, shot_takes, talk_segments,
)
from .talk import (  # noqa: F401
    JOIN_KIND, JOIN_SECTIONS, _advance_long_talk, _cutaway_job, _existing_join, _join_job, _join_key, _join_state,
    _talk_job, _take_history,
    advance_join, join_plan, join_talk, join_timeout, join_transitions, record_join, render_join, talk_repair,
)
from .voice import (  # noqa: F401
    _advance_voice, _designed_voice, _marked_voice_lines, _sentence_lines, _voice_job, _voice_state, _with_audio,
    voice_lines,
)
