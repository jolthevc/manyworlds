"""The captions plan of a finished video: each piece's spoken words timed in the joined video (voice lines, word
timestamps, estimates), the caption style and where the captions go."""

from __future__ import annotations

from psycopg.types.json import Jsonb

from ..policy import Policy
from .common import _pieces, _shot, unmarked
from .takes import _take_index, _take_script


def caption_lines(meta: dict) -> list[tuple[dict, int]]:
    """A voice line's timings as captions.estimate_timings' lines ({text, start_s, end_s}), from voice.speak's meta:
    one per clause when it has clauses (finer spans), else one per line; each with the number of filler words in
    front. A clause the voice began with a filler (meta.fillers, or the clause's own `filler`) gets the filler in
    front of its text: the clause's span holds the filler too, so the filler takes its share of the time, and those
    leading words are then dropped from the captions."""
    def timed(items) -> bool:
        return bool(items) and all(isinstance(x, dict) and x.get("text") and x.get("start_s") is not None
                                   and x.get("end_s") is not None for x in items)
    clauses, lines = meta.get("clauses") or [], meta.get("lines") or []
    if timed(clauses):
        said = {f.get("clause"): f.get("word") for f in meta.get("fillers") or [] if isinstance(f, dict)}
        out = []
        for i, c in enumerate(clauses):
            filler = said.get(c.get("index", i)) or c.get("filler")
            text = f"{filler}, {c['text']}" if filler else c["text"]
            out.append(({"text": text, "start_s": float(c["start_s"]), "end_s": float(c["end_s"])},
                        len(str(filler).split()) if filler else 0))
        return out
    if timed(lines):
        return [({"text": ln["text"], "start_s": float(ln["start_s"]), "end_s": float(ln["end_s"])}, 0)
                for ln in lines]
    return []


def _audio_meta(conn, *, uri: str | None = None, job_id=None) -> dict:
    row = (conn.execute("SELECT meta FROM assets WHERE uri = %s AND kind = 'audio' LIMIT 1", (uri,)).fetchone() if uri
           else conn.execute("SELECT meta FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri LIMIT 1",
                             (job_id,)).fetchone() if job_id else None)
    return (row["meta"] or {}) if row else {}


def _piece_timing(conn, piece: dict, script: str) -> tuple[list[tuple[dict, int]], float | None]:
    """A piece's caption lines in its own time, and its length. Its designed voice line's clauses or lines (a narrated
    piece's voice starts after the edit's lead); else a Kokoro edit's own cues; else the script over the spoken part
    (a Kokoro talk clip's own audio, or the edit between its lead and tail)."""
    from ..worker.handlers.edit_narrated import LEAD_S, TAIL_S
    job = piece["job"] or {}
    payload, vmeta = job.get("payload") or {}, piece["meta"] or {}
    narrated = job.get("kind") == "image.narrated"
    lead = float(vmeta.get("lead_s", LEAD_S)) if narrated else 0.0
    tail = float(vmeta.get("tail_s", TAIL_S)) if narrated else 0.0
    voice = _audio_meta(conn, uri=payload["audio_uri"]) if payload.get("audio_uri") else {}
    seconds = vmeta.get("seconds")
    if seconds is None and voice.get("seconds") is not None:
        seconds = float(voice["seconds"]) + lead + tail
    seconds = float(seconds) if seconds is not None else None
    lines = caption_lines(voice) if voice else []
    if lines:
        return [({**ln, "start_s": ln["start_s"] + lead, "end_s": ln["end_s"] + lead}, f) for ln, f in lines], seconds
    cues = vmeta.get("cues") if narrated else None
    if cues and all(isinstance(c, dict) and c.get("text") and c.get("start_s") is not None for c in cues):
        return [({"text": c["text"], "start_s": float(c["start_s"]), "end_s": float(c["end_s"])}, 0)
                for c in cues], seconds
    spoken = None
    if not narrated and job.get("id"):
        spoken = _audio_meta(conn, job_id=job["id"]).get("seconds")
    if spoken is None and seconds is not None:
        spoken = seconds - lead - tail
    if not script.strip() or not spoken or float(spoken) <= 0:
        return [], seconds
    # One block over the spoken span (nothing finer known): captions.block_timings shares it out
    return [({"text": script, "start_s": lead, "end_s": lead + float(spoken), "block": True}, 0)], seconds


def _voice_audio(conn, piece: dict) -> dict | None:
    """The audio asset a piece speaks: its designed voice line (payload.audio_uri), else a talk clip's own audio
    (Kokoro on the talk worker). A narrated edit without a separate voice line has none."""
    job = piece["job"] or {}
    payload = job.get("payload") or {}
    if payload.get("audio_uri"):
        return conn.execute("SELECT id, uri, meta, job_id, creator_id FROM assets WHERE uri = %s AND kind = 'audio' "
                            "LIMIT 1", (payload["audio_uri"],)).fetchone()
    if job.get("kind") != "image.narrated" and job.get("id"):
        return conn.execute("SELECT id, uri, meta, job_id, creator_id FROM assets WHERE job_id = %s AND kind = 'audio' "
                            "ORDER BY uri LIMIT 1", (job["id"],)).fetchone()
    return None


def voice_words(conn, policy: Policy, audio: dict, store=None, *, transcriber=None) -> list[dict] | None:
    """A voice line's word timestamps [{word, start, end}] (burned_captions.word_timestamps): the asset's cached
    meta.words, else transcribed once (speech.timed_words, charged to the ledger) and cached there. None on any
    failure (the captions then fall back to estimated timings)."""
    import tempfile
    from pathlib import Path

    from .. import speech, storage
    meta = audio["meta"] or {}
    if meta.get("words"):
        return list(meta["words"])
    transcriber = transcriber or speech.timed_words
    try:
        store = store or storage.from_url()
        with tempfile.TemporaryDirectory(prefix="mw-words-") as tmp:
            path = store.get_file(audio["uri"], Path(tmp) / f"voice{Path(audio['uri']).suffix or '.wav'}")
            words = transcriber(conn, policy, path, float(meta.get("seconds") or 0), job_id=audio.get("job_id"),
                                creator_id=audio.get("creator_id"))
    except Exception:  # noqa: BLE001 - estimated timings are the fallback
        return None
    if not words:
        return None
    conn.execute("UPDATE assets SET meta = COALESCE(meta, '{}'::jsonb) || %s WHERE id = %s",
                 (Jsonb({"words": words}), audio["id"]))
    return list(words)


def _fillers(meta: dict) -> set[str]:
    from .. import captions
    said = [f.get("word") for f in meta.get("fillers") or [] if isinstance(f, dict)]
    said += [c.get("filler") for c in meta.get("clauses") or [] if isinstance(c, dict)]
    return {captions.key(str(w)) for w in said if w}


def _drop_unscripted(words: list, script: str, drop: set[str]) -> list:
    """Transcribed words in `drop` (the voice's fillers) that the script doesn't have at that place."""
    from difflib import SequenceMatcher

    from .. import captions
    if not drop:
        return words
    a, b = [captions.key(w.text) for w in words], [captions.key(t) for t, _ in captions.parse_emphasis(script)]
    gone = set()
    for op, i1, i2, _, _ in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("insert", "delete", "replace"):
            gone.update(i for i in range(i1, i2) if a[i] in drop)
    return [w for i, w in enumerate(words) if i not in gone]


def _transcribed(conn, policy: Policy, piece: dict, script: str, offset: float, store, transcriber,
                 fmt: str | None = None) -> list | None:
    """A piece's words from its voice line's word timestamps, placed at `offset` (plus a narrated edit's lead), or
    None when word timestamps are off for this format (captions.word_timestamps_for), the piece has no voice line,
    or the transcription failed."""
    from .. import captions
    if not captions.word_timestamps_for(policy, fmt):
        return None
    audio = _voice_audio(conn, piece)
    if audio is None:
        return None
    words = voice_words(conn, policy, audio, store, transcriber=transcriber)
    if not words:
        return None
    from ..worker.handlers.edit_narrated import LEAD_S
    job = piece["job"] or {}
    lead = float((piece["meta"] or {}).get("lead_s", LEAD_S)) if job.get("kind") == "image.narrated" else 0.0
    try:
        timed = captions.word_timings(words, script=script or None, offset=offset + lead)
    except Exception:  # noqa: BLE001 - a malformed cache entry: estimates instead
        return None
    return _drop_unscripted(timed, script, _fillers(audio["meta"] or {})) or None


def caption_words(conn, policy: Policy, body: dict, video: dict, *, store=None, transcriber=None,
                  fmt: str | None = None) -> list | None:
    """The post's spoken words, timed in its finished video (captions.Word). Each piece's lines (_piece_timing) are
    placed at the piece's start in the joined video (the join's measured segment lengths, else each piece's own),
    fillers dropped, and the post's script (with *emphasis*) lines up the shown words. With word timestamps on for
    the post's format `fmt` (captions.word_timestamps_for), a piece with a voice line is timed from its transcript's
    word timestamps instead (voice_words; estimates when that fails). None when a piece's length is unknown, so
    later captions can't be placed."""
    from .. import captions
    row, pieces = _pieces(conn, video)
    measured = (row["meta"] or {}).get("segment_seconds") or []
    segmented = len(body.get("talk_segments") or []) == len(pieces) > 1
    words, scripts, t = [], [], 0.0
    for i, piece in enumerate(pieces):
        script = _take_script(body, i if segmented else None)
        lines, seconds = _piece_timing(conn, piece, unmarked(script))
        heard = _transcribed(conn, policy, piece, script, t, store, transcriber, fmt) if lines else None
        if heard:
            words += heard
            lines = []
        for line, fillers in lines:
            if line.get("block"):   # a Kokoro take's script over its own audio
                words += captions.block_timings(line["text"], line["start_s"], line["end_s"], offset=t,
                                                policy=policy)[fillers:]
            else:
                words += captions.estimate_timings([line], offset=t, policy=policy)[fillers:]
        scripts.append(script)
        if i < len(pieces) - 1:
            length = measured[i] if i < len(measured) and measured[i] is not None else seconds
            if length is None:
                return None
            t += float(length)
    return captions._align_script(words, " ".join(scripts)) if words else []


def caption_placement(policy: Policy, shot: dict | None) -> str:
    """Where a video's captions go: the first entry of burned_captions.placement_by_shot_size whose words appear in
    the shot's shot size, else default_placement."""
    cfg = policy.get("burned_captions", default={}) or {}
    size = str(((shot or {}).get("framing") or {}).get("shot_size") or "").lower()
    for words, place in (cfg.get("placement_by_shot_size") or {}).items():
        if str(words).lower() in size:
            return str(place)
    return str(cfg.get("default_placement") or "lower")


def _caption_style(conn, creator_id) -> str | None:
    """The creator's caption style (profile dial caption_style), or None for burned_captions.default_style."""
    from ..creator import profile
    current = profile.current(conn, creator_id)
    style = ((current[1] if current else {}).get("dials") or {}).get("caption_style")
    return str(style) if style else None


def _style_keys(post: dict, body: dict) -> list[str]:
    """What burned_captions.style_by_format is looked up by: the post's template (a shot-sheet archetype such as an
    outfit check, when the director or breadth driver names one) before its format."""
    return [str(k) for k in (body.get("template"), post["format"]) if k]


def caption_style(conn, policy: Policy, post: dict, body: dict) -> str | None:
    """The post's caption style (captions.caption_style_for: the opening shot's edit.caption_style, the template or
    format, the creator's dial), or None for no spoken captions (on-screen labels only)."""
    from .. import captions
    shot = _shot(body, _take_index(body, 0 if body.get("talk_segments") else None))
    return captions.caption_style_for(policy, creator=_caption_style(conn, post["creator_id"]),
                                      fmt=_style_keys(post, body), shot=shot)


def caption_plan(conn, policy: Policy, post: dict, body: dict, video: dict, *, width: int, height: int, store=None,
                 transcriber=None, face_box=None):
    """The captions for a post's finished video of width x height, or None when nothing is said (or can't be timed)
    or its style is none. Placed by the opening shot's size, off the face (`face_box`, shares of the frame)."""
    from .. import captions
    style = caption_style(conn, policy, post, body)
    if style is None:
        return None
    words = caption_words(conn, policy, body, video, store=store, transcriber=transcriber, fmt=post["format"])
    if not words:
        return None
    shot = _shot(body, _take_index(body, 0 if body.get("talk_segments") else None))
    return captions.make_plan(words, policy, width=width, height=height, style=style,
                              placement=captions.placement_for_shot(policy, shot, face_box))
