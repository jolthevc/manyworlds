"""Final-file review: the finished video (final.mp4, after sound, captions, labels, hooks and the finish) judged once.

The clip gates (manyworlds.review.video) see each take before it is joined and finished; nothing until now looked at
the file that is posted. This stage does, once per post, from frames cut out of the finished file:

  face_clear  no caption, label, hook or sticker covers the face or the eyes (frames are taken while each text span
              is on screen, and the judge is told what text should be there)
  seams       the joins (given as times in the finished file) don't show: the frames either side of each join are
              shown in pairs; a jump in the face, light, clothes or framing that isn't a deliberate cut costs
  sync        the sound still matches the picture at the end: the audio and video streams of the file end within
              `max_end_drift_seconds` of each other (measured with ffmpeg, free), and the judge's `end_sync` for the
              last frames against the last words is at `min_end_sync` or more (when someone speaks)
  real        would a viewer take it for real phone footage (`phone_real` 1-10, at `min_phone_real` or more)

Report-only by default (`review.final.enforce: false`): the verdicts are recorded and `passed` is True whatever they
say; enforced, `passed` is False when any gate fails, and production decides what that means for the post. The
result lands on the post (body.final_review), as one `reviews` row per gate on the finished video asset (when the
asset is found), and as a `review.final` decision; the judge's tokens are charged to the post's latest job, so they
count in the post's cost like every other review (manyworlds.ledger). Policy `review.final`.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ..creator.lab import appearance_prompt
from ..llm import StructuredLLM, image_block
from ..policy import Policy
from . import common

SYSTEM = """You are the last quality gate for an AI creator studio whose creators are openly labeled as AI.
You see frames from one finished vertical phone video, captions and labels burned in, exactly as it will be posted.
Judge it strictly, as a viewer scrolling past would see it. Score what you see, not what was intended."""

GATES = ("face_clear", "seams", "sync", "real")


class FinalReview(BaseModel):
    face_covered: bool = Field(description="a caption, label, hook text or sticker covers the face or the eyes in "
                               "any frame")
    covered_frames: list[int] = Field(default_factory=list, description="the FRAME numbers where text covers the "
                                      "face or eyes, empty if none")
    seams: int | None = Field(default=None, description="only when JOIN pairs are given: 1-10 for the worst join; "
                              "10 = it reads as one continuous phone video or a clean deliberate cut, 1 = an obvious "
                              "jump (face, light, clothes or framing change for no reason); null when no joins")
    seam_defects: list[str] = Field(default_factory=list, description="what jumps at which join, empty if none")
    end_sync: int | None = Field(default=None, description="only when LAST WORDS are given and a face is visible "
                                 "at the END frames: 1-10, the mouth plausibly says them (10) or is clearly out of "
                                 "step (1); null otherwise")
    phone_real: int = Field(description="1-10: would a viewer believe this is real phone footage of a real person "
                            "(10) or see a render (1)")
    defects: list[str] = Field(default_factory=list, description="concrete visible problems with the FRAME number")
    notes: str = Field(default="", description="one sentence")


def cfg(policy: Policy | None) -> dict:
    return (policy.get("review", "final", default={}) or {}) if policy is not None else {}


def enabled(policy: Policy | None) -> bool:
    return bool(cfg(policy).get("enabled"))


# --------------------------------------------------------------------------------------------------- the frames

def _at(j) -> float | None:
    """A join's time in seconds: a number, or a dict with at_s / t / time / start_s."""
    if isinstance(j, (int, float)):
        return float(j)
    if isinstance(j, dict):
        for k in ("at_s", "t", "time", "start_s"):
            if j.get(k) is not None:
                return float(j[k])
    return None


def _span(s) -> dict | None:
    if not isinstance(s, dict) or s.get("start_s") is None or s.get("end_s") is None:
        return None
    return {"text": str(s.get("text") or "").strip(), "start_s": float(s["start_s"]), "end_s": float(s["end_s"]),
            "kind": str(s.get("kind") or "text")}


def sample_times(seconds: float, policy: Policy | None, *, joins=None, text_spans=None) -> list[dict]:
    """[{t, why}] in time order: `keyframes` spread over the file, a frame either side of each join
    (`join_offset_seconds`, at most `max_joins` joins), one in the middle of each text span (at most
    `max_text_frames`), and the end (`end_window_seconds` before it and the last moment). Frames closer than
    `min_gap_seconds` merge; at most `max_frames`, the even spread dropped first."""
    c = cfg(policy)
    d = max(0.0, float(seconds or 0))
    if d <= 0:
        return []
    last = max(0.0, d - float(c.get("end_margin_seconds", 0.1)))

    def clamp(t: float) -> float:
        return round(min(last, max(0.0, t)), 3)

    n = int(c.get("keyframes", 6))
    off, w = float(c.get("join_offset_seconds", 0.1)), float(c.get("end_window_seconds", 0.5))
    marked: list[tuple[float, str]] = []
    for k, t in enumerate(t for t in (_at(j) for j in (joins or [])) if t is not None):
        if k >= int(c.get("max_joins", 6)):
            break
        marked += [(clamp(t - off), f"join {k + 1}, before"), (clamp(t + off), f"join {k + 1}, after")]
    spans = [s for s in (_span(s) for s in (text_spans or [])) if s and s["text"]]
    for s in spans[:int(c.get("max_text_frames", 4))]:
        marked.append((clamp((s["start_s"] + s["end_s"]) / 2), f"text on screen: \"{s['text'][:80]}\""))
    marked += [(clamp(d - w), "end"), (last, "end, last moment")]
    even = [(clamp((i + 0.5) * d / n), "spread") for i in range(n)] if n > 0 else []
    cap, gap = int(c.get("max_frames", 20)), float(c.get("min_gap_seconds", 0.15))
    out: list[dict] = []
    for t, why in sorted(marked) + even:   # marked frames claim their place first; the spread fills in
        near = next((o for o in out if abs(o["t"] - t) < gap), None)
        if near is not None:
            if why != "spread" and why not in near["why"]:
                near["why"] = f"{near['why']}; {why}" if near["why"] != "spread" else why
            continue
        if len(out) >= cap:
            continue
        out.append({"t": t, "why": why})
    return sorted(out, key=lambda o: o["t"])


def _frame(video: Path, t: float, out: Path, height: int, timeout: float, *, back: float = 0.1,
           tries: int = 3) -> Path:
    """The frame at `t` s as a JPEG `height` px tall. A time past the last picture (the container can run longer
    than the video stream) steps back by `back` s, up to `tries` times."""
    from .. import media
    err = ""
    for k in range(max(1, tries)):
        at = max(0.0, t - k * back)
        cmd = [media.ffmpeg(), "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1",
               "-vf", f"scale=-2:{height}", "-q:v", "3", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
        err = (r.stderr or "")[-300:]
    raise RuntimeError(f"frame at {t:.3f} s of {video.name}: {err or 'no picture'}")


_TIME = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")


def _stream_end(video: Path, stream: str, timeout: float) -> float | None:
    """Where one stream of the file ends (seconds), from ffmpeg copying it to nothing (no decode); None without that
    stream."""
    from .. import media
    r = subprocess.run([media.ffmpeg(), "-hide_banner", "-nostdin", "-stats", "-i", str(video), "-map", f"0:{stream}:0",
                        "-c", "copy", "-f", "null", "-"], capture_output=True, text=True, timeout=timeout)
    times = _TIME.findall(r.stderr or "")
    if r.returncode != 0 or not times:
        return None
    h, m, s = times[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def end_drift(video: Path, policy: Policy | None) -> dict:
    """{video_s, audio_s, drift_s}: how far apart the file's audio and video streams end. drift_s None when the
    file has no audio (nothing to drift) or a stream can't be read."""
    timeout = float(cfg(policy).get("timeout_seconds", 120))
    v, a = _stream_end(Path(video), "v", timeout), _stream_end(Path(video), "a", timeout)
    return {"video_s": v, "audio_s": a, "drift_s": round(abs(a - v), 3) if (a is not None and v is not None) else None}


# ------------------------------------------------------------------------------------------------- the verdict

def decide(r: FinalReview, drift: float | None, policy: Policy | None) -> dict[str, bool]:
    c = cfg(policy)
    sync = drift is None or drift <= float(c.get("max_end_drift_seconds", 0.1))
    if r.end_sync is not None:
        sync = sync and r.end_sync >= int(c.get("min_end_sync", 6))
    return {"face_clear": not r.face_covered,
            "seams": r.seams is None or r.seams >= int(c.get("min_seams", 6)),
            "sync": sync,
            "real": r.phone_real >= int(c.get("min_phone_real", 6))}


def _last_words(spans: list[dict], policy: Policy | None) -> str | None:
    """The words spoken at the end: the last caption span's text (spans of kind caption, else the last span)."""
    kinds = set(cfg(policy).get("spoken_kinds") or ["caption"])
    said = [s for s in spans if s["kind"] in kinds] or []
    return said[-1]["text"] if said else None


def _judge(llm: StructuredLLM, policy: Policy, creator: dict | None, frames: list[tuple[dict, bytes]],
           spans: list[dict], joins: int, last_words: str | None) -> FinalReview:
    c = cfg(policy)
    content: list = []
    if creator and creator.get("bible"):
        content.append({"type": "text", "text": f"The person: {appearance_prompt(creator['bible'])}."})
    for k, (f, b) in enumerate(frames):
        content += [{"type": "text", "text": f"FRAME {k + 1} of {len(frames)} at {f['t']:.2f} s ({f['why']}):"},
                    image_block(b, "image/jpeg")]
    if spans:
        content.append({"type": "text", "text": "TEXT ON SCREEN (burned in; check it never covers the face or eyes): "
                                                + "; ".join(f"{s['start_s']:.1f}-{s['end_s']:.1f} s \"{s['text'][:80]}\""
                                                            for s in spans[:int(c.get("max_spans_listed", 12))])})
    if joins:
        content.append({"type": "text", "text": f"JOINS: {joins}. The frames marked 'join N, before' and 'after' "
                                                f"are either side of join N: score `seams` on the worst."})
    if last_words:
        content.append({"type": "text", "text": f"LAST WORDS (spoken over the END frames): \"{last_words[:160]}\""})
    content.append({"type": "text", "text": "Review the finished video as a whole from these frames."})
    r = llm.structured(str(c.get("role", "judge")), system=SYSTEM, content=content, schema=FinalReview,
                       effort=str(c.get("effort", "low")), creator_id=(creator or {}).get("id"))
    r.phone_real = max(1, min(10, int(r.phone_real)))
    r.seams = max(1, min(10, int(r.seams))) if (joins and r.seams is not None) else None
    r.end_sync = max(1, min(10, int(r.end_sync))) if (last_words and r.end_sync is not None) else None
    return r


def review_final(conn, policy: Policy, llm: StructuredLLM, post: dict, video_path, *, joins=None,
                 text_spans=None) -> dict:
    """Judge one post's finished video file once. Returns {passed, accepted, enforced, gates, scores, drift, frames,
    model, ...}; `passed` is what production acts on (always True while report-only). Off (`review.final.enabled`
    false): {"skipped": "off", "passed": True}. Also stored on the post (body.final_review; the caller's `post["body"]`
    dict is updated in place too, so a body it writes back keeps it)."""
    c = cfg(policy)
    if not c.get("enabled"):
        return {"skipped": "off", "passed": True}
    import tempfile

    from .. import media
    video_path = Path(video_path)
    enforce = bool(c.get("enforce"))
    spans = [s for s in (_span(s) for s in (text_spans or [])) if s and s["text"]]
    join_times = [t for t in (_at(j) for j in (joins or [])) if t is not None]
    seconds = media.duration(video_path) or 0.0
    times = sample_times(seconds, policy, joins=join_times, text_spans=spans)
    if not times:
        out = {"skipped": "unreadable", "passed": not enforce, "accepted": False, "enforced": enforce}
        _store(conn, post, out)
        return out
    height, timeout = int(c.get("frame_height", 960)), float(c.get("timeout_seconds", 120))
    with tempfile.TemporaryDirectory(prefix="mw-final-review-") as tmp:
        back = float(c.get("end_margin_seconds", 0.1))
        frames = [(f, _frame(video_path, f["t"], Path(tmp) / f"final_{k}.jpg", height, timeout, back=back).read_bytes())
                  for k, f in enumerate(times)]
    drift = end_drift(video_path, policy)
    creator = conn.execute("SELECT id, bible FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()
    last_words = _last_words(spans, policy)
    job = conn.execute("SELECT id FROM jobs WHERE post_id = %s ORDER BY updated_at DESC LIMIT 1",
                       (post["id"],)).fetchone()
    with common.charged_to(llm, job["id"] if job else None):   # counts in the post's cost
        r = _judge(llm, policy, creator, frames, spans, len(join_times), last_words)
    verdict = decide(r, drift["drift_s"], policy)
    accepted = all(verdict.values())
    model = llm.model_for(str(c.get("role", "judge")))
    scores = {"face_clear": 0 if r.face_covered else 10, "seams": r.seams, "sync": r.end_sync,
              "real": r.phone_real}
    out = {"passed": accepted or not enforce, "accepted": accepted, "enforced": enforce, "gates": verdict,
           "scores": scores, "drift": drift, "joins": join_times, "frames": times, "model": model,
           "policy_version": policy.version, **r.model_dump()}
    _store(conn, post, out)
    _rows(conn, post, out, verdict, scores)
    conn.execute("INSERT INTO decisions (kind, creator_id, inputs, output, model, policy_version) "
                 "VALUES ('review.final', %s, %s, %s, %s, %s)",
                 (post["creator_id"], Jsonb({"post_id": str(post["id"]), "joins": join_times, "text_spans": len(spans),
                                             "frames": len(times), "seconds": seconds}),
                  Jsonb({k: out[k] for k in ("passed", "accepted", "enforced", "gates", "scores", "drift")}),
                  model, policy.version))
    return out


def _store(conn, post: dict, out: dict) -> None:
    conn.execute("UPDATE posts SET body = body || %s, updated_at = now() WHERE id = %s",
                 (Jsonb({"final_review": out}), post["id"]))
    if isinstance(post.get("body"), dict):
        post["body"]["final_review"] = out


def _rows(conn, post: dict, out: dict, verdict: dict[str, bool], scores: dict) -> None:
    """One `reviews` row per gate on the post's finished video asset (assets/<creator>/post-<id>/final.mp4), as the
    other gates write; while report-only a row is written passed, with details.would_pass. No asset: no rows."""
    asset = conn.execute("SELECT id FROM assets WHERE kind = 'video' AND uri LIKE %s ORDER BY created_at DESC LIMIT 1",
                         (f"%/post-{post['id']}/final.mp4",)).fetchone()
    if not asset:
        return
    details = {k: out[k] for k in ("model", "enforced", "drift", "defects", "seam_defects", "covered_frames", "notes")}
    common.write_rows(conn, [(asset["id"], f"final_{g}", ok or not out["enforced"], scores.get(g),
                              Jsonb({**details, "would_pass": ok, "stage": "final"})) for g, ok in verdict.items()])
