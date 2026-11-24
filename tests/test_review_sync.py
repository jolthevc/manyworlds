"""The finished talking video's CPU lip-sync score: a mouth that follows the voice scores high, one shifted past the
lag window or inverted scores low, and the silent-mouth-moving / talking-mouth-closed shares say why. Report-only
until enforced. Landmarks are injected (a drawn mouth has no face for MediaPipe to find); the real extractor loads
mediapipe lazily."""

from __future__ import annotations

import subprocess
import sys
import wave

import numpy as np
from psycopg.types.json import Jsonb

from manyworlds import media, policy as policy_mod
from manyworlds.review import sync

from test_production import creator

FPS, RATE = 25, 16000


def _with_sync(policy, **over):
    review = policy.body["review"]
    return policy_mod.from_dict({**policy.body, "review": {**review, "sync": {**review["sync"], **over}}})


def _envelope(seconds=8.0, seed=3):
    """Syllable-like on/off speech, one value per video frame (0 silent, 1 voiced)."""
    rng = np.random.default_rng(seed)
    out, on = [], False
    while len(out) < int(seconds * FPS):
        out += [1.0 if on else 0.0] * int(rng.integers(12, 30))   # phrases and pauses, longer than the lag window
        on = not on
    return np.array(out[:int(seconds * FPS)])


def _beat(seconds=8.0):
    """A steady beat: 12 frames voiced, 13 silent (one a second). Half a beat off, no lag within +-200 ms aligns it."""
    return np.array([1.0 if i % 25 < 12 else 0.0 for i in range(int(seconds * FPS))])


def _loud(env):
    """Log RMS of a tone shaped by `env`, as review.sync measures it."""
    from manyworlds.worker import lipsync
    t = np.arange(int(len(env) / FPS * RATE)) / RATE
    tone = 0.5 * np.sin(2 * np.pi * 220 * t) * (np.repeat(env, RATE // FPS)[:len(t)] + 1e-3)
    return lipsync.loudness(tone, RATE, FPS, len(env))


def test_a_mouth_following_the_voice_scores_high_and_out_of_step_low(policy):
    env = _envelope()
    loud = _loud(env)
    opening = env * 0.4   # mouth gap over mouth width
    good = sync.score(opening, loud, FPS, policy)
    assert good["corr"] > 0.9 and abs(good["lag_ms"]) <= 40
    assert good["silent_mouth_moving"] == 0 and good["talking_mouth_closed"] == 0
    ok, why = sync.decide(good, policy)
    assert ok and not why

    late = sync.score(np.roll(opening, 3), loud, FPS, policy)   # 120 ms late: inside the window, found
    assert late["corr"] > 0.9 and late["lag_ms"] == 120.0

    beat = _beat()   # out of step: a mouth half a beat behind (480 ms), past +-max_lag_ms
    far = sync.score(np.roll(beat, 12) * 0.4, _loud(beat), FPS, policy)
    assert far["corr"] is None or far["corr"] < 0.3
    assert sync.decide(far, policy)[0] is False

    inverted = sync.score(0.4 - opening, loud, FPS, policy)
    assert inverted["corr"] is None or inverted["corr"] < 0
    assert inverted["silent_mouth_moving"] > 0.9 and inverted["talking_mouth_closed"] > 0.9
    ok, why = sync.decide(inverted, policy)
    assert not ok and any("silence" in w for w in why) and any("talking" in w for w in why)


def test_no_face_frames_are_left_out_and_counted(policy):
    env = _envelope()
    opening = env * 0.4
    opening[:80] = np.nan
    s = sync.score(opening, _loud(env), FPS, policy)
    assert s["face_share"] == round(1 - 80 / len(env), 3) and s["corr"] > 0.9
    assert sync.decide(s, policy)[0] is False   # face in too few frames


def _clip(path, env, *, mouth=None):
    """A drawn 'mouth': each frame's brightness is the mouth opening, the sound a tone shaped by `env`."""
    frames = env if mouth is None else mouth
    wav = path.with_suffix(".wav")
    t = np.arange(int(len(env) / FPS * RATE)) / RATE
    tone = 0.5 * np.sin(2 * np.pi * 220 * t) * (np.repeat(env, RATE // FPS)[:len(t)] + 1e-3)
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes((tone * 32767).astype(np.int16).tobytes())
    raw = b"".join(np.full((64, 64, 3), int(30 + 200 * v), np.uint8).tobytes() for v in frames)
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x64",
                    "-r", str(FPS), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(path)], input=raw, check=True)
    return path


def brightness(frames, fps):
    """Stub extractor: the frame's brightness is its mouth opening."""
    return np.array([(float(f.mean()) - 30) / 200 * 0.4 for f in frames])


def test_measure_reads_the_file_frames_and_sound_in_step(policy, tmp_path):
    env = _envelope()
    in_step = sync.measure(_clip(tmp_path / "in.mp4", env), policy, brightness)
    assert in_step["corr"] > 0.8 and abs(in_step["lag_ms"]) <= 80 and in_step["frames"] >= 90
    off = sync.measure(_clip(tmp_path / "off.mp4", _beat(), mouth=np.roll(_beat(), 12)), policy, brightness)
    assert off["corr"] is None or off["corr"] < 0.3


def test_sampling_stays_under_the_frame_cap(policy):
    cap = int(policy.get("review", "sync", "max_frames"))
    assert sync.sample_rate(10, 30, policy) == float(policy.get("review", "sync", "sample_fps"))
    assert sync.sample_rate(600, 30, policy) * 600 <= cap + 1e-6


def _post(conn, cid, fmt="talking_video"):
    pid = conn.execute("INSERT INTO posts (creator_id, platform, format, status, body) "
                       "VALUES (%s, 'instagram', %s, 'producing', %s) RETURNING id",
                       (cid, fmt, Jsonb({"script": "hi"}))).fetchone()["id"]
    conn.execute("INSERT INTO assets (creator_id, kind, uri, sha256, bytes) VALUES (%s, 'video', %s, 'x', 1)",
                 (cid, f"s3://b/assets/{cid}/post-{pid}/final.mp4"))
    return conn.execute("SELECT * FROM posts WHERE id = %s", (pid,)).fetchone()


def test_report_only_records_the_score_and_never_drops(conn, policy, tmp_path):
    cid = creator(conn)
    post = _post(conn, cid)
    clip = _clip(tmp_path / "off.mp4", _envelope(), mouth=1 - _envelope())   # inverted: out of step
    out = sync.review_sync(conn, policy, post, clip, extractor=brightness)
    assert out["accepted"] is False and out["passed"] is True and out["enforced"] is False
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (post["id"],)).fetchone()["body"]
    assert body["sync_review"]["accepted"] is False and post["body"]["sync_review"] == body["sync_review"]
    row = conn.execute("SELECT r.passed, r.details FROM reviews r JOIN assets a ON a.id = r.asset_id "
                       "WHERE a.creator_id = %s AND r.gate = 'final_lip_sync'", (cid,)).fetchone()
    assert row["passed"] is True and row["details"]["would_pass"] is False
    assert conn.execute("SELECT count(*) AS n FROM decisions WHERE kind = 'review.sync' AND creator_id = %s",
                        (cid,)).fetchone()["n"] == 1

    enforced = sync.review_sync(conn, _with_sync(policy, enforce=True), post, clip, extractor=brightness)
    assert enforced["passed"] is False
    good = sync.review_sync(conn, _with_sync(policy, enforce=True), post, _clip(tmp_path / "in.mp4", _envelope()),
                            extractor=brightness)
    assert good["passed"] is True and good["accepted"] is True


def test_other_formats_and_failures_are_skipped_not_fatal(conn, policy, tmp_path):
    cid = creator(conn)
    assert sync.review_sync(conn, policy, _post(conn, cid, "story_clip"), tmp_path / "x.mp4")["skipped"] == "off"
    post = _post(conn, cid)

    def broken(frames, fps):
        raise RuntimeError("no landmarker")

    out = sync.review_sync(conn, policy, post, _clip(tmp_path / "in.mp4", _envelope()), extractor=broken)
    assert out["passed"] is True and "no landmarker" in out["error"]


def test_the_real_extractor_loads_mediapipe_only_when_it_runs(policy):
    code = ("import sys; from manyworlds.review import sync; from manyworlds.production import finish; "
            "sync.default_extractor(None); assert 'mediapipe' not in sys.modules, 'eager import'")
    subprocess.run([sys.executable, "-c", code], check=True)
    extract = sync.mediapipe_extractor("/nonexistent/face_landmarker.task")
    try:
        extract(iter([]), FPS)
    except ImportError:
        pass   # no `sync` extra here: review_sync records the error
    except FileNotFoundError as e:
        assert "face_landmarker" in str(e)   # mediapipe installed: the model path is checked
    else:
        raise AssertionError("a missing model must not score")
