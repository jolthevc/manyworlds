"""The finished file is judged once, as it will be posted: text over the face, seams at the joins, sound against
picture at the end, and whether it reads as real phone footage. Report-only until enforced."""

from __future__ import annotations

import subprocess

from psycopg.types.json import Jsonb

from manyworlds import media, policy as policy_mod
from manyworlds.review import final

from test_media import _clip
from test_production import creator


class FinalStub:
    """Stands in for the judge: returns the given scores, records calls."""

    def __init__(self, **kw):
        self.kw, self.calls = kw, []

    def model_for(self, role):
        return f"stub-{role}"

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        self.calls.append({"role": role, "content": content})
        return schema(**{"face_covered": False, "phone_real": 8, "seams": 8, "end_sync": 8, **self.kw})


def _with_final(policy, **over):
    review = policy.body["review"]
    return policy_mod.from_dict({**policy.body, "review": {**review, "final": {**review["final"], **over}}})


def _post(conn, cid, with_asset=True):
    pid = conn.execute("INSERT INTO posts (creator_id, platform, format, status, body) "
                       "VALUES (%s, 'instagram', 'talking_video', 'producing', %s) RETURNING id",
                       (cid, Jsonb({"script": "hi"}))).fetchone()["id"]
    if with_asset:
        conn.execute("INSERT INTO assets (creator_id, kind, uri, sha256, bytes) VALUES (%s, 'video', %s, 'x', 1)",
                     (cid, f"s3://b/assets/{cid}/post-{pid}/final.mp4"))
    return conn.execute("SELECT * FROM posts WHERE id = %s", (pid,)).fetchone()


SPANS = [{"text": "so here's the thing", "start_s": 0.1, "end_s": 0.9, "kind": "caption"},
         {"text": "fight me", "start_s": 1.2, "end_s": 1.9, "kind": "caption"},
         {"text": "11:40 PM", "start_s": 0.0, "end_s": 2.0, "kind": "label"}]


def test_frames_are_taken_at_the_joins_the_text_and_the_end(policy):
    times = final.sample_times(10.0, policy, joins=[4.0, {"at_s": 7.0}], text_spans=SPANS)
    whys = " | ".join(t["why"] for t in times)
    off = float(policy.get("review", "final", "join_offset_seconds"))
    at = {round(t["t"], 3) for t in times}
    assert {round(4.0 - off, 3), round(4.0 + off, 3), round(7.0 - off, 3), round(7.0 + off, 3)} <= at
    assert "join 2, after" in whys and "fight me" in whys and "end, last moment" in whys
    assert [t["t"] for t in times] == sorted(t["t"] for t in times) and max(at) < 10.0
    assert len(times) <= int(policy.get("review", "final", "max_frames"))
    gap = float(policy.get("review", "final", "min_gap_seconds"))
    assert all(b["t"] - a["t"] >= gap for a, b in zip(times, times[1:]))
    capped = final.sample_times(10.0, _with_final(policy, max_frames=3), joins=[4.0], text_spans=SPANS)
    assert len(capped) == 3 and all(t["why"] != "spread" for t in capped)   # the even spread goes first
    assert final.sample_times(0, policy) == []


def test_the_end_drift_is_measured_between_the_audio_and_video_streams(policy, tmp_path):
    good = _clip(tmp_path / "good.mp4", seconds=2.0)
    assert final.end_drift(good, policy)["drift_s"] <= float(policy.get("review", "final", "max_end_drift_seconds"))
    bad = tmp_path / "bad.mp4"   # 2 s of picture, 1 s of sound
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x288:rate=30:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(bad)], check=True)
    d = final.end_drift(bad, policy)
    assert d["drift_s"] > 0.5 and d["video_s"] > d["audio_s"]
    silent = _clip(tmp_path / "silent.mp4", seconds=1.0, audio=False)
    assert final.end_drift(silent, policy)["drift_s"] is None


def test_the_finished_file_is_judged_once_and_recorded_report_only(conn, policy, tmp_path):
    cid = creator(conn)
    post = _post(conn, cid)
    video = _clip(tmp_path / "final.mp4", seconds=2.0)
    llm = FinalStub(face_covered=True, covered_frames=[2], phone_real=8)
    out = final.review_final(conn, policy, llm, post, video, joins=[1.0], text_spans=SPANS)
    assert out["accepted"] is False and out["passed"] is True and out["enforced"] is False   # report-only
    assert out["gates"] == {"face_clear": False, "seams": True, "sync": True, "real": True}
    assert len(llm.calls) == 1 and llm.calls[0]["role"] == policy.get("review", "final", "role")
    content = llm.calls[0]["content"]
    texts = [str(b.get("text", "")) for b in content]
    assert sum(t.startswith("FRAME") for t in texts) == len(out["frames"]) > 0
    assert all(b["source"]["media_type"] == "image/jpeg" for b in content if b.get("type") == "image")
    assert any(t.startswith("JOINS: 1") for t in texts) and any('"fight me"' in t and t.startswith("LAST WORDS")
                                                               for t in texts)
    assert any(t.startswith("TEXT ON SCREEN") and "11:40 PM" in t for t in texts)
    # on the post (and on the caller's body), as rows on the finished asset, and as a decision
    stored = conn.execute("SELECT body FROM posts WHERE id = %s", (post["id"],)).fetchone()["body"]
    assert stored["final_review"]["gates"]["face_clear"] is False and stored["script"] == "hi"
    assert post["body"]["final_review"]["accepted"] is False
    rows = {r["gate"]: r for r in conn.execute("SELECT gate, passed, details FROM reviews").fetchall()}
    assert set(rows) == {"final_face_clear", "final_seams", "final_sync", "final_real"}
    assert rows["final_face_clear"]["passed"] is True and rows["final_face_clear"]["details"]["would_pass"] is False
    d = conn.execute("SELECT kind, output, policy_version FROM decisions").fetchone()
    assert d["kind"] == "review.final" and d["output"]["accepted"] is False and d["policy_version"] == policy.version


def test_enforced_a_failed_gate_fails_the_post_and_a_drifting_end_fails_sync(conn, policy, tmp_path):
    cid = creator(conn)
    strict = _with_final(policy, enforce=True)
    good = _clip(tmp_path / "good.mp4", seconds=2.0)
    out = final.review_final(conn, strict, FinalStub(), _post(conn, cid), good, joins=[], text_spans=[])
    assert out["passed"] is True and out["accepted"] is True and out["seams"] is None   # no joins: no seam score
    out = final.review_final(conn, strict, FinalStub(phone_real=2), _post(conn, cid), good, joins=[], text_spans=[])
    assert out["passed"] is False and out["gates"]["real"] is False
    bad = tmp_path / "bad.mp4"
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x288:rate=30:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(bad)], check=True)
    out = final.review_final(conn, strict, FinalStub(), _post(conn, cid, with_asset=False), bad, joins=[],
                             text_spans=[])
    assert out["gates"]["sync"] is False and out["passed"] is False
    assert conn.execute("SELECT count(*) AS n FROM decisions WHERE kind = 'review.final'").fetchone()["n"] == 3


def test_the_final_review_can_be_switched_off(conn, policy, tmp_path):
    cid = creator(conn)
    llm = FinalStub()
    out = final.review_final(conn, _with_final(policy, enabled=False), llm, _post(conn, cid),
                             tmp_path / "missing.mp4", joins=[], text_spans=[])
    assert out == {"skipped": "off", "passed": True} and llm.calls == []


def test_the_shipped_final_review_is_report_only(raw_policy):
    assert raw_policy.get("review", "final", "enabled") is True
    assert raw_policy.get("review", "final", "enforce") is False
