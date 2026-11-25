"""Automatic casting on the control plane (creator/voice_lab.cast, pick_cast, settle_casts; owner 2026-11-25: no
human selection gates), the engine payloads production sends, the launch's voice step, and the bake-off that
flips the default engine (manyworlds/voice_bakeoff.py)."""

from __future__ import annotations

import uuid

from psycopg.types.json import Jsonb

from manyworlds import voices
from manyworlds.creator import voice_lab

from voice_helpers import cast_meta, finished_cast

BIBLE = {"display_name": "Maren", "one_line": "slow mornings, fast opinions",
         "appearance": {"apparent_age": 27, "gender_presentation": "woman"},
         "voice_notes": "warm and slightly husky, unhurried"}
KOKORO = {"voice": {"engine": "kokoro", "voice": "af_bella:0.6+af_nova:0.4"}}


def creator(conn, anchors=None, bible=None):
    return conn.execute("INSERT INTO creators (handle, bible, anchors) VALUES (%s, %s, %s) RETURNING id",
                        (f"c{uuid.uuid4().hex[:6]}", Jsonb(bible or BIBLE), Jsonb(anchors or KOKORO))).fetchone()["id"]


def anchors(conn, cid):
    return conn.execute("SELECT anchors FROM creators WHERE id = %s", (cid,)).fetchone()["anchors"]


def test_casting_waits_for_the_bakeoff_then_uses_its_winner(conn, policy):
    assert policy.get("voice", "engine") == "indextts"
    assert voice_lab.default_engine(conn, policy) == policy.get("voice", "engine_gate", "until_bakeoff")
    conn.execute("INSERT INTO decisions (kind, output, step) VALUES ('voice_engine', %s, 'voice_engine:x')",
                 (Jsonb({"winner": "indextts"}),))
    assert voice_lab.default_engine(conn, policy) == "indextts"


def test_a_cast_asks_for_candidates_from_the_bible_with_the_casting_script(conn, policy):
    cid = creator(conn)
    job = voice_lab.cast(conn, policy, cid, engine="indextts", batch="test-voice-x")
    j = conn.execute("SELECT kind, payload, batch FROM jobs WHERE id = %s", (job,)).fetchone()
    p = j["payload"]
    assert j["kind"] == "voice.cast" and p["engine"] == "indextts" and p["purpose"] == "creator"
    assert p["designer"] == policy.get("voice", "casting", "designer")
    assert "27-year-old woman" in p["description"] and p["n"] == policy.get("voice", "casting", "candidates")
    assert [s["name"] for s in p["script"]] == [s["name"] for s in policy.get("voice", "casting", "script")]
    by_name = {s["name"]: s["emotion"] for s in p["script"]}
    assert by_name["neutral"] is None and by_name["excited"]["name"] == "excited"
    assert set(p["params"]["engines"]) == {"indextts", "qwen3"}          # the engine and the designer
    assert p["params"]["line_check"]["repairs"] == policy.get("voice", "line_check", "repairs")
    assert voice_lab.cast(conn, policy, cid, engine="indextts") == job               # idempotent
    assert voice_lab.cast(conn, policy, cid, engine="indextts", attempt=1) != job    # a new attempt, new seeds


def test_the_best_qualifying_candidate_becomes_her_voice_and_the_decision_says_why(conn, policy):
    cid = creator(conn)
    job = finished_cast(conn, cid, [cast_meta(0, gender="male", p=0.95, mos=4.5),     # best numbers, wrong gender
                                    cast_meta(1, mos=3.6), cast_meta(2, mos=4.2)])
    out = voice_lab.pick_cast(conn, policy, cid, job)
    assert out["picked"]["index"] == 2 and out["ranked"] == [2, 1]
    v = anchors(conn, cid)["voice"]
    assert v["engine"] == "indextts" and v["cast_job_id"] == str(job) and v["voice"] == KOKORO["voice"]["voice"]
    assert v["pace_wps"] and v["fit"] == {"gender": "female", "age": 27.0} and v["scores"]["qualifies"]
    assert voices.engine(anchors(conn, cid)) == "indextts" and voices.designed(anchors(conn, cid))
    d = conn.execute("SELECT kind, inputs, rationale FROM decisions WHERE job_id = %s", (job,)).fetchone()
    assert d["kind"] == "voice_cast" and "candidate 2" in d["rationale"]
    assert any("sounds male" in r for c in d["inputs"]["candidates"] if c["index"] == 0 for r in c["reasons"])


def test_no_voice_is_stored_when_nothing_qualifies(conn, policy):
    cid = creator(conn)
    job = finished_cast(conn, cid, [cast_meta(0, gender="male"), cast_meta(1, mos=2.0)])
    out = voice_lab.pick_cast(conn, policy, cid, job)
    assert out["picked"] is None and "no candidate qualified" in out["reason"]
    assert voices.engine(anchors(conn, cid)) == "kokoro"


def test_a_voice_too_like_another_creators_is_passed_over_but_distance_is_never_maximised(conn, policy):
    creator(conn, {"voice": {"engine": "indextts", "ref_uri": "s3://o.wav", "ref_text": "x",
                             "embedding": [1.0, 0.0, 0.0]}})
    cid = creator(conn)
    job = finished_cast(conn, cid, [cast_meta(0, embedding=[1.0, 0.01, 0.0], mos=4.5),   # her twin
                                    cast_meta(1, embedding=[0.5, 0.866, 0.0], mos=4.2),  # 0.5 away: enough
                                    cast_meta(2, embedding=[0.0, 0.0, 1.0], mos=3.6)])   # far, but worse
    assert voice_lab.pick_cast(conn, policy, cid, job)["picked"]["index"] == 1


def test_finished_casts_settle_by_themselves_but_a_bakeoff_cast_never_replaces_a_voice(conn, policy):
    cid, other = creator(conn), creator(conn)
    finished_cast(conn, cid, [cast_meta(0)])
    finished_cast(conn, other, [cast_meta(0)], purpose="bakeoff")
    out = voice_lab.settle_casts(conn, policy)
    assert out == {"judged": 1, "picked": 1}
    assert voices.engine(anchors(conn, cid)) == "indextts" and voices.engine(anchors(conn, other)) == "kokoro"
    assert voice_lab.settle_casts(conn, policy) == {"judged": 0, "picked": 0}       # judged once


def test_an_indextts_voice_speaks_through_its_own_kind_with_emotions_never_words(policy):
    voice = {"engine": "indextts", "ref_uri": "s3://b/v.wav", "ref_text": "Okay so.", "pace_wps": [3.4, 4.2]}
    p = voice_lab.speak_payload(policy, voice, [
        {"text": "One thing nobody tells you.", "delivery": "quiet, sincere"},
        {"text": "(amused) I tripped.", "pause_after_s": 0.4}, {"text": "Okay."}])
    assert voice_lab.speak_kind(policy, "indextts") == "voice.speak.indextts"
    assert [line["text"] for line in p["lines"]] == ["One thing nobody tells you.", "I tripped.", "Okay."]
    assert [(line.get("emotion") or {}).get("name") for line in p["lines"]] == ["confiding", "amused", None]
    assert p["lines"][1]["delivery"] == "amused" and p["lines"][1]["pause_after_s"] == 0.4
    assert p["params"]["line_check"]["pace_wps"] == [3.4, 4.2] and "indextts" in p["params"]["engines"]
    assert "emotions" not in p["params"]["engines"]["indextts"]


def test_a_voxcpm_voice_keeps_its_payload_and_gains_the_line_check(policy):
    voice = {"engine": "voxcpm", "ref_uri": "s3://b/v.wav", "ref_text": "Okay so."}
    p = voice_lab.speak_payload(policy, voice, [{"text": "Hi.", "delivery": "softly"}])
    assert p["lines"] == [{"text": "Hi."}] and p["params"]["line_check"]["enabled"] is True
    assert voice_lab.speak_kind(policy) == voice_lab.speak_kind(policy, "voxcpm") == "voice.speak"


def test_production_voices_an_indextts_creator_with_its_kind(conn, policy):
    from manyworlds.production import voice as pv
    cid = creator(conn, {"voice": {"engine": "indextts", "ref_uri": "s3://b/v.wav", "ref_text": "Okay so."}})
    post = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'talking_video', 'planned', "
                        "'{}') RETURNING *", (cid,)).fetchone()
    assert pv._designed_voice(conn, post)["engine"] == "indextts"
    job = pv._voice_job(conn, policy, post, pv._designed_voice(conn, post), [{"text": "Hi there."}], 1)
    assert conn.execute("SELECT kind FROM jobs WHERE id = %s", (job,)).fetchone()["kind"] == "voice.speak.indextts"


def test_the_new_kinds_run_on_their_own_worker_family(policy):
    """Never on an image older than their handlers: talkcomfy v20 would crash on a kind it has no handler for."""
    for kind in ("voice.speak.indextts", "voice.speak.qwen3", "voice.cast"):
        assert policy.family(kind) == "tts"
    # talkcomfy v23 is the first image with these handlers (built 2026-11-25 from the merged release)
    assert int(str(policy.get("worker_images", "tts", "tag")).lstrip("v")) >= 23


def test_every_engine_records_its_licence(policy):
    for name, e in policy.get("voice", "engines").items():
        assert e.get("licence") and isinstance(e.get("licence_ok"), bool), name
