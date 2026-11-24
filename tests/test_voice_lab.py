"""A designed voice per creator (creator/voice_lab.py): candidates from the bible, picked automatically (within the
WER limit, unlike every other creator's voice, then lowest WER, then least monotone), Kokoro until then."""

from __future__ import annotations

import uuid

import pytest
from psycopg.types.json import Jsonb

from manyworlds import voices
from manyworlds.creator import lab, voice_lab

BIBLE = {"display_name": "Maren", "one_line": "slow mornings, fast opinions",
         "appearance": {"apparent_age": 27, "gender_presentation": "woman"},
         "voice_notes": "warm and slightly husky (laughs mid-sentence), unhurried"}
KOKORO = {"voice": {"engine": "kokoro", "voice": "af_bella:0.6+af_nova:0.4"}}
OTHER, FAR = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]


def creator(conn, anchors=None):
    return conn.execute("INSERT INTO creators (handle, bible, anchors) VALUES (%s, %s, %s) RETURNING id",
                        (f"c{uuid.uuid4().hex[:6]}", Jsonb(BIBLE), Jsonb(anchors or KOKORO))).fetchone()["id"]


def finished_design(conn, cid, candidates, status="done"):
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('voice.design', %s, %s, %s) "
                       "RETURNING id", (status, cid, Jsonb({"description": "warm, husky", "text": "Okay so this happened."}))
                       ).fetchone()["id"]
    for i, meta in enumerate(candidates):
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                     "VALUES (%s, %s, 'audio', %s, 'x', 1, %s)", (job, cid, f"s3://b/voice/{i:02d}.wav", Jsonb({"index": i, **meta})))
    return job


def anchors(conn, cid):
    return conn.execute("SELECT anchors FROM creators WHERE id = %s", (cid,)).fetchone()["anchors"]


def test_design_asks_for_candidates_from_the_bible(conn, policy):
    cid = creator(conn)
    job = voice_lab.design(conn, policy, cid)
    row = conn.execute("SELECT kind, payload, creator_id FROM jobs WHERE id = %s", (job,)).fetchone()
    p = row["payload"]
    assert row["kind"] == "voice.design" and row["creator_id"] == cid
    assert p["n"] == policy.get("voice", "candidates") and p["text"] == policy.get("voice", "anchor_text").strip()
    assert "27-year-old woman" in p["description"] and "husky" in p["description"]
    assert "(" not in p["description"] and ")" not in p["description"]     # parentheses delimit it for the model
    assert p["params"]["model"] == "openbmb/VoxCPM2"
    assert p["params"]["asr_model"] == policy.get("review", "video", "intelligibility", "model")   # the gate's ASR
    assert voice_lab.design(conn, policy, cid) == job                     # the same bible designs once


def test_candidates_rank_by_wer_limit_distance_then_wer_then_pitch_variance():
    cands = [{"index": 0, "wer": 0.3, "embedding": FAR, "pitch_var": 5.0},     # says the passage badly
             {"index": 1, "wer": 0.0, "embedding": OTHER, "pitch_var": 9.0},   # sounds like another creator
             {"index": 2, "wer": 0.05, "embedding": FAR, "pitch_var": 3.0},
             {"index": 3, "wer": 0.0, "embedding": FAR, "pitch_var": 1.0},
             {"index": 4, "wer": 0.0, "embedding": FAR, "pitch_var": 2.0}]
    ranked = voice_lab.rank(cands, [OTHER], max_wer=0.1, min_distance=0.35)
    assert [c["index"] for c in ranked] == [4, 3, 2]
    assert ranked[0]["distance"] == 1.0
    # with no other creator's voice, distance does not exclude anyone
    assert [c["index"] for c in voice_lab.rank(cands, [], max_wer=0.1, min_distance=0.35)] == [1, 4, 3, 2]


def test_pick_stores_the_best_candidate_as_the_creators_voice_and_records_why(conn, policy):
    creator(conn, {"voice": {"engine": "voxcpm", "ref_uri": "s3://b/o.wav", "ref_text": "x", "embedding": OTHER}})
    cid = creator(conn)
    job = finished_design(conn, cid, [{"wer": 0.3, "embedding": FAR, "pitch_var": 5.0},
                                      {"wer": 0.0, "embedding": OTHER, "pitch_var": 9.0},
                                      {"wer": 0.05, "embedding": FAR, "pitch_var": 3.0},
                                      {"wer": 0.0, "embedding": FAR, "pitch_var": 1.0},
                                      {"wer": 0.0, "embedding": FAR, "pitch_var": 2.0}])
    out = voice_lab.pick(conn, policy, cid, job)
    assert out["picked"]["index"] == 4 and out["ranked"] == [4, 3, 2]
    v = anchors(conn, cid)["voice"]
    assert (v["engine"], v["ref_uri"], v["ref_text"], v["description"]) == (
        "voxcpm", "s3://b/voice/04.wav", "Okay so this happened.", "warm, husky")
    assert v["embedding"] == FAR and v["voice"] == KOKORO["voice"]["voice"]   # Kokoro kept as the fallback
    assert voices.engine(anchors(conn, cid)) == "voxcpm"
    voice_lab.pick(conn, policy, cid, {"id": job})                        # a rerun decides nothing new
    rows = conn.execute("SELECT kind, output FROM decisions WHERE job_id = %s", (job,)).fetchall()
    assert len(rows) == 1 and rows[0]["kind"] == "voice_pick" and rows[0]["output"]["picked"]["index"] == 4


def test_no_voice_is_stored_when_no_candidate_qualifies(conn, policy):
    cid = creator(conn)
    job = finished_design(conn, cid, [{"wer": 0.4, "embedding": FAR, "pitch_var": 2.0}])
    out = voice_lab.pick(conn, policy, cid, job)
    assert out["picked"] is None and anchors(conn, cid) == KOKORO
    assert conn.execute("SELECT count(*) AS n FROM decisions WHERE job_id = %s", (job,)).fetchone()["n"] == 1


def test_an_unfinished_design_is_not_picked_from(conn, policy):
    cid = creator(conn)
    job = finished_design(conn, cid, [], status="running")
    assert voice_lab.pick(conn, policy, cid, job)["picked"] is None
    assert conn.execute("SELECT count(*) AS n FROM decisions WHERE job_id = %s", (job,)).fetchone()["n"] == 0


def test_kokoro_until_voice_designed(conn, policy):
    """Review Focus 4: a creator without a designed voice (Maren today) keeps speaking with Kokoro, also while a
    design is pending."""
    cid = creator(conn)
    voice_lab.design(conn, policy, cid)
    assert voices.engine(anchors(conn, cid)) == "kokoro"
    sample = conn.execute("SELECT kind, payload FROM jobs WHERE id = %s",
                          (lab.enqueue_voice_sample(conn, cid),)).fetchone()
    assert sample["kind"] == "image.voice" and sample["payload"]["voice"] == KOKORO["voice"]["voice"]
    assert voices.engine({"voice": {"engine": "voxcpm"}}) == "kokoro"       # no reference clip yet: not usable
    assert voices.engine(None) == "kokoro"


def test_a_speak_payload_carries_the_designed_voice_and_the_takes(policy):
    voice = {"engine": "voxcpm", "ref_uri": "s3://b/v.wav", "ref_text": "Okay so this happened."}
    p = voice_lab.speak_payload(policy, voice, [{"text": "Hi.", "delivery": "softly"}, {"text": "Bye."}], seed=3)
    assert (p["ref_uri"], p["ref_text"], p["takes"], p["seed"]) == ("s3://b/v.wav", voice["ref_text"],
                                                                    policy.get("voice", "takes"), 3)
    assert p["lines"] == [{"text": "Hi."}, {"text": "Bye."}]      # ultimate cloning: the delivery would be read aloud
    assert p["params"]["pause_s"] == policy.get("voice", "pause_s")


def test_deliveries_reach_the_model_only_where_it_follows_them(policy):
    """Breadth rerun 2026-11-24: with ultimate cloning VoxCPM read the delivery aloud ("A quiet, sincere, one thing
    nobody tells you...") and every directed line failed intelligibility. Controllable cloning keeps it."""
    from manyworlds import policy as policy_mod

    voice = {"ref_uri": "s3://b/v.wav", "ref_text": "Okay so this happened."}
    lines = [{"text": "One thing nobody tells you.", "delivery": "quiet, sincere"},
             {"text": "(musing) Walking home.", "pause_after_s": 0.4}]
    assert voice_lab.speak_payload(policy, voice, lines)["lines"] == [
        {"text": "One thing nobody tells you."}, {"text": "Walking home.", "pause_after_s": 0.4}]
    body = dict(policy.body)
    body["voice"] = {**body["voice"], "clone": "controllable"}
    ctl = policy_mod.from_dict(body)
    assert voice_lab.speak_payload(ctl, voice, lines)["lines"] == [
        {"text": "One thing nobody tells you.", "delivery": "quiet, sincere"},
        {"text": "(musing) Walking home.", "pause_after_s": 0.4}]


def test_the_speak_payload_carries_natural_speech_as_policy_sets_it(policy):
    """Item 7 (plan 2026-11-22-video-quality-autonomous): clause splitting, the gap ranges and the line ceiling reach
    the worker from policy and parse there; breaths and fillers are experimental and ship off."""
    from manyworlds.worker.handlers import tts_voxcpm

    voice = {"ref_uri": "s3://b/v.wav", "ref_text": "Hi."}
    params = voice_lab.speak_payload(policy, voice, [{"text": "Hi."}])["params"]
    for key in ("synthesis", "clone", "clause_gap_s", "sentence_gap_s", "trim", "fade_ms", "line_max_wer", "breaths",
                "disfluency"):
        assert params[key] == policy.get("voice", key)
    cfg = tts_voxcpm.speech_config(params)
    assert cfg["clause_gap"][1] < cfg["sentence_gap"][0]                            # clause pauses shorter
    assert cfg["breaths"] is None and cfg["disfluency"] is None
    assert cfg["max_wer"] == policy.get("voice", "max_wer") and cfg["retry_factor"] >= 1
    # whole lines until the comparison (breadth test), and every line cloned with the reference's transcript
    assert policy.get("voice", "synthesis") == "whole_line" and cfg["split"] is False
    assert tts_voxcpm.clone_mode(params) == "ultimate"


def test_a_comparison_is_asked_for_only_by_the_breadth_test(policy):
    voice = {"ref_uri": "s3://b/v.wav", "ref_text": "Hi."}
    assert "compare" not in voice_lab.speak_payload(policy, voice, [{"text": "Hi."}])
    assert voice_lab.speak_payload(policy, voice, [{"text": "Hi."}], compare=True)["compare"] is True


def test_candidates_list_every_candidate_with_its_scores_best_first(conn, policy):
    creator(conn, {"voice": {"engine": "voxcpm", "ref_uri": "s3://b/o.wav", "ref_text": "x", "embedding": OTHER}})
    cid = creator(conn)
    job = finished_design(conn, cid, [{"wer": 0.3, "embedding": FAR, "pitch_var": 5.0, "seconds": 14.0},
                                      {"wer": 0.0, "embedding": OTHER, "pitch_var": 9.0},
                                      {"wer": 0.05, "embedding": FAR, "pitch_var": 3.0},
                                      {"wer": 0.0, "embedding": FAR, "pitch_var": 1.0}])
    listed = voice_lab.candidates(conn, policy, job)
    assert [c["index"] for c in listed] == [3, 2, 1, 0]                   # qualifying by WER first, then the rest
    assert [c["qualifies"] for c in listed] == [True, True, False, False]
    first = listed[0]
    assert first["uri"] == "s3://b/voice/03.wav" and first["wer"] == 0.0 and first["distance"] == 1.0
    assert listed[2]["distance"] == 0.0                                    # sounds like the other creator
    assert set(first) == {"index", "uri", "wer", "distance", "pitch_var", "seconds", "transcript", "qualifies"}
    assert voice_lab.candidates(conn, policy, finished_design(conn, cid, [], status="running")) == []


def test_the_owner_can_pick_a_candidate_by_index_but_never_one_like_another_creator(conn, policy):
    creator(conn, {"voice": {"engine": "voxcpm", "ref_uri": "s3://b/o.wav", "ref_text": "x", "embedding": OTHER}})
    cid = creator(conn)
    job = finished_design(conn, cid, [{"wer": 0.3, "embedding": FAR, "pitch_var": 5.0},
                                      {"wer": 0.0, "embedding": OTHER, "pitch_var": 9.0},
                                      {"wer": 0.0, "embedding": FAR, "pitch_var": 1.0}])
    out = voice_lab.pick(conn, policy, cid, job, candidate=0)            # over max_wer, but the owner heard it
    assert out["picked"]["index"] == 0 and out["override"] is True
    assert anchors(conn, cid)["voice"]["ref_uri"] == "s3://b/voice/00.wav"
    too_close = voice_lab.pick(conn, policy, cid, job, candidate=1)
    assert too_close["picked"] is None and "min_distance" in too_close["reason"]
    assert anchors(conn, cid)["voice"]["ref_uri"] == "s3://b/voice/00.wav"   # unchanged
    with pytest.raises(ValueError):
        voice_lab.pick(conn, policy, cid, job, candidate=9)
    rows = conn.execute("SELECT step, output FROM decisions WHERE job_id = %s", (job,)).fetchall()
    assert [r["step"] for r in rows] == ["voice_pick:candidate:0"] and rows[0]["output"]["override"] is True
    # the automatic pick is still its own decision
    assert voice_lab.pick(conn, policy, cid, job)["picked"]["index"] == 2
