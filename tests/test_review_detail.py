"""Finer checks (manyworlds.review.detail): hands and teeth, lips closing on b/p/m, background flicker, and set and
wardrobe consistency against reference images. Each is recorded as its own row and rejects only when enforced;
outputs without the signal behave exactly as before."""

from __future__ import annotations

import io
import uuid

import numpy as np
import pytest
from PIL import Image, ImageDraw
from psycopg.types.json import Jsonb

from manyworlds import policy as policy_mod
from manyworlds.review import detail, gates, video
from manyworlds.worker import lipsync

from test_production import creator, real_refs
from test_review import JudgeStub, creator_with_job, ok
from test_video_review import ClipStub, _put, clip, clip_job, rendered


def _detail(policy, **checks):
    d = {**policy.body["review"]["detail"]}
    for name, over in checks.items():
        d[name] = {**d.get(name, {}), **over}
    return policy_mod.from_dict({**policy.body, "review": {**policy.body["review"], "detail": d}})


def _gates(conn, asset_id):
    return {x["gate"]: x for x in conn.execute("SELECT gate, passed, score, details FROM reviews WHERE asset_id = %s",
                                               (asset_id,))}


def test_every_check_ships_switched_on_but_not_enforced(policy):
    for name in ("hands", "teeth", "lip_closure", "background", "consistency"):
        c = policy.get("review", "detail", name)
        assert c["enabled"] is True and c["enforce"] is False


def test_a_check_that_isnt_enforced_records_but_never_rejects(policy):
    assert detail.verdict(policy, "hands", 2) == (True, {"would_pass": False, "enforced": False, "bar": 5.0})
    assert detail.verdict(policy, "hands", 8)[1]["would_pass"] is True
    assert detail.verdict(policy, "hands", None) is None                                 # no hands in frame
    on = _detail(policy, hands={"enforce": True})
    assert detail.verdict(on, "hands", 2)[0] is False and detail.verdict(on, "hands", 5)[0] is True
    off = _detail(policy, hands={"enabled": False})
    assert detail.verdict(off, "hands", 2) is None
    bar = policy.get("review", "detail", "background", "max_flicker")
    assert detail.verdict(policy, "background", bar + 0.1, lower_is_better=True)[1]["would_pass"] is False


# --- (a) hands and teeth, from the judge ----------------------------------------------------------------------

def test_hands_and_teeth_scores_become_rows_only_when_the_judge_gives_them(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = clip_job(conn, store, tmp_path, cid)
    rendered(conn, store, tmp_path, job)
    llm = ClipStub(clip(hands=2, defects=["keyframe 2: six fingers on the left hand"]), clip(hands=3))
    [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
    assert r["accepted"] and r["hands"] is True and "teeth" not in r          # recorded, not enforced
    rows = _gates(conn, r["asset_id"])
    assert "teeth" not in rows and rows["hands"]["passed"] is True and rows["hands"]["score"] == 2
    assert rows["hands"]["details"]["would_pass"] is False and rows["hands"]["details"]["enforced"] is False
    ask = next(b["text"] for b in llm.calls[0]["content"] if str(b.get("text", "")).startswith("DETAIL CHECKS"))
    assert "`hands`" in ask and "`teeth`" in ask and "`consistency`" not in ask      # no references, no consistency


def test_enforced_bad_hands_reject_the_clip_at_the_screen(conn, policy, store, tmp_path):
    pol = _detail(policy, hands={"enforce": True})
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = clip_job(conn, store, tmp_path, cid)
    rendered(conn, store, tmp_path, job)
    llm = ClipStub(clip(hands=2), clip())
    [r] = video.review_job(conn, pol, llm, store, job, tmp_path)
    assert not r["accepted"] and r["hands"] is False and r["identity"] and len(llm.calls) == 1   # no confirm paid


def test_photos_get_hands_and_teeth_rows_and_enforced_teeth_reject(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path, n_images=2)
    judge = JudgeStub([ok(hands=9, teeth=2), ok()], confirm=[ok(hands=9, teeth=2), ok()])
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert [r["accepted"] for r in results] == [True, True]
    rows = _gates(conn, results[0]["asset_id"])
    assert rows["teeth"]["passed"] is True and rows["teeth"]["details"]["would_pass"] is False
    assert "hands" not in _gates(conn, results[1]["asset_id"])
    _, job = creator_with_job(conn, store, tmp_path, n_images=1)
    results = gates.review_job(conn, _detail(policy, teeth={"enforce": True}), JudgeStub([ok(teeth=2)]), store, job,
                               tmp_path)
    assert results[0]["accepted"] is False and results[0]["teeth"] is False


def test_a_switched_off_check_drops_the_judges_score_and_its_prompt(conn, policy, store, tmp_path):
    pol = _detail(policy, hands={"enabled": False}, teeth={"enabled": False})
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = clip_job(conn, store, tmp_path, cid)
    rendered(conn, store, tmp_path, job)
    llm = ClipStub(clip(hands=1, teeth=1), clip(hands=1, teeth=1))
    [r] = video.review_job(conn, pol, llm, store, job, tmp_path)
    assert r["accepted"] and "hands" not in _gates(conn, r["asset_id"])
    assert not any(str(b.get("text", "")).startswith("DETAIL CHECKS") for b in llm.calls[0]["content"])


# --- (b) lip closure on bilabials ------------------------------------------------------------------------------

def test_bilabial_onsets_follow_spelling_rules():
    assert lipsync.bilabial_onsets([{"text": "Bob", "start_s": 0.0, "end_s": 0.6}]) == pytest.approx([0.0, 0.4])
    assert lipsync.bilabial_onsets([{"text": "phone", "start_s": 0, "end_s": 1}]) == []          # ph is an f
    for word in ("summer", "jumped", "lamb", "happy"):                                          # one closure each
        assert len(lipsync.bilabial_onsets([{"text": word, "start_s": 0, "end_s": 1}])) == 1, word


def test_estimated_word_times_cover_each_line_or_the_script():
    words = lipsync.estimate_words("one two", 2.0)
    assert words[0]["start_s"] == 0 and words[-1]["end_s"] == pytest.approx(2.0)
    lines = [{"text": "hi there", "start_s": 0.5, "end_s": 1.5}, {"text": "bye", "start_s": 2.0, "end_s": 2.4}]
    words = lipsync.estimate_words(lines=lines)
    assert [w["text"] for w in words] == ["hi", "there", "bye"] and words[2]["start_s"] == 2.0


def test_lips_must_close_near_each_bilabial():
    fps = 25
    mouth = np.full(50, 0.3)
    mouth[[10, 32]] = 0.0                                       # closes at 0.4 s and at 1.28 s
    s = lipsync.lip_closure(mouth, fps, [0.4, 1.2, 1.8], tolerance_frames=2, closed_max=0.05)
    assert (s["bilabials"], s["closed"], s["share"]) == (3, 2, pytest.approx(0.667, abs=1e-3))
    assert s["missed_at_s"] == [1.8]
    # the sync lag shifts the window: the mouth that follows the voice by 2 frames still counts
    late = np.full(50, 0.3)
    late[[12, 45]] = 0.0                                        # the second closure is on time, so 2 frames early
    assert lipsync.lip_closure(late, fps, [0.4, 1.8], tolerance_frames=0, offset_frames=2)["share"] == 0.5
    # no face in the window: not counted; too few checked: no share
    blind = [None] * 50
    assert lipsync.lip_closure(blind, fps, [0.4], min_bilabials=1)["bilabials"] == 0
    assert lipsync.lip_closure(mouth, fps, [0.4], min_bilabials=3)["share"] is None
    # a mouth that opens wide closes relative to its own range
    wide = np.full(50, 0.6)
    wide[10] = 0.1
    assert lipsync.lip_closure(wide, fps, [0.4], relative_to_p90=0.25)["closed"] == 1


SCRIPT = "Maybe pop music by me, probably."


def _talk_with_mouth(conn, store, tmp_path, cid, openness):
    payload = {"image_uri": _put(store, tmp_path, b"\x89PNG start"), "script": SCRIPT, "scene": "at the counter"}
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('talk.infinitetalk', 'queued', %s, %s) "
                       "RETURNING id", (cid, Jsonb(payload))).fetchone()["id"]
    rendered(conn, store, tmp_path, job)
    sync = {"corr": 0.6, "offset_frames": 0, "face_share": 1.0, "frames": len(openness), "fps": 25.0,
            "openness": openness}
    conn.execute("UPDATE assets SET meta = %s WHERE job_id = %s AND kind = 'video'", (Jsonb({"sync": sync}), job))
    return job


def test_lip_closure_is_measured_from_the_mouth_series_and_the_script(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = _talk_with_mouth(conn, store, tmp_path, cid, [0.3] * 100)        # never closes
    llm = ClipStub(clip(), clip())
    [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
    assert r["accepted"] and r["lip_closure"] is True and llm.calls           # recorded only
    row = _gates(conn, r["asset_id"])["lip_closure"]
    assert row["score"] == 0.0 and row["details"]["would_pass"] is False and row["details"]["timing"] == "script"
    assert row["details"]["bilabials"] >= policy.get("review", "detail", "lip_closure", "min_bilabials")
    assert "openness" not in _gates(conn, r["asset_id"])["lip_sync"]["details"]     # the series stays on the asset


def test_enforced_lip_closure_rejects_before_any_paid_review(conn, policy, store, tmp_path):
    pol = _detail(policy, lip_closure={"enforce": True})
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    mumbling, closing = (_talk_with_mouth(conn, store, tmp_path, cid, [0.3] * 100),
                         _talk_with_mouth(conn, store, tmp_path, cid, [0.0, 0.0, 0.3, 0.3] * 25))
    llm = ClipStub(clip(), clip())
    [r] = video.review_job(conn, pol, llm, store, mumbling, tmp_path)
    assert not r["accepted"] and r["lip_closure"] is False and llm.calls == []
    [r] = video.review_job(conn, pol, llm, store, closing, tmp_path)
    assert r["accepted"] and r["lip_closure"] is True


def test_a_clip_without_a_mouth_series_or_script_gets_no_lip_closure_row(conn, policy, store, tmp_path):
    assert detail.lip_closure({"corr": 0.5}, policy, script=SCRIPT) is None                 # an older clip
    assert detail.lip_closure({"openness": [0.1] * 10, "fps": 25}, policy, script="") is None
    assert detail.lip_closure({"error": "no face"}, policy, script=SCRIPT) is None
    worker = {"bilabial": {"bilabials": 4, "closed": 4, "share": 1.0}, "openness": [0.3] * 10, "fps": 25}
    assert detail.lip_closure(worker, policy, script=SCRIPT)["timing"] == "words"           # the worker's own wins


# --- (c) background warp ---------------------------------------------------------------------------------------

def _scene(subject_x: int, background: str, seed: int = 0) -> bytes:
    w, h = 160, 284
    if background == "noise":
        rng = np.random.default_rng(seed)
        img = Image.fromarray((rng.random((h, w)) > 0.5).astype(np.uint8) * 255).convert("RGB")
    else:
        img = Image.new("RGB", (w, h), (200, 190, 170))
        d = ImageDraw.Draw(img)
        for x in range(0, w, 12):                  # shelves and door frames: straight lines
            d.line([(x, 0), (x, h)], fill=(60, 50, 40), width=2)
    ImageDraw.Draw(img).rectangle([subject_x, 60, subject_x + 40, 220], fill=(230, 180, 150))   # inside the box
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_background_flicker_ignores_the_subject_and_catches_a_warping_background(policy):
    steady, moved, warped = _scene(52, "lines"), _scene(68, "lines"), _scene(68, "noise")
    out = detail.background_flicker([steady, moved, warped], policy)
    assert out["pairs"][0] < 0.05                                   # only the subject moved
    assert out["pairs"][1] > policy.get("review", "detail", "background", "max_flicker")
    assert out["flicker"] == max(out["pairs"])
    assert detail.background_flicker([steady], policy) is None
    assert detail.background_flicker([b"jpg 0", b"jpg 1"], policy) is None        # not images: no measure
    assert detail.background_flicker([steady, moved], _detail(policy, background={"enabled": False})) is None


# --- (d) set and wardrobe consistency --------------------------------------------------------------------------

BODEGA = {"set": {"location": "a corner bodega", "set_id": "bodega"}}


def test_references_are_only_those_the_shot_names(policy):
    refs = [{"kind": "set", "uri": "s3://r/set.png"}, {"kind": "outfit", "uri": "s3://r/coat.png"}, {"kind": "x"}]
    assert detail.references_for({"references": refs}, BODEGA, policy) == [
        {"kind": "set", "id": "bodega", "uri": "s3://r/set.png"}]
    both = {**BODEGA, "wardrobe": {"outfit_id": "green-coat"}}
    assert [r["kind"] for r in detail.references_for({"references": refs}, both, policy)] == ["set", "outfit"]
    assert detail.references_for({"references": refs}, None, policy) == []
    assert detail.references_for({}, BODEGA, policy) == []
    assert detail.references_for({"references": refs}, both, _detail(policy, consistency={"enabled": False})) == []


def _directed_clip(conn, store, tmp_path, cid, references):
    payload = {"image_uri": _put(store, tmp_path, b"\x89PNG start"), "scene": "in the bodega", "shot": BODEGA,
               "references": references}
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('video.wan_i2v', 'queued', %s, %s) "
                       "RETURNING id", (cid, Jsonb(payload))).fetchone()["id"]
    rendered(conn, store, tmp_path, job)
    return job


def test_a_clip_is_compared_with_its_set_reference(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    refs = [{"kind": "set", "uri": _put(store, tmp_path, b"\x89PNG bodega")},
            {"kind": "outfit", "uri": _put(store, tmp_path, b"\x89PNG coat")}]
    job = _directed_clip(conn, store, tmp_path, cid, refs)
    llm = ClipStub(clip(consistency=3), clip(consistency=3))
    [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
    texts = [str(b.get("text", "")) for b in llm.calls[0]["content"]]
    assert "SET REFERENCE (bodega):" in texts and not any(t.startswith("OUTFIT REFERENCE") for t in texts)
    assert any(t.startswith("DETAIL CHECKS") and "SET REFERENCE (bodega)" in t for t in texts)
    row = _gates(conn, r["asset_id"])["consistency"]
    assert r["accepted"] and row["passed"] is True and row["score"] == 3 and row["details"]["would_pass"] is False
    pol = _detail(policy, consistency={"enforce": True})
    job = _directed_clip(conn, store, tmp_path, cid, refs)
    [r] = video.review_job(conn, pol, ClipStub(clip(consistency=3)), store, job, tmp_path)
    assert not r["accepted"] and r["consistency"] is False


def test_without_references_a_consistency_score_is_ignored(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = _directed_clip(conn, store, tmp_path, cid, [])
    [r] = video.review_job(conn, _detail(policy, consistency={"enforce": True}), ClipStub(clip(consistency=1)),
                           store, job, tmp_path)
    assert r["accepted"] and "consistency" not in _gates(conn, r["asset_id"])


def test_a_photo_is_compared_with_its_outfit_reference(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path, n_images=1)
    p = tmp_path / f"{uuid.uuid4().hex}.png"
    p.write_bytes(b"\x89PNG coat")
    coat = store.put_file(p, f"t/{p.name}").uri
    conn.execute("UPDATE jobs SET payload = %s WHERE id = %s",
                 (Jsonb({"shot": {"wardrobe": {"outfit_id": "green-coat"}},
                         "references": [{"kind": "outfit", "uri": coat}]}), job))
    judge = JudgeStub([ok(consistency=9)])
    [r] = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert r["accepted"] and _gates(conn, r["asset_id"])["consistency"]["score"] == 9
    assert any(b.get("text") == "OUTFIT REFERENCE (green-coat):" for b in judge.calls[0]["content"])


# --- (f) hand and mouth close-ups for the judge (review.detail.crops, worker/crops.py) ---------------------------

def test_a_crop_box_is_square_padded_and_inside_the_frame():
    from manyworlds.worker import crops
    assert crops.square_box([(0.5, 0.5), (0.6, 0.55)], 1000, 1000, margin=0.5) == (450, 425, 650, 625)
    x0, y0, x1, y1 = crops.square_box([(0.0, 0.98), (0.02, 1.0)], 480, 832, margin=0.5, min_side=64)
    assert x1 - x0 == y1 - y0 == 64 and x0 == 0 and y1 == 832                  # moved inside, not shrunk
    assert crops.square_box([(1.5, 0.5)], 480, 832) is None                     # nothing in the frame
    assert crops.square_box([(0.1, 0.1), (0.9, 0.9)], 480, 832)[2] - crops.square_box(
        [(0.1, 0.1), (0.9, 0.9)], 480, 832)[0] == 480                            # never larger than the frame


def test_the_worker_turns_landmark_crops_into_crop_outputs(tmp_path):
    from manyworlds.worker import crops
    from manyworlds.worker.runner import OutputFile
    keys = [OutputFile(tmp_path / f"keyframe_{i}.jpg", "keyframe", {"index": i}) for i in range(2)]
    made = tmp_path / "crops" / "crop_1_hands_1.png"
    made.parent.mkdir()
    made.write_bytes(b"\x89PNG hand")
    seen = {}

    def run(spec):
        import json
        seen.update(json.loads(spec.read_text()))
        return [{"path": str(made), "of": 1, "part": "hands", "box": [0, 0, 10, 10]},
                {"path": str(tmp_path / "missing.png"), "of": 0, "part": "mouth"}]
    on = {"enabled": True, "hands": True, "mouth": True, "size": 384}
    [c] = crops.crops_in_venv(keys, tmp_path, on, run=run)
    assert c.kind == "crop" and c.meta["of"] == 1 and c.meta["part"] == "hands" and c.path == made
    assert seen["size"] == 384 and [f[0] for f in seen["frames"]] == [0, 1]
    assert crops.crops_in_venv(keys, tmp_path, {**on, "enabled": False}, run=run) == []
    assert crops.crops_in_venv(keys, tmp_path, None, run=run) == []

    def broken(spec):
        raise RuntimeError("no mediapipe")
    assert crops.crops_in_venv(keys, tmp_path, on, run=broken) == []          # a clip stands without crops


def test_clip_jobs_ask_their_worker_for_crops_when_switched_on(policy):
    from manyworlds import production
    assert production._with_crops({}, policy)["crops"]["enabled"] is True
    assert "crops" not in production._with_crops({}, _detail(policy, crops={"enabled": False}))


def _cropped(conn, store, tmp_path, cid):
    job = clip_job(conn, store, tmp_path, cid)
    rendered(conn, store, tmp_path, job)
    for of, part in ((1, "hands"), (3, "mouth")):
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                     "VALUES (%s, %s, 'crop', %s, 'x', 1, %s)",
                     (job, cid, _put(store, tmp_path, b"\x89PNG crop"), Jsonb({"of": of, "part": part})))
    return job


def test_the_judge_sees_hand_and_mouth_close_ups_beside_the_keyframes(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = _cropped(conn, store, tmp_path, cid)
    llm = ClipStub(clip(hands=8), clip(hands=8))
    [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
    texts = [str(b.get("text", "")) for b in llm.calls[0]["content"]]
    assert "HAND CLOSE-UP from KEYFRAME 2:" in texts and "MOUTH CLOSE-UP from KEYFRAME 4:" in texts
    ask = next(t for t in texts if t.startswith("DETAIL CHECKS"))
    assert "HAND CLOSE-UPs" in ask and "MOUTH CLOSE-UPs" in ask
    assert r["accepted"]
    # Teeth off: no mouth close-up. Crops off: none, and the prompt is as before.
    llm = ClipStub(clip(), clip())
    video.review_job(conn, _detail(policy, teeth={"enabled": False}), llm, store, _cropped(conn, store, tmp_path, cid),
                     tmp_path)
    texts = [str(b.get("text", "")) for b in llm.calls[0]["content"]]
    assert "HAND CLOSE-UP from KEYFRAME 2:" in texts and not any(t.startswith("MOUTH CLOSE-UP") for t in texts)
    llm = ClipStub(clip(), clip())
    video.review_job(conn, _detail(policy, crops={"enabled": False}), llm, store, _cropped(conn, store, tmp_path, cid),
                     tmp_path)
    texts = [str(b.get("text", "")) for b in llm.calls[0]["content"]]
    assert not any("CLOSE-UP" in t for t in texts)
