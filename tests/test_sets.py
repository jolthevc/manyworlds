"""Set and wardrobe bibles (manyworlds.creator.sets): a set or outfit the director names for the first time is
registered with its canonical description and a plate job; the image gate's rows (not its identity row) choose the
plate; later scenes in that set carry the canonical text and the references; reference conditioning (image.qwen_edit)
happens only when switched on."""

from __future__ import annotations

import dataclasses

from psycopg.types.json import Jsonb

from manyworlds import production
from manyworlds.brain import director
from manyworlds.creator import sets
from manyworlds.creator.lab import photo_prompt
from manyworlds.review import gates

from test_director import DirectorLLM
from test_production import creator, planned, real_refs
from test_review import BIBLE, JudgeStub, ok

DRESSING = "white metro tiles, a pale oak worktop, a green kettle, one window over the sink"
HOODIE = "an oversized heather-grey hoodie and black leggings"
KITCHEN = {"set": {"location": "her small galley kitchen", "set_id": "home_kitchen", "set_dressing": DRESSING},
           "wardrobe": {"outfit_id": "grey_hoodie", "outfit": HOODIE}}


def _plates(conn, cid):
    return conn.execute("SELECT id, kind, payload, idempotency_key FROM jobs WHERE batch = %s "
                        "ORDER BY idempotency_key", (f"post-sets-{cid}",)).fetchall()


def test_a_new_set_and_outfit_are_registered_from_the_directed_shot_with_their_plates(conn, policy):
    cid = creator(conn)
    director.direct(conn, policy, DirectorLLM({"shots": [KITCHEN]}, conn=conn), {"scenes": ["making toast"]}, cid,
                    fmt="photo")
    b = sets.bibles(conn, cid)
    kitchen, hoodie = b["sets"]["home_kitchen"], b["outfits"]["grey_hoodie"]
    assert kitchen["description"] == DRESSING and kitchen["name"] == "her small galley kitchen"
    assert kitchen["plate_uri"] is None and len(kitchen["plate_jobs"]) == 1
    assert hoodie["description"] == HOODIE and hoodie["reference_uri"] is None and len(hoodie["reference_jobs"]) == 1
    outfit_job, set_job = _plates(conn, cid)                       # ordered by key: "outfit:..." then "set:..."
    assert [outfit_job["kind"], set_job["kind"]] == ["image.zimage", "image.zimage"]
    assert str(set_job["id"]) == kitchen["plate_jobs"][0] and str(outfit_job["id"]) == hoodie["reference_jobs"][0]
    p = set_job["payload"]
    assert DRESSING in p["prompt"] and "no people" in p["prompt"] and "lora_uri" not in p
    assert "nobody" in p["scene"] and DRESSING in p["scene"]       # what the image gate scores it against
    assert p["plate"] == {"kind": "set", "id": "home_kitchen"}
    assert HOODIE in outfit_job["payload"]["prompt"] and "hanger" in outfit_job["payload"]["prompt"]

    # A later post naming the same set: nothing is overwritten, no new plate, and the director is shown the canon.
    other = {"set": {**KITCHEN["set"], "set_dressing": "a different kitchen"}, "wardrobe": KITCHEN["wardrobe"]}
    llm = DirectorLLM({"shots": [other]}, conn=conn)
    director.direct(conn, policy, llm, {"scenes": ["washing up"]}, cid, fmt="photo")
    assert sets.bibles(conn, cid)["sets"]["home_kitchen"]["description"] == DRESSING
    assert len(_plates(conn, cid)) == 2
    assert DRESSING in str(llm.calls[0]["content"]) and HOODIE in str(llm.calls[0]["content"])


def test_a_shot_without_ids_or_with_sets_off_registers_nothing(conn, policy):
    cid = creator(conn)
    sets.register(conn, policy, cid, [{"set": {"location": "a bus stop"}, "wardrobe": {"outfit": "a raincoat"}}])
    off = dataclasses.replace(policy, body={**policy.body, "sets": {**policy.body["sets"], "enabled": False}})
    sets.register(conn, off, cid, [KITCHEN])
    assert sets.bibles(conn, cid) == {"sets": {}, "outfits": {}} and _plates(conn, cid) == []


def _finish(conn, store, tmp_path, job_id, cid, n=4) -> list[str]:
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job_id,))
    uris = []
    for i in range(n):
        p = tmp_path / f"plate_{job_id}_{i}.png"
        p.write_bytes(b"\x89PNG plate" + bytes([i]))
        uri = store.put_file(p, f"plates/{job_id}/{i}.png").uri
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'image', %s, "
                     "'x', 1)", (job_id, cid, uri))
        uris.append(uri)
    return uris


def test_the_plate_is_chosen_by_the_image_gate_without_its_identity_row(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    sets.register(conn, policy, cid, [KITCHEN])
    job = sets.bibles(conn, cid)["sets"]["home_kitchen"]["plate_jobs"][0]
    uris = _finish(conn, store, tmp_path, job, cid)
    assert sets.refresh(conn, policy, cid)["chosen"] == []         # not reviewed yet: wait
    judge = JudgeStub([ok(identity=0, scene=5), ok(identity=0, scene=9), ok(identity=0, scene=9, realism=3),
                       ok(identity=0, scene=8)])
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    # Nobody to match, so no identity gate: plates are accepted or rejected on realism, safety and scene alone, and
    # don't count as Z-Image rejects.
    assert [r["accepted"] for r in results] == [True, True, False, True]
    assert not conn.execute("SELECT 1 FROM reviews WHERE gate = 'identity'").fetchone()
    assert "identity" not in results[0]
    out = sets.refresh(conn, policy, cid)
    assert out["chosen"] == ["set:home_kitchen"] and out["retried"] == []
    kitchen = sets.bibles(conn, cid)["sets"]["home_kitchen"]
    assert kitchen["plate_uri"] == uris[1]                         # best scene + realism among the good ones
    assert sets.bibles(conn, cid)["outfits"]["grey_hoodie"]["reference_uri"] is None   # its job is still queued


def test_the_producer_tick_picks_up_reviewed_plates(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    sets.register(conn, policy, cid, [KITCHEN])
    job = sets.bibles(conn, cid)["sets"]["home_kitchen"]["plate_jobs"][0]
    uris = _finish(conn, store, tmp_path, job, cid, n=1)
    assert sets.refresh_all(conn, policy) == {}                   # not reviewed yet
    gates.review_job(conn, policy, JudgeStub([ok(identity=0, scene=9)]), store, job, tmp_path)
    assert sets.refresh_all(conn, policy) == {str(cid): {"chosen": ["set:home_kitchen"], "retried": [], "gave_up": []}}
    assert sets.bibles(conn, cid)["sets"]["home_kitchen"]["plate_uri"] == uris[0]
    assert sets.refresh_all(conn, policy) == {}                   # the outfit's job is still queued


def test_a_rejected_plate_is_made_again_up_to_the_limit(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    sets.register(conn, policy, cid, [{"set": KITCHEN["set"]}])
    limit = int(policy.get("sets", "max_plate_attempts"))
    for attempt in range(limit):
        job = sets.bibles(conn, cid)["sets"]["home_kitchen"]["plate_jobs"][-1]
        _finish(conn, store, tmp_path, job, cid, n=2)
        gates.review_job(conn, policy, JudgeStub([ok(identity=0, scene=2)] * 2), store, job, tmp_path)
        out = sets.refresh(conn, policy, cid)
        if attempt < limit - 1:
            assert out["retried"] == ["set:home_kitchen"]
        else:
            assert out["gave_up"] == ["set:home_kitchen"]
    kitchen = sets.bibles(conn, cid)["sets"]["home_kitchen"]
    assert len(kitchen["plate_jobs"]) == limit and kitchen["plate_uri"] is None and kitchen["plate_failed"] is True
    seeds = {j["payload"]["seed"] for j in _plates(conn, cid)}
    assert len(seeds) == limit                                     # every attempt a new seed


def _registered(conn, cid):
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (cid,)).fetchone()["anchors"]
    anchors["sets"] = {"home_kitchen": {"set_id": "home_kitchen", "name": "her small galley kitchen",
                                        "description": DRESSING, "plate_uri": "s3://b/plate.png", "plate_jobs": []}}
    anchors["outfits"] = {"grey_hoodie": {"outfit_id": "grey_hoodie", "description": HOODIE,
                                          "reference_uri": "s3://b/hoodie.png", "reference_jobs": []}}
    conn.execute("UPDATE creators SET anchors = %s WHERE id = %s", (Jsonb(anchors), cid))


SHOT = {"set": {"set_id": "home_kitchen", "location": "her small galley kitchen"},
        "wardrobe": {"outfit_id": "grey_hoodie"}}


def _started_payload(conn, policy, cid, shot=SHOT):
    pid = planned(conn, policy, cid, scenes=("making toast in her kitchen",))
    conn.execute("UPDATE posts SET body = body || %s WHERE id = %s", (Jsonb({"shots": [shot]}), pid))
    post = conn.execute("SELECT * FROM posts WHERE id = %s", (pid,)).fetchone()
    production.start(conn, policy, post)
    job = conn.execute("SELECT kind, payload, pools FROM jobs WHERE post_id = %s", (pid,)).fetchone()
    return post, job


def test_a_scene_in_a_registered_set_carries_the_canonical_text_and_the_references(conn, policy):
    cid = creator(conn)
    _registered(conn, cid)
    post, job = _started_payload(conn, policy, cid)
    p = job["payload"]
    assert job["kind"] == "image.zimage" and job["pools"] is None     # reference_conditioning is off
    assert p["references"] == [{"kind": "set", "uri": "s3://b/plate.png"}, {"kind": "outfit", "uri": "s3://b/hoodie.png"}]
    assert DRESSING in p["prompt"] and HOODIE in p["prompt"] and p["prompt"].startswith("mw_m, ")
    assert p["shot"]["set"]["set_dressing"] == DRESSING and p["shot"]["wardrobe"]["outfit"] == HOODIE
    assert p["lora_uri"] == "s3://b/m.safetensors"
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (post["id"],)).fetchone()["body"]
    assert body["shots"] == [SHOT]                                    # the post's own shot is not rewritten


def test_a_shot_in_an_unregistered_set_is_made_as_before(conn, policy):
    cid = creator(conn)
    shot = {"set": {"set_id": "canal_path", "location": "the canal path at dusk"}}
    post, job = _started_payload(conn, policy, cid, shot)
    p = job["payload"]
    assert "references" not in p and job["kind"] == "image.zimage"
    assert p["prompt"] == photo_prompt(BIBLE | {"display_name": "M"}, "making toast in her kitchen", trigger="mw_m",
                                       shot=shot, policy=policy)
    assert p["shot"] == shot


def _conditioned(policy, **sets_over):
    return dataclasses.replace(policy, body={**policy.body, "sets": {**policy.body["sets"],
                                                                     "reference_conditioning": True, **sets_over}})


def test_reference_conditioning_makes_the_scene_with_the_edit_kind_from_anchor_plate_and_outfit(conn, policy):
    cid = creator(conn)
    _registered(conn, cid)
    on = _conditioned(policy)
    _, job = _started_payload(conn, on, cid)
    p = job["payload"]
    assert job["kind"] == "image.qwen_edit"
    assert job["pools"] == policy.get("job_kinds", "image.qwen_edit", "pools")   # pinned: 48-80 GB cards only
    assert p["images"] == ["s3://b/r0", "s3://b/plate.png", "s3://b/hoodie.png"]  # anchor first
    assert p["prompt"].startswith("Picture 1 is the person") and "picture 2" in p["prompt"] and "picture 3" in p["prompt"]
    assert "mw_m" not in p["prompt"] and "lora_uri" not in p and "lora_scale" not in p
    edit = policy.get("sets", "edit")
    assert p["params"] == {"steps": edit["steps"], "true_cfg": edit["true_cfg"],
                           "negative_prompt": edit["negative_prompt"], "lightning": edit["lightning"]}
    assert p["references"] and p["shot"]["set"]["set_dressing"] == DRESSING   # still judged against them


def test_reference_conditioning_needs_the_edit_kind_in_policy(conn, policy):
    cid = creator(conn)
    _registered(conn, cid)
    on = _conditioned(policy)
    kinds = {k: v for k, v in on.body["job_kinds"].items() if k != "image.qwen_edit"}
    without = dataclasses.replace(on, body={**on.body, "job_kinds": kinds})
    _, job = _started_payload(conn, without, cid)
    assert job["kind"] == "image.zimage" and "images" not in job["payload"] and job["payload"]["references"]
