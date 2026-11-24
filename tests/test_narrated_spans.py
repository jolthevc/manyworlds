"""A narrated edit's on-screen text is timed where each shot's photo really plays (the image.narrated video's
`photo_spans`, kept in body.photo_spans), not as if the photos split the video evenly; an older edit without spans
keeps the even share."""

from __future__ import annotations

from psycopg.types.json import Jsonb

from manyworlds import production
from manyworlds.direction import ShotSheet
from manyworlds.worker.handlers import edit_narrated

from test_production import creator

SPANS = [{"start_s": 0.0, "end_s": 2.0}, {"start_s": 2.0, "end_s": 9.0}]


def _labelled(text):
    return ShotSheet.model_validate({"edit": {"on_screen_text": [{"kind": "label", "text": text}]}}
                                    ).model_dump(mode="json")


def _edit_video(conn, meta):
    cid = creator(conn)
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('image.narrated', 'done', %s, %s) "
                       "RETURNING id", (cid, Jsonb({}))).fetchone()["id"]
    video = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                         "VALUES (%s, %s, 'video', 's3://b/edit.mp4', 'x', 1, %s) RETURNING id, uri",
                         (job, cid, Jsonb(meta))).fetchone()
    return {"id": None, "creator_id": cid, "format": "narrated_edit"}, video


def test_the_handler_records_where_each_photo_plays():
    frames = edit_narrated.photo_frames(3, 10.0, 30)
    spans = edit_narrated.photo_spans(3, frames, 30)
    assert len(spans) == 3 and spans[0]["start_s"] == 0.0
    assert all(a["end_s"] == b["start_s"] for a, b in zip(spans, spans[1:]))     # back to back, in order
    assert spans[-1]["end_s"] <= 10.0 and spans[-1]["end_s"] == round(3 * frames / 30, 3)


def test_labels_follow_the_photos_real_spans(conn):
    post, video = _edit_video(conn, {"seconds": 9.0, "photo_spans": SPANS})
    body = {"scenes": ["the kitchen", "the table"], "shots": [_labelled("Step one"), _labelled("Soup day")]}
    production._keep_photo_spans(conn, body, video)
    assert body["photo_spans"] == SPANS
    assert production.shot_spans(conn, post, body, video, 9.0) == [(0, 0.0, 2.0), (1, 2.0, 9.0)]
    items = production.overlay_items(conn, post, body, video, 9.0)
    assert [(i["text"], i["start_s"]) for i in items] == [("Step one", 0.0), ("Soup day", 2.0)]
    # the finished video ran shorter: no span past its end
    assert production.shot_spans(conn, post, body, video, 8.0)[-1] == (1, 2.0, 8.0)


def test_an_older_edit_without_spans_shares_the_video_evenly(conn):
    post, video = _edit_video(conn, {"seconds": 9.0})
    body = {"scenes": ["a", "b"], "shots": [_labelled("one"), _labelled("two")], "photo_spans": SPANS}
    production._keep_photo_spans(conn, body, video)                 # a remade edit without spans: none kept
    assert "photo_spans" not in body
    assert production.shot_spans(conn, post, body, video, 9.0) == [(0, 0.0, 4.5), (1, 4.5, 9.0)]
    # spans that don't hold one per shot are not trusted either
    body["photo_spans"] = SPANS[:1]
    assert production.shot_spans(conn, post, body, video, 9.0) == [(0, 0.0, 4.5), (1, 4.5, 9.0)]


def test_the_finisher_reads_the_spans_from_the_video_meta_when_the_body_has_none():
    body = {"shots": [{}, {}]}
    assert production._spans_from("narrated_edit", body, {"photo_spans": SPANS}, [], 9.0) == [(0, 0.0, 2.0),
                                                                                             (1, 2.0, 9.0)]
