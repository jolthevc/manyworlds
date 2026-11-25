"""The data policy (owner 2026-11-25; manyworlds.research.data_policy): maximise permissible intelligence, minimise
retained third-party content. Green facts stay; amber media and caption text are held only until read; red is never
collected. Enforced in code: the router, the direct pulls, ingestion and the purge."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest
from psycopg.types.json import Jsonb

from manyworlds.research import data_policy, providers, pulls, references
from manyworlds.storage import LocalStore


def _source(policy, name, **over):
    body = dict(policy.body)
    dp = dict(body["data_policy"])
    dp["sources"] = {**dp["sources"], name: {**dp["sources"].get(name, {}), **over}}
    body["data_policy"] = dp
    return dataclasses.replace(policy, body=body)


def test_only_approved_sources_and_tasks_run(policy):
    assert data_policy.refused(policy, "apify", "posts") is None
    assert "blocked" in data_policy.refused(policy, "hikerapi", "posts")
    assert "not in data_policy.sources" in data_policy.refused(policy, "someapi", "posts")
    assert "task posts not approved" in data_policy.refused(policy, "claude_web", "posts")
    assert "not in data_policy.sources" in data_policy.refused(policy, "feeds", "feed")   # publisher RSS removed
    paused = _source(policy, "apify", status="paused", why="terms under review")
    assert "terms under review" in data_policy.refused(paused, "apify", "posts")
    with pytest.raises(data_policy.SourcePaused):
        pulls.candidates(None, paused, lambda *a: [], {"tiktok": ["x"]})


def test_the_router_never_calls_a_paused_source(conn, policy, monkeypatch):
    monkeypatch.setenv("APIFY_TOKEN", "t")
    called = []

    def fake(actor, payload):
        called.append(actor)
        return []
    paused = _source(policy, "apify", status="paused", why="unclear")
    router = providers.Router(conn, paused, adapters={"apify": providers.ApifyProvider(paused, fetch=fake)})
    items, used, errors = router._run("posts", "tiktok", lambda a: a.fetch_posts("tiktok", "mira", 12, None))
    assert not called and used is None and any("unclear" in str(e.get("skipped")) for e in errors)


def test_bios_lose_contact_details_and_captions_leave_only_features():
    bio = "NYC stylist ✦ collabs: hello@mira.studio | +1 (212) 555-0199 ✦ she/her"
    clean = data_policy.clean_bio(bio)
    assert "@mira.studio" not in clean and "555" not in clean and "stylist" in clean
    f = data_policy.caption_features("Five things nobody tells you about thrifting?\n#thrift #OOTD @zara link in bio #ad")
    assert f["hashtags"] == ["ad", "ootd", "thrift"] and f["mentions"] == ["zara"] and f["question"]
    assert f["disclosure"] and "link in bio" in f["ctas"] and f["words"] == 14
    assert not any("thrifting" in str(v) for v in f.values())          # no words of the caption itself


def _post(conn, *, media=None, annotated_hours_ago=None, fetched_days_ago=0.0, caption="a caption #x"):
    cid = pulls.upsert_creator(conn, "tiktok", "mira.lou", bio="hi, mail me at a@b.co")
    pid = pulls.upsert_post(conn, None, cid, "tiktok", {"external_id": f"p{datetime.now().timestamp()}",
                                                         "media_type": "photo", "caption": caption})
    conn.execute("UPDATE reference_posts SET media = %s, media_purge_after = now() + interval '10 days', "
                 "fetched_at = now() - %s * interval '1 day' WHERE id = %s",
                 (Jsonb(media or []), fetched_days_ago, pid))
    if annotated_hours_ago is not None:
        conn.execute("INSERT INTO reference_annotations (post_id, schema, updated_at) VALUES "
                     "(%s, 'iia_v1', now() - %s * interval '1 hour')", (pid, annotated_hours_ago))
    return cid, pid


def test_ingestion_keeps_features_and_a_clean_bio(conn):
    cid, pid = _post(conn, caption="Morning walk #coffee @bluebottle")
    row = conn.execute("SELECT caption_features FROM reference_posts WHERE id = %s", (pid,)).fetchone()
    assert row["caption_features"]["mentions"] == ["bluebottle"]
    assert "a@b.co" not in conn.execute("SELECT bio FROM reference_creators WHERE id = %s", (cid,)).fetchone()["bio"]


def test_read_media_and_old_captions_are_purged_and_what_was_learned_stays(conn, policy, tmp_path):
    store = LocalStore(tmp_path / "s")
    f = tmp_path / "img.jpg"
    f.write_bytes(b"jpg")
    uri = store.put_file(f, references.media_key("x", "img0.jpg")).uri
    _, read = _post(conn, media=[{"uri": uri, "kind": "image"}], annotated_hours_ago=48, fetched_days_ago=45)
    _, unread = _post(conn, media=[{"uri": uri, "kind": "image"}], fetched_days_ago=1)
    out = references.purge(conn, policy, store)
    rows = {str(r["id"]): r for r in conn.execute("SELECT id, media, caption, caption_features FROM reference_posts")}
    assert rows[read]["media"] == [] and rows[read]["caption"] is None and rows[read]["caption_features"]["words"]
    assert rows[unread]["media"] and rows[unread]["caption"]           # not read yet: kept, within its 14 days
    assert out["read_media_purged"] == 1 and out["captions_cleared"] == 1


def test_open_weight_hosts_read_no_research_media_until_approved(policy):
    assert data_policy.processor_ok(policy, "claude-haiku-5-5")
    assert not data_policy.processor_ok(policy, "glm-5.3")
    assert not data_policy.processor_ok(policy, None)


def test_media_is_held_at_most_two_weeks(policy):
    now = datetime.now(timezone.utc)
    assert references.media_purge_after(policy, now) - now <= timedelta(days=14)


def test_the_audit_cleans_bios_and_backfills_features(conn, policy):
    cid = pulls.upsert_creator(conn, "instagram", "x.y")
    conn.execute("UPDATE reference_creators SET bio = 'model, book: agent@agency.com' WHERE id = %s", (cid,))
    out = data_policy.audit(conn, policy)
    assert out["red"]["bios_with_contact_details"] == 1 and not out["red"]["face_or_voice_columns"]
    data_policy.audit(conn, policy, apply=True)
    assert data_policy.audit(conn, policy)["red"]["bios_with_contact_details"] == 0
