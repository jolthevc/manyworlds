"""Instagram's JPEG copies (public/ in the bucket) go once their post is out or given up and the keep window has
passed; copies of posts still on their way, or published recently, stay."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds.deploy import autoscale
from manyworlds.publish import instagram, public_copies

from brain_helpers import creator

NOW = datetime.now(timezone.utc)


def _post(conn, store, tmp_path, cid, name, *, status, days_old, published=False, tried=True):
    src = tmp_path / f"{name}.png"
    src.write_bytes(b"png")
    uri = store.put_file(src, f"images/{name}.png").uri
    cover = store.put_file(src, f"covers/{name}.png").uri
    copies = []
    for u in (uri, cover):
        j = tmp_path / f"{name}-{len(copies)}.jpg"
        j.write_bytes(b"jpg")
        copies.append(store.put_file(j, instagram.public_key(u)))
    body = {"images": [{"uri": uri}], "cover": {"uri": cover}, **({"ig_attempts": 1} if tried else {})}
    when = NOW - timedelta(days=days_old)
    pid = conn.execute("INSERT INTO posts (creator_id, format, status, body, updated_at) VALUES (%s, 'photo', %s, %s, %s) "
                       "RETURNING id", (cid, status, Jsonb(body), when)).fetchone()["id"]
    if published:
        conn.execute("INSERT INTO publications (post_id, platform, simulated, published_at) "
                     "VALUES (%s, 'instagram', false, %s)", (pid, when))
    return pid, uri, [store.root / instagram.public_key(u) for u in (uri, cover)]


def test_copies_of_finished_posts_past_the_window_are_deleted_and_the_rest_kept(conn, policy, store, tmp_path):
    keep = policy.get("publishing", "public_copy_keep_days")
    cid = creator(conn)
    old_pub, old_uri, old_copies = _post(conn, store, tmp_path, cid, "a", status="published", days_old=keep + 1,
                                         published=True)
    held, _, held_copies = _post(conn, store, tmp_path, cid, "b", status="held", days_old=keep + 1)
    recent, _, recent_copies = _post(conn, store, tmp_path, cid, "c", status="published", days_old=1, published=True)
    waiting, _, waiting_copies = _post(conn, store, tmp_path, cid, "d", status="approved", days_old=keep + 1)
    out = public_copies.purge(conn, policy, store, now=NOW)
    assert out == {"posts": 2, "deleted": 4, "failed": 0}
    for p in old_copies + held_copies:
        assert not p.exists(), p
    for p in recent_copies + waiting_copies:
        assert p.exists(), p
    assert (store.root / "images" / "a.png").exists()          # the original stays (retention decides on it)
    marked = {r["id"] for r in conn.execute("SELECT id FROM posts WHERE body ? 'public_copies_purged_at'")}
    assert marked == {old_pub, held}
    assert public_copies.purge(conn, policy, store, now=NOW)["posts"] == 0   # not visited again


def test_housekeeping_purges_copies_beside_research_items(conn, policy, store, tmp_path):
    keep = policy.get("publishing", "public_copy_keep_days")
    _, _, copies = _post(conn, store, tmp_path, creator(conn), "e", status="dropped", days_old=keep + 1)
    acts = autoscale.housekeeping(conn, policy, store=store, now=NOW)
    assert {"action": "purged_public_copies", "posts": 1, "deleted": 2, "failed": 0} in acts
    assert not any(p.exists() for p in copies)


def test_nothing_due_never_opens_the_store(conn, policy):
    def no_store():
        raise AssertionError("the store was opened with nothing to delete")
    assert public_copies.purge(conn, policy, no_store, now=NOW) == {"posts": 0, "deleted": 0, "failed": 0}


def test_a_storage_error_leaves_the_post_for_the_next_run(conn, policy, store, tmp_path):
    keep = policy.get("publishing", "public_copy_keep_days")
    pid, _, _ = _post(conn, store, tmp_path, creator(conn), "f", status="held", days_old=keep + 1)

    class Broken:
        def uri_for(self, key):
            return store.uri_for(key)

        def delete(self, uri):
            raise OSError("bucket unreachable")

    assert public_copies.purge(conn, policy, Broken(), now=NOW) == {"posts": 0, "deleted": 0, "failed": 1}
    assert not conn.execute("SELECT body ? 'public_copies_purged_at' AS m FROM posts WHERE id = %s",
                            (pid,)).fetchone()["m"]
