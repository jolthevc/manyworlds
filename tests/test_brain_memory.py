"""A creator's memory: the places, habits, tastes, people and canon that keep their life coherent."""

from __future__ import annotations

from psycopg.types.json import Jsonb

from manyworlds.brain import memory

from brain_helpers import creator


def _with_bible(conn):
    cid = creator(conn)
    bible = {"home_and_places": ["a top-floor flat above a bakery", "the canal path at dawn"],
             "interests": "fermenting things; secondhand cookbooks",
             "opinions_and_running_jokes": ["the kettle named Gerald is the only reliable man in her life"]}
    conn.execute("UPDATE creators SET bible = %s WHERE id = %s", (Jsonb(bible), cid))
    conn.execute("INSERT INTO creator_profiles (creator_id, version, body, author) VALUES (%s, 1, %s, 'brain')",
                 (cid, Jsonb({"look": {"signature_item": "thin gold wire glasses"}})))
    conn.execute("UPDATE creators SET profile_version = 1 WHERE id = %s", (cid,))
    return cid


def test_seeding_from_the_profile_is_idempotent(conn):
    cid = _with_bible(conn)
    n = memory.seed_from_profile(conn, cid)
    assert n == 6 and memory.seed_from_profile(conn, cid) == 0
    kinds = {f["kind"] for f in memory.facts(conn, cid, limit=50)}
    assert kinds == {"place", "taste", "opinion", "signature"}
    assert any("Gerald" in f["text"] for f in memory.facts(conn, cid, limit=50))


def test_a_newer_single_valued_fact_replaces_the_old_one(conn):
    cid = creator(conn)
    memory.add(conn, cid, "place", "lives in Leeds", key="home")
    memory.add(conn, cid, "place", "lives in Copenhagen", key="home")
    texts = [f["text"] for f in memory.facts(conn, cid, limit=10)]
    assert texts == ["lives in Copenhagen"]
    closed = conn.execute("SELECT count(*) n FROM creator_memory WHERE valid_until IS NOT NULL").fetchone()["n"]
    assert closed == 1


def test_facts_are_open_ones_newest_first(conn):
    cid = creator(conn)
    for t in ("a", "b", "c"):
        memory.add(conn, cid, "canon", t)
    assert [f["text"] for f in memory.facts(conn, cid, limit=2)] == ["c", "b"]
