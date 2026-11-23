"""Production's state in posts.body is declared (production/state.py): a key the package reads or writes that is
declared nowhere fails here, so a misspelt counter is caught in CI, not in a live post. And one producer per post:
a post whose lease another producer (connection) holds is skipped, not advanced twice."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from manyworlds import db, production
from manyworlds.production import state

from test_production import NOW, creator, planned

PKG = Path(production.__file__).parent

# Where production names a body key in its code.
KEY_PATTERNS = [
    r'\bbody\[\s*"([^"]+)"\s*\]',                          # body["key"]
    r'\bbody\.(?:get|setdefault|pop)\(\s*"([^"]+)"',       # body.get("key"), .setdefault, .pop
    r'\["body"\]\[\s*"([^"]+)"\s*\]',                      # post["body"]["key"]
    r'\["body"\]\s*or\s*\{\}\)\.get\(\s*"([^"]+)"',        # (post["body"] or {}).get("key")
    r'\(\s*"([a-z_]*(?:repairs|failures))"\s*,',           # a Step's counters: ("clip_repairs", limit)
    r'\(\s*"([a-z_]+_jobs)"\s*,',                          # FOLLOW_UPS' job lists: ("edit_jobs", _edit_job, ...)
    r"\bbody \? '([a-z_]+)'",                              # SQL: body ? 'key'
    r"jsonb_build_object\('([a-z_]+)'",                    # SQL: body || jsonb_build_object('key', ...)
    r'\bpatch\[\s*"([^"]+)"\s*\]',                         # _advance_failed's patch["key"]
]
PER_INDEX = r'f"([a-z_]+_)\{\w+\}"'                         # f"clip_repairs_{i}": a prefix plus an index


def _sources() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(PKG.glob("*.py")) if p.name != "state.py"}


def scanned_keys() -> tuple[set[str], set[str]]:
    """Every body key and per-index key prefix named in the production package's source."""
    keys, prefixes = set(), set()
    for src in _sources().values():
        for pattern in KEY_PATTERNS:
            keys.update(re.findall(pattern, src))
        for args in re.findall(r"\bbody\.update\(([^)]*)\)", src):        # body.update(key=...)
            keys.update(re.findall(r"(\w+)=(?!=)", args))
        for items in re.findall(r"\bpatch\s*=\s*\{([^}]*)\}", src):         # patch = {"key": ...}
            keys.update(re.findall(r'"(\w+)"\s*:', items))
        prefixes.update(re.findall(PER_INDEX, src))
    return keys, prefixes


def test_every_body_key_production_uses_is_declared():
    keys, prefixes = scanned_keys()
    assert len(keys) > 40, "the scan found too few keys: has the code's shape changed under these patterns?"
    undeclared = sorted(k for k in keys if not state.declared(k))
    assert undeclared == [], f"body keys used by production but not declared in production/state.py: {undeclared}"
    assert prefixes == set(state.STATE_PREFIXES), "per-index keys (f'..._{i}') and STATE_PREFIXES differ"


def test_every_declared_key_is_used():
    keys, _ = scanned_keys()
    assert sorted(state.STATE_KEYS - keys) == [], "declared state that production no longer uses"
    assert sorted(state.BRIEF_KEYS - keys) == [], "declared brief fields that production no longer reads"
    assert not state.STATE_KEYS & state.BRIEF_KEYS


def test_a_misspelt_key_is_caught():
    assert state.declared("scene_jobs") and state.declared("clip_repairs_3") and state.declared("scenes")
    assert not state.declared("scene_job") and not state.declared("clip_repair_3")


def test_follow_up_formats_name_the_follow_ups():
    assert production.FOLLOW_UP_FORMATS == tuple(production.FOLLOW_UPS)


def test_the_package_reexports_its_modules_names():
    for module in (production.common, production.steps, production.scenes, production.takes, production.voice,
                   production.motion, production.talk, production.followups, production.captioning,
                   production.mixing, production.finish, production.core, production.locks):
        for name, value in vars(module).items():
            if callable(value) and getattr(value, "__module__", None) == module.__name__:
                assert getattr(production, name) is value, f"production.{name} is not {module.__name__}.{name}"


# --- one producer per post ------------------------------------------------------------------------------------------

@pytest.fixture()
def other(database_url):
    """A second producer's connection (the Fly producer beside a manual `mw` run)."""
    c = db.connect(database_url)
    yield c
    c.close()


def test_a_post_lease_is_held_by_one_connection_at_a_time(conn, other, policy):
    pid = "00000000-0000-0000-0000-000000000001"
    assert production.take_post_lease(conn, policy, pid, "mine")
    assert not production.take_post_lease(other, policy, pid, "theirs")
    production.release_post_lease(other, pid, "theirs")          # not theirs to release
    assert not production.take_post_lease(other, policy, pid, "theirs")
    production.release_post_lease(conn, pid, "mine")
    assert production.take_post_lease(other, policy, pid, "theirs")
    # A lease past its time (a producer that died mid-advance) is taken over.
    other.execute("UPDATE leases SET until = now() - interval '1 second' WHERE name = %s",
                  (production.lease_name(pid),))
    assert production.take_post_lease(conn, policy, pid, "mine")


def test_post_lease_context_releases_on_error(conn, other, policy):
    pid = "00000000-0000-0000-0000-000000000002"
    with pytest.raises(RuntimeError):
        with production.post_lease(conn, policy, pid) as held:
            assert held
            with production.post_lease(other, policy, pid) as theirs:
                assert theirs is False
            raise RuntimeError("advance failed")
    with production.post_lease(other, policy, pid) as held:
        assert held


def test_a_post_another_producer_holds_is_skipped(conn, other, policy):
    pid = planned(conn, policy, creator(conn))
    assert production.take_post_lease(other, policy, pid, "theirs")
    out = production.advance(conn, policy, now=NOW)               # a planned post: not started under their lease
    assert out["started"] == 0 and out["locked"] == 1
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (pid,)).fetchone()["status"] == "planned"
    production.release_post_lease(other, pid, "theirs")
    assert production.advance(conn, policy, now=NOW)["started"] == 1
    before = conn.execute("SELECT body, updated_at FROM posts WHERE id = %s", (pid,)).fetchone()
    assert production.take_post_lease(other, policy, pid, "theirs")
    out = production.advance(conn, policy, now=NOW)               # producing: not advanced under their lease
    assert out["producing"] == 0 and out["locked"] == 1 and "advance_errors" not in out
    after = conn.execute("SELECT body, updated_at FROM posts WHERE id = %s", (pid,)).fetchone()
    assert after == before
    production.release_post_lease(other, pid, "theirs")
    out = production.advance(conn, policy, now=NOW)
    assert out["producing"] == 1 and "locked" not in out
    assert conn.execute("SELECT count(*) n FROM leases WHERE name = %s",
                        (production.lease_name(pid),)).fetchone()["n"] == 0   # released after the advance
