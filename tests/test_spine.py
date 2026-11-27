"""The analytical spine, gold agreement, exploratory methods and the research loop's wiring (owner 2026-11-26:
docs/2026-11-26-unified-intelligence.md, Phases 0-1)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds.brain import inquiry, predictions, recipes, research_questions
from manyworlds.research import gold, spine

NOW = datetime(2026, 11, 20, tzinfo=timezone.utc)


def _ref_creator(conn, handle="ana", archetype="soft_luxury"):
    return conn.execute("INSERT INTO reference_creators (platform, handle, scores) VALUES ('instagram', %s, %s) "
                        "RETURNING id", (handle, Jsonb({"archetype": archetype}))).fetchone()["id"]


def _ref_post(conn, creator, i, likes, *, days_old=10.0, media="photo", coded=None):
    posted = NOW - timedelta(days=days_old + i)
    pid = conn.execute(
        "INSERT INTO reference_posts (creator_id, platform, external_id, media_type, posted_at, metrics, snapshots, "
        "fetched_at) VALUES (%s, 'instagram', %s, %s, %s, %s, %s, %s) RETURNING id",
        (creator, f"{creator}-{i}", media, posted, Jsonb({"likes": likes, "comments": 1}),
         Jsonb([{"at": (posted + timedelta(days=1)).isoformat(), "likes": likes // 2}]), posted + timedelta(days=days_old))
    ).fetchone()["id"]
    if coded is not None:
        conn.execute("INSERT INTO reference_annotations (post_id, schema, model, observed) VALUES (%s, 'iia_v2', "
                     "'claude-haiku-5-5', %s)", (pid, Jsonb({"coded": coded})))
    return pid


# --- the spine ------------------------------------------------------------------------------------------------------
def test_the_sync_mirrors_posts_reads_and_counts_and_normalises_against_the_creators_median(conn, policy):
    c = _ref_creator(conn)
    likes = [400, 100, 100, 100, 100, 100, 100]          # the newest post: four times the six before it
    ids = [_ref_post(conn, c, i, n, coded={"candid_level": "candid", "garments": ["dress", "blazer"]} if i == 0 else None)
           for i, n in enumerate(likes)]
    out = spine.sync(conn, policy)
    assert out["external"]["posts"] == 7 and out["external"]["creators"] >= 1
    unit = conn.execute("SELECT id, format, rights, parent_id FROM content_units WHERE ref_table = 'reference_posts' "
                        "AND ref_id = %s", (str(ids[0]),)).fetchone()
    assert unit["format"] == "photo" and unit["rights"] == "amber" and unit["parent_id"]
    vals = {(r["attribute"], r["value"]) for r in conn.execute(
        "SELECT attribute, value FROM attribute_values WHERE unit_id = %s", (unit["id"],)).fetchall()}
    assert {("candid_level", "candid"), ("garments", "blazer"), ("garments", "dress")} <= vals
    row = conn.execute("SELECT raw, normalized, baseline, baseline_n FROM outcomes WHERE unit_id = %s "
                       "AND metric = 'likes' AND horizon = 'd7'", (unit["id"],)).fetchone()   # read at 10 days
    assert row is not None and float(row["raw"]) == 400
    assert row["baseline_n"] >= 5 and abs(float(row["normalized"]) - math.log(4)) < 1e-6
    assert conn.execute("SELECT count(*) n FROM outcomes WHERE unit_id = %s AND metric = 'engagement'",
                        (unit["id"],)).fetchone()["n"] >= 1                           # likes + comments... kept too
    again = spine.sync(conn, policy)                                                    # idempotent
    assert again["external"]["posts"] == 7
    assert conn.execute("SELECT count(*) n FROM content_units WHERE ref_table = 'reference_posts'").fetchone()["n"] == 7


def test_a_name_outside_the_registry_is_stored_as_other_and_reported(policy):
    problems: list = []
    assert spine._known(policy, "platforms", "myspace", problems) == "other" and problems
    assert spine._known(policy, "formats", "reel") == "reel"
    assert spine.horizon(policy, 2 * 86400) == "d3" and spine.horizon(policy, 3600) is None


def test_a_coded_read_becomes_long_form_rows():
    rows = spine.coded_values({"candid_level": "high", "garments": [], "aspiration": 4, "slide_roles": ["hook", "list"]},
                              "carousel")
    assert ("candid_level", "high", None) in rows and ("garments", "(none)", None) in rows
    assert ("aspiration", None, 4.0) in rows and ("slide_roles", "0:hook", None) in rows
    assert not any(a == "slide_roles" for a, _, _ in spine.coded_values({"slide_roles": ["hook"]}, "photo"))


# --- agreement ------------------------------------------------------------------------------------------------------
def test_krippendorffs_alpha_on_known_cases():
    assert gold.alpha({1: {"a": "x", "b": "x"}, 2: {"a": "y", "b": "y"}}) == 1.0
    assert abs(gold.alpha({1: {"a": "x", "b": "x"}, 2: {"a": "y", "b": "y"}, 3: {"a": "x", "b": "y"}}) - 0.4444) < 1e-3
    assert gold.alpha({1: {"a": "x", "b": "x"}}) is None                    # no expected disagreement
    assert gold.alpha({1: {"a": 1.0, "b": 2.0}, 2: {"a": 5.0, "b": 5.0}}, "interval") > 0.8


def test_the_gold_report_flags_an_attribute_the_reference_disagrees_with(conn, policy):
    c = _ref_creator(conn)
    units = []
    for i in range(40):
        pid = _ref_post(conn, c, i, 100)
        u = spine.upsert_unit(conn, source="external", level="post", ref_table="reference_posts", ref_id=pid)
        units.append(u)
        good = "candid" if i % 2 else "posed"
        spine.put_value(conn, u, "candid_level", value=good, method="model", annotator="haiku")
        spine.put_value(conn, u, "candid_level", value=good, method="model", annotator="opus", purpose="gold")
        spine.put_value(conn, u, "camera_angle", value="high_angle" if i % 2 else "eye_level", method="model",
                        annotator="haiku")
        spine.put_value(conn, u, "camera_angle", value="eye_level" if i % 3 else "high_angle", method="model",
                        annotator="opus",
                        purpose="gold")
    rep = gold.report(conn, policy, unit_ids=units)
    assert rep["attributes"]["candid_level"]["verdict"] == "reliable"
    assert "camera_angle" in rep["unreliable"] and gold.unreliable(conn) == {"camera_angle"}
    assert rep["attributes"]["candid_level"]["class"] == "subjective" and "subjective" in rep["by_class"]


def test_unreliable_fields_leave_every_row(policy):
    r = recipes.Row(post_id="p", creator_id="c", medium="photo", y=0.1, attrs=frozenset({"camera_angle=high",
                    "pose=standing"}), known=frozenset({"camera_angle", "pose"}), values={"camera_angle": "high"})
    [out] = recipes.drop_fields([r], {"camera_angle"})
    assert out.attrs == frozenset({"pose=standing"}) and out.excluded["camera_angle"] == "unreliable"


# --- exploration and specs ------------------------------------------------------------------------------------------
def _row(i, creator, y, attrs, month="2026-01", known=("candid_level", "palette")):
    return recipes.Row(post_id=f"p{i}", creator_id=creator, medium="photo", y=y, attrs=frozenset(attrs),
                       known=frozenset(known), context={"medium": "photo", "month": month})


def test_contrast_shows_what_winners_use_that_we_do_not():
    ext = [_row(i, f"c{i % 4}", 0.5, {"candid_level=high"}) for i in range(12)]
    own = [_row(100 + i, "own", None, {"candid_level=low"}) for i in range(6)]
    out = inquiry.contrast_sources(ext, own)
    assert out["under_used"][0]["attr"] == "candid_level=high" and out["under_used"][0]["gap"] == 1.0
    assert out["over_used"][0]["attr"] == "candid_level=low"


def test_a_trend_finds_its_change_point():
    rows = []
    for m in range(1, 9):
        for j in range(20):
            rising = m >= 5 and j < 15 or m < 5 and j < 3
            rows.append(_row(m * 100 + j, f"c{j % 5}", 0.0, {"palette=pastel"} if rising else set(),
                             month=f"2026-{m:02d}"))
    out = inquiry.trend(rows, "palette=pastel")
    cp = out["change_point"]
    assert cp["from_month"] == "2026-05" and cp["direction"] == "rising" and cp["p_approx"] < 0.001


def test_a_spec_keeps_known_explore_looks_and_outcomes_and_drops_the_rest(policy):
    cat = research_questions.catalogue(policy)
    attr = cat["attributes"][0]
    spec, dropped = research_questions.clean_spec(policy, {
        "explore": [{"method": "trend", "attribute": attr}, {"method": "astrology"}, {"method": "groups"}],
        "outcome": "shares:d7"})
    assert spec["explore"] == [{"method": "trend", "attr": attr}, {"method": "groups", "by": "medium"}]
    assert spec["outcome"] == ["shares", "d7"] and any("astrology" in d for d in dropped)
    spec, dropped = research_questions.clean_spec(policy, {"outcome": "vibes:d7"})
    assert spec["outcome"] is None and "outcome vibes:d7" in dropped


def test_every_analysis_is_counted_in_the_trial_ledger(conn, policy):
    inquiry.record(conn, policy, requester="t", method="trend", spec={"attr": "x=y"}, result={}, family="f1")
    inquiry.record(conn, policy, requester="t", method="recipes", spec={}, result={}, family="f1", family_size=12)
    assert inquiry.trials(conn, "f1") == 13


# --- predictions against a naive baseline ---------------------------------------------------------------------------
def test_the_scorecard_compares_each_role_with_the_base_rate(conn):
    for i, held in enumerate([True, True, True, False]):
        conn.execute("INSERT INTO predictions (role, claim, metric, comparator, threshold, probability, horizon_days, "
                     "mode, outcome, brier, resolved_at, resolve_at) VALUES ('showrunner', 'c', 'median_reach', "
                     "'above', 1, 0.9, 7, 'real', %s, %s, %s, %s)",
                     (held, (0.9 - (1.0 if held else 0.0)) ** 2, NOW + timedelta(hours=i), NOW))
    base = predictions.baseline_briers(conn, mode="real")
    assert len(base) == 4 and list(base.values())[0] == 0.25                 # the first: a coin before any history
    [row] = predictions.scorecard(conn, mode="real", role="showrunner")
    assert row["brier_base_rate"] is not None and row["skill"] is not None
