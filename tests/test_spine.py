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


# --- Pegasus on our own videos (off until the owner's account and training opt-out) --------------------------------
def _with_pegasus(policy, **over):
    import dataclasses
    intel = {**policy.body["intelligence"], "pegasus": {**policy.body["intelligence"]["pegasus"], **over}}
    return dataclasses.replace(policy, body={**policy.body, "intelligence": intel})


def test_pegasus_refuses_until_enabled_opted_out_keyed_and_own(conn, policy, monkeypatch):
    from manyworlds.research import pegasus
    assert "enabled is false" in pegasus.refused(conn, policy, "x")
    on = _with_pegasus(policy, enabled=True)
    assert "training opt-out" in pegasus.refused(conn, on, "x")
    on = _with_pegasus(policy, enabled=True, training_opt_out_confirmed=True)
    monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
    assert "TWELVELABS_API_KEY" in pegasus.refused(conn, on, "x")
    monkeypatch.setenv("TWELVELABS_API_KEY", "test-key")
    assert "own posts" in pegasus.refused(conn, on, "00000000-0000-0000-0000-000000000000")


def test_a_pegasus_read_lands_as_time_ordered_beats_and_is_charged(conn, policy, store, tmp_path, monkeypatch):
    from brain_helpers import creator, make_post

    from manyworlds.research import pegasus
    monkeypatch.setenv("TWELVELABS_API_KEY", "test-key")
    on = _with_pegasus(policy, enabled=True, training_opt_out_confirmed=True)
    cid = creator(conn)
    post = make_post(conn, cid)
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, post_id) VALUES ('talk.infinitetalk', 'done', %s, "
                       "%s) RETURNING id", (cid, post)).fetchone()["id"]
    clip = tmp_path / "take.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    asset = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'video', "
                         "%s, 'x', 1) RETURNING id", (job, cid, store.put_file(clip, "t/take.mp4").uri)).fetchone()["id"]
    seen = []

    def fake(path, c):
        seen.append(path)
        return {"duration_s": 60.0, "usage": {"output_tokens": 1000}, "deleted": True, "segments": {
            "beat": [{"start_time": 2.0, "end_time": 3.5, "metadata": {"kind": "gesture", "energy": 2}},
                     {"start_time": 0.0, "end_time": 1.0, "metadata": {"kind": "opening", "energy": 1}}],
            "overall": [{"start_time": 0, "end_time": 6, "metadata": {"naturalness": 7, "defects": ["frozen hand"]}}]}}

    out = pegasus.read_assets(conn, on, store, tmp_path, [str(asset)], analyse_=fake)
    [r] = out["reads"]
    assert r["deleted"] and abs(r["usd"] - (0.029 + 0.0075)) < 1e-6 and not clip.with_name(f"{asset}.mp4").exists()
    beats = conn.execute("SELECT u.seq, u.start_s, v.value FROM content_units u JOIN attribute_values v ON v.unit_id = u.id "
                         "WHERE u.level = 'beat' AND v.attribute = 'av_kind' ORDER BY u.seq").fetchall()
    assert [(b["seq"], float(b["start_s"]), b["value"]) for b in beats] == [(0, 0.0, "opening"), (1, 2.0, "gesture")]
    assert conn.execute("SELECT usd FROM cost_entries WHERE agent = 'research:pegasus'").fetchone()["usd"] > 0
    # a reference post's media is never sent
    assert "own posts" in pegasus.refused(conn, on, "00000000-0000-0000-0000-000000000000")


# --- audience propositions (owner 2026-11-26: why people follow, apart from what she posts about) -------------------
def test_the_proposition_library_names_the_owners_ten_and_four_jobs(policy):
    from manyworlds.brain import archetypes
    names = archetypes.proposition_names(policy)
    assert len(names) == 10 and {"aspirational_icon", "protagonist", "comfort_figure"} <= set(names)
    assert archetypes.jobs(policy) == ["attention", "follow", "return", "affinity"]
    for p in archetypes.propositions(policy)["propositions"].values():
        assert all(p.get(j) for j in ("follow_reason", "attention", "follow", "return", "affinity", "signals"))


def test_a_mix_is_normalised_and_unknown_reasons_become_candidates(policy):
    from manyworlds.brain import archetypes
    mix = archetypes.clean_mix(policy, [{"name": "Muse", "share": 3}, {"name": "cozy chaos", "share": 1},
                                        {"name": "x", "share": 0}])
    assert mix == [{"name": "muse", "share": 0.75}, {"name": "other:cozy_chaos", "share": 0.25}]
    assert archetypes.clean_mix(policy, []) == []


def test_the_dossier_keeps_propositions_verticals_jobs_and_arcs_apart_from_playbooks(policy):
    from manyworlds.brain import archetypes
    from manyworlds.creator import dossier
    play = archetypes.names(policy)[0]
    pick = dossier.ArchetypePick(primary=play, why="w", verticals=["Fashion", "astrology"],
                                 propositions=[{"name": "tastemaker", "share": 0.6}, {"name": "maverick", "share": 0.4}])
    out = dossier.clean(policy, "archetypes", pick, {})
    assert out["archetypes"] == [play] and out["verticals"] == ["fashion"]
    assert [m["name"] for m in out["propositions"]] == ["tastemaker", "maverick"]
    cs = dossier.ContentSystem(mix={"photo": 1, "carousel": 1, "story": 0, "video": 0}, talk_share=0.2, hooks=[],
                               never_makes=[], idea_engines=["Sunday market run"],
                               job_mix={"attention": 2, "follow": 1, "bring_back": 1, "affinity": 0},
                               series=[{"name": "Finds", "premise": "p", "archetype": play, "proposition": "nope",
                                        "job": "return", "arc": "episodic", "capabilities": [], "per_week": 2,
                                        "hook": "h", "why_it_works": "w"},
                                       {"name": "Odd", "premise": "p", "archetype": play, "job": "viral",
                                        "arc": "forever", "capabilities": [], "per_week": 1, "hook": "h",
                                        "why_it_works": "w"}])
    got = dossier.clean(policy, "content_system", cs, {"archetypes": [play], "propositions": out["propositions"]})
    s = got["content_system"]["series"]
    assert (s[0]["job"], s[0]["arc"], s[0]["proposition"]) == ("return", "episodic", "tastemaker")
    assert (s[1]["job"], s[1]["arc"]) == ("", "")
    assert got["content_system"]["job_mix"] == {"attention": 0.5, "follow": 0.25, "return": 0.25, "affinity": 0.0}


def test_reference_creators_are_coded_for_why_people_follow_them(conn, policy):
    from manyworlds.research import propositions

    class FakeLLM:
        def tagged(self, *a):
            import contextlib
            return contextlib.nullcontext()

        def model_for(self, role):
            return "haiku"

        def structured(self, role, *, system, schema, effort, content):
            assert "Propositions" in content and "Creator 1:" in content
            return schema.model_validate({"creators": [{"index": 1, "verticals": ["fashion"], "propositions": [
                {"name": "aspirational_icon", "share": 0.7}, {"name": "quiet rebel", "share": 0.3}]}]})

    c = _ref_creator(conn)
    spine.sync(conn, policy)
    out = propositions.read(conn, policy, FakeLLM(), [str(c)], usd=1.0)
    assert out["creators"] == 1
    vals = {(r["attribute"], r["value"]) for r in conn.execute(
        "SELECT v.attribute, v.value FROM attribute_values v JOIN content_units u ON u.id = v.unit_id "
        "WHERE u.ref_id = %s AND u.level = 'creator'", (str(c),)).fetchall()}
    assert {("proposition", "aspirational_icon"), ("proposition", "other:quiet_rebel"), ("vertical", "fashion")} <= vals
    assert propositions.candidates(conn)[0]["value"] == "other:quiet_rebel"
