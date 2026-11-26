"""The learning harness (docs/2026-11-26-learning-harness.md section 4): defect-only rejection, the exploration
temperature, hypotheses not beliefs, cross-portfolio propagation, the noise guard and policy rollback."""

from __future__ import annotations

import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

from manyworlds import voice_eval as ve
from manyworlds.brain import events, evidence, experiments, exploration, hypotheses, overrides, patterns

from brain_helpers import creator

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone.utc)


# --- 1. defect-only rejection ---------------------------------------------------------------------------------------
def _gate_names_in_code() -> set[str]:
    """Every gate name a review row is written under (the modules' gate lists, and literal names in the review and
    production code), so a new gate written anywhere must be classified."""
    from manyworlds.review import common, detail, final, gates, video
    names = set(gates.GATES) | set(video.GATES) | set(gates.BODY_CHECKS) | set(detail.JUDGED) | \
        {f"final_{g}" for g in final.GATES} | set(common.CONTEXT_SCORES) | {"scene", "direction", "performance"} | \
        {"set_match", "dance_identity", "lip_closure", "background"}   # checks keyed in code (gates.py, video.py)
    row = re.compile(r"\(\s*(?:a|v|aid|asset_id|a\[\"id\"\]|v\[\"id\"\])(?:\[\"id\"\])?,\s*\"([a-z_]+)\",")
    sql = re.compile(r"INSERT INTO reviews \([^)]*\) VALUES \(%s, '([a-z_]+)'")
    for folder in ("review", "production"):
        for p in (ROOT / "manyworlds" / folder).glob("*.py"):
            text = p.read_text(encoding="utf-8")
            names |= set(row.findall(text)) | set(sql.findall(text))
    return names


def test_every_post_media_gate_that_can_reject_maps_to_a_material_defect_class(raw_policy):
    ro = raw_policy.get("review", "reject_only")
    names = _gate_names_in_code()
    assert {"identity", "lip_sync", "intelligibility", "final_lip_sync", "set_match", "brand"} <= names
    unclassified = names - set(ro["gates"]) - set(ro["rank_only"])
    assert unclassified == set(), f"gates with no material defect class: {sorted(unclassified)}"
    assert set(ro["gates"].values()) <= set(ro["classes"])
    assert not set(ro["gates"]) & set(ro["rank_only"])
    assert {"brand", "scene", "direction", "distribution", "performance"} <= set(ro["rank_only"])


def test_rank_only_gates_never_reject_under_the_shipped_policy(raw_policy):
    from manyworlds.review import common

    class Read:
        brand_alignment, distribution_risk, distribution_label = 0, 10, "limit boosting"
    v = common.context_verdicts(raw_policy.get("review", "image"), Read())
    assert v["brand"][0] and v["brand"][1]["off_register"] and v["brand"][1]["rank_only"]
    assert v["distribution"][0]
    assert common.direction_passes(0, raw_policy)                       # direction scores, never rejects
    assert raw_policy.get("review", "image", "min_scene") == 0 and raw_policy.get("review", "video", "min_scene") == 0


def test_brand_alignment_still_ranks_candidates_in_selection(conn, policy, store, tmp_path):
    from manyworlds.production import steps
    from manyworlds.review import gates
    from test_review import JudgeStub, creator_with_job, ok
    _, job = creator_with_job(conn, store, tmp_path, n_images=2)
    judge = JudgeStub([ok(), ok()], confirm=[ok(brand_alignment=2), ok(brand_alignment=9)])
    res = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert all(r["accepted"] for r in res)                               # the off-register one is kept...
    best = steps._best_confirmed(conn, [str(job)], policy)
    assert str(best["id"]) == res[1]["asset_id"]                         # ...and loses in selection


def test_only_material_voice_defects_regenerate_a_line(policy):
    cfg = {**policy.get("voice", "judge"), **policy.get("voice", "line_check")}
    good = {"wer": 0.0, "mos": 4.0, "f0_std_st": 2.5, "wps": 4.0, "words": 8, "ref_distance": 0.2,
            "energy_std_db": 5.0, "f0_range_st": 6.0}
    band = (3.6, 4.4)
    flat = ve.line_verdict({**good, "f0_std_st": 0.3}, cfg, band)
    slow = ve.line_verdict({**good, "wps": 1.5}, cfg, band)
    assert not flat["passed"] and not flat["regenerate"] and not slow["regenerate"]   # recorded, kept
    for over, word in (({"wer": 0.5}, "unclear"), ({"mos": 2.0}, "robotic"), ({"ref_distance": 0.8}, "inconsistent")):
        v = ve.line_verdict({**good, **over}, cfg, band)
        assert v["regenerate"] and v["defects"][0].startswith(word)
    assert ve.line_verdict({**good, "f0_std_st": 0.3}, {**cfg, "regenerate_on": ["flat"]}, band)["regenerate"]
    # a take without a defect beats one with, whatever the composite
    assert ve.better({"passed": False, "regenerate": False, "composite": 0.4},
                     {"passed": False, "regenerate": True, "composite": 0.9})


# --- 2. the exploration temperature -----------------------------------------------------------------------------------
def _stage(conn, cid, stage):
    conn.execute("INSERT INTO creator_states (creator_id, stage) VALUES (%s, %s) ON CONFLICT (creator_id) "
                 "DO UPDATE SET stage = EXCLUDED.stage", (cid, stage))


def _ideas(conn, cid, level, n, *, days_ago=1):
    for _ in range(n):
        conn.execute("INSERT INTO ideas (creator_id, batch, concept, format, body, attributes, status, created_at) "
                     "VALUES (%s, 'b', 'c', 'photo', '{}', %s, 'proposed', %s)",
                     (cid, Jsonb({"exploration": level}), NOW - timedelta(days=days_ago)))


def test_shares_follow_the_creator_stage(conn, policy):
    launch, settled = creator(conn), creator(conn)
    _stage(conn, launch, "exploring")
    _stage(conn, settled, "growing")
    a = exploration.shares(conn, policy, launch)["shares"]
    b = exploration.shares(conn, policy, settled)["shares"]
    assert a["novel"] > b["novel"] and a["refine"] < b["refine"]
    assert abs(sum(a.values()) - 1) < 1e-3 and abs(sum(b.values()) - 1) < 1e-3


def test_a_creator_temperature_is_set_only_within_authority_and_is_reversible(conn, policy):
    cid = creator(conn)
    path = f"brain.exploration_mix.creators.{cid}.novel"
    assert overrides.propose(conn, policy, path, 0.35, author="talent_manager", reason="plateau")["applied"]
    eff = overrides.effective_policy(conn, policy)
    s = exploration.shares(conn, eff, cid)
    assert s["own"] and s["shares"]["novel"] == 0.35
    beyond = overrides.propose(conn, policy, path, 0.9, author="talent_manager", reason="more")
    assert not beyond["applied"] and beyond["approval_id"]               # beyond authority: the owner decides
    oid = conn.execute("SELECT id FROM policy_overrides WHERE path = %s", (path,)).fetchone()["id"]
    with pytest.raises(ValueError, match="no value"):                    # nothing before it to restore
        overrides.revert(conn, policy, oid, author="owner", reason="r")


def test_the_portfolio_total_corrects_every_creator_toward_its_target(conn, policy):
    cid, other = creator(conn), creator(conn)
    before = exploration.shares(conn, policy, cid)
    _ideas(conn, other, "refine", 30)                                    # the portfolio has explored nothing
    after = exploration.shares(conn, policy, cid)
    mc = policy.get("brain", "exploration_mix", "portfolio", "max_correction")
    assert after["correction"] == pytest.approx(1 + mc)
    assert after["shares"]["novel"] > before["shares"]["novel"]


def test_batch_targets_are_exact_and_close_the_gap_recent_batches_left(conn, policy):
    cid = creator(conn)
    fresh = exploration.batch_targets(conn, policy, cid, 10)
    assert sum(fresh["targets"].values()) == 10
    _ideas(conn, cid, "refine", 10)                                      # last batch explored nothing
    carry = exploration.batch_targets(conn, policy, cid, 10)
    assert sum(carry["targets"].values()) == 10
    assert carry["targets"]["refine"] < fresh["targets"]["refine"]
    assert carry["targets"]["novel"] + carry["targets"]["adjacent"] > fresh["targets"]["novel"] + \
        fresh["targets"]["adjacent"]


def test_reconcile_trims_the_most_over_target_level_and_flags_without_relabelling():
    levels = ["refine", "novel", "novel", "novel", "refine", "adjacent"]
    fit = exploration.reconcile(levels, {"refine": 3, "adjacent": 1, "novel": 1}, 5, keep={3})
    assert fit["trimmed"] == [2]                                         # a novel one, never the kept slot (3)
    assert fit["planned"] == {"refine": 2, "adjacent": 1, "novel": 2}
    assert [levels[i] for i in fit["flagged"]] == ["novel"]
    assert fit["shortfall"] == {"refine": 1, "adjacent": 0, "novel": 0}
    assert exploration.allocate(7, {"refine": 0.6, "adjacent": 0.3, "novel": 0.1}) == \
        {"refine": 4, "adjacent": 2, "novel": 1}


def test_the_showrunner_gets_exact_counts_and_code_fits_the_batch(conn, policy):
    from test_brain_showrunner import PlanLLM, _maren, _run, idea
    cid = _maren(conn)
    many = [idea(i % 3, caption=f"moment number {i}, quiet and good", exploration="novel") for i in range(40)]
    llm = PlanLLM(many, conn=conn)
    out, _ = _run(conn, policy, cid, llm)
    prompt = str(llm.calls[0]["content"])
    d = conn.execute("SELECT inputs FROM decisions WHERE kind = 'content_batch'").fetchone()["inputs"]
    mix, total = d["exploration_mix"], d["requested"]["total"]
    t = mix["targets"]
    assert f"exactly {t['refine']} refine" in prompt and f"{t['novel']} novel" in prompt
    size = max(total, math.ceil(total * (1 + policy.get("brain", "exploration_mix", "trim_over_share"))))
    assert out["ideas"] == size == mix["planned"]["novel"] and mix["trimmed"] == 40 - size
    assert any("exploration mix" in x["why"] for x in out["dropped"])
    rows = conn.execute("SELECT body, attributes FROM ideas").fetchall()
    assert all(r["attributes"]["exploration"] == "novel" and r["body"]["exploratory"] for r in rows)
    assert sum(bool(r["body"].get("exploration_off_target")) for r in rows) == size - t["novel"]
    window = conn.execute("SELECT evidence_window FROM creator_states WHERE creator_id = %s", (cid,)).fetchone()
    assert window["evidence_window"]["exploration"]["planned"]["novel"] == size


# --- 3. hypotheses, not permanent beliefs ---------------------------------------------------------------------------
def test_findings_carry_a_hypothesis_record_with_defaults(conn, policy):
    cid = creator(conn)
    fid = evidence.record_finding(conn, scope="creator", creator_id=cid, statement="s", evidence="simulated",
                                  grade="tentative", sources=[{"experiment_id": "e"}])
    f = evidence.finding(conn, fid)
    assert f["origin"] == "experiment" and f["level"] == "character" and f["last_tested"] is not None
    assert f["replications"] == [] and f["contradictions"] == [] and f["label"] == evidence.LABEL
    post = evidence.record_finding(conn, scope="post", statement="p", evidence="simulated", grade="speculative")
    assert evidence.finding(conn, post)["origin"] == "post"


def _f(**kw):
    return {"statement": "s", "grade": "proven", "origin": "experiment", "created_at": NOW, "last_tested": NOW,
            "replications": [], "contradictions": [], **kw}


def test_stale_proven_findings_decay_and_are_due_for_retest(policy):
    days = policy.get("brain", "hypotheses", "retest_days", "proven")
    fresh, stale = _f(), _f(last_tested=NOW - timedelta(days=days + 400))
    assert not hypotheses.due_for_retest(policy, fresh, NOW)
    assert hypotheses.view(policy, stale, NOW)["due_for_retest"]
    floor = policy.get("brain", "hypotheses", "decay", "floor")
    assert hypotheses.weight(policy, stale, NOW) == pytest.approx(hypotheses.weight(policy, fresh, NOW) * floor)
    assert hypotheses.view(policy, stale, NOW)["age_days"] >= days + 400


def test_our_own_experiments_outrank_observation_and_replication_raises_weight(policy):
    obs = _f(origin="creator")
    assert hypotheses.weight(policy, _f(), NOW) > hypotheses.weight(policy, obs, NOW)
    held = _f(replications=[{"creator_id": "a"}, {"creator_id": "b"}])
    failed = _f(replications=[{"creator_id": "a"}], contradictions=[{"creator_id": "b"}])
    assert hypotheses.weight(policy, held, NOW) > hypotheses.weight(policy, _f(), NOW) > \
        hypotheses.weight(policy, failed, NOW)


def test_planning_prompts_show_level_replications_and_age(conn, policy):
    from test_brain_showrunner import PlanLLM, _maren, _run, idea
    cid = _maren(conn)
    fid = evidence.record_finding(conn, scope="creator", creator_id=cid, statement="steam shots get saved",
                                  evidence="simulated", grade="proven", sources=[{"experiment_id": "x"}])
    conn.execute("UPDATE simulated_findings SET last_tested = now() - interval '400 days' WHERE id = %s", (fid,))
    llm = PlanLLM([idea(0)], conn=conn)
    _run(conn, policy, cid, llm)
    prompt = str(llm.calls[0]["content"])
    assert "steam shots get saved" in prompt and '"level": "character"' in prompt
    assert '"replicated_in": 0' in prompt and '"due_for_retest": true' in prompt and '"age_days": 4' in prompt


def test_card_lines_and_weight_carry_the_hypothesis_record(policy):
    old = NOW - timedelta(days=300)
    card = {"key": "k", "medium": "photo", "situation": "s", "mechanism": "m", "sources": [], "evidence": {},
            "replications": [{"creator_id": "a"}, {"creator_id": "b"}], "contradictions": [{"creator_id": "c"}],
            "last_tested": old, "retest_after": old + timedelta(days=60), "level": "archetype"}
    line = patterns.brief([card])
    assert "held in 2 of our creators, failed in 1" in line and "DUE FOR RETEST" in line
    fresh = {**card, "last_tested": NOW, "retest_after": NOW + timedelta(days=60)}
    assert patterns.weight(policy, card) < patterns.weight(policy, fresh)
    held, failed = patterns._held({"min_posts_per_creator": 2, "hold_ratio": 1.0, "fail_ratio": 0.8},
                                  {"a": [1.5, 1.2], "b": [0.5, 0.6], "c": [3.0]})
    assert [h["creator_id"] for h in held] == ["a"] and [f["creator_id"] for f in failed] == ["b"]


# --- 4. cross-portfolio propagation ---------------------------------------------------------------------------------
def _archetyped(conn, archetype, publish_mode="simulated"):
    cid = creator(conn, publish_mode=publish_mode)
    conn.execute("UPDATE creators SET bible = %s WHERE id = %s", (Jsonb({"archetypes": [archetype]}), cid))
    return cid


def _rows(n, diff):
    return [{"arm": "A", "lift": diff + 0.01 * (i % 5), "post_id": f"a{i}"} for i in range(n)] + \
           [{"arm": "B", "lift": 0.01 * (i % 5), "post_id": f"b{i}"} for i in range(n)]


def _test_on(conn, policy, cid, diff):
    exp = experiments.open_for(conn, cid)[0]
    return experiments.conclude(conn, policy, exp, _rows(exp["min_sample"], diff))


def test_a_proven_result_becomes_a_hypothesis_for_comparable_creators_never_a_rule(conn, policy):
    a, b, d = (_archetyped(conn, "fashion_style") for _ in range(3))
    other = _archetyped(conn, "food_home")
    key = "attribute:lead=personality"
    evidence.record_finding(conn, scope="creator", creator_id=a, statement="lead=personality may help",
                            evidence="simulated", grade="tentative", effect=0.4, interval=(0.1, 0.7), n=30,
                            sources=[{"creator_id": str(a)}, {"table": "simulated_metrics"}, {"effects": [key]}])
    experiments.open_from_findings(conn, policy, a)
    out = _test_on(conn, policy, a, 0.8)
    assert out["status"] == "confirmed" and out["hypothesis"]["opened"]
    h = evidence.finding(conn, out["hypothesis"]["hypothesis_id"])
    assert (h["scope"], h["level"], h["grade"], h["creator_id"]) == ("archetype", "archetype", "speculative", None)
    assert [x["creator_id"] for x in h["replications"]] == [str(a)] and h["parent_id"] == uuid.UUID(out["finding_id"])
    # offered to a comparable creator as an experiment, not to another archetype, never as her finding
    assert [x["id"] for x in hypotheses.hypotheses_for(conn, policy, b)] == [h["id"]]
    assert hypotheses.hypotheses_for(conn, policy, other) == [] and hypotheses.hypotheses_for(conn, policy, a) == []
    assert evidence.findings(conn, mode="sandbox", creator_id=b) == []
    assert hypotheses.planning_views(conn, policy, b)[0]["tests"] == key
    # b tests it and it holds: a replication, and it grades tentative
    assert experiments.open_from_findings(conn, policy, b)
    assert experiments.open_for(conn, b)[0]["source_finding"] == h["id"]
    held = _test_on(conn, policy, b, 0.8)["hypothesis"]
    assert held["held"] and held["grade"] == "tentative"
    w_before = hypotheses.weight(policy, evidence.finding(conn, h["id"]))
    # d tests it and finds nothing: a contradiction lowers its weight
    experiments.open_from_findings(conn, policy, d)
    failed = _test_on(conn, policy, d, 0.0)["hypothesis"]
    assert not failed["held"]
    h2 = evidence.finding(conn, h["id"])
    assert len(h2["replications"]) == 2 and len(h2["contradictions"]) == 1
    assert hypotheses.weight(policy, h2) < w_before


def test_contradictions_that_dominate_close_a_hypothesis(policy):
    reps = [{"creator_id": "a", "archetype": "fashion_style"}]
    cons = [{"creator_id": x} for x in "bcd"]
    assert hypotheses.standing(policy, {"replications": reps, "contradictions": cons})["status"] == "contradicted"
    across = [{"creator_id": "a", "archetype": "fashion_style"}, {"creator_id": "b", "archetype": "beauty_grwm"}]
    assert hypotheses.standing(policy, {"replications": across, "contradictions": []}) == \
        {"grade": "tentative", "level": "portfolio", "status": "open"}


def test_casting_reads_archetype_hypotheses_with_their_replications(conn, policy):
    from manyworlds.brain import casting
    a, b = _archetyped(conn, "fashion_style", "instagram"), _archetyped(conn, "fashion_style", "instagram")
    evidence.record_finding(conn, scope="archetype", statement="outfit-of-the-week carousels grow reach",
                            evidence="experimental", grade="tentative", effect=0.3, origin="experiment",
                            level="archetype", applies_to={"archetype": "fashion_style", "effect_key": "format:x"},
                            replications=[{"creator_id": str(a)}, {"creator_id": str(b)}])
    text = casting._hypotheses_text(conn, policy)
    assert "outfit-of-the-week" in text and '"replicated_in": 2' in text and "fashion_style" in text


# --- 5. the noise guard ------------------------------------------------------------------------------------------------
def _breakout(conn, cid, setting):
    pid = conn.execute("INSERT INTO posts (creator_id, format, status, caption) VALUES (%s, 'photo', 'simulated', "
                       "'c') RETURNING id", (cid,)).fetchone()["id"]
    conn.execute("INSERT INTO ideas (creator_id, batch, concept, format, body, attributes, status, post_id) "
                 "VALUES (%s, 'b', 'c', 'photo', '{}', %s, 'scheduled', %s)", (cid, Jsonb({"setting": setting}), pid))
    conn.execute("INSERT INTO events (type, subject, payload, routed_to, creator_id, dedupe_key) VALUES "
                 "('breakout', %s, %s, 'analyst', %s, %s)",
                 (Jsonb({"creator_id": str(cid), "post_id": str(pid)}),
                  Jsonb({"ratio": 4.0, "caption": setting, "format": "photo", "mode": "sandbox"}), cid,
                  f"breakout:{pid}"))
    return pid


def test_a_breakout_reaches_planning_as_one_unreplicated_post_until_it_repeats(conn, policy):
    her, peer = _archetyped(conn, "fashion_style"), _archetyped(conn, "fashion_style")
    _breakout(conn, her, "rooftop at dusk")
    notes = events.for_planning(conn, her, policy=policy)
    assert notes[0]["evidence"] == events.UNREPLICATED
    for _ in range(policy.get("brain", "events", "noise_guard", "min_posts") - 1):
        _breakout(conn, peer, "rooftop at dusk")                         # a comparable creator's posts count too
    notes = events.for_planning(conn, her, policy=policy)
    assert notes[0]["evidence"].startswith("replicated: ") and "setting=rooftop at dusk in 3 breakouts" in         notes[0]["evidence"]


def test_attribution_joins_exploration_controls_directed_fields_and_all_in_cost(conn, policy):
    cid = _archetyped(conn, "lifestyle_slow", "instagram")
    for i, (level, size, reach, usd) in enumerate((("novel", "close-up", 3000, 0.2), ("refine", "wide", 1000, 0.1),
                                                   ("refine", "wide", 1500, 0.1))):
        pid = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'photo', 'published', %s) "
                           "RETURNING id", (cid, Jsonb({"shots": [{"shot_type": "x", "framing": {"shot_size": size},
                                                                   "performance": {"energy": "low"}}]}))).fetchone()["id"]
        conn.execute("INSERT INTO ideas (creator_id, batch, concept, format, body, attributes, status, post_id) "
                     "VALUES (%s, 'b', 'c', 'photo', '{}', %s, 'scheduled', %s)",
                     (cid, Jsonb({"exploration": level}), pid))
        job = conn.execute("INSERT INTO jobs (kind, status, creator_id, post_id) VALUES ('image.zimage', 'done', %s, "
                           "%s) RETURNING id", (cid, pid)).fetchone()["id"]
        asset = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES "
                             "(%s, %s, 'image', %s, 'x', 1) RETURNING id", (job, cid, f"s3://b/{i}.png")).fetchone()["id"]
        conn.execute("INSERT INTO performance_observations (asset_id, post_id, creator_id, stage, requested) "
                     "VALUES (%s, %s, %s, 'confirm', %s)", (asset, pid, cid, Jsonb({"smile": 2 if i == 0 else 1})))
        conn.execute("INSERT INTO cost_entries (job_id, post_id, category, provider, quantity, unit, usd) VALUES "
                     "(%s, %s, 'gpu', 'runpod', 1, 'gpu_second', %s)", (job, pid, usd))
        conn.execute("INSERT INTO publications (post_id, platform, external_id, ai_label, simulated, published_at) "
                     "VALUES (%s, 'instagram', %s, true, false, %s)", (pid, f"m{i}", NOW - timedelta(days=5)))
        conn.execute("INSERT INTO metric_snapshots (post_id, captured_at, simulated, metrics, post_age_s) VALUES "
                     "(%s, %s, false, %s, 300000)", (pid, NOW, Jsonb({"reach": reach})))
    dims = policy.get("brain", "talent_manager", "brand", "attribution_dims")
    a = evidence.attribution(conn, [cid], "real", policy=policy, dims=dims)
    assert a["exploration"]["novel"]["n"] == 1 and a["exploration"]["refine"]["n"] == 2
    assert a["control"]["smile=2"]["n"] == 1 and a["shot_size"]["wide"]["n"] == 2 and a["energy"]["low"]["n"] == 3
    assert a["exploration"]["novel"]["usd_per_post"] == 0.2 and a["exploration"]["novel"]["reach_per_usd"] == 15000.0


# --- 6. rollback -----------------------------------------------------------------------------------------------------
def test_a_policy_override_is_reverted_as_a_new_recorded_override(conn, policy):
    path = "brain.exploration_share"
    overrides.propose(conn, policy, path, 0.4, author="strategist", reason="explore")
    overrides.propose(conn, policy, path, 0.5, author="strategist", reason="more")
    first, second = (r["id"] for r in conn.execute("SELECT id FROM policy_overrides ORDER BY version").fetchall())
    with pytest.raises(ValueError, match="revert that one first"):
        overrides.revert(conn, policy, first, author="owner", reason="r")
    out = overrides.revert(conn, policy, second, author="owner", reason="the lift was noise")
    assert out["applied"] and out["value"] == 0.4 and out["restored_from"] == "override v1"
    assert overrides.effective_policy(conn, policy).get("brain", "exploration_share") == 0.4
    row = conn.execute("SELECT author, reason, version FROM policy_overrides ORDER BY version DESC LIMIT 1").fetchone()
    assert row["author"] == "owner" and "revert of override" in row["reason"] and "noise" in row["reason"]
    third = conn.execute("SELECT id FROM policy_overrides ORDER BY version DESC LIMIT 1").fetchone()["id"]
    back = overrides.revert(conn, policy, third, author="owner", reason="undo the undo")   # reversible itself
    assert back["value"] == 0.5 and overrides.effective_policy(conn, policy).get("brain", "exploration_share") == 0.5


def test_a_revert_beyond_authority_waits_for_the_owner(conn, policy):
    path = "brain.exploration_share"
    r = overrides.propose(conn, policy, path, 0.7, author="strategist", reason="r")
    overrides.decide(conn, r["approval_id"], approved=True, by="owner")
    overrides.propose(conn, policy, path, 0.6, author="strategist", reason="r")
    last = conn.execute("SELECT id FROM policy_overrides ORDER BY version DESC LIMIT 1").fetchone()["id"]
    out = overrides.revert(conn, policy, last, author="strategist", reason="back")   # 0.7 is beyond authority
    assert not out["applied"] and out["approval_id"]
    assert overrides.effective_policy(conn, policy).get("brain", "exploration_share") == 0.6


def test_the_cli_reverts_an_override(conn, policy, capsys):
    from manyworlds import cli
    overrides.propose(conn, policy, "brain.exploration_share", 0.45, author="strategist", reason="r")
    oid = conn.execute("SELECT id FROM policy_overrides").fetchone()["id"]
    args = cli.build_parser().parse_args(["policy", "revert", str(oid), "--reason", "noise"])
    assert (args.cmd, args.policy_cmd, args.override_id, args.author) == ("policy", "revert", oid, "owner")
    assert cli._policy_overrides(args, conn, policy) == 0
    assert '"restored_from": "the policy file"' in capsys.readouterr().out
    assert overrides.effective_policy(conn, policy).get("brain", "exploration_share") == \
        policy.get("brain", "exploration_share")
