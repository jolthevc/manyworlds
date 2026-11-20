"""Experiments (M6): a promising observational finding becomes a randomised test with two arms, a
minimum sample per arm and a fixed stopping point; only its result can make a finding `proven`."""

from __future__ import annotations

import random

from manyworlds.brain import evidence, experiments

from brain_helpers import creator


def _finding(conn, cid, effect_id, *, grade="tentative", effect=0.4):
    return evidence.record_finding(conn, scope="creator", creator_id=cid, statement=f"{effect_id} may help",
                                   evidence="simulated", grade=grade, effect=effect, interval=(0.1, 0.7), n=30,
                                   sources=[{"creator_id": str(cid)}, {"table": "simulated_metrics"},
                                            {"effects": [effect_id]}])


def test_a_tentative_finding_opens_one_experiment_with_a_control_arm(conn, policy):
    cid = creator(conn, publish_mode="simulated")
    _finding(conn, cid, "attribute:lead=personality")
    opened = experiments.open_from_findings(conn, policy, cid)
    assert len(opened) == 1
    exp = experiments.open_for(conn, cid)[0]
    cfg = policy.get("brain", "experiments")
    assert exp["metric"] == "reach_lift" and exp["min_sample"] == experiments.sample_size(policy, cfg["default_sd"])
    a, b = exp["arms"]
    assert a["name"] == "A" and a["attribute"] == {"lead": "personality"} and "lead" in a["instruction"]
    assert b["name"] == "B" and b["attribute"] == {} and "not" in b["instruction"].lower()
    assert experiments.open_from_findings(conn, policy, cid) == []          # never twice for the same effect


def test_samples_are_sized_to_detect_the_smallest_effect_worth_finding(policy):
    cfg = policy.get("brain", "experiments")
    noisy, calm = experiments.sample_size(policy, 0.9), experiments.sample_size(policy, 0.3)
    assert noisy > calm >= cfg["min_sample"] and noisy <= cfg["max_sample"]
    assert experiments.sample_size(policy, 0.6) == 36        # 2 * (1.96 + 0.84)^2 * 0.6^2 / 0.4^2, rounded up


def test_speculative_findings_and_unsupported_effects_open_nothing(conn, policy):
    cid = creator(conn, publish_mode="simulated")
    _finding(conn, cid, "attribute:mood=warm", grade="speculative")
    _finding(conn, cid, "format_dimension:story:personality")
    assert experiments.open_from_findings(conn, policy, cid) == []


def test_open_experiments_are_capped_per_creator(conn, policy):
    cid = creator(conn, publish_mode="simulated")
    for key in ("attribute:lead=personality", "dimension:aspiration", "format:carousel", "attribute:mood=calm"):
        _finding(conn, cid, key)
    experiments.open_from_findings(conn, policy, cid)
    assert len(experiments.open_for(conn, cid)) == policy.get("brain", "experiments", "max_open_per_creator")


def test_slots_are_random_balanced_and_repeatable(conn, policy):
    cid = creator(conn, publish_mode="simulated")
    _finding(conn, cid, "dimension:personality")
    experiments.open_from_findings(conn, policy, cid)
    slots = experiments.assign(conn, policy, cid, start="2026-11-20", days=3)
    per = int(policy.get("brain", "experiments", "slots_per_batch"))
    assert len(slots) == per and sorted(s["arm"] for s in slots).count("A") == per // 2
    assert all(0 <= s["day"] < 3 and s["instruction"] for s in slots)
    assert slots == experiments.assign(conn, policy, cid, start="2026-11-20", days=3)       # same batch, same slots
    other = [experiments.assign(conn, policy, cid, start=f"2026-12-{d:02d}", days=3) for d in range(1, 12)]
    assert len({tuple((s["day"], s["arm"]) for s in o) for o in other}) > 1                # randomised across batches


def _arms(n, diff, rng):
    return [{"arm": "A", "y": rng.gauss(diff, 0.5)} for _ in range(n)] + \
           [{"arm": "B", "y": rng.gauss(0.0, 0.5)} for _ in range(n)]


def test_a_planted_difference_is_detected_with_the_right_sign(policy):
    res = experiments.compare(_arms(30, 0.6, random.Random(1)), policy)
    assert res["effect"] > 0 and res["low"] > 0 and res["significant"]


def test_null_experiments_rarely_claim_an_effect(policy):
    alpha = float(policy.get("brain", "experiments", "alpha"))
    hits = sum(experiments.compare(_arms(20, 0.0, random.Random(s)), policy, seed=s)["significant"]
               for s in range(200))
    assert hits / 200 <= alpha + 0.05


def _exp_rows(n_a, n_b, diff, seed=0):
    """Both arms share the same spread of outcomes; arm A is shifted by `diff` (0: exactly no difference)."""
    rng = random.Random(seed)
    base = [rng.gauss(0.0, 0.4) for _ in range(max(n_a, n_b))]
    return [{"arm": "A", "lift": base[i] + diff, "post_id": f"a{i}"} for i in range(n_a)] + \
           [{"arm": "B", "lift": base[i], "post_id": f"b{i}"} for i in range(n_b)]


def test_an_experiment_waits_for_its_sample_then_concludes_once(conn, policy):
    cid = creator(conn, publish_mode="simulated")
    _finding(conn, cid, "attribute:lead=personality")
    experiments.open_from_findings(conn, policy, cid)
    exp = experiments.open_for(conn, cid)[0]
    need = exp["min_sample"]
    assert experiments.conclude(conn, policy, exp, _exp_rows(need, need - 1, 0.8)) is None     # B is short
    out = experiments.conclude(conn, policy, exp, _exp_rows(need, need, 0.8))
    assert out["status"] == "confirmed" and out["finding_id"]
    row = conn.execute("SELECT status, result FROM experiments WHERE id = %s", (exp["id"],)).fetchone()
    assert row["status"] == "concluded" and row["result"]["status"] == "confirmed"
    proven = evidence.findings(conn, mode="sandbox", creator_id=cid, grades=("proven",))
    assert len(proven) == 1 and proven[0]["evidence"] == "simulated" and proven[0]["label"]
    assert evidence.findings(conn, mode="real", creator_id=cid) == []                         # never real evidence
    assert experiments.open_for(conn, cid) == []
    evidence.supersede(conn, creator_id=cid)                                                  # the next analysis
    assert len(evidence.findings(conn, mode="sandbox", creator_id=cid, grades=("proven",))) == 1


def test_no_difference_is_a_result_too_and_makes_no_finding(conn, policy):
    cid = creator(conn, publish_mode="simulated")
    _finding(conn, cid, "attribute:lead=personality")
    experiments.open_from_findings(conn, policy, cid)
    exp = experiments.open_for(conn, cid)[0]
    need = exp["min_sample"]
    out = experiments.conclude(conn, policy, exp, _exp_rows(need, need, 0.0, seed=3))
    assert out["status"] == "no_difference" and out["finding_id"] is None
    assert experiments.concluded_for(conn, cid)[0]["result"]["status"] == "no_difference"
