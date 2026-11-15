"""The simulated audience: deterministic scenarios with hidden effects the analyst must discover."""

from __future__ import annotations

import math
import statistics

from manyworlds.brain import evidence, simulation

from brain_helpers import creator, make_post

KEYS = {"reach", "views", "likes", "comments", "saves", "shares", "follows"}


def _features(**kw):
    return {"format": "photo", "scores": {"personality": 5, "aspiration": 5}, "attributes": {}, **kw}


def _reach(rows):
    return max(r["metrics"]["reach"] for r in rows)


def test_the_same_seed_gives_the_same_outcomes(policy):
    sc = simulation.load("personality_wins")
    a = simulation.simulate(sc, _features(), index=3, rng_key="post-1:0", policy=policy)
    b = simulation.simulate(sc, _features(), index=3, rng_key="post-1:0", policy=policy)
    c = simulation.simulate(sc, _features(), index=3, rng_key="post-2:0", policy=policy)
    assert a == b and a != c


def test_a_planted_effect_moves_reach_in_its_direction(policy):
    sc = simulation.load("personality_wins")
    hi = [math.log(_reach(simulation.simulate(sc, _features(scores={"personality": 9}), index=0, rng_key=f"h{i}",
                                              policy=policy))) for i in range(300)]
    lo = [math.log(_reach(simulation.simulate(sc, _features(scores={"personality": 2}), index=0, rng_key=f"l{i}",
                                              policy=policy))) for i in range(300)]
    assert statistics.mean(hi) - statistics.mean(lo) > 0.5


def test_breakouts_happen_at_their_rate(policy):
    sc = simulation.load("breakout")
    flags = [simulation.simulate(sc, _features(), index=0, rng_key=f"b{i}", policy=policy)[-1]["breakout"]
             for i in range(2000)]
    assert abs(sum(flags) / 2000 - sc["breakout"]["probability"]) < 0.03


def test_metric_keys_match_real_ingestion_and_ages_follow_policy(policy):
    rows = simulation.simulate(simulation.load("null"), _features(), index=0, rng_key="k", policy=policy)
    assert [r["post_age_s"] for r in rows] == policy.get("simulation", "ages_s")
    assert all(set(r["metrics"]) == KEYS for r in rows)
    assert [r["metrics"]["reach"] for r in rows] == sorted(r["metrics"]["reach"] for r in rows)   # grows with age


def test_simulate_post_writes_simulated_metrics_once(conn, policy):
    post = make_post(conn, creator(conn))
    simulation.simulate_post(conn, policy, post, scenario="null", seed=0)
    simulation.simulate_post(conn, policy, post, scenario="null", seed=0)
    rows = evidence.metrics(conn, post, mode="sandbox")
    assert len(rows) == len(policy.get("simulation", "ages_s")) and all(r["label"] == "SIMULATED" for r in rows)
    assert evidence.metrics(conn, post, mode="real") == []


def test_stored_outcomes_hide_the_scenarios_truth(conn, policy):
    # The analyst must find breakouts and effects from the numbers, as with real data.
    cid = creator(conn)
    posts = [make_post(conn, cid) for _ in range(30)]          # breakout rate 0.1: some of these break out
    for p in posts:
        simulation.simulate_post(conn, policy, p, scenario="breakout", seed=1)
    assert all(set(r["metrics"]) == KEYS for p in posts for r in evidence.metrics(conn, p, mode="sandbox"))
