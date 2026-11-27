"""Recipes (manyworlds.brain.recipes; docs/research/2026-11-26-recipes.md) and the research questions the Brain asks
(manyworlds.brain.research_questions). Synthetic rows where the truth is known: a combination that works only
together is found and confirmed on holdout creators; one creator's quirk is not a finding (the creator is the unit);
a pure-noise screen of many pairs confirms nothing (BH over the declared family, then holdout); posts without the
metric are excluded, never zero; reads under another vocabulary are excluded for that field; an effect that differs
by context is detected; a question filed with a spec is answered and stored; the translator drops unknown names.
Stub LLMs, no spend; a small bootstrap keeps it fast."""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from types import SimpleNamespace as NS

from psycopg.types.json import Jsonb

from brain_helpers import StubLLM, make_ctx
from manyworlds import policy as policy_mod, queue
from manyworlds.brain import recipes, research_questions as rq
from manyworlds.brain.agents import intelligence
from manyworlds.research import annotation as ann

A, B = "garments=blazer", "lighting_kind=golden_hour"
POSTED, FETCHED = datetime(2026, 10, 5, tzinfo=timezone.utc), datetime(2026, 11, 20, tzinfo=timezone.utc)
CURRENT = object()


def _policy(policy, intelligence_over: dict | None = None, **recipes_over):
    """The shipped policy with test floors (5 creators) and a small bootstrap."""
    intel = policy.body["intelligence"]
    rec = {**intel["recipes"], "min_creators": 5, "min_posts_per_arm": 3, "min_creators_both_arms": 5,
           "bootstrap": {"iterations": 200, "seed": 7, "ci": 0.95}, **recipes_over}
    return policy_mod.from_dict({**policy.body, "intelligence": {**intel, "recipes": rec, **(intelligence_over or {})}})


def _creators(pol, n_each: int) -> tuple[list[str], list[str]]:
    """n_each discovery and n_each holdout creator ids (uuid-shaped), by the policy's own split."""
    disc, hold, i = [], [], 0
    while len(disc) < n_each or len(hold) < n_each:
        cid = f"00000000-0000-4000-8000-{i:012d}"
        bucket = hold if recipes.side(cid, pol) == "holdout" else disc
        if len(bucket) < n_each:
            bucket.append(cid)
        i += 1
    return disc, hold


def _raw(pol, cid, coded, y, *, medium="photo", versions=CURRENT, metrics=None, archetype=None):
    observed = {"coded": coded}
    if versions is CURRENT:
        observed["vocab_versions"] = ann.vocab_versions(pol)
    elif versions is not None:
        observed["vocab_versions"] = versions
    return {"post_id": f"p{random.random()}", "creator_id": cid, "platform": "instagram", "media_type": medium,
            "posted_at": POSTED, "fetched_at": FETCHED, "schema": ann.ANNOTATION_SCHEMA, "observed": observed,
            "over_own_median": None if y is None else math.exp(y),
            "metrics": {"likes": 10} if metrics is None else metrics, "archetype": archetype, "followers": 50000}


def _combo_data(pol, creators, *, posts=48, effect=0.6, noise=0.3, seed=1):
    """Every creator: a blazer on half the posts, golden hour on half, crossed evenly; engagement lifts only when both
    are there (no effect of either alone)."""
    rng = random.Random(seed)
    out = []
    for cid in creators:
        for i in range(posts):
            a, b = i % 2 == 0, (i // 2) % 2 == 0
            y = (effect if a and b else 0.0) + rng.gauss(0, noise)
            out.append(_raw(pol, cid, {"garments": ["blazer" if a else "jeans"],
                                       "lighting_kind": "golden_hour" if b else "studio"}, y))
    return out


# --- the analysis ---------------------------------------------------------------------------------------------------
def test_an_interaction_that_exists_only_in_combination_is_found_and_confirmed_on_holdout(policy):
    pol = _policy(policy)
    disc, hold = _creators(pol, 8)
    rows = recipes.prepare(_combo_data(pol, disc + hold), pol)
    out = recipes.run(rows, pol, [{"kind": "interaction", "attrs": [A, B]}, {"kind": "main", "attrs": [A]},
                                  {"kind": "recipe", "attrs": [A, B]}])
    assert out["family_size"] == 3 and out["creators"] == {"discovery": 8, "holdout": 8}
    inter, main_a, combo = out["results"]
    assert inter["stage"] == "confirmed" and inter["reason"] == "held on the holdout creators"
    assert 1.4 < inter["effect"] < 2.4 and inter["ci"][0] > 1.0        # exp(0.6) = 1.82
    assert inter["discovery"]["n_creators_both_arms"] == 8 and inter["holdout"]["n_creators_both_arms"] == 8
    assert inter["creators_positive"] == 8 and inter["q"] <= pol.get("intelligence", "recipes", "fdr_q")
    assert inter["discovery"]["p"] == round(2 / 2 ** 8, 5)           # exact sign flips: all 8 the same way
    assert combo["stage"] == "confirmed" and combo["effect"] > main_a["effect"]   # alone, a blazer is half the story
    assert set(inter["balance"]["month"]) == {"both", "a_only", "b_only", "neither"}
    assert inter["coverage"]["share_with_metric"] == 1.0 and set(inter["vocab_versions"]) == {"garments",
                                                                                               "lighting_kind"}
    assert recipes.card_tests(inter)[0]["q"] == inter["q"] and recipes.card_tests(main_a | {"stage": "discovery"}) == []


def test_a_single_creator_artefact_is_not_a_finding(policy):
    """One creator's sunglasses posts do 7x; nobody else's differ. Pooled over posts it looks real; over creators it is
    one creator out of sixteen."""
    pol = _policy(policy)
    disc, hold = _creators(pol, 8)
    odd = disc[0]
    raw = []
    for n, cid in enumerate(disc + hold):
        rng = random.Random(100 + n)
        for _ in range(24):
            noise = rng.gauss(0, 0.3)
            for on in (True, False):         # pairs share their noise: with and without differ only by the effect
                y = noise + (2.0 if cid == odd and on else 0.0)
                raw.append(_raw(pol, cid, {"accessories": ["sunglasses"] if on else []}, y))
    rows = recipes.prepare(raw, pol)
    with_, without = ([r.y for r in rows if (attr in r.attrs) == on] for attr, on in
                      (("accessories=sunglasses", True), ("accessories=sunglasses", False)))
    se = math.sqrt(_var(with_) / len(with_) + _var(without) / len(without))
    assert (sum(with_) / len(with_) - sum(without) / len(without)) / se > 3   # a naive post-level test would bite
    out = recipes.run(rows, pol, [{"kind": "main", "attrs": ["accessories=sunglasses"]}])
    (res,) = out["results"]
    assert res["stage"] == "failed" and res["discovery"]["p"] == 1.0
    assert res["discovery"]["creators_positive"] == 1 and res["discovery"]["n_creators_both_arms"] == 8


def _var(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def test_a_pure_noise_screen_of_many_pairs_confirms_nothing(policy):
    pol = _policy(policy, screen={"prevalence_floor": 0.15, "max_attributes": 30, "max_pairs": 30,
                                  "main_effects": True})
    disc, hold = _creators(pol, 10)
    fields = {"lighting_kind": ("golden_hour", "studio"), "candid_level": ("candid", "posed"),
              "palette": ("neutral", "bright_bold"), "makeup": ("natural", "full_glam"),
              "composition": ("centred", "off_centre")}
    rng = random.Random(3)
    raw = []
    for cid in disc + hold:
        for _ in range(40):
            coded = {f: rng.choice(vals) for f, vals in fields.items()}
            coded["garments"] = [rng.choice(("blazer", "jeans"))]
            raw.append(_raw(pol, cid, coded, rng.gauss(0, 0.3)))
    rows = recipes.prepare(raw, pol)
    tests = recipes.screen(rows, pol)
    assert len(tests) == 12 + 30                                       # 12 attributes, pairs capped at max_pairs
    assert all(recipes.field_of(t["attrs"][0]) != recipes.field_of(t["attrs"][1])
               for t in tests if t["kind"] == "interaction")
    out = recipes.run(rows, pol, tests)
    assert out["family_size"] == 42 and out["confirmed"] == 0
    assert all(r["stage"] in ("failed", "insufficient", "discovery") for r in out["results"])


def test_posts_without_the_metric_are_excluded_and_counted_never_zero(policy):
    pol = _policy(policy)
    disc, hold = _creators(pol, 4)
    rng = random.Random(5)
    base = [_raw(pol, cid, {"garments": ["blazer" if i % 2 else "jeans"]}, rng.gauss(0, 0.3))
            for cid in disc + hold for i in range(24)]
    missing = [_raw(pol, cid, {"garments": ["blazer"]}, None, metrics={}) for cid in disc + hold for _ in range(5)]
    clean = recipes.main_effect(recipes.prepare(base, pol), A, pol)
    full = recipes.main_effect(recipes.prepare(base + missing, pol), A, pol)
    for k in ("effect", "ci", "p", "n_posts", "n_creators_both_arms"):
        assert full[k] == clean[k], k
    assert full["coverage"]["missing"] == {"no_public_counts": 40}
    assert full["coverage"]["with_metric"] == len(base) and full["coverage"]["share_with_metric"] < 1
    # had they been read as zero (clipped at min_ratio), the blazer arm would have sunk
    zeroed = [{**r, "over_own_median": 0.0} for r in missing]
    sunk = recipes.main_effect(recipes.prepare(base + zeroed, pol), A, pol)
    assert sunk["effect"] < clean["effect"] and sunk["coverage"]["clipped"] == 40


def test_reads_under_another_vocabulary_are_excluded_for_that_field(policy):
    pol = _policy(policy)
    disc, hold = _creators(pol, 4)
    cur = ann.vocab_versions(pol)
    rng = random.Random(6)
    good = [_raw(pol, cid, {"garments": ["blazer" if i % 2 else "jeans"], "lighting_kind": "studio"},
                 rng.gauss(0, 0.3)) for cid in disc + hold for i in range(24)]
    stale = [_raw(pol, cid, {"garments": ["blazer"], "lighting_kind": "studio"}, 2.0,
                  versions={**cur, "garments": "deadbeef"}) for cid in disc + hold for _ in range(4)]
    v1 = [_raw(pol, cid, {"garments": ["blazer"]}, 2.0, versions=None) for cid in disc + hold for _ in range(3)]
    rows = recipes.prepare(good + stale + v1, pol)
    s, old = rows[len(good)], rows[-1]
    assert "garments" not in s.known and "lighting_kind" in s.known and s.excluded["garments"] == "old_vocab"
    assert not old.known and old.excluded["garments"] == "unversioned"
    assert "sound" not in s.excluded and "sound" not in s.known       # a video field is not asked of a photo
    res = recipes.main_effect(rows, A, pol)
    clean = recipes.main_effect(recipes.prepare(good, pol), A, pol)
    assert res["effect"] == clean["effect"] and res["n_posts"] == clean["n_posts"]
    assert res["coverage"]["fields_excluded"]["garments"] == {"old_vocab": 32, "unversioned": 24}
    # declared compatible (values only added): the older reads count again
    compat = policy_mod.from_dict({**pol.body, "intelligence": {**pol.body["intelligence"], "annotation": {
        **pol.body["intelligence"]["annotation"], "vocab_compatible": {"garments": ["deadbeef"]}}}})
    again = recipes.main_effect(recipes.prepare(good + stale + v1, compat), A, compat)
    assert again["n_posts"] == clean["n_posts"] + 32 and again["effect"] > clean["effect"]


def test_vocab_versions_change_only_for_the_field_whose_list_changed(policy):
    before = ann.vocab_versions(policy)
    voc = policy.body["intelligence"]["annotation"]["vocab"]
    changed = policy_mod.from_dict({**policy.body, "intelligence": {**policy.body["intelligence"], "annotation": {
        **policy.body["intelligence"]["annotation"], "vocab": {**voc, "garments": [*voc["garments"], "cape"]}}}})
    after = ann.vocab_versions(changed)
    assert set(before) == set(ann.CODED_FIELDS)
    assert [f for f in before if before[f] != after[f]] == ["garments"]
    assert len(before["garments"]) == ann.VERSION_CHARS


def test_conditional_heterogeneity_is_detected(policy):
    pol = _policy(policy)
    disc, hold = _creators(pol, 8)
    everyone = disc + hold

    def data(effect_b, seed):
        rng = random.Random(seed)
        out = []
        for n, cid in enumerate(everyone):
            arch = "a" if n % 2 == 0 else "b"
            lift = 0.6 if arch == "a" else effect_b
            for i in range(24):
                on = i % 2 == 0
                out.append(_raw(pol, cid, {"garments": ["blazer" if on else "jeans"]},
                                (lift if on else 0.0) + rng.gauss(0, 0.3), archetype=arch))
        return recipes.prepare(out, pol)

    res = recipes.conditional(data(0.0, 8), A, "archetype", pol)
    assert res["sufficient"] and res["heterogeneity"]["df"] == 1 and res["heterogeneity"]["p"] < 0.01
    assert res["by_context"]["a"]["effect"] > 1.5 and abs(res["by_context"]["b"]["log_effect"]) < 0.2
    same = recipes.conditional(data(0.6, 9), A, "archetype", pol)
    assert same["heterogeneity"]["p"] > 0.001
    assert abs(recipes.chi2_sf(3.841, 1) - 0.05) < 1e-3 and abs(recipes.chi2_sf(9.488, 4) - 0.05) < 1e-3
    assert abs(recipes.chi2_sf(0.5, 3) - 0.9189) < 1e-3                # the series branch


def test_bh_and_the_sign_test():
    assert recipes.bh([0.01, 0.04, 0.03, 0.5]) == [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5]
    assert recipes.bh([0.01], m=10) == [0.1]                          # the family is the declared one
    assert recipes.sign_test_p([1, 2, 3, 4, 5]) == 2 / 32 and recipes.sign_test_p([0.0, 0.0]) == 1.0


# --- research questions ---------------------------------------------------------------------------------------------
def _store_rows(conn, pol, raw):
    """Synthetic rows as reference creators, posts and annotations (what evidence.annotated_posts reads)."""
    for cid in sorted({r["creator_id"] for r in raw}):
        conn.execute("INSERT INTO reference_creators (id, platform, handle, followers) VALUES (%s, 'instagram', %s, "
                     "50000)", (cid, f"h{cid[-6:]}"))
    for n, r in enumerate(raw):
        pid = conn.execute(
            "INSERT INTO reference_posts (creator_id, platform, external_id, media_type, posted_at, fetched_at, metrics)"
            " VALUES (%s, 'instagram', %s, %s, %s, %s, %s) RETURNING id",
            (r["creator_id"], f"x{n}", r["media_type"], r["posted_at"], r["fetched_at"],
             Jsonb({**r["metrics"], "over_own_median": r["over_own_median"]}))).fetchone()["id"]
        conn.execute("INSERT INTO reference_annotations (post_id, schema, observed) VALUES (%s, %s, %s)",
                     (pid, ann.ANNOTATION_SCHEMA, Jsonb(r["observed"])))


def test_a_question_filed_with_a_spec_is_answered_and_its_result_stored(conn, policy):
    conn.execute("DELETE FROM research_questions")
    pol = _policy(policy)
    disc, hold = _creators(pol, 6)
    _store_rows(conn, pol, _combo_data(pol, disc + hold, posts=32, noise=0.2, seed=11))
    qid = rq.ask(conn, pol, "talent_manager:maren", "Does a blazer in golden hour beat either alone?",
                 spec={"tests": [{"kind": "interaction", "attrs": [A, B]}], "uses": ["showrunner"]})
    row = conn.execute("SELECT * FROM research_questions WHERE id = %s", (qid,)).fetchone()
    assert row["status"] == "queued" and row["spec"]["tests"] == [{"kind": "interaction", "attrs": [A, B]}]
    llm = StubLLM(conn=conn)
    out = rq.run_due(conn, pol, llm)
    assert out["answered"] == [{"id": qid, "status": "answered", "confirmed": 1}] and llm.calls == []
    row = conn.execute("SELECT * FROM research_questions WHERE id = %s", (qid,)).fetchone()
    (res,) = row["result"]["data"]["results"]
    assert res["stage"] == "confirmed" and res["effect"] > 1.3 and row["result"]["data"]["family_size"] == 1
    assert row["answered_at"] is not None and row["policy_version"] == pol.version
    assert "confirmed on holdout creators" in rq.brief(conn, pol, "talent_manager")
    assert "engagement x" in rq.brief(conn, pol, "showrunner")       # named in the spec's uses
    assert rq.brief(conn, pol, "casting") == ""


class TranslateLLM(StubLLM):
    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema, effort=effort)
        return rq.SpecDraft(tests=[
            rq.TestDraft(kind="interaction", attributes=["garments=blazer", "garments=toga", "moon=yes"]),
            rq.TestDraft(kind="conditional", attributes=["lighting_kind=golden_hour"], context="zodiac"),
            rq.TestDraft(kind="main", attributes=["Lighting_Kind = Golden Hour"]),
            rq.TestDraft(kind="vibes", attributes=["garments=blazer"])], medium="hologram")


def test_the_translator_drops_unknown_names_and_its_spec_is_stored_before_testing(conn, policy):
    conn.execute("DELETE FROM research_questions")
    qid = rq.ask(conn, policy, "showrunner", "Does golden hour help?")
    llm = TranslateLLM(conn=conn)
    rq.run_due(conn, policy, llm)
    row = conn.execute("SELECT * FROM research_questions WHERE id = %s", (qid,)).fetchone()
    assert row["spec"]["tests"] == [{"kind": "main", "attrs": ["lighting_kind=golden_hour"]}]
    assert row["spec"]["filters"] == {}
    dropped = set(row["spec"]["dropped"])
    assert {"attribute garments=toga", "attribute moon=yes", "context zodiac", "kind vibes",
            "medium hologram", "interaction test without enough known attributes"} <= dropped
    assert [c["role"] for c in llm.calls] == [policy.get("intelligence", "questions", "role")]
    assert llm.calls[0]["agent"] == rq.AGENT and "garments=blazer" in llm.calls[0]["content"]
    assert row["status"] == "unanswerable"                           # no data: said so, no tests over zero rows
    assert row["result"]["data"]["no_data"] and row["result"]["data"]["posts"] == 0


def test_a_question_needing_a_call_waits_when_the_day_is_spent(conn, policy):
    conn.execute("DELETE FROM research_questions")
    broke = policy_mod.from_dict({**policy.body, "intelligence": {**policy.body["intelligence"], "questions": {
        **policy.body["intelligence"]["questions"], "daily_usd": 0}}})
    qid = rq.ask(conn, broke, "ceo", "What works for travel carousels?")
    out = rq.run_due(conn, broke, StubLLM(conn=conn))
    assert out["held"] == [{"id": qid, "why": "questions daily budget reached"}]
    assert conn.execute("SELECT status FROM research_questions WHERE id = %s", (qid,)).fetchone()["status"] == "queued"


def test_the_intelligence_agent_answers_questions_in_its_questions_mode(conn, policy):
    conn.execute("DELETE FROM research_questions")
    qid = rq.ask(conn, policy, "ceo", "Does a blazer help?", spec={"tests": [{"kind": "main", "attrs": [A]}]})
    jid = queue.enqueue(conn, "brain.intelligence", {"mode": "questions"})
    out = intelligence.Intelligence().run(make_ctx(conn, policy, jid))
    assert out["questions"]["answered"][0]["id"] == qid
    assert conn.execute("SELECT status FROM research_questions WHERE id = %s", (qid,)).fetchone()["status"] == \
        "unanswerable"                                                   # no posts yet: no analysis over zero rows


def test_new_reads_store_the_vocabulary_versions(conn, policy):
    cid = conn.execute("INSERT INTO reference_creators (platform, handle) VALUES ('instagram', 'vv') "
                       "RETURNING id").fetchone()["id"]
    pid = conn.execute("INSERT INTO reference_posts (creator_id, platform, external_id, media_type) VALUES "
                       "(%s, 'instagram', 'vv1', 'photo') RETURNING id", (cid,)).fetchone()["id"]
    intelligence.Intelligence()._store_read(conn, NS(policy=policy), {"id": pid, "metrics": {}, "content_hash": None},
                                            ann.Annotation(), "stub")
    obs = conn.execute("SELECT observed FROM reference_annotations WHERE post_id = %s", (pid,)).fetchone()["observed"]
    assert obs["vocab_versions"] == ann.vocab_versions(policy)
    assert ann.current_fields(obs["vocab_versions"], ann.vocab_versions(policy), {}) == set(ann.CODED_FIELDS)
