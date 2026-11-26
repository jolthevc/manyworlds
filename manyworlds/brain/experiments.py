"""Experiments: the step from "this pattern showed up" to "this causes more reach".

The analyst's findings are observational: a pattern in what the creator happened to post. A tentative one
(it cleared the corrected interval) becomes an experiment with two arms: A follows the finding (an
attribute value, a high score on a rubric dimension, or a format), B is the control (anything else). The
showrunner gets randomised slots each batch, a day and an arm, and writes a natural idea for each, so arms
are spread across days and the plan's other content instead of being chosen where they'd look good.

The outcome is reach as log-lift over the creator's baseline (as in manyworlds.brain.analysis), read once
each arm has its sample: a fixed horizon, so looking every batch can't inflate false positives. The sample
is sized when the experiment opens, for `power` to detect `mde` given the creator's outcome spread. A clear difference becomes a `proven` finding (evidence `experimental`, or `simulated` for
sandbox creators, through manyworlds.brain.evidence); no difference is recorded on the experiment, which
is a result too. One experiment per creator and effect, ever.

Cross-portfolio (manyworlds.brain.hypotheses): a proven result opens a speculative archetype-level hypothesis for
comparable creators, offered to them here as an experiment (after their own tentative findings), never applied as a
rule; their results replicate or contradict it.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from statistics import NormalDist

from psycopg.types.json import Jsonb

from ..policy import Policy
from . import analysis, evidence, hypotheses


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "experiments", default={}) or {}


def arms_for(effect_key: str) -> list[dict] | None:
    """Arm A follows the effect, arm B is the control. None for effects an idea can't be told to follow."""
    kind, _, key = effect_key.partition(":")
    if kind == "attribute" and "=" in key:
        k, v = key.split("=", 1)
        return [{"name": "A", "attribute": {k: v}, "instruction": f"{k}: {v} (make the idea genuinely that)"},
                {"name": "B", "attribute": {}, "instruction": f"{k}: not {v} (any other {k}; otherwise a normal idea)"}]
    if kind == "dimension" and key:
        return [{"name": "A", "attribute": {}, "dimension": key,
                 "instruction": f"an idea where {key} is the heart of it (it should score high on {key})"},
                {"name": "B", "attribute": {}, "instruction": f"a good ordinary idea where {key} is not the point"}]
    if kind == "format" and key:
        return [{"name": "A", "attribute": {}, "format": key, "instruction": f"format {key}"},
                {"name": "B", "attribute": {}, "instruction": f"any format except {key}"}]
    return None


def sample_size(policy: Policy, sd: float) -> int:
    """Posts per arm for `power` to detect a difference of `mde` at two-sided `alpha`, given outcome spread sd."""
    cfg = _cfg(policy)
    z = NormalDist().inv_cdf(1 - float(cfg.get("alpha", 0.05)) / 2) + NormalDist().inv_cdf(float(cfg.get("power", 0.8)))
    n = math.ceil(2 * z * z * sd * sd / float(cfg.get("mde", 0.4)) ** 2)
    return max(int(cfg.get("min_sample", 10)), min(int(cfg.get("max_sample", 60)), n))


def _outcome_sd(conn, policy: Policy, creator_id, mode) -> float:
    lifts = [r["lift"] for r in analysis.normalise(analysis.dataset(conn, creator_id, mode=mode, policy=policy), policy)
             if r.get("lift") is not None]
    if len(lifts) >= int(policy.get("brain", "analysis", "min_n_any", default=8)):
        return statistics.stdev(lifts)
    return float(_cfg(policy).get("default_sd", 0.6))


def open_for(conn, creator_id, *, domain: str = "content") -> list[dict]:
    """Open experiments in the creator's current evidence mode (a creator who went live leaves sandbox ones), of one
    domain: content (post slots for the showrunner) or music (song slots per release batch, effect keys music:...)."""
    music_key = "(effect_key LIKE 'music:%%')"
    cond = music_key if domain == "music" else f"NOT coalesce({music_key}, false)"
    return conn.execute(f"SELECT * FROM experiments WHERE creator_id = %s AND status = 'open' AND mode = %s AND {cond} "
                        "ORDER BY created_at", (creator_id, evidence.mode_for(conn, creator_id))).fetchall()


def concluded_for(conn, creator_id) -> list[dict]:
    return conn.execute("SELECT * FROM experiments WHERE creator_id = %s AND status = 'concluded' "
                        "ORDER BY concluded_at", (creator_id,)).fetchall()


def open_from_findings(conn, policy: Policy, creator_id) -> list[str]:
    """Open experiments for the creator's tentative findings, strongest first (effect x evidence weight:
    manyworlds.brain.hypotheses.weight), up to the cap; with room left, for the archetype-level hypotheses comparable
    creators proved (brain.hypotheses.propagation.offer_as_experiment): a suggestion she tests, never a rule."""
    cfg = _cfg(policy)
    room = int(cfg.get("max_open_per_creator", 1)) - len(open_for(conn, creator_id))
    mode = evidence.mode_for(conn, creator_id)
    found = evidence.findings(conn, mode=mode, creator_id=creator_id, grades=("tentative",), domain="content")
    found.sort(key=lambda f: -abs(float(f["effect"] or 0)) * hypotheses.weight(policy, f))
    if (policy.get("brain", "hypotheses", "propagation", default={}) or {}).get("offer_as_experiment", True):
        found += hypotheses.hypotheses_for(conn, policy, creator_id)
    tested = {r["effect_key"] for r in conn.execute(
        "SELECT effect_key FROM experiments WHERE creator_id = %s AND mode = %s AND effect_key IS NOT NULL",
        (creator_id, mode)).fetchall()}
    opened, need = [], None
    for f in found:
        if room <= 0:
            break
        keys = next((s.get("effects") for s in f["sources"] or [] if isinstance(s, dict) and s.get("effects")), None)
        key = keys[0] if keys else None
        arms = arms_for(key) if key else None
        if not arms or key in tested:
            continue
        need = need or sample_size(policy, _outcome_sd(conn, policy, creator_id, mode))
        row = conn.execute(   # a concurrent analyst may have opened it first: then this one opens nothing
            "INSERT INTO experiments (creator_id, hypothesis, arms, metric, min_sample, status, effect_key, source_finding, mode) "
            "VALUES (%s, %s, %s, 'reach_lift', %s, 'open', %s, %s, %s) "
            "ON CONFLICT (creator_id, effect_key, mode) WHERE effect_key IS NOT NULL DO NOTHING RETURNING id",
            (creator_id, f["statement"], Jsonb(arms), need, key, f["id"], mode)).fetchone()
        if row:
            opened.append(str(row["id"]))
        tested.add(key)
        room -= 1
    return opened


def assign(conn, policy: Policy, creator_id, *, start, days: int) -> list[dict]:
    """This batch's experiment slots: for each open experiment, `slots_per_batch` slots split evenly between
    the arms, each on a random day. Seeded by experiment and batch start, so a rerun gets the same slots."""
    per = int(_cfg(policy).get("slots_per_batch", 4))
    # One platform for every slot, so a format or attribute arm can't drift onto another platform's audience.
    platform = (policy.get("scope", "platforms", default=None) or ["instagram"])[0]
    slots = []
    for exp in open_for(conn, creator_id):
        rng = random.Random(int(hashlib.sha256(f"{exp['id']}:{start}".encode()).hexdigest()[:12], 16))
        arms = (["A", "B"] * (per // 2 + 1))[:per]
        rng.shuffle(arms)
        by_name = {a["name"]: a for a in exp["arms"]}
        for i, arm in enumerate(arms):
            slots.append({"slot": f"{exp['id']}:{i}", "experiment_id": str(exp["id"]), "arm": arm,
                          "day": rng.randrange(days), "platform": platform,
                          "instruction": f"{by_name[arm]['instruction']}, on {platform}"})
    return slots


def compare(rows: list[dict], policy: Policy, *, seed: int = 0) -> dict:
    """Difference in mean outcome `y` between arm A and arm B, with a normal interval at `alpha`.
    (`seed` is accepted for symmetry with the analysis code; the interval is analytic.)"""
    a = [r["y"] for r in rows if r["arm"] == "A"]
    b = [r["y"] for r in rows if r["arm"] == "B"]
    alpha = float(_cfg(policy).get("alpha", 0.05))
    if len(a) < 2 or len(b) < 2:
        return {"n_a": len(a), "n_b": len(b), "effect": None, "significant": False}
    est = statistics.fmean(a) - statistics.fmean(b)
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    low, high = est - z * se, est + z * se
    mde = float(_cfg(policy).get("mde", 0.4))
    power = NormalDist().cdf(mde / se - z) if se > 0 else 1.0   # chance this test would catch a true `mde`
    return {"n_a": len(a), "n_b": len(b), "effect": round(est, 4), "se": round(se, 4), "low": round(low, 4),
            "high": round(high, 4), "significant": low > 0 or high < 0, "power": round(power, 3)}


def rows_for(conn, policy: Policy, exp: dict, mode: evidence.Mode) -> list[dict]:
    """The experiment's posts with outcomes, by the arm they were assigned. Ideas that never became a post with
    an outcome (dropped, failed renders) are missing; the result reports slots assigned per arm beside them."""
    data = analysis.normalise(analysis.dataset(conn, exp["creator_id"], mode=mode, policy=policy), policy)
    return [{"post_id": r["post_id"], "arm": r["arm"], "lift": r["lift"]} for r in data
            if r.get("experiment_id") == str(exp["id"]) and r.get("arm") and r.get("lift") is not None]


def conclude(conn, policy: Policy, exp: dict, rows: list[dict]) -> dict | None:
    """Read the result once each arm has its minimum sample; None while it hasn't."""
    need = int(exp["min_sample"] or _cfg(policy).get("min_sample", 10))
    if min(sum(r["arm"] == "A" for r in rows), sum(r["arm"] == "B" for r in rows)) < need:
        return None
    cap = math.log(float(policy.get("brain", "analysis", "breakout_ratio", default=3.0)))
    res = compare([{"arm": r["arm"], "y": max(-cap, min(cap, r["lift"]))} for r in rows], policy)
    status = ("confirmed" if res["effect"] > 0 else "reversed") if res["significant"] else "no_difference"
    finding_id = None
    with conn.transaction():
        # The finding and the closing land together, once: a rerun or a concurrent analyst finds it closed.
        if not conn.execute("SELECT 1 FROM experiments WHERE id = %s AND status = 'open' FOR UPDATE",
                            (exp["id"],)).fetchone():
            return None
        if status != "no_difference":
            mode = evidence.mode_for(conn, exp["creator_id"])
            a = next(x for x in exp["arms"] if x["name"] == "A")
            verb = "beat" if status == "confirmed" else "did worse than"
            statement = (f"Experiment: posts with {a['instruction']} {verb} the control by {abs(res['effect']):.2f} "
                         f"log-lift in reach ({res['n_a']} vs {res['n_b']} posts). Tested: {exp['hypothesis']}")
            source = [{"creator_id": str(exp["creator_id"])}, {"experiment_id": str(exp["id"])},
                      {"posts": [r["post_id"] for r in rows]}] + \
                ([{"table": "simulated_metrics"}] if mode == "sandbox" else [])
            finding_id = evidence.record_finding(
                conn, scope="creator", creator_id=exp["creator_id"], statement=statement,
                evidence="simulated" if mode == "sandbox" else "experimental", grade="proven", effect=res["effect"],
                interval=(res["low"], res["high"]), n=res["n_a"] + res["n_b"], sources=source)
        # Slots assigned vs posts with outcomes, per arm: attrition (dropped ideas, failed renders) shows here.
        assigned = {r["arm"]: r["n"] for r in conn.execute(
            "SELECT arm, count(*) n FROM ideas WHERE experiment_id = %s GROUP BY arm", (exp["id"],)).fetchall()}
        result = {"status": status, **res, "assigned": assigned, "finding_id": str(finding_id) if finding_id else None}
        # Across the portfolio: replicate or contradict the hypothesis it tested, or open one for comparable creators.
        spread = hypotheses.on_experiment(conn, policy, exp, result)
        if spread:
            result["hypothesis"] = spread
        conn.execute("UPDATE experiments SET status = 'concluded', result = %s, concluded_at = now(), updated_at = now() "
                     "WHERE id = %s", (Jsonb(result), exp["id"]))
    return result


def review(conn, policy: Policy, creator_id) -> dict:
    """The analyst's experiment step: read results that are due, then open new experiments."""
    mode = evidence.mode_for(conn, creator_id)
    # A creator who left the sandbox doesn't continue its simulated experiments.
    conn.execute("UPDATE experiments SET status = 'abandoned', updated_at = now() "
                 "WHERE creator_id = %s AND status = 'open' AND mode <> %s", (creator_id, mode))
    concluded = []
    for exp in open_for(conn, creator_id):
        out = conclude(conn, policy, exp, rows_for(conn, policy, exp, mode))
        if out:
            concluded.append({"experiment_id": str(exp["id"]), **out})
    return {"concluded": concluded, "opened": open_from_findings(conn, policy, creator_id)}


# --- music experiments (spec 2026-11-23-music-department.md, section 2) ---------------------------------------------
# A tentative music finding ("songs with an earlier first chorus did better") becomes a two-arm test on songs: arm A
# follows the finding (the feature high, or low when the finding's effect is negative), arm B the other side. Slots
# are assigned per release batch (music_assign: one arm per song, balanced and shuffled, seeded by the experiment
# and the batch), the arm's instruction goes into the song's brief, and the songs carry the arm (songs.experiment_id,
# songs.arm). The outcome is each song's outcome (brain.music_analysis: posts' log-lift, net of format), read once
# both arms have music.experiments.min_songs_per_arm songs with outcomes.

def _music_cfg(policy: Policy) -> dict:
    return policy.get("music", "experiments", default={}) or {}


def music_arms(policy: Policy, effect_key: str, *, positive: bool = True) -> list[dict] | None:
    """Arms for music:feature:<name> (A follows the finding's direction); None for anything else."""
    parts = effect_key.split(":")
    if len(parts) != 3 or parts[0] != "music" or parts[1] != "feature" or not parts[2]:
        return None
    name = parts[2]
    words = (_music_cfg(policy).get("instructions") or {}).get(name) or {}
    hi = words.get("high") or f"{name} clearly high for this artist"
    lo = words.get("low") or f"{name} clearly low for this artist"
    a, b = (hi, lo) if positive else (lo, hi)
    return [{"name": "A", "music_feature": name, "side": "high" if positive else "low", "instruction": a},
            {"name": "B", "music_feature": name, "side": "low" if positive else "high", "instruction": b}]


def music_open_from_findings(conn, policy: Policy, creator_id) -> list[str]:
    """Open music experiments for the artist's tentative music findings about one feature, strongest first, up to
    music.experiments.max_open_per_artist. One experiment per artist, effect and mode, ever."""
    from . import music_analysis
    c = _music_cfg(policy)
    room = int(c.get("max_open_per_artist", 1)) - len(open_for(conn, creator_id, domain="music"))
    mode = evidence.mode_for(conn, creator_id)
    found = [f for f in music_analysis.findings(conn, mode=mode, creator_id=creator_id) if f["grade"] == "tentative"]
    found.sort(key=lambda f: -abs(float(f["effect"] or 0)))
    opened = []
    for f in found:
        if room <= 0:
            break
        effect = next((s.get("effect") for s in f["sources"] or [] if isinstance(s, dict) and s.get("effect")), "")
        if not str(effect).startswith("feature:"):
            continue
        key = f"music:{effect}"
        arms = music_arms(policy, key, positive=float(f["effect"] or 0) > 0)
        row = conn.execute(
            "INSERT INTO experiments (creator_id, hypothesis, arms, metric, min_sample, status, effect_key, "
            "source_finding, mode) VALUES (%s, %s, %s, 'song_reach_lift', %s, 'open', %s, %s, %s) "
            "ON CONFLICT (creator_id, effect_key, mode) WHERE effect_key IS NOT NULL DO NOTHING RETURNING id",
            (creator_id, f["statement"], Jsonb(arms), int(c.get("min_songs_per_arm", 6)), key, f["id"],
             mode)).fetchone()
        if row:
            opened.append(str(row["id"]))
            room -= 1
    return opened


def music_assign(conn, policy: Policy, creator_id, *, batch_key: str, n_songs: int) -> list[dict | None]:
    """The release batch's arms, one entry per song (None: no experiment for that song): the first open music
    experiment's arms, balanced and shuffled (seeded by experiment and batch, so a rerun gets the same)."""
    exps = open_for(conn, creator_id, domain="music")
    if not exps or n_songs <= 0:
        return [None] * max(n_songs, 0)
    exp = exps[0]
    rng = random.Random(int(hashlib.sha256(f"{exp['id']}:{batch_key}".encode()).hexdigest()[:12], 16))
    arms = (["A", "B"] * (n_songs // 2 + 1))[:n_songs]
    rng.shuffle(arms)
    by = {a["name"]: a for a in exp["arms"]}
    return [{"experiment_id": str(exp["id"]), "arm": a, "instruction": by[a]["instruction"]} for a in arms]


def music_conclude(conn, policy: Policy, exp: dict) -> dict | None:
    """Read a music experiment once both arms have their songs with outcomes; a clear difference is a proven music
    finding (experimental, or simulated for sandbox artists). None while the sample isn't there."""
    from . import music_analysis
    mode = evidence.mode_for(conn, exp["creator_id"])
    songs_in = conn.execute("SELECT id, arm FROM songs WHERE experiment_id = %s AND arm IS NOT NULL",
                            (exp["id"],)).fetchall()
    outs = music_analysis.outcomes(conn, policy, [str(exp["creator_id"])], mode=mode)
    rows = [{"arm": r["arm"], "y": outs[str(r["id"])]["y"], "song_id": str(r["id"])}
            for r in songs_in if str(r["id"]) in outs]
    need = int(exp["min_sample"] or _music_cfg(policy).get("min_songs_per_arm", 6))
    if min(sum(r["arm"] == "A" for r in rows), sum(r["arm"] == "B" for r in rows)) < need:
        return None
    res = compare(rows, policy)
    status = ("confirmed" if res["effect"] > 0 else "reversed") if res["significant"] else "no_difference"
    finding_id = None
    with conn.transaction():
        if not conn.execute("SELECT 1 FROM experiments WHERE id = %s AND status = 'open' FOR UPDATE",
                            (exp["id"],)).fetchone():
            return None
        if status != "no_difference":
            a = next(x for x in exp["arms"] if x["name"] == "A")
            verb = "beat" if status == "confirmed" else "did worse than"
            statement = (f"Music experiment: songs with {a['instruction']} {verb} the control by "
                         f"{abs(res['effect']):.2f} log-lift ({res['n_a']} vs {res['n_b']} songs). "
                         f"Tested: {exp['hypothesis']}")
            source = [{"creator_id": str(exp["creator_id"])}, {"experiment_id": str(exp["id"])},
                      {"songs": [r["song_id"] for r in rows]}] + \
                ([{"table": "simulated_metrics"}] if mode == "sandbox" else [])
            finding_id = evidence.record_finding(
                conn, scope="creator", creator_id=exp["creator_id"], statement=statement,
                evidence="simulated" if mode == "sandbox" else "experimental", grade="proven", effect=res["effect"],
                interval=(res["low"], res["high"]), n=res["n_a"] + res["n_b"], sources=source, domain="music")
        result = {"status": status, **res, "finding_id": str(finding_id) if finding_id else None}
        conn.execute("UPDATE experiments SET status = 'concluded', result = %s, concluded_at = now(), updated_at = now() "
                     "WHERE id = %s", (Jsonb(result), exp["id"]))
    return result


def music_review(conn, policy: Policy, creator_id) -> dict:
    """Read due music experiments, then open new ones from the artist's tentative music findings."""
    concluded = []
    for exp in open_for(conn, creator_id, domain="music"):
        out = music_conclude(conn, policy, exp)
        if out:
            concluded.append({"experiment_id": str(exp["id"]), **out})
    return {"concluded": concluded, "opened": music_open_from_findings(conn, policy, creator_id)}
