"""The music department (spec 2026-11-23-music-department.md, sections 3 and 6): the roles, the pipeline with its
revision loop and takes, the portfolio's music view, music experiments, and the sandbox runner end to end with a
scripted stub LLM and stubbed GPU (no music.song job runs)."""

from __future__ import annotations

import math
import random
import re
from datetime import datetime, timedelta, timezone

import pytest
from psycopg.types.json import Jsonb

from manyworlds import music, music_pipeline, music_sandbox, queue, songs
from manyworlds import policy as policy_mod
from manyworlds.brain import evidence, experiments, lyrics, music_roles, portfolio, proposals
from manyworlds.brain.agents import (a_and_r, curator, head_of_music, melody, mix_master, music_critic,
                                     music_talent_manager, producer, songwriter, vocal_director)

from brain_helpers import StubLLM, creator, make_ctx

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
LYRICS = ("[Verse]\nThe kettle sings before I do\nSteam on the window, writing you\n"
          "[Chorus]\nWindow seat, the night is ours\nCounting the streetlights instead of stars\n"
          "[Chorus]\nWindow seat, the night is ours\nCounting the streetlights instead of stars")


class RoleLLM(StubLLM):
    """Answers by schema class: a dict, a list of dicts (one per call, the last repeating) or fn(content) -> dict.
    A schema without an answer gets its empty instance (every role output has defaults)."""

    def __init__(self, answers: dict, cost: float = 0.0, conn=None):
        super().__init__(cost=cost, conn=conn)
        self.answers, self.used = answers, {}

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        self.calls[-1]["schema"] = schema
        a = self.answers.get(schema)
        if a is None:
            return schema.model_construct()
        if isinstance(a, list):
            i = self.used.get(schema, 0)
            self.used[schema] = i + 1
            a = a[min(i, len(a) - 1)]
        if callable(a):
            a = a(content)
        return schema.model_validate(a)


def _ids(content: str, field: str) -> list[str]:
    return re.findall(rf'"{field}":\s*"({UUID})"', content)


DRAFT_REVISE = {"dimensions": [{"dimension": "structure", "finding": "too slow for the genre",
                                "evidence": ["tempo_bpm", "band.tempo_bpm"], "severity": "major"}],
                "fixes": [{"role": "producer", "problem": "tempo below the band", "change": "raise to 104 bpm",
                           "evidence": ["tempo_bpm"]}], "verdict": "revise", "summary": "tempo"}
SHIP = {"dimensions": [], "fixes": [], "verdict": "ship", "summary": "good to make"}
SONG_REVIEW = {"dimensions": [{"dimension": "mix", "finding": "quiet", "evidence": ["lufs_integrated"],
                               "severity": "minor"}],
               "fixes": [{"role": "mix_master", "problem": "quiet master", "change": "master louder",
                          "evidence": ["lufs_integrated"]}], "verdict": "revise",
               "prediction": {"metric": "song_reach_lift", "comparator": "above", "threshold": 0.0,
                              "probability": 0.5, "claim": "as usual"}, "summary": "mix"}


def answers():
    return {
        music_talent_manager.Dossier: {"sonic_identity": {"genre": "indie", "subgenre": "bedroom pop",
                                                          "voice": "soft, close, low alto",
                                                          "voice_tags": ["soft female vocal"],
                                                          "palette": ["nylon guitar", "tape hiss"]},
                                       "story": "songs written on night buses", "release_plan": ["three singles"]},
        music_talent_manager.Career: {"state": "starting", "next_songs": [
            {"concept": "a window seat on the last bus", "mood": "tender", "genre": "indie", "why": "her world"},
            {"concept": "the next Taylor Swift anthem", "genre": "pop"},            # dropped: names a real artist
            {"concept": "laundry at midnight", "mood": "wry", "genre": "indie"}],
            "predictions": [{"claim": "reach above 500", "metric": "median_reach", "comparator": "above",
                             "threshold": 500, "probability": 0.6, "horizon_days": 21}]},
        songwriter.Output: {"title_idea": "Window Seat", "concept": "the last bus home", "hook_line": "window seat",
                            "hook_at_s": 6, "structure": [{"name": "verse", "seconds": 8},
                                                          {"name": "chorus", "seconds": 8}]},
        producer.Output: [{"tags": ["indie pop", "nylon guitar", "in the style of a famous band"], "bpm": 84,
                           "key": "A minor", "time_signature": "4", "duration_s": 30},
                          {"tags": ["indie pop", "nylon guitar"], "bpm": 104, "key": "A minor",
                           "time_signature": "4", "duration_s": 30}],
        lyrics.SongAnswer: {"title": "Window Seat", "lyrics": LYRICS, "tags": ["indie pop"], "bpm": 100,
                            "key": "A minor", "time_signature": "4", "duration_s": 30},
        vocal_director.Output: {"vocal_tags": ["breathy", "close-mic"], "conversion_strength": 3},
        music_critic.Assessment: [DRAFT_REVISE, SHIP, SONG_REVIEW, DRAFT_REVISE, SHIP, SONG_REVIEW],
        melody.Output: {"take_index": 1, "why": "earlier hook"},
        mix_master.Output: {"target_lufs": -30, "true_peak_ceiling_dbtp": 0.5, "why": "louder"},
        curator.Output: lambda c: {"kind": "single", "title": "Window Seat (single)",
                                   "tracks": [{"song_id": _ids(c, "song_id")[0], "why": "the hook"}]},
        a_and_r.Output: {"pitches": [
            {"concept": "a coastal folk singer who records in lighthouses", "genre": "folk", "why_now": "rising"},
            {"concept": "the next Taylor Swift", "genre": "pop"}]},
        head_of_music.Slate: lambda c: {
            "genre_priorities": [{"genre": "folk", "move": "grow", "why": "gap"}],
            "budget_split": [{"creator_id": i, "share": 2} for i in _ids(c, "creator_id")[:1]],
            "cadence": [{"creator_id": i, "songs_per_week": 9} for i in _ids(c, "creator_id")[:1]],
            "concepts": [{"proposal_id": p, "accept": True, "reason": "fills the folk gap"}
                         for p in _ids(c, "proposal_id")]},
    }


def _sandbox_ctx(conn, policy, llm, handle="maren-music"):
    cid = music_sandbox.create_artist(conn, handle, genre="indie")
    return cid


def test_the_sandbox_runs_the_department_end_to_end_without_the_gpu(conn, policy):
    cid = _sandbox_ctx(conn, policy, None)
    llm = RoleLLM(answers(), conn=conn)
    rep = music_sandbox.run(conn, policy, llm, "maren-music", songs_n=2, usd=5, run_id="r1")
    assert "stopped" not in rep and rep["dossier"]["dossier"]["sonic_identity"]["genre"] == "indie"
    assert rep["career"]["dropped_briefs"] == 1 and rep["career"]["predictions"] == 1
    assert [s["state"] for s in rep["songs"]] == ["done", "done"]
    assert all(s["verdict"] == "revise" and s["revisions"] == 1 and s["title"] == "Window Seat" for s in rep["songs"])
    # no GPU: every music.song job is a done stub, nothing is queued for a machine
    jobs = conn.execute("SELECT status, result FROM jobs WHERE kind = 'music.song'").fetchall()
    assert len(jobs) == 4 and all(j["status"] == "done" and j["result"] == {"stub": True} for j in jobs)
    rows = conn.execute("SELECT id, status, take, spec FROM songs WHERE creator_id = %s ORDER BY pipeline_key, take",
                        (cid,)).fetchall()
    assert [(r["take"], r["status"]) for r in rows] == [(0, "dropped"), (1, "made"), (0, "dropped"), (1, "made")]
    chosen = [r for r in rows if r["status"] == "made"]
    plan = chosen[0]["spec"]["mix_plan"]
    assert plan["target_lufs"] == -14 and plan["true_peak_ceiling_dbtp"] == -1.0       # clamped to policy
    assert songs.feedback_for(conn, "mix_master", song_id=chosen[0]["id"])            # the critic's fix routed
    # the revision loop: the producer's second answer had the critic's fix in its input; the bad tag was dropped
    prod = conn.execute("SELECT inputs, output FROM decisions WHERE kind = 'music_producer' ORDER BY created_at"
                        ).fetchall()
    assert len(prod) == 4 and prod[1]["inputs"]["input"]["fixes"] == ["tempo below the band -> raise to 104 bpm"]
    draft = chosen[0]["spec"]
    payload = conn.execute("SELECT payload FROM jobs j JOIN songs s ON s.job_id = j.id WHERE s.id = %s",
                           (chosen[0]["id"],)).fetchone()["payload"]
    assert payload["bpm"] == 104 and "style of" not in payload["tags"]
    assert payload["tags"].startswith("indie pop, nylon guitar, soft female vocal, breathy")   # the fixed voice leads
    # the release, A&R and the slate
    assert rep["release"]["kind"] == "single" and len(rep["release"]["tracks"]) == 1
    assert len(rep["a_and_r"]["filed"]) == 1 and len(rep["a_and_r"]["dropped"]) == 1
    concept = rep["slate"]["concepts"][0]
    assert concept["status"] == "accepted"
    thesis = conn.execute("SELECT body FROM theses WHERE id = %s", (concept["applied"]["thesis_id"],)).fetchone()
    assert thesis["body"]["vertical"] == "music" and thesis["body"]["music"]["genre"] == "folk"
    assert rep["slate"]["budget_split"][0]["share"] == 1.0
    assert rep["slate"]["cadence"][0]["songs_per_week"] == policy.get("music", "roles", "head_of_music",
                                                                      "max_songs_per_week")
    kinds = {r["kind"] for r in conn.execute("SELECT kind FROM decisions WHERE job_id = (SELECT id FROM jobs WHERE "
                                             "kind = 'brain.music_sandbox')").fetchall()}
    assert {"music_talent_manager", "music_songwriter", "music_producer", "music_lyricist", "music_vocal_director",
            "music_draft_review", "music_melody", "music_critique", "music_mix_master", "music_curator",
            "music_a_and_r", "music_head_of_music"} <= kinds
    assert rep["llm_calls"] == len(llm.calls) - 0 or rep["llm_calls"] <= len(llm.calls)
    assert draft


def test_the_sandbox_refuses_a_real_creator_and_a_cap_above_policy(conn, policy):
    real = creator(conn, publish_mode="instagram")
    handle = conn.execute("SELECT handle FROM creators WHERE id = %s", (real,)).fetchone()["handle"]
    with pytest.raises(music_sandbox.NotSandbox):
        music_sandbox.run(conn, policy, StubLLM(conn=conn), handle, songs_n=1)
    music_sandbox.create_artist(conn, "sandy", genre="pop")
    with pytest.raises(ValueError, match="max_usd"):
        music_sandbox.run(conn, policy, StubLLM(conn=conn), "sandy", usd=50)


def test_the_run_stops_at_its_cap(conn, policy):
    music_sandbox.create_artist(conn, "capped", genre="pop")
    llm = RoleLLM(answers(), cost=1.0, conn=conn)
    rep = music_sandbox.run(conn, policy, llm, "capped", songs_n=2, usd=2.0, run_id="cap")
    assert "cap" in rep["stopped"] and len(llm.calls) == 2 and rep["usd_spent"] == 2.0


def test_a_refused_song_never_reaches_the_gpu(conn, policy):
    cid = music_sandbox.create_artist(conn, "refused", genre="pop")
    a = answers()
    a[lyrics.SongAnswer] = {"title": "x", "lyrics": "[Verse]\nin the style of a famous band", "tags": ["pop"],
                            "bpm": 100, "key": "A minor", "time_signature": "4", "duration_s": 30}
    ctx = make_ctx(conn, policy, queue.enqueue(conn, "brain.music_pipeline", {}, creator_id=cid))
    ctx.llm = RoleLLM(a, conn=conn)
    out = music_pipeline.run_song(ctx, cid, {"concept": "x"}, key="k", takes=music_pipeline.StubTakes())
    assert out["state"] == "refused" and conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'music.song'"
                                                      ).fetchone()["n"] == 0


def test_real_takes_are_queued_and_the_pipeline_resumes_when_measured(conn, policy):
    cid = music_sandbox.create_artist(conn, "queued", genre="indie")
    ctx = make_ctx(conn, policy, queue.enqueue(conn, "brain.music_pipeline", {}, creator_id=cid))
    ctx.llm = RoleLLM({**answers(), music_critic.Assessment: [SHIP, SONG_REVIEW]}, conn=conn)
    out = music_pipeline.run_song(ctx, cid, {"concept": "bus"}, key="q1")
    assert out["state"] == "waiting"
    jobs = conn.execute("SELECT id, payload, idempotency_key FROM jobs WHERE kind = 'music.song' ORDER BY "
                        "idempotency_key").fetchall()
    assert [j["idempotency_key"] for j in jobs] == ["music:q1:take:0", "music:q1:take:1"]
    assert jobs[0]["payload"]["seed"] != jobs[1]["payload"]["seed"] and jobs[0]["payload"]["check"]
    assert music_pipeline.run_song(ctx, draft=out["draft"])["state"] == "waiting"    # idempotent: still 2 jobs
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'music.song'").fetchone()["n"] == 2
    for j in jobs:
        conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (j["id"],))
        for role in music.ROLES:
            conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes, meta) VALUES (%s, 'audio', %s, 'x', 1, "
                         "%s)", (j["id"], f"s3://b/{j['id']}/{role}.wav", Jsonb({"role": role})))
    assert music_pipeline.run_song(ctx, draft=out["draft"])["state"] == "waiting"    # registered, not yet measured
    for s in conn.execute("SELECT id FROM songs WHERE pipeline_key = 'q1'").fetchall():
        songs.store_features(conn, s["id"], {"duration_s": 30, "lufs_integrated": -15, "time_to_first_chorus_s": 9})
    done = music_pipeline.run_song(ctx, draft=out["draft"])
    assert done["state"] == "done" and done["song_id"] and done["verdict"] == "revise"


def test_the_pipeline_agent_waits_by_enqueueing_its_continuation(conn, policy):
    cid = music_sandbox.create_artist(conn, "agent", genre="indie")
    job = queue.enqueue(conn, "brain.music_pipeline", {"creator_id": str(cid), "brief": {"concept": "x"}, "key": "a1",
                                                       "test": True}, creator_id=cid)
    ctx = make_ctx(conn, policy, job)
    ctx.llm = RoleLLM({**answers(), music_critic.Assessment: [SHIP]}, conn=conn)
    from manyworlds.brain import agent
    assert agent.get("music_pipeline").run(ctx)["state"] == "waiting"
    nxt = conn.execute("SELECT payload, run_after FROM jobs WHERE kind = 'brain.music_pipeline' AND parent_job_id = %s",
                       (job,)).fetchone()
    assert nxt["payload"]["draft"]["stage"] == "takes" and nxt["run_after"] is not None
    off = make_ctx(conn, policy, queue.enqueue(conn, "brain.music_pipeline", {"creator_id": str(cid)}, creator_id=cid))
    assert agent.get("music_pipeline").run(off) == {"skipped": "music.enabled is off"}


def test_the_pipeline_is_composed_from_policy_per_artist(policy):
    body = dict(policy.body)
    pipe = {**body["music"]["pipeline"], "skip": ["vocal_director"],
            "artists": {"maren": {"writing": ["songwriter", "lyricist"], "takes": 3}}}
    pol = policy_mod.from_dict({**body, "music": {**body["music"], "pipeline": pipe}})
    assert "vocal_director" not in music_pipeline.plan_for(pol)["writing"]
    m = music_pipeline.plan_for(pol, "maren")
    assert m["writing"] == ["songwriter", "lyricist"] and m["takes"] == 3 and m["after"][0] == "melody"
    roles = music_pipeline._roles()
    assert {"songwriter", "lyricist", "producer", "vocal_director", "melody", "mix_master"} <= set(roles)
    for name, role in roles.items():                     # strict inputs, defaulted outputs, an llm role in policy
        if name == "lyricist":
            continue
        with pytest.raises(Exception):
            role.Input.model_validate({"brief": {}, "not_a_field": 1})
        role.Output.model_validate({})
        assert policy.get("llm", "roles", music_roles.llm_role(policy, name))
        assert policy.get("brain", "llm_usd_per_day", name) > 0
    for name in ("head_of_music", "a_and_r", "music_talent_manager", "curator"):
        assert policy.get("llm", "roles", music_roles.llm_role(policy, name))
        assert name in policy.get("brain", "chief_of_staff", "decision_agents")


def test_a_song_role_runs_alone_on_a_songs_draft_with_the_critics_fixes(conn, policy):
    music_sandbox.create_artist(conn, "solo", genre="indie")
    llm = RoleLLM(answers(), conn=conn)
    rep = music_sandbox.run(conn, policy, llm, "solo", songs_n=1, usd=5, run_id="s1")
    sid = rep["songs"][0]["song_id"]
    conn.execute("INSERT INTO song_feedback (song_id, assessment_id, role, problem, change) SELECT %s, id, 'producer', "
                 "'too slow', 'faster' FROM song_assessments WHERE song_id = %s LIMIT 1", (sid, sid))
    cid = conn.execute("SELECT creator_id FROM songs WHERE id = %s", (sid,)).fetchone()["creator_id"]
    ctx = make_ctx(conn, policy, queue.enqueue(conn, "brain.producer", {"song_id": sid}, creator_id=cid))
    ctx.llm = RoleLLM(answers(), conn=conn)
    from manyworlds.brain import agent
    out = agent.get("producer").run(ctx)
    assert out["draft"]["production"]["bpm"] == 84
    assert "too slow -> faster" in ctx.llm.calls[0]["content"]


def test_originality_checks(raw_policy):
    pol = raw_policy
    assert music_roles.word_issues(pol, "the next Taylor Swift")
    assert music_roles.word_issues(pol, "sounds like a famous band")
    assert music_roles.word_issues(pol, "a coastal folk singer, lighthouses, wind") == []


# --- portfolio -----------------------------------------------------------------------------------------------------

def _artist(conn, handle, genre, feats, *, n=3, mode="instagram"):
    cid = conn.execute("INSERT INTO creators (handle, publish_mode, vertical, bible) VALUES (%s, %s, 'music', %s) "
                       "RETURNING id", (handle, mode, Jsonb({"music": {"genre": genre}}))).fetchone()["id"]
    rng = random.Random(handle)
    for i in range(n):
        sid = conn.execute("INSERT INTO songs (creator_id, title, genre) VALUES (%s, %s, %s) RETURNING id",
                           (cid, f"{handle} {i}", genre)).fetchone()["id"]
        songs.store_features(conn, sid, {k: v + rng.uniform(-0.01, 0.01) for k, v in feats.items()})
    return cid


A = {"tempo_bpm": 90, "lufs_integrated": -10, "onset_density": 2, "chorus_repeats": 3, "vocal_clarity_db": 1}
B = {"tempo_bpm": 140, "lufs_integrated": -7, "onset_density": 5, "chorus_repeats": 2, "vocal_clarity_db": -3}


def test_the_music_portfolio_measures_diversification_and_gaps(conn, policy):
    from manyworlds.brain import music_portfolio
    _artist(conn, "one", "indie", A)
    _artist(conn, "two", "indie", A)          # sounds like one
    _artist(conn, "three", "rap", B)
    snap = music_portfolio.snapshot(conn, policy)
    assert {a["handle"]: a["genre"] for a in snap["artists"]} == {"one": "indie", "two": "indie", "three": "rap"}
    close = snap["diversification"]["too_close"]
    assert [sorted([p["a"], p["b"]]) for p in close] == [["one", "two"]]
    far = [p for p in snap["diversification"]["pairs"] if "three" in (p["a"], p["b"])]
    assert all(p["distance"] > snap["diversification"]["min_distance"] for p in far)
    gap_genres = {g["genre"] for g in snap["gaps"] if g["kind"] == "genre"}
    assert {"pop", "country", "rock", "lofi"} <= gap_genres and "indie" not in gap_genres
    assert snap["budget"]["share"] == policy.get("music", "portfolio", "budget_share") and snap["budget"]["usd"] > 0
    ceo = portfolio.snapshot(conn, policy)
    assert {c["vertical"] for c in ceo["creators"]} == {"music"} and ceo["music"]["genres"] == {"indie": 2, "rap": 1}


def test_content_only_portfolios_carry_no_music_section(conn, policy):
    creator(conn, publish_mode="instagram")
    assert "music" not in portfolio.snapshot(conn, policy)


def test_an_accepted_artist_concept_becomes_a_music_thesis(conn, policy):
    p = proposals.file(conn, policy, kind="artist_concept", from_role="a_and_r", subject={"concept": "a folk singer",
                                                                                          "genre": "folk"},
                       argument="rising")
    assert p["to_role"] == "head_of_music"
    assert proposals.decide(conn, policy, p["id"], decided_by="strategist", accept=True, reason="x").get("refused")
    out = proposals.decide(conn, policy, p["id"], decided_by="head_of_music", accept=True, reason="gap")
    body = conn.execute("SELECT body FROM theses WHERE id = %s", (out["applied"]["thesis_id"],)).fetchone()["body"]
    assert body["vertical"] == "music" and body["music"]["genre"] == "folk"


# --- experiments ---------------------------------------------------------------------------------------------------

def test_a_tentative_music_finding_becomes_a_song_experiment(conn, policy):
    cid = creator(conn, publish_mode="instagram")
    evidence.record_finding(conn, scope="creator", creator_id=cid, statement="an earlier chorus did better",
                            evidence="observational", grade="tentative", effect=-0.4, domain="music",
                            sources=[{"effect": "feature:time_to_first_chorus_s"}])
    opened = experiments.music_open_from_findings(conn, policy, cid)
    assert len(opened) == 1 and experiments.music_open_from_findings(conn, policy, cid) == []
    exp = experiments.open_for(conn, cid, domain="music")[0]
    assert exp["effect_key"] == "music:feature:time_to_first_chorus_s" and exp["metric"] == "song_reach_lift"
    a = next(x for x in exp["arms"] if x["name"] == "A")
    assert a["side"] == "low" and "first 8 seconds" in a["instruction"]      # the finding said earlier is better
    assert experiments.open_for(conn, cid) == []                             # never a content slot
    slots = experiments.music_assign(conn, policy, cid, batch_key="b1", n_songs=6)
    assert sorted(s["arm"] for s in slots) == ["A"] * 3 + ["B"] * 3
    assert slots == experiments.music_assign(conn, policy, cid, batch_key="b1", n_songs=6)
    # outcomes: arm A songs do better; read once each arm has its songs
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    rng = random.Random(3)
    need = policy.get("music", "experiments", "min_songs_per_arm")
    arms = ["base"] * 4 + ["A", "B"] * need
    for i, arm in enumerate(arms):
        when = t0 + timedelta(hours=12 * i)
        sid = None
        if arm != "base":
            sid = conn.execute("INSERT INTO songs (creator_id, title, experiment_id, arm) VALUES (%s, 's', %s, %s) "
                               "RETURNING id", (cid, exp["id"], arm)).fetchone()["id"]
        post = conn.execute("INSERT INTO posts (creator_id, format, status, published_at, song_id) VALUES "
                            "(%s, 'music_video', 'published', %s, %s) RETURNING id", (cid, when, sid)).fetchone()["id"]
        conn.execute("INSERT INTO publications (post_id, platform, simulated, published_at) VALUES (%s, 'instagram', "
                     "false, %s)", (post, when))
        reach = round(1000 * math.exp((0.8 if arm == "A" else 0.0) + rng.gauss(0, 0.05)))
        conn.execute("INSERT INTO metric_snapshots (post_id, metrics, post_age_s) VALUES (%s, %s, 259200)",
                     (post, Jsonb({"reach": reach})))
    out = experiments.music_review(conn, policy, cid)
    assert out["concluded"][0]["status"] == "confirmed"
    f = evidence.findings(conn, mode="real", creator_id=cid, grades=("proven",), domain="music")
    assert len(f) == 1 and f[0]["evidence"] == "experimental"
