"""The music department's song pipeline (spec 2026-11-23-music-department.md, section 3): the roles composed per
artist from policy `music.pipeline`, a cheap revision loop before any GPU, candidate takes, the pick, the critic and
the mix plan.

    plan_for(policy, handle)          the pipeline for one artist: music.pipeline with music.pipeline.artists.<handle>
                                      merged over it (writing roles in order, draft review, revision_limit, takes,
                                      after-take steps, skip). A role is swapped by its llm role
                                      (music.roles.<role>.llm_role), skipped by `skip`, reordered by the lists.
    run_song(ctx, creator_id, brief, key=, takes=)   one song through the pipeline; resumable: a run that waits for
                                      its takes returns state 'waiting' with the draft, and run_song(draft=...) goes on
                                      from there (the music_pipeline agent re-enqueues itself).
    payload(policy, draft, take)      the music.song job for one take (manyworlds.music.song_payload: checked)
    QueueTakes / StubTakes            the take makers: queued music.song GPU jobs (measured by the producer tick), or,
                                      in the sandbox, done jobs with stand-in measurements (no GPU, no music.song run)
    STEPS                             after-take steps by name; `voice_conversion` is a slot another branch fills
                                      (register_step), a no-op until then

Before the GPU: the writing roles (songwriter, producer, lyricist, vocal_director by default), then the critic's
draft review (brain.agents.music_critic.review_draft); its fixes go back to the roles that wrote the parts, up to
revision_limit rounds, and a draft still judged 'rethink' never reaches the GPU while stop_on_rethink. After the
takes: melody picks one, the critic reviews it (its fixes routed through song_feedback), mix_master plans the mix.
Only the chosen take stays 'made'; the others are 'dropped'. Each role's call is a recorded decision.
"""

from __future__ import annotations

import copy
import hashlib
import random
from typing import Any, Callable, Protocol

from psycopg.types.json import Jsonb

from . import music, queue, song_measure, songs
from .policy import Policy

STEPS: dict[str, Callable] = {}


def register_step(name: str, fn: Callable) -> None:
    """An after-take step (fn(ctx, draft) -> None), e.g. the voice conversion a later branch builds."""
    STEPS[name] = fn


def cfg(policy: Policy) -> dict:
    return policy.get("music", "pipeline", default={}) or {}


def plan_for(policy: Policy, handle: str | None = None) -> dict:
    c = {k: v for k, v in cfg(policy).items() if k != "artists"}
    over = ((cfg(policy).get("artists") or {}).get(handle) or {}) if handle else {}
    out = {**c, **over}
    skip = set(out.get("skip") or [])
    out["writing"] = [r for r in out.get("writing") or [] if r not in skip]
    out["after"] = [r for r in out.get("after") or [] if r not in skip]
    return out


def _roles():
    from .brain import agents  # noqa: F401 - registers the roles
    from .brain.music_roles import SONG_ROLES
    return SONG_ROLES


# --- takes ---------------------------------------------------------------------------------------------------------

class TakeMaker(Protocol):
    def make(self, ctx, draft: dict, payloads: list[dict]) -> list[str] | None:
        """The takes' song ids once made and measured, else None (waiting)."""


def _register_take(conn, job_id, draft: dict, i: int) -> str:
    sid = songs.from_job(conn, job_id, creator_id=draft["creator_id"], brief=_brief(draft),
                         genre=(draft.get("brief") or {}).get("genre") or None,
                         subgenre=(draft.get("brief") or {}).get("subgenre") or None,
                         parent_id=draft.get("parent_id"))
    exp = draft.get("experiment") or {}
    conn.execute("UPDATE songs SET take = %s, pipeline_key = %s, experiment_id = %s, arm = %s WHERE id = %s",
                 (i, draft["key"], exp.get("experiment_id"), exp.get("arm"), sid))
    return str(sid)


class QueueTakes:
    """Real takes: one music.song job per take (idempotent per pipeline key and take), on the GPU; measured on the
    control plane by the producer's tick (songs.measure_pending). Waits until every take is done and measured."""

    def make(self, ctx, draft, payloads):
        conn = ctx.conn
        ids = []
        for i, p in enumerate(payloads):
            ids.append(queue.enqueue(conn, music.kind(ctx.policy), p, creator_id=draft["creator_id"],
                                     batch=f"music-{draft['key']}", idempotency_key=f"music:{draft['key']}:take:{i}"))
        rows = conn.execute("SELECT id, status FROM jobs WHERE id = ANY(%s::uuid[])", ([str(j) for j in ids],)).fetchall()
        status = {str(r["id"]): r["status"] for r in rows}
        if any(status[str(j)] in ("queued", "running") for j in ids):
            return None
        made = [_register_take(conn, j, draft, i) for i, j in enumerate(ids) if status[str(j)] == "done"]
        made = [s for s in made if s]
        feats = songs.latest_features(conn, made)
        if any("duration_s" not in (feats.get(s) or {}) for s in made):
            return None   # the producer's tick measures them
        return made


class StubTakes:
    """Sandbox takes: no GPU and no music.song run. Each take is a done job row with stand-in audio rows and
    stand-in measurements (the words are counted for real; sound numbers are drawn around the spec, seeded by the
    take), stored as measurements with detail.stub = true. Never used for a real creator (the sandbox runner refuses
    one)."""

    def make(self, ctx, draft, payloads):
        conn, policy = ctx.conn, ctx.policy
        out = []
        for i, p in enumerate(payloads):
            key = f"music:{draft['key']}:take:{i}"
            row = conn.execute("SELECT id FROM jobs WHERE idempotency_key = %s", (key,)).fetchone()
            if row is None:
                row = conn.execute(
                    "INSERT INTO jobs (kind, status, payload, creator_id, batch, idempotency_key, result) VALUES "
                    "(%s, 'done', %s, %s, %s, %s, %s) RETURNING id",
                    (music.kind(policy), Jsonb(p), draft["creator_id"], f"music-sandbox-{draft['key']}", key,
                     Jsonb({"stub": True}))).fetchone()
                for role in music.ROLES:
                    conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) VALUES "
                                 "(%s, %s, 'audio', %s, 'stub', 0, %s)",
                                 (row["id"], draft["creator_id"], f"stub://{key}/{role}.wav",
                                  Jsonb({"role": role, "seconds": p.get("duration_s"), "stub": True,
                                         **({"check": {"passed": True, "stub": True}} if role == "mix" else {})})))
            sid = _register_take(conn, row["id"], draft, i)
            if "duration_s" not in (songs.latest_features(conn, [sid]).get(sid) or {}):
                songs.store_features(conn, sid, stub_features(policy, draft, p, i), detail={"stub": True},
                                     policy_version=policy.version)
            out.append(sid)
        return out


def stub_features(policy: Policy, draft: dict, payload: dict, i: int) -> dict:
    """Stand-in measurements for a sandbox take: the lyric features counted from the words, the sound drawn around
    the spec (policy music.pipeline.stub: spreads), seeded by the pipeline key and take."""
    s = cfg(policy).get("stub") or {}
    rng = random.Random(int(hashlib.sha256(f"{draft['key']}:{i}".encode()).hexdigest()[:12], 16))
    f = dict(song_measure.lyric_features(payload.get("lyrics") or "", payload.get("title") or "", policy=policy))
    f.pop("sung_syllables", None)
    hook = float((draft.get("concept") or {}).get("hook_at_s") or s.get("default_hook_s", 10))
    lo, hi = s.get("lufs") or (-14, -8)
    f.update({"duration_s": float(payload.get("duration_s") or 30),
              "tempo_bpm": round(float(payload.get("bpm") or 96) * rng.uniform(0.98, 1.02), 1),
              "key": payload.get("key"), "mode": str(payload.get("key") or "C major").split()[-1],
              "time_to_first_chorus_s": round(max(0.0, hook + rng.gauss(0, float(s.get("hook_sd_s", 2.0)))), 2),
              "chorus_repeats": max(1, int(f.get("written_choruses") or 1) + rng.choice([-1, 0, 0, 1])),
              "lufs_integrated": round(rng.uniform(float(lo), float(hi)), 2),
              "true_peak_dbtp": round(rng.uniform(-2.0, 0.5), 2),
              "time_to_first_vocal_s": round(rng.uniform(0.0, float(s.get("max_vocal_entry_s", 8))), 2),
              "vocal_clarity_db": round(rng.gauss(0, 3), 2),
              "chorus_energy_lift_db": round(rng.gauss(3, 2), 2), "onset_density": round(rng.uniform(1, 4), 2)})
    return f


# --- the draft -----------------------------------------------------------------------------------------------------

def _brief(draft: dict) -> dict:
    """What the song row keeps: the brief and a slim draft (the roles run alone from it: SongRoleAgent)."""
    slim = {k: v for k, v in draft.items() if k not in ("context", "bible", "takes", "take_context", "trace")}
    return {**(draft.get("brief") or {}), "draft": copy.deepcopy(slim)}


def payload(policy: Policy, draft: dict, take: int) -> dict:
    words, prod, vocal = draft.get("words") or {}, draft.get("production") or {}, draft.get("vocal") or {}
    seconds = float(cfg(policy).get("song_seconds", 30))
    spec = {"title": words.get("title"), "lyrics": words.get("lyrics"),
            "tags": [*(prod.get("tags") or []), *(vocal.get("vocal_tags") or [])],
            "bpm": prod.get("bpm"), "key": prod.get("key"), "time_signature": prod.get("time_signature"),
            "duration_s": prod.get("duration_s") or seconds}
    seed = int(hashlib.sha256(f"{draft['key']}:{take}".encode()).hexdigest()[:8], 16)
    return music.song_payload(policy, spec, seed=seed, check=True, max_seconds=seconds)


def new_draft(conn, policy: Policy, creator_id, brief: dict, *, key: str, experiment: dict | None = None,
              parent_id=None) -> dict:
    from .brain import music_analysis
    from .brain.agents.music_talent_manager import latest_dossier
    row = conn.execute("SELECT handle, bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    dossier = (latest_dossier(conn, creator_id) or {}).get("output") or {}
    from .brain import evidence
    mode = evidence.mode_for(conn, creator_id)
    genre = brief.get("genre") or (dossier.get("sonic_identity") or {}).get("genre")
    best = sorted([r for r in music_analysis.song_rows(conn, policy, mode=mode, creator_id=creator_id)
                   if r.get("y") is not None], key=lambda r: -r["y"])[:int(cfg(policy).get("best_songs_shown", 3))]
    b = dict(brief)
    if experiment:
        b["why"] = (b.get("why") or "") + f" [experiment arm {experiment['arm']}: {experiment['instruction']}]"
    return {"key": key, "creator_id": str(creator_id), "handle": row["handle"], "bible": row["bible"] or {},
            "artist": dossier, "brief": {**b, "genre": genre or ""}, "experiment": experiment, "parent_id": parent_id,
            "context": {"bands": music_analysis.bands(conn, policy, creator_id=creator_id, genre=genre, mode=mode),
                        "findings": [{"statement": f["statement"], "grade": f["grade"], "label": f.get("label")}
                                     for f in music_analysis.findings(conn, mode=mode, creator_id=creator_id)],
                        "nearest": [{"title": r["title"], "features": r["features"], "outcome": r["outcome"]}
                                    for r in best]},
            "fixes": {}, "revisions": 0, "stage": "writing"}


def _group(fixes: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in fixes:
        out.setdefault(f["role"], []).append(f"{f['problem']} -> {f['change']}")
    return out


def _write(ctx, draft: dict, roles: list[str], rnd: int) -> None:
    reg = _roles()
    for name in roles:
        if name in reg:
            reg[name].run(ctx, draft, step=f"{draft['key']}:{name}:r{rnd}")


def run_song(ctx, creator_id=None, brief: dict | None = None, *, key: str | None = None, takes: TakeMaker | None = None,
             draft: dict | None = None, experiment: dict | None = None) -> dict:
    """One song through the pipeline (see the module docstring): {state: done | waiting | refused | rethink, draft,
    song_id?, verdict?}."""
    from .brain.agents import music_critic
    conn, policy = ctx.conn, ctx.policy
    takes = takes or QueueTakes()
    if draft is None:
        draft = new_draft(conn, policy, creator_id, brief or {}, key=key or str(ctx.job.id), experiment=experiment)
    plan = plan_for(policy, draft.get("handle"))
    if draft["stage"] == "writing":
        _write(ctx, draft, plan["writing"], 0)
        limit = int(plan.get("revision_limit", 2))
        review = None
        for rnd in range(limit + 1):
            if draft.get("lyrics_refused"):
                return {"state": "refused", "why": draft["lyrics_refused"], "draft": draft}
            if not plan.get("draft_review", True):
                break
            review = music_critic.review_draft(ctx, draft, step=f"{draft['key']}:r{rnd}")
            draft.setdefault("reviews", []).append({"verdict": review["verdict"], "fixes": review["fixes"]})
            if review["verdict"] == "ship" or not review["fixes"] or rnd == limit:
                break
            draft["fixes"] = _group(review["fixes"])
            draft["revisions"] = rnd + 1
            _write(ctx, draft, [r for r in plan["writing"] if r in draft["fixes"]], rnd + 1)
        if review and review["verdict"] == "rethink" and plan.get("stop_on_rethink", True):
            return {"state": "rethink", "why": review.get("summary"), "draft": draft}
        draft["fixes"] = {}
        try:
            n = min(int(plan.get("takes", 2)), int((policy.get("music", "roles", "melody", default={}) or {})
                                                   .get("max_takes", 4)))
            draft["payloads"] = [payload(policy, draft, i) for i in range(max(1, n))]
        except music.SongRefused as e:
            return {"state": "refused", "why": str(e), "draft": draft}
        draft["stage"] = "takes"
    if draft["stage"] == "takes":
        made = takes.make(ctx, draft, draft["payloads"])
        if made is None:
            return {"state": "waiting", "draft": draft}
        from .brain import music_analysis
        draft["takes"] = []
        for i, sid in enumerate(made):
            c = music_analysis.song_context(conn, policy, sid, nearest_k=0)
            draft["takes"].append({"index": i, "song_id": sid, "features": c["features"], "position": c["position"]})
        draft["stage"] = "after"
    reg = _roles()
    for step in plan["after"]:
        if step == "melody" and step in reg:
            reg[step].run(ctx, draft, step=f"{draft['key']}:melody")
            if draft.get("chosen_song_id"):
                conn.execute("UPDATE songs SET status = 'dropped', updated_at = now() WHERE pipeline_key = %s "
                             "AND id <> %s", (draft["key"], draft["chosen_song_id"]))
        elif step == "music_critic" and draft.get("chosen_song_id"):
            out = music_critic.critique(ctx, draft["chosen_song_id"])
            draft["assessment"] = out
            fixes = conn.execute("SELECT role, problem, change FROM song_feedback WHERE song_id = %s AND status = 'open'",
                                 (draft["chosen_song_id"],)).fetchall()
            draft["fixes"] = _group([dict(f) for f in fixes])
        elif step in reg and draft.get("chosen_song_id"):
            if step == "mix_master":
                c = music_analysis_context(conn, policy, draft["chosen_song_id"])
                draft["take_context"] = c
            reg[step].run(ctx, draft, step=f"{draft['key']}:{step}")
        elif step in STEPS:
            STEPS[step](ctx, draft)
    sid = draft.get("chosen_song_id")
    if sid:
        conn.execute("UPDATE songs SET brief = %s, updated_at = now() WHERE id = %s", (Jsonb(_brief(draft)), sid))
    draft["stage"] = "done"
    return {"state": "done", "song_id": sid, "verdict": (draft.get("assessment") or {}).get("verdict"),
            "draft": draft}


def music_analysis_context(conn, policy: Policy, song_id) -> dict:
    from .brain import music_analysis
    c = music_analysis.song_context(conn, policy, song_id, nearest_k=0)
    return {"features": c["features"], "position": c["position"]}


def next_briefs(conn, creator_id, n: int) -> list[dict]:
    """The talent manager's latest song briefs for the artist (brain.agents.music_talent_manager, review), at most n."""
    from .brain.agents.music_talent_manager import latest_career
    row = latest_career(conn, creator_id)
    return list(((row or {}).get("output") or {}).get("next_songs") or [])[:n]


def jsonable(x: Any) -> Any:
    import json
    return json.loads(json.dumps(x, default=str))
