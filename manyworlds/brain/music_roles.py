"""What every music role shares (spec 2026-11-23-music-department.md, section 3).

A role is an agent in brain/agents/ with a strict input model (extra fields refused) and an output model (every field
defaulted, so an empty answer is still valid), its own llm role (policy music.roles.<role>.llm_role, a model in
llm.roles) and its own daily budget (brain.llm_usd_per_day.<role>). It runs in-process inside the department's
pipeline (manyworlds.music_pipeline) or alone as a `brain.<role>` job. `ask()` makes the one call a role makes:

  - charged to the role (ledger agent tag, `sandbox:<role>` for scenario creators), its budget checked before the
    call, or, inside `spend_cap()` (the sandbox runner), the run's own cap instead;
  - recorded once per job step as a decision (kind music_<role>), so a rerun of the job asks nothing again;
  - given the critic's open fixes for the role (manyworlds.songs.feedback_for) when the input carries a song.

`clean_words()` is the originality check every role's free text goes through: no real artist names and no phrase
that points at someone else's work (music.lyrics.banned_names / banned_phrases), no "the next <someone>".
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import re
from datetime import datetime, timezone
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

from .. import music
from ..policy import Policy
from . import budget
from .agent import AgentContext, remember

T = TypeVar("T", bound=BaseModel)

ROLES = ("head_of_music", "a_and_r", "music_talent_manager", "songwriter", "lyricist", "melody", "producer",
         "vocal_director", "mix_master", "curator", "music_critic")


class StrictInput(BaseModel):
    """A role's input: typed, nothing extra (a misspelt field is an error, not silently ignored)."""
    model_config = ConfigDict(extra="forbid")


def cfg(policy: Policy, role: str) -> dict:
    return policy.get("music", "roles", role, default={}) or {}


def llm_role(policy: Policy, role: str) -> str:
    return str(cfg(policy, role).get("llm_role") or role)


def enabled(policy: Policy, role: str) -> bool:
    return bool(cfg(policy, role).get("enabled", True))


# --- spend ---------------------------------------------------------------------------------------------------------

_CAP: contextvars.ContextVar[dict | None] = contextvars.ContextVar("music_spend_cap", default=None)


@contextlib.contextmanager
def spend_cap(conn, usd: float):
    """Inside this block every role's call is checked against one cap for the whole run (the LLM spend the ledger
    records from the block's start), instead of each role's daily budget: the sandbox runner's `--usd`."""
    start = conn.execute("SELECT now() AS t").fetchone()["t"]
    token = _CAP.set({"conn": conn, "usd": float(usd), "since": start})
    try:
        yield
    finally:
        _CAP.reset(token)


def spent_in_cap() -> float | None:
    cap = _CAP.get()
    if cap is None:
        return None
    row = cap["conn"].execute("SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE category = 'llm' "
                              "AND created_at >= %s", (cap["since"],)).fetchone()
    return float(row["u"])


def _guard(conn, policy: Policy, tag: str):
    cap = _CAP.get()
    if cap is None:
        return lambda: budget.check(conn, policy, tag)

    def check():
        spent = spent_in_cap() or 0.0
        if spent >= cap["usd"]:
            raise budget.OverBudget(f"the run's LLM cap of ${cap['usd']:.2f} is used (${spent:.2f})")
    return check


# --- originality ---------------------------------------------------------------------------------------------------

_NEXT = re.compile(r"\b(the next|our own|a female|a male|the [a-z]+ version of)\s+[A-Z][a-z]+", re.UNICODE)


def word_issues(policy: Policy, text: str) -> list[str]:
    """Why free text a role wrote points at someone else (empty when it doesn't)."""
    lc = music.cfg(policy).get("lyrics") or {}
    out = [f"names a real artist ({n!r})" for n in music._name_hits(text, lc.get("banned_names"))]
    out += [f"points at someone else's work ({p!r})" for p in music._phrase_hits(text, lc.get("banned_phrases"))]
    if _NEXT.search(str(text or "")):
        out.append("pitches a copy of someone ('the next ...'): concepts are original")
    return out


def clean_words(policy: Policy, obj: Any) -> list[str]:
    return word_issues(policy, json.dumps(obj, ensure_ascii=False, default=str))


def name_issues(policy: Policy, obj: Any) -> list[str]:
    """The narrower check for text that describes an artist in words (a dossier, a song brief): real names and
    "the next <someone>" only. Phrases like "sounds like" or "inspired by" are fine there when no one is named;
    they stay refused in lyrics, titles and tags (music.issues) and in A&R pitches (clean_words)."""
    lc = music.cfg(policy).get("lyrics") or {}
    text = json.dumps(obj, ensure_ascii=False, default=str)
    out = [f"names a real artist ({n!r})" for n in music._name_hits(text, lc.get("banned_names"))]
    if _NEXT.search(text):
        out.append("pitches a copy of someone ('the next ...'): concepts are original")
    return out


# --- the call ------------------------------------------------------------------------------------------------------

def ask(ctx: AgentContext, role: str, *, system: str, inp: BaseModel, out: type[T], creator_id, step: str,
        context: dict | None = None, rationale_field: str = "why") -> T:
    """The role's one call for this step (see the module docstring). Returns the output model."""
    conn, policy = ctx.conn, ctx.policy
    c = cfg(policy, role)
    name = llm_role(policy, role)

    def compute():
        tag = budget.tag_for(conn, role, creator_id)
        content = f"INPUT ({type(inp).__name__}):\n{inp.model_dump_json()}"
        if context:
            content += "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False, default=str)
        with ctx.llm.tagged(tag, ctx.job.id, before_call=_guard(conn, policy, tag), attempt=ctx.job.attempts):
            answer = ctx.llm.structured(name, system=system, content=content, schema=out,
                                        effort=str(c.get("effort", "medium")), creator_id=creator_id)
        dumped = answer.model_dump()
        kind = role if role.startswith("music_") else f"music_{role}"
        return {"kind": kind, "creator_id": creator_id, "inputs": {"step": step, "input": inp.model_dump()},
                "output": dumped, "rationale": str(dumped.get(rationale_field) or "")[:2000],
                "model": ctx.llm.model_for(name)}
    return out.model_validate(remember(ctx, f"{role}:{step}", compute))


def fixes_for(conn, role: str, song_id) -> list[str]:
    """The critic's open fixes for this role on this song, as lines a prompt can carry."""
    if song_id is None:
        return []
    from .. import songs
    return [f"{f['problem']} -> {f['change']} (evidence: {', '.join(f['evidence'] or [])})"
            for f in songs.feedback_for(conn, role, song_id=song_id)]


def now() -> datetime:
    return datetime.now(timezone.utc)


# --- the song roles' shared shapes ---------------------------------------------------------------------------------

class SongBrief(BaseModel):
    """What one song is for: the music talent manager writes it, the songwriter starts from it."""
    concept: str = ""
    theme: str = ""
    pov: str = ""
    mood: str = ""
    genre: str = ""
    subgenre: str = ""
    target_listener: str = ""
    reference_pattern: str = ""      # a series pattern from the artist's archetype playbook, by name
    why: str = ""


class SongRole:
    """A role in the song pipeline: builds its input from the draft (the pipeline's state, a dict), asks, and
    writes its output back into the draft. Subclasses set name, Input, Output, SYSTEM and the two hooks."""
    name = ""
    Input: type[BaseModel] = StrictInput
    Output: type[BaseModel] = BaseModel
    SYSTEM = ""

    def build(self, ctx: AgentContext, draft: dict) -> BaseModel:
        raise NotImplementedError

    def apply(self, ctx: AgentContext, draft: dict, out: BaseModel) -> None:
        raise NotImplementedError

    def run(self, ctx: AgentContext, draft: dict, *, step: str) -> BaseModel:
        inp = self.build(ctx, draft)
        out = ask(ctx, self.name, system=self.SYSTEM, inp=inp, out=self.Output, creator_id=draft.get("creator_id"),
                  step=step)
        self.apply(ctx, draft, out)
        draft.setdefault("trace", []).append({"role": self.name, "step": step})
        return out


SONG_ROLES: dict[str, SongRole] = {}


def register_song_role(role: SongRole) -> SongRole:
    SONG_ROLES[role.name] = role
    return role


def draft_fixes(draft: dict, role: str) -> list[str]:
    """The fixes the draft carries for a role: the critic's draft review this run, plus its open fixes on the song
    this draft revises (songs.feedback_for)."""
    return list((draft.get("fixes") or {}).get(role) or [])


class SongRoleAgent:
    """A song role run alone as `brain.<role>` ({song_id}): the song's stored draft (songs.brief.draft) with the
    critic's open fixes for the role, the role's answer recorded and the draft it makes kept in the result."""

    def __init__(self, role: SongRole):
        self.role = role
        self.name = role.name

    def run(self, ctx: AgentContext) -> dict:
        from ..worker.runner import NonRetryable
        from .. import songs
        p = ctx.job.payload or {}
        song = songs.get(ctx.conn, p.get("song_id")) if p.get("song_id") else None
        if song is None:
            raise NonRetryable(f"{self.name} payload needs a song_id of a song with a draft")
        draft = dict((song["brief"] or {}).get("draft") or {})
        if not draft:
            raise NonRetryable(f"song {song['id']} has no draft (made outside the music pipeline)")
        draft["creator_id"] = str(song["creator_id"])
        draft.setdefault("fixes", {})[self.name] = fixes_for(ctx.conn, self.name, song["id"])
        out = self.role.run(ctx, draft, step=f"song:{song['id']}")
        return {"song_id": str(song["id"]), "output": out.model_dump(), "draft": draft}
