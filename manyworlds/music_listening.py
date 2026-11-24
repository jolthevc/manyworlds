"""What the critic hears, behind two adapters (spec 2026-11-23-music-department.md, section 4).

    aesthetics   Meta Audiobox Aesthetics (CC-BY-4.0) on the mix and the vocal stem: production quality (PQ),
                 production complexity (PC), content enjoyment (CE), content usefulness (CU), each 1-10.
                 Adapters: stub (nothing; shipped), worker (read from the song's music.listen job).
    listener     an Apache-2.0 audio LLM's factual answers about the song (ListenerReport: genre, mood, instruments,
                 vocal timbre, hook entry, audible lyrics, mix problems). Adapters: stub (nothing; shipped), hosted (an
                 OpenAI-compatible audio model through manyworlds.llm, role music.critic.listener.hosted_role, which
                 must be in llm.roles: added by the owner once the host, model and price are checked), worker (the
                 music.listen GPU job: Qwen3-Omni, Step-Audio-2-mini or Qwen2.5-Omni on a card; not built yet: no
                 handler, image or job_kinds entry).

`aesthetics(policy, ...)` and `listener(policy, ...)` pick the adapter from music.critic.<name>.adapter. Every
adapter answers None when it has nothing (yet): the critic then reasons without it and says so. A worker adapter
asks once (an idempotent music.listen job per song) and answers once the job is done. None of the real adapters is
exercised by the tests against a real model or machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from .policy import Policy

AESTHETIC_AXES = ("PQ", "PC", "CE", "CU")


class ListenerReport(BaseModel):
    """Checkable answers, not opinions: what a listener can hear and say."""
    genre: str = Field(default="", description="the genre and subgenre heard")
    mood: str = Field(default="", description="the mood in a few words")
    instruments: list[str] = Field(default_factory=list, description="instruments heard")
    vocal_timbre: str = Field(default="", description="the voice: register, texture, delivery")
    hook_entry_s: float | None = Field(default=None, description="seconds from the start to the first hook")
    audible_lyrics: list[str] = Field(default_factory=list, description="lines heard clearly, as heard")
    mix_problems: list[str] = Field(default_factory=list, description="audible mix problems (muddy, harsh, buried "
                                                                      "vocal, clipping, too quiet...)")


class Song(Protocol):
    """What an adapter is given: the songs row's mapping (id, mix_uri, vocal_uri, lyrics, title)."""
    def __getitem__(self, key: str) -> Any: ...


class Aesthetics(Protocol):
    def scores(self, song) -> dict | None:
        """{"mix": {PQ, PC, CE, CU}, "vocals": {...}} or None."""


class Listener(Protocol):
    def describe(self, song) -> ListenerReport | None: ...


def _cfg(policy: Policy, name: str) -> dict:
    return policy.get("music", "critic", name, default={}) or {}


class StubAesthetics:
    def __init__(self, scores: dict | None = None):
        self._scores = scores

    def scores(self, song) -> dict | None:
        return self._scores


class StubListener:
    def __init__(self, report: ListenerReport | None = None):
        self._report = report

    def describe(self, song) -> ListenerReport | None:
        return self._report


def _listen_job(conn, policy: Policy, song) -> dict | None:
    """The song's music.listen job (asked once; idempotent), its result once done, else None."""
    from . import queue
    kind = str(_cfg(policy, "listener").get("worker_kind") or "music.listen")
    key = f"song:{song['id']}:listen"
    row = conn.execute("SELECT status, result FROM jobs WHERE idempotency_key = %s", (key,)).fetchone()
    if row is None:
        queue.enqueue(conn, kind, {"song_id": str(song["id"]), "mix_uri": song["mix_uri"],
                                   "vocal_uri": song["vocal_uri"], "questions": list(ListenerReport.model_fields),
                                   "aesthetics_axes": list(AESTHETIC_AXES)},
                      creator_id=song["creator_id"], idempotency_key=key)
        return None
    return row["result"] if row["status"] == "done" else None


class WorkerAesthetics:
    """Audiobox Aesthetics scores from the song's music.listen job (result.aesthetics)."""
    def __init__(self, conn, policy: Policy):
        self.conn, self.policy = conn, policy

    def scores(self, song) -> dict | None:
        result = _listen_job(self.conn, self.policy, song)
        return (result or {}).get("aesthetics") or None


class WorkerListener:
    """The listener's answers from the song's music.listen job (result.listener)."""
    def __init__(self, conn, policy: Policy):
        self.conn, self.policy = conn, policy

    def describe(self, song) -> ListenerReport | None:
        result = _listen_job(self.conn, self.policy, song)
        heard = (result or {}).get("listener")
        return ListenerReport.model_validate(heard) if heard else None


LISTENER_SYSTEM = ("You listen to one song and answer factual questions about what you hear. Say only what is "
                   "audible; leave a field empty rather than guess. Times are seconds from the start.")


class HostedListener:
    """An OpenAI-compatible hosted audio model, called through manyworlds.llm (role music.critic.listener
    .hosted_role; the role must be in llm.roles, its model in llm.open_models with `audio: true`)."""
    def __init__(self, llm, policy: Policy, store):
        self.llm, self.policy, self.store = llm, policy, store

    def describe(self, song) -> ListenerReport | None:
        import tempfile

        from .llm import audio_block
        c = _cfg(self.policy, "listener")
        role = str(c.get("hosted_role") or "music_listener")
        if not self.policy.get("llm", "roles", role):
            return None   # not configured: the owner adds the role once the hosted model is verified
        with tempfile.TemporaryDirectory(prefix="mw-listen-") as tmp:
            local = self.store.get_file(song["mix_uri"], Path(tmp) / ("mix" + Path(song["mix_uri"]).suffix))
            data = Path(local).read_bytes()
        if len(data) > float(c.get("max_audio_mb", 20)) * 1e6:
            return None
        fmt = Path(song["mix_uri"]).suffix.lstrip(".") or str(c.get("audio_format") or "mp3")
        return self.llm.structured(role, system=LISTENER_SYSTEM, schema=ListenerReport, effort="low",
                                   creator_id=song["creator_id"],
                                   content=[audio_block(data, fmt),
                                            {"type": "text", "text": f"The song's title: {song['title']!r}."}])


AdapterName = Literal["stub", "worker", "hosted"]


def aesthetics(policy: Policy, *, conn=None) -> Aesthetics:
    name = str(_cfg(policy, "aesthetics").get("adapter") or "stub")
    if name == "stub":
        return StubAesthetics()
    if name == "worker":
        return WorkerAesthetics(conn, policy)
    raise ValueError(f"unknown aesthetics adapter {name!r} (music.critic.aesthetics.adapter)")


def listener(policy: Policy, *, conn=None, llm=None, store=None) -> Listener:
    name = str(_cfg(policy, "listener").get("adapter") or "stub")
    if name == "stub":
        return StubListener()
    if name == "worker":
        return WorkerListener(conn, policy)
    if name == "hosted":
        if store is None:
            from . import storage
            store = storage.from_url()
        return HostedListener(llm, policy, store)
    raise ValueError(f"unknown listener adapter {name!r} (music.critic.listener.adapter)")
