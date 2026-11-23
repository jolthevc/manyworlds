"""The brain's call journal and Message Batches (manyworlds.llm_batch): half-price calls for work nobody waits on."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel

from manyworlds import llm_batch, policy as policy_mod, queue
from manyworlds.brain import runtime
from manyworlds.llm import LLM, Deferred


class Score(BaseModel):
    score: float
    why: str = ""


def _ns(**kw):
    return type("NS", (), kw)


class FakeClaude:
    """Answers sync calls (messages.parse) and batches (messages.batches) with `answer(params) -> dict`."""

    def __init__(self, answer=lambda params: {"score": 7, "why": "fine"}, fail=()):
        self.parsed, self.batches, self.answer, self.fail = [], {}, answer, set(fail)
        usage = _ns(input_tokens=1000, output_tokens=100, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        outer = self

        class batches:
            @staticmethod
            def create(requests):
                bid = f"msgbatch_{len(outer.batches)}"
                outer.batches[bid] = {"requests": list(requests), "ended": False}
                return _ns(id=bid, processing_status="in_progress")

            @staticmethod
            def retrieve(bid):
                b = outer.batches[bid]
                return _ns(id=bid, processing_status="ended" if b["ended"] else "in_progress",
                           request_counts={"processing": 0 if b["ended"] else len(b["requests"])})

            @staticmethod
            def results(bid):
                for r in outer.batches[bid]["requests"]:
                    if r["custom_id"] in outer.fail:
                        yield _ns(custom_id=r["custom_id"], result=_ns(type="errored", error=_ns(type="api_error")))
                        continue
                    text = json.dumps(outer.answer(r["params"]))
                    msg = _ns(model=r["params"]["model"], usage=usage, stop_reason="end_turn",
                              content=[_ns(type="text", text=text)])
                    yield _ns(custom_id=r["custom_id"], result=_ns(type="succeeded", message=msg))

        class messages:
            @staticmethod
            def parse(**kw):
                outer.parsed.append(kw)
                return _ns(model=kw["model"], usage=usage, stop_reason="end_turn",
                           parsed_output=kw["output_format"](**outer.answer(kw)))

        messages.batches = batches
        self.messages = messages

    def end_all(self):
        for b in self.batches.values():
            b["ended"] = True


@pytest.fixture()
def bpolicy(policy):
    llm = {**policy.body["llm"], "batch": {**policy.body["llm"]["batch"], "gather_seconds": 0, "poll_seconds": 0},
           "roles": {**policy.body["llm"]["roles"], "brain": "claude-opus-5-5", "showrunner": "claude-sonnet-5-5"}}
    return policy_mod.from_dict({**policy.body, "llm": llm})


def _llm(conn, policy, client):
    llm = LLM(conn, policy, client=client)
    llm.batching = True
    return llm


def _usd(conn, agent="echo"):
    return float(conn.execute("SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE agent = %s",
                              (agent,)).fetchone()["u"])


def test_a_batched_role_waits_for_its_batch_then_replays_at_half_price(conn, bpolicy):
    client = FakeClaude()
    llm = _llm(conn, bpolicy, client)
    job = queue.enqueue(conn, "brain.echo", {})
    with llm.tagged("echo", job, attempt=1), pytest.raises(Deferred):
        llm.structured("brain", system="s", content="rate this", schema=Score)
    assert client.parsed == [] and _usd(conn) == 0
    assert llm_batch.flush(conn, bpolicy, client)["sent"] == 1
    params = client.batches["msgbatch_0"]["requests"][0]["params"]
    assert params["model"] == "claude-opus-5-5" and params["output_config"]["format"]["type"] == "json_schema"
    assert "fallbacks" not in json.dumps(params)                    # a batch request can't carry them
    assert llm_batch.flush(conn, bpolicy, client)["collected"] == 0   # not ended yet
    client.end_all()
    assert llm_batch.flush(conn, bpolicy, client)["collected"] == 1
    full = (1000 * 4.00 + 100 * 20.00) / 1e6
    assert _usd(conn) == pytest.approx(full / 2)                     # charged when it landed, at the batch price
    with llm.tagged("echo", job, attempt=1):                         # the rerun
        assert llm.structured("brain", system="s", content="rate this", schema=Score).score == 7
    with llm.tagged("echo", job, attempt=1):                         # and again: still nothing new to pay
        assert llm.structured("brain", system="s", content="rate this", schema=Score).score == 7
    assert client.parsed == [] and _usd(conn) == pytest.approx(full / 2)


def test_a_role_that_does_not_batch_is_journaled_and_replayed_for_nothing(conn, bpolicy):
    client = FakeClaude()
    llm = _llm(conn, bpolicy, client)
    job = queue.enqueue(conn, "brain.echo", {})
    for _ in range(3):
        with llm.tagged("echo", job, attempt=1):
            assert llm.structured("showrunner", system="s", content="ideas", schema=Score).score == 7
    assert len(client.parsed) == 1


def test_the_same_prompt_asked_twice_in_a_run_is_two_calls(conn, bpolicy):
    client = FakeClaude()
    llm = _llm(conn, bpolicy, client)
    job = queue.enqueue(conn, "brain.echo", {})
    with llm.tagged("echo", job, attempt=1):
        llm.structured("showrunner", system="s", content="sample", schema=Score)
        llm.structured("showrunner", system="s", content="sample", schema=Score)   # a second sample, not a replay
    assert len(client.parsed) == 2


def test_a_slightly_changed_prompt_matches_its_row_and_a_new_one_does_not(conn, bpolicy):
    client = FakeClaude()
    llm = _llm(conn, bpolicy, client)
    job = queue.enqueue(conn, "brain.echo", {})
    base = " ".join(f"word{i}" for i in range(60))
    with llm.tagged("echo", job, attempt=1):
        llm.structured("showrunner", system="s", content=base + " spent 1.20", schema=Score)
    with llm.tagged("echo", job, attempt=1):   # the snapshot's spend moved since: the same call
        llm.structured("showrunner", system="s", content=base + " spent 1.35", schema=Score)
    assert len(client.parsed) == 1
    with llm.tagged("echo", job, attempt=1):   # a different question
        llm.structured("showrunner", system="s", content="something else entirely", schema=Score)
    assert len(client.parsed) == 2


def test_a_retried_job_asks_afresh(conn, bpolicy):
    client = FakeClaude()
    llm = _llm(conn, bpolicy, client)
    job = queue.enqueue(conn, "brain.echo", {})
    with llm.tagged("echo", job, attempt=1):
        llm.structured("showrunner", system="s", content="ideas", schema=Score)
    with llm.tagged("echo", job, attempt=2):   # the first attempt failed after this answer: don't replay it
        llm.structured("showrunner", system="s", content="ideas", schema=Score)
    assert len(client.parsed) == 2


def test_an_errored_batch_call_is_answered_directly(conn, bpolicy):
    client = FakeClaude(fail={"mw1"})
    llm = _llm(conn, bpolicy, client)
    job = queue.enqueue(conn, "brain.echo", {})
    with llm.tagged("echo", job, attempt=1), pytest.raises(Deferred):
        llm.structured("brain", system="s", content="x", schema=Score)
    llm_batch.flush(conn, bpolicy, client)
    client.end_all()
    assert llm_batch.flush(conn, bpolicy, client)["failed"] == 1
    with llm.tagged("echo", job, attempt=1):
        assert llm.structured("brain", system="s", content="x", schema=Score).score == 7
    assert len(client.parsed) == 1
    row = conn.execute("SELECT mode, status FROM llm_journal").fetchone()
    assert (row["mode"], row["status"]) == ("batch_fallback", "done")


def test_a_call_that_waited_too_long_is_answered_directly(conn, bpolicy):
    client = FakeClaude()
    llm = _llm(conn, bpolicy, client)
    job = queue.enqueue(conn, "brain.echo", {})
    with llm.tagged("echo", job, attempt=1), pytest.raises(Deferred):
        llm.structured("brain", system="s", content="x", schema=Score)
    wait = float(bpolicy.get("llm", "batch", "max_wait_minutes"))
    conn.execute("UPDATE llm_journal SET created_at = now() - %s * interval '1 minute'", (wait + 1,))
    with llm.tagged("echo", job, attempt=1):
        assert llm.structured("brain", system="s", content="x", schema=Score).score == 7
    assert len(client.parsed) == 1


def test_calls_outside_the_brain_worker_or_a_job_are_not_journaled(conn, bpolicy):
    client = FakeClaude()
    llm = LLM(conn, bpolicy, client=client)            # batching off: the producer, reviewer, sandbox, tests
    job = queue.enqueue(conn, "brain.echo", {})
    with llm.tagged("echo", job, attempt=1):
        llm.structured("brain", system="s", content="x", schema=Score)
    on = _llm(conn, bpolicy, client)
    on.structured("brain", system="s", content="x", schema=Score)   # no job
    assert len(client.parsed) == 2 and conn.execute("SELECT count(*) n FROM llm_journal").fetchone()["n"] == 0


def test_the_runtime_gives_a_waiting_job_back_without_charging_an_attempt(conn, bpolicy):
    client = FakeClaude(answer=lambda params: {"ok": True})
    llm = _llm(conn, bpolicy, client)
    job = queue.enqueue(conn, "brain.echo", {"say": "hi"})
    [out] = runtime.tick(conn, bpolicy, llm, worker_id="w")
    assert out["status"] == "deferred"
    row = conn.execute("SELECT status, attempts, run_after FROM jobs WHERE id = %s", (job,)).fetchone()
    assert row["status"] == "queued" and row["attempts"] == 0 and row["run_after"] > datetime.now(timezone.utc)
    llm_batch.flush(conn, bpolicy, client)
    client.end_all()
    llm_batch.flush(conn, bpolicy, client)                       # the answer landed: the job is woken
    assert conn.execute("SELECT run_after <= now() AS due FROM jobs WHERE id = %s", (job,)).fetchone()["due"]
    [out] = runtime.tick(conn, bpolicy, llm, worker_id="w")
    assert out["status"] == "done"
    assert conn.execute("SELECT status, attempts FROM jobs WHERE id = %s", (job,)).fetchone()["attempts"] == 1


def test_a_refused_batch_marks_its_calls_failed_and_wakes_their_jobs(conn, bpolicy):
    client = FakeClaude()

    def boom(requests):
        raise RuntimeError("invalid_request")
    client.messages.batches.create = staticmethod(boom)
    llm = _llm(conn, bpolicy, client)
    job = queue.enqueue(conn, "brain.echo", {})
    with llm.tagged("echo", job, attempt=1), pytest.raises(Deferred):
        llm.structured("brain", system="s", content="x", schema=Score)
    assert llm_batch.flush(conn, bpolicy, client)["failed"] == 1
    assert conn.execute("SELECT status FROM llm_journal").fetchone()["status"] == "failed"


def test_old_rows_are_purged(conn, bpolicy):
    client = FakeClaude()
    llm = _llm(conn, bpolicy, client)
    job = queue.enqueue(conn, "brain.echo", {})
    with llm.tagged("echo", job, attempt=1):
        llm.structured("showrunner", system="s", content="x", schema=Score)
    conn.execute("UPDATE llm_journal SET created_at = now() - interval '30 days'")
    llm_batch.flush(conn, bpolicy, client, now=datetime.now(timezone.utc) + timedelta(seconds=1))
    assert conn.execute("SELECT count(*) n FROM llm_journal").fetchone()["n"] == 0


def test_status_reports_what_batching_saved(conn, bpolicy):
    client = FakeClaude()
    llm = _llm(conn, bpolicy, client)
    job = queue.enqueue(conn, "brain.echo", {})
    with llm.tagged("echo", job, attempt=1), pytest.raises(Deferred):
        llm.structured("brain", system="s", content="x", schema=Score)
    llm_batch.flush(conn, bpolicy, client)
    client.end_all()
    llm_batch.flush(conn, bpolicy, client)
    s = llm_batch.status(conn, bpolicy)
    assert s["saved_usd"] == pytest.approx(s["batched_usd"]) and s["batched_usd"] > 0


def test_shipped_batch_roles_exist_and_are_never_on_the_critical_path(raw_policy):
    roles = raw_policy.get("llm", "batch", "roles")
    assert set(roles) <= set(raw_policy.get("llm", "roles"))
    assert not {"director", "review", "review_confirm", "comment_reply"} & set(roles)
