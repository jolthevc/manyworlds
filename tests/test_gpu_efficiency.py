"""GPU dollars that do work (2026-11-22: 24% of billed GPU time ran a job): a code bug fails at once instead of
retrying on fresh machines, a card below a kind's memory floor never takes it, and a machine serving several
families runs one model's work back to back and frees the last engine's memory when it switches."""

from __future__ import annotations

from manyworlds import queue
from manyworlds.worker import runner

from test_worker import make_worker


def test_a_code_fault_fails_once_and_a_machine_fault_is_still_released():
    assert runner.is_code_fault(TypeError("process() takes no keyword arguments"))
    assert runner.is_code_fault(RuntimeError("ComfyUI failed: Failed to find C compiler"))
    assert not runner.is_code_fault(RuntimeError("CUDA error: out of memory"))
    assert runner.is_worker_fault(OSError("[Errno 122] Disk quota exceeded"))


def test_a_code_bug_uses_one_attempt_not_three(conn, database_url, store, policy):
    job = queue.enqueue(conn, "fake.image", {"fail": "code"}, max_attempts=3)
    make_worker(database_url, store, policy).run(max_jobs=1)
    row = conn.execute("SELECT status, attempts, error FROM jobs WHERE id = %s", (job,)).fetchone()
    assert row["status"] in ("failed", "dead") and row["attempts"] == 1 and row["error"].startswith("code fault")


def test_kinds_below_their_memory_floor_are_not_served(policy):
    floor = policy.get("job_kinds", "comfy.wan_animate2", "min_vram_gb")
    kinds = ["comfy.wan_animate2", "voice.speak"]
    assert runner.kinds_that_fit(kinds, policy, floor - 20) == ["voice.speak"]
    assert runner.kinds_that_fit(kinds, policy, floor) == kinds
    assert runner.kinds_that_fit(kinds, policy, None) == kinds          # no nvidia-smi: unknown, serve all


def test_a_worker_prefers_the_loaded_models_kind_within_a_priority(conn):
    older = queue.enqueue(conn, "fake.image", {}, priority=0)
    newer = queue.enqueue(conn, "fake.video", {}, priority=0)
    urgent = queue.enqueue(conn, "fake.image", {}, priority=5)
    got = [queue.claim(conn, ["fake.image", "fake.video"], "w", 60, prefer=["fake.video"]).id for _ in range(3)]
    assert got == [urgent, newer, older]                                 # priority first, then the loaded model


def test_switching_family_releases_the_last_engine(policy, monkeypatch):
    freed = []
    monkeypatch.setitem(runner.RELEASE, "voice.speak", lambda: freed.append("voice"))
    monkeypatch.setitem(runner.RELEASE, "voice.design", lambda: freed.append("voice"))
    w = runner.Worker.__new__(runner.Worker)
    w.policy, w.last_kind = policy, None
    for kind in ("voice.speak", "voice.design", "talk.infinitetalk", "talk.infinitetalk"):
        w._switch_to(kind)
    assert freed == ["voice"]                                            # once, at the voice -> talk switch
