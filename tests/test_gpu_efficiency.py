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
    assert not runner.is_code_fault(RuntimeError("ComfyUI exited at start (1): CUDA error: no CUDA GPUs are available"))
    assert not runner.is_code_fault(AttributeError("'NoneType' object has no attribute 'json'"))
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
    voice = lambda: freed.append("voice")                                # one module's release()
    monkeypatch.setitem(runner.RELEASE, "voice.speak", voice)
    monkeypatch.setitem(runner.RELEASE, "voice.design", voice)
    w = runner.Worker.__new__(runner.Worker)
    w.policy, w.last_kind, w.ran = policy, None, []
    for kind in ("voice.speak", "voice.design", "talk.infinitetalk", "talk.infinitetalk"):
        w._switch_to(kind)
    assert freed == ["voice"]                                            # once, at the voice -> talk switch


def test_an_engine_loaded_earlier_in_the_family_is_freed_too(policy, monkeypatch):
    """Z-Image stays loaded through an image.narrated job (no engine of its own): the switch to ComfyUI frees it."""
    freed = []
    monkeypatch.setitem(runner.RELEASE, "image.zimage", lambda: freed.append("zimage"))
    monkeypatch.delitem(runner.RELEASE, "image.narrated", raising=False)
    w = runner.Worker.__new__(runner.Worker)
    w.policy, w.last_kind, w.ran = policy, None, []
    for kind in ("image.zimage", "image.narrated", "comfy.wan_i2v"):
        w._switch_to(kind)
    assert freed == ["zimage"]


def test_a_code_fault_quoted_from_a_subprocess_or_a_node_fails_once():
    # 2026-11-22..24: InfiniteTalk's and SVC's own tracebacks end in an AttributeError / TypeError, but reach the
    # worker as a RuntimeError, so each job ran three times (an L40S for 5-35 min a try) and failed the same way.
    sub = RuntimeError("InfiniteTalk exited 1:   File \"/opt/InfiniteTalk/wan/multitalk.py\", line 714, in gen\n"
                       "    x = context[0].to(context[0].dtype)\nAttributeError: 'list' object has no attribute 'dtype'\n")
    svc = RuntimeError("cli.inference_svc failed (1):     return forward_call(*args, **kwargs)\nTypeError: "
                       "LlamaAttention.forward() missing 1 required positional argument: 'position_embeddings'\n")
    node = RuntimeError("ComfyUI failed: node sample_high (WanVideoSampler): ImportError: cannot import name 'x'")
    assert runner.is_code_fault(sub) and runner.is_code_fault(svc) and runner.is_code_fault(node)
    # The last exception decides: a code-looking error the program recovered from, then a machine or a transient one
    assert not runner.is_code_fault(RuntimeError("LongCat exited 1: TypeError: x\nDuring handling...\n"
                                                 "torch.OutOfMemoryError: CUDA out of memory"))
    assert not runner.is_code_fault(RuntimeError("tool exited 1: TypeError: bad\nConnectionResetError: reset by peer"))
    # An attribute of None stays a transient empty answer, quoted or not
    assert not runner.is_code_fault(RuntimeError("tool exited 1:\nAttributeError: 'NoneType' object has no attribute 'json'"))
    # A plain failure with no exception named is retried as before
    assert not runner.is_code_fault(RuntimeError("InfiniteTalk timed out after 3000 s"))
