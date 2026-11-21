"""The owner hears about what needs them, once, on their phone (manyworlds/notify.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds import notify, policy as policy_mod
from manyworlds.deploy import autoscale

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


class Phone:
    def __init__(self, fail_on: str | None = None):
        self.got: list[tuple[str, str, str]] = []
        self.fail_on = fail_on

    def __call__(self, title, body, priority):
        if self.fail_on and self.fail_on in title:
            raise OSError("network down")
        self.got.append((title, body, priority))


def _event(conn, type_, at=NOW, **payload):
    conn.execute("INSERT INTO events (type, payload, routed_to, created_at) VALUES (%s, %s, 'owner', %s)",
                 (type_, Jsonb(payload), at))


def test_each_owner_event_is_pushed_once(conn, policy):
    _event(conn, "publish_held", reason="instagram said no")
    conn.execute("INSERT INTO events (type, routed_to, created_at) VALUES ('breakout', 'analyst', %s)", (NOW,))
    phone = Phone()
    assert [s["type"] for s in notify.deliver(conn, policy, send=phone, now=NOW)] == ["publish_held"]
    assert phone.got == [("Manyworlds: publish held", '{"reason": "instagram said no"}', "high")]
    assert notify.deliver(conn, policy, send=phone, now=NOW) == [] and len(phone.got) == 1


def test_pending_approvals_are_pushed(conn, policy):
    conn.execute("INSERT INTO approvals (kind, subject, reason, requested_by, created_at) "
                 "VALUES ('launch', '{}', 'best thesis', 'strategist', %s)", (NOW,))
    phone = Phone()
    notify.deliver(conn, policy, send=phone, now=NOW)
    assert phone.got[0][0] == "Manyworlds: approval launch" and "best thesis" in phone.got[0][1]


def test_without_a_channel_nothing_is_sent_or_marked(conn, policy, monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    _event(conn, "brain_step_failed")
    assert notify.deliver(conn, policy, now=NOW) == []
    assert len(notify.pending(conn, policy, NOW)) == 1


def test_a_burst_becomes_one_summary(conn, policy):
    for _ in range(7):
        _event(conn, "brain_step_failed")
    _event(conn, "publish_held")
    phone = Phone()
    assert len(notify.deliver(conn, policy, send=phone, now=NOW)) == 8
    assert len(phone.got) == 1 and "8 things" in phone.got[0][0] and phone.got[0][2] == "high"
    assert "brain_step_failed x7" in phone.got[0][1]


def test_old_items_are_never_pushed(conn, policy):
    _event(conn, "publish_held", at=NOW - timedelta(days=3))
    assert notify.deliver(conn, policy, send=Phone(), now=NOW) == []


def test_quiet_types_are_skipped(conn, policy):
    pol = policy_mod.from_dict({**policy.body, "notify": {**policy.body["notify"], "quiet": ["account_needed"]}})
    _event(conn, "account_needed")
    assert notify.deliver(conn, pol, send=Phone(), now=NOW) == []


def test_a_failed_send_is_retried_next_round(conn, policy):
    _event(conn, "publish_held")
    _event(conn, "launch_failed")
    phone = Phone(fail_on="launch")
    assert [s["type"] for s in notify.deliver(conn, policy, send=phone, now=NOW)] == ["publish_held"]
    assert [s["type"] for s in notify.deliver(conn, policy, send=Phone(), now=NOW)] == ["launch_failed"]


def test_secrets_never_reach_the_phone(conn, policy, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_supersecretvalue123")
    _event(conn, "autoscaler_failed", error="401 for key rp_supersecretvalue123")
    phone = Phone()
    notify.deliver(conn, policy, send=phone, now=NOW)
    assert "supersecret" not in phone.got[0][1] and "<RUNPOD_API_KEY>" in phone.got[0][1]


def test_autoscaler_trouble_becomes_owner_events_once(conn, policy):
    actions = [{"group": "mw-image-3090-runpod-1", "action": "stopped", "reason": "max_usd"},
               {"group": "mw-image-3090-runpod-2", "action": "stopped", "reason": "idle"},
               {"action": "provider_unfunded", "provider": "vast"},
               {"action": "not_launching", "reason": "worker_credentials", "missing": ["WORKER_DATABASE_URL"]}]
    assert notify.from_autoscaler(conn, policy, actions, NOW) == 3
    assert notify.from_autoscaler(conn, policy, actions, NOW) == 0          # same trouble, same day: once
    types = sorted(r["type"] for r in conn.execute("SELECT type FROM events"))
    assert types == ["machine_stopped", "not_launching", "provider_unfunded"]


def test_the_autoscaler_loop_notifies_and_survives_a_broken_channel(conn, policy, monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "test-topic")

    def broken(*a, **k):
        raise OSError("ntfy down")
    monkeypatch.setattr(notify.urllib.request, "urlopen", broken)
    out = autoscale.notify_owner(conn, policy, [{"action": "provider_unfunded", "provider": "vast"}], NOW)
    assert out == []                                     # nothing sent, nothing raised
    assert len(notify.pending(conn, policy, NOW)) == 1   # still waiting to be told


def test_ntfy_posts_to_the_topic(policy, monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "secret-topic")
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout):
        seen.update(url=req.full_url, data=req.data, prio=req.get_header("Priority"))
        return Resp()
    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    notify.ntfy_sender(policy)("Manyworlds: x", "body", "high")
    assert seen == {"url": "https://ntfy.sh/secret-topic", "data": b"body", "prio": "high"}
