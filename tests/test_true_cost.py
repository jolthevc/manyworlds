"""A machine's true cost, not just its hourly price. 2026-11-20: Vast billed $40.40 over a week where the
ledger metered $4.27 of GPU time; $29.11 was download bandwidth (hosts charge per GB; every boot downloads
20-94 GB of weights). Offers are now ranked by expected total cost, a machine's spending cap counts its
expected download, and the provider's itemised charges are reconciled into the ledger and machine costs."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from manyworlds import ledger
from manyworlds.deploy import providers, reconcile
from manyworlds.deploy.enforce import tick as enforce_tick
from manyworlds.deploy.providers import LaunchSpec, Vast

from test_enforce import T0, FakeAPI, register

SPEC = LaunchSpec(image="i", gpu_name="RTX 3090", cpu=4, ram_gb=24, disk_gb=80, max_usd_per_hour=0.3, download_gb=25)


def test_vast_offers_are_ranked_by_hourly_plus_download_cost(policy, monkeypatch):
    cfg = policy.get("autoscale", "vast")
    v = Vast(max_inet_down_usd_per_gb=float(cfg["max_inet_down_usd_per_gb"]),
             expected_hours=float(cfg["expected_hours"]), offers_to_try=2)
    sent = {}
    offers = [{"id": 1, "dph_total": 0.14, "inet_down_cost": 0.0156},     # cheap hourly, pricey download: $0.53
              {"id": 2, "dph_total": 0.19, "inet_down_cost": 0.0006},     # $0.20
              {"id": 3, "dph_total": 0.17, "inet_down_cost": 0.004}]      # $0.27
    monkeypatch.setattr(Vast, "_req", lambda self, m, p, body=None: sent.update(body or {}) or {"offers": offers})
    got = v.offers(SPEC)
    assert sent["inet_down_cost"] == {"lte": float(cfg["max_inet_down_usd_per_gb"])}
    assert [o["id"] for o in got] == [2, 3]
    assert abs(v.expected_cost(got[0], SPEC) - (0.19 * float(cfg["expected_hours"]) + 25 * 0.0006)) < 1e-9


def test_a_vast_launch_reports_its_expected_download_cost(monkeypatch):
    v = Vast(max_inet_down_usd_per_gb=0.01, expected_hours=1.0)
    monkeypatch.setattr(Vast, "offers", lambda self, spec: [{"id": 7, "dph_total": 0.2, "inet_down_cost": 0.004}])
    monkeypatch.setattr(Vast, "_req", lambda self, m, p, body=None: {"new_contract": 99})
    m = v.launch("mw-x", SPEC)
    assert m.ref == "99" and abs(m.upfront_usd - 25 * 0.004) < 1e-9


def test_the_spending_cap_counts_the_expected_download(conn, policy):
    register(conn, "mw-image-3090-a", hours=12, max_usd=0.5)
    conn.execute("UPDATE gpu_groups SET est_extra_usd = 0.45 WHERE name = 'mw-image-3090-a'")
    from manyworlds import queue
    queue.enqueue(conn, "image.zimage", {})
    api = FakeAPI(["mw-image-3090-a"])
    actions = enforce_tick(conn, policy, api, now=T0 + timedelta(minutes=30))   # 0.08 metered + 0.45 expected
    assert api.stopped == ["mw-image-3090-a"] and actions[0]["reason"] == "max_usd"


class ChargingVast:
    name = "vast"

    def __init__(self, rows):
        self.rows = rows

    def configured(self):
        return True

    def charges(self, since, until):
        return self.rows


def test_reconciling_writes_the_providers_actual_charges_once(conn, policy):
    register(conn, "mw-talk-4090-vast-081b", hours=1, max_usd=1.25)
    conn.execute("UPDATE gpu_groups SET provider = 'vast', billed_usd = 0.16, stopped_at = %s, "
                 "provider_ref = '61596442' WHERE name = 'mw-talk-4090-vast-081b'", (T0 + timedelta(minutes=22),))
    day = date(2026, 11, 12)
    rows = [{"ref": "61596442", "label": "mw-talk-4090-vast-081b", "day": day, "component": c, "usd": u}
            for c, u in (("gpu", 0.096), ("disk", 0.04), ("bwd", 4.93), ("bwu", 0.04))]
    ps = {"vast": ChargingVast(rows)}
    before = ledger.bill_total(conn)
    out = reconcile.run(conn, policy, ps, now=T0 + timedelta(days=1))   # stopped long before: settled
    g = conn.execute("SELECT actual_usd, cost_usd FROM gpu_groups WHERE name = 'mw-talk-4090-vast-081b'").fetchone()
    assert abs(float(g["actual_usd"]) - 5.106) < 1e-9 and abs(float(g["cost_usd"]) - 5.106) < 1e-9
    assert abs((ledger.bill_total(conn) - before) - (5.106 - 0.16)) < 1e-6     # the unmetered difference, once
    assert out["groups"] == 1
    reconcile.run(conn, policy, ps, now=T0 + timedelta(days=1))
    assert abs((ledger.bill_total(conn) - before) - (5.106 - 0.16)) < 1e-6
    by = {r["component"]: float(r["usd"]) for r in conn.execute("SELECT component, usd FROM provider_charges").fetchall()}
    assert by["bwd"] == 4.93


def test_pool_scores_use_true_cost(conn, policy):
    from manyworlds.deploy import scoring
    register(conn, "mw-image-3090-vast-z", hours=1, max_usd=10)
    conn.execute("UPDATE gpu_groups SET provider = 'vast', pool = 'vast_3090', family = 'image', billed_usd = 0.1, "
                 "actual_usd = 2.0, stopped_at = now() WHERE name = 'mw-image-3090-vast-z'")
    ms = scoring.machine_stats(conn, "image", ["vast_3090"], datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert ms[0]["billed_usd"] == 2.0


def test_the_autoscaler_reconciles_at_most_every_period(conn, policy):
    from manyworlds.deploy import autoscale
    calls = []

    class P(ChargingVast):
        def charges(self, since, until):
            calls.append(1)
            return []
    assert autoscale.maybe_reconcile(conn, policy, {"vast": P([])}) == []
    autoscale.maybe_reconcile(conn, policy, {"vast": P([])})
    assert len(calls) == 1


def test_backfilled_charges_count_on_the_day_they_were_incurred(conn, policy):
    from manyworlds.deploy.enforce import billed_today
    register(conn, "mw-old-vast", hours=1, max_usd=1)
    conn.execute("UPDATE gpu_groups SET provider = 'vast', billed_usd = 0.1, stopped_at = now() - interval '5 days', "
                 "provider_ref = '1' WHERE name = 'mw-old-vast'")
    rows = [{"ref": "1", "label": "mw-old-vast", "day": date(2026, 11, 15), "component": "bwd", "usd": 30.0}]
    reconcile.run(conn, policy, {"vast": ChargingVast(rows)}, now=datetime.now(timezone.utc))
    assert billed_today(conn) < 1                     # a week-old bill must not block today's machines


def test_vasts_real_charge_format_parses_into_components():
    import json
    from pathlib import Path
    from manyworlds.deploy.providers import parse_vast_charges
    res = json.loads((Path(__file__).parent / "fixtures" / "vast_charges_tree.json").read_text())
    rows = parse_vast_charges(res)
    one = {r["component"]: r["usd"] for r in rows if r["ref"] == "61596442"}
    assert one == {"gpu": 0.096, "disk": 0.036, "bwd": 4.927, "bwu": 0.035}
    assert {r["label"] for r in rows} == {"mw-talk-4090-vast-081b", "mw-talk-4090-vast-41a6"}


def test_split_charge_rows_add_up(monkeypatch):
    v = Vast()
    day = {"start": 1795132800, "type": "instance", "source": "instance-9", "metadata": {"label": "mw-a"}}
    pages = [{"results": [{**day, "items": [{"type": "bwd", "amount": 1.0}]}], "next_token": "t"},
             {"results": [{**day, "items": [{"type": "bwd", "amount": 2.0}]}], "next_token": None}]
    monkeypatch.setattr(Vast, "_req", lambda self, m, p, body=None: pages.pop(0))
    rows = v.charges(datetime(2026, 11, 20, tzinfo=timezone.utc), datetime(2026, 11, 21, tzinfo=timezone.utc))
    assert [(r["component"], r["usd"]) for r in rows] == [("bwd", 3.0)]


def test_a_machine_is_reconciled_only_after_it_settles(conn, policy):
    register(conn, "mw-fresh-vast", hours=1, max_usd=1)
    now = datetime.now(timezone.utc)
    conn.execute("UPDATE gpu_groups SET provider = 'vast', billed_usd = 0.3, started_at = %s, stopped_at = %s, "
                 "provider_ref = '5' WHERE name = 'mw-fresh-vast'", (now - timedelta(hours=1), now - timedelta(minutes=5)))
    rows = [{"ref": "5", "label": "mw-fresh-vast", "day": now.date(), "component": "gpu", "usd": 0.1}]   # not posted yet
    out = reconcile.run(conn, policy, {"vast": ChargingVast(rows)}, now=now)
    assert out["groups"] == 0 and conn.execute("SELECT actual_usd FROM gpu_groups").fetchone()["actual_usd"] is None


def test_todays_cap_sees_expected_downloads_of_running_machines(conn, policy):
    from manyworlds.deploy.enforce import billed_today
    register(conn, "mw-run-vast", hours=1, max_usd=1, started=datetime.now(timezone.utc))
    conn.execute("UPDATE gpu_groups SET est_extra_usd = 0.7 WHERE name = 'mw-run-vast'")
    assert billed_today(conn) >= 0.7
