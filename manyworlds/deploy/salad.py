"""SaladCloud launcher for GPU worker container groups (public API, no extra deps).

    python -m manyworlds.deploy.salad gpu-classes
    python -m manyworlds.deploy.salad up image --gpu "RTX 4090" --replicas 1 --priority batch
    python -m manyworlds.deploy.salad status image
    python -m manyworlds.deploy.salad down image        # stop and delete: billing stops

Needs SALAD_API_KEY, SALAD_ORG, SALAD_PROJECT and MW_REGISTRY (e.g. docker.io/<user>).
For a private Docker Hub repo also set DOCKERHUB_USERNAME and DOCKERHUB_TOKEN.
Workers get DATABASE_URL and the R2 settings as environment variables, so they write
straight into the same database and bucket as the control plane.

API shapes follow SaladCloud's public API docs as of late 2026; if a call is rejected,
the full error body is printed so the field can be corrected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

API = "https://api.salad.com/api/public"

FAMILIES = {
    # family: (image name, job kinds, pool name prefix, vCPU, RAM MB, disk GB)
    "image": ("mw-worker-image", "image.zimage", "salad", 4, 24576, 60),
    "tts": ("mw-worker-tts", "tts.kokoro", "salad", 2, 8192, 30),
    "video": ("mw-worker-video", "video.wan_i2v", "salad", 8, 49152, 120),
    "train": ("mw-worker-train", "train.lora", "salad", 8, 32768, 100),
}
PASS_THROUGH = ["DATABASE_URL", "STORAGE_URL", "S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "HF_TOKEN", "MW_IMAGE_MODEL",
                "MW_VIDEO_MODEL", "MW_VIDEO_DISTILL_LORA"]


def _req(method: str, path: str, body: dict | None = None):
    org, project = os.environ["SALAD_ORG"], os.environ["SALAD_PROJECT"]
    path = path.format(org=org, project=project)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{API}{path}", data=data, method=method, headers={
        "Salad-Api-Key": os.environ["SALAD_API_KEY"], "Content-Type": "application/json",
        "Accept": "application/json",
        # Cloudflare in front of the API rejects urllib's default agent (error 1010).
        "User-Agent": "manyworlds-deploy/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        sys.exit(f"SaladCloud API {method} {path} -> {e.code}: {e.read().decode()[:2000]}")


def gpu_classes() -> list[dict]:
    res = _req("GET", "/organizations/{org}/gpu-classes")
    return res.get("items", res) if isinstance(res, dict) else res


def _gpu_ids(match: str) -> list[str]:
    """Exact model match: 'RTX 3090' must not also pick 'RTX 3090 Ti' or laptop parts,
    or one benchmark would mix GPU types."""
    want = match.lower().strip()
    ids = [g["id"] for g in gpu_classes() if g["name"].lower().split(" (")[0].strip() == want]
    if not ids:
        sys.exit(f"no SaladCloud GPU class named exactly {match!r}; run gpu-classes")
    return ids


def gpu_slug(gpu: str) -> str:
    m = re.search(r"\b(?:RTX\s*)?(A?\d{4})\b", gpu, re.I)
    return m.group(1).lower() if m else re.sub(r"\W+", "", gpu).lower()


def group_name(family: str, gpu: str) -> str:
    # SaladCloud keeps a deleted group's name reserved for a while, so each run can carry
    # its own suffix (MW_SALAD_RUN), e.g. mw-image-4090-r2.
    run = os.environ.get("MW_SALAD_RUN", "").strip()
    return f"mw-{family}-{gpu_slug(gpu)}" + (f"-{run}" if run else "")


def register(family: str, gpu: str, replicas: int, priority: str, max_hours: float | None,
             max_usd: float | None) -> dict:
    """Record the group with a hard deadline and spend cap BEFORE creating it, so the
    cloud-side enforcer can always stop it. Refuses runs longer than policy allows."""
    from datetime import datetime, timedelta, timezone

    from .. import db
    from .. import policy as policy_mod

    pol = policy_mod.load()
    cfg = pol.get("gpu_runs", default={})
    hours = float(max_hours if max_hours is not None else cfg.get("default_max_hours", 2))
    limit = float(cfg.get("max_hours_limit", 12))
    if not 0 < hours <= limit:
        sys.exit(f"--max-hours must be between 0 and {limit} (policy gpu_runs.max_hours_limit)")
    usd_cap = float(max_usd if max_usd is not None else cfg.get("default_max_usd", 1.0))
    pool = f"salad_{priority}_{gpu_slug(gpu)}"
    price = float(pol.pool(pool)["usd_per_hour"])
    conn = db.connect()
    db.migrate(conn)
    now = datetime.now(timezone.utc)
    row = {"name": group_name(family, gpu), "pool": pool, "usd_per_hour": price, "replicas": replicas,
           "job_kinds": FAMILIES[family][1].split(","), "deadline": now + timedelta(hours=hours),
           "max_usd": usd_cap, "family": family}
    conn.execute(
        """
        INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, replicas, job_kinds, deadline, max_usd, family)
        VALUES (%(name)s, 'salad', %(pool)s, %(usd_per_hour)s, %(replicas)s, %(job_kinds)s, %(deadline)s, %(max_usd)s,
                %(family)s)
        ON CONFLICT (name) DO UPDATE SET deadline = EXCLUDED.deadline, max_usd = EXCLUDED.max_usd,
            started_at = now(), billed_until = now(), last_busy_at = now(), stopped_at = NULL, stop_reason = NULL
        """, row)
    return row


def up(family: str, gpu: str, replicas: int, priority: str, tag: str,
       max_hours: float | None = None, max_usd: float | None = None) -> None:
    reg = register(family, gpu, replicas, priority, max_hours, max_usd)
    print(f"registered {reg['name']}: deadline {reg['deadline']:%Y-%m-%d %H:%M} UTC, cap ${reg['max_usd']:.2f}, "
          f"~${reg['usd_per_hour'] * replicas:.2f}/hour")
    image, kinds, _, cpu, mem, disk = FAMILIES[family]
    env = {k: os.environ[k] for k in PASS_THROUGH if os.environ.get(k)}
    env.update({"MW_WORKER_KINDS": kinds, "MW_POOL": f"salad_{priority}_{gpu_slug(gpu)}"})
    container = {
        "image": f"{os.environ['MW_REGISTRY'].rstrip('/')}/{image}:{tag}",
        "resources": {"cpu": cpu, "memory": mem, "gpu_classes": _gpu_ids(gpu),
                      "storage_amount": disk * 1024 ** 3},
        "environment_variables": env,
    }
    if os.environ.get("DOCKERHUB_TOKEN"):
        container["registry_authentication"] = {"docker_hub": {
            "username": os.environ["DOCKERHUB_USERNAME"],
            "personal_access_token": os.environ["DOCKERHUB_TOKEN"]}}
    body = {"name": group_name(family, gpu), "container": container, "replicas": replicas,
            "restart_policy": "always", "autostart_policy": True, "priority": priority}
    res = _req("POST", "/organizations/{org}/projects/{project}/containers", body)
    print(json.dumps({"created": res.get("name"), "status": res.get("current_state")}, indent=2, default=str))


def status(family: str, gpu: str) -> None:
    base = f"/organizations/{{org}}/projects/{{project}}/containers/{group_name(family, gpu)}"
    g = _req("GET", base)
    inst = _req("GET", f"{base}/instances")
    print(json.dumps({"group": group_name(family, gpu), "state": g.get("current_state"),
                      "replicas": g.get("replicas"), "instances": (inst or {}).get("instances", inst)},
                     indent=2, default=str))


def down(family: str, gpu: str) -> None:
    name = group_name(family, gpu)
    base = f"/organizations/{{org}}/projects/{{project}}/containers/{name}"
    _req("POST", f"{base}/stop")
    _req("DELETE", base)
    try:  # final meter + mark stopped, so the ledger holds the full billed time
        from datetime import datetime, timezone

        from .. import db, policy as policy_mod
        conn = db.connect()
        g = conn.execute("SELECT * FROM gpu_groups WHERE name = %s AND stopped_at IS NULL", (name,)).fetchone()
        if g:
            from .. import ledger
            now = datetime.now(timezone.utc)
            secs = max((now - g["billed_until"]).total_seconds(), 0)
            usd = secs / 3600 * float(g["usd_per_hour"]) * int(g["replicas"])
            ledger.record(conn, category="gpu_billed", provider=g["pool"], quantity=round(secs, 1),
                          unit="instance_second", usd=usd, policy_version=policy_mod.load().version)
            conn.execute("UPDATE gpu_groups SET billed_until = %s, billed_usd = billed_usd + %s, "
                         "stopped_at = %s, stop_reason = 'manual' WHERE name = %s", (now, usd, now, name))
    except Exception as e:  # noqa: BLE001 - stopping must never fail because of metering
        print(f"warning: could not meter final time for {name}: {e}")
    print(f"stopped and deleted {name}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="salad")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpu-classes")
    u = sub.add_parser("up")
    u.add_argument("family", choices=FAMILIES)
    u.add_argument("--gpu", default="RTX 4090")
    u.add_argument("--replicas", type=int, default=1)
    u.add_argument("--priority", default="batch", choices=["batch", "low", "medium", "high"])
    u.add_argument("--tag", default="day1")
    u.add_argument("--max-hours", type=float, help="hard stop (default policy gpu_runs.default_max_hours)")
    u.add_argument("--max-usd", type=float, help="spend cap (default policy gpu_runs.default_max_usd)")
    for name in ("status", "down"):
        s = sub.add_parser(name)
        s.add_argument("family", choices=FAMILIES)
        s.add_argument("--gpu", default="RTX 4090")
    a = p.parse_args(argv)
    if a.cmd == "gpu-classes":
        for g in gpu_classes():
            print(json.dumps({k: g.get(k) for k in ("id", "name", "prices", "is_high_demand")}, default=str))
    elif a.cmd == "up":
        up(a.family, a.gpu, a.replicas, a.priority, a.tag, a.max_hours, a.max_usd)
    elif a.cmd == "status":
        status(a.family, a.gpu)
    else:
        down(a.family, a.gpu)


if __name__ == "__main__":
    main()
