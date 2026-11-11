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


def up(family: str, gpu: str, replicas: int, priority: str, tag: str) -> None:
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
    base = f"/organizations/{{org}}/projects/{{project}}/containers/{group_name(family, gpu)}"
    _req("POST", f"{base}/stop")
    _req("DELETE", base)
    print(f"stopped and deleted {group_name(family, gpu)}")


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
    for name in ("status", "down"):
        s = sub.add_parser(name)
        s.add_argument("family", choices=FAMILIES)
        s.add_argument("--gpu", default="RTX 4090")
    a = p.parse_args(argv)
    if a.cmd == "gpu-classes":
        for g in gpu_classes():
            print(json.dumps({k: g.get(k) for k in ("id", "name", "prices", "is_high_demand")}, default=str))
    elif a.cmd == "up":
        up(a.family, a.gpu, a.replicas, a.priority, a.tag)
    elif a.cmd == "status":
        status(a.family, a.gpu)
    else:
        down(a.family, a.gpu)


if __name__ == "__main__":
    main()
