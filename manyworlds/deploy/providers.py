"""GPU providers behind one small interface, so the autoscaler can start and stop
machines on SaladCloud, Vast.ai or RunPod and the policy decides which.

Each provider can: launch one worker machine under a name, list our live machines
(names start with "mw-"), and destroy one. Errors raise ProviderError with the
provider's response body (never request headers, which carry keys).

API shapes follow each provider's public docs as of Nov 2026:
  SaladCloud  https://api.salad.com/api/public   (container groups)
  Vast.ai     https://console.vast.ai/api/v0     (offers -> instances)
  RunPod      https://rest.runpod.io/v1          (pods)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

USER_AGENT = "manyworlds-deploy/0.2"  # some APIs sit behind Cloudflare, which rejects urllib's default


class ProviderError(RuntimeError):
    pass


class NoCapacity(ProviderError):
    """The provider has no machine matching the request right now; try another pool."""


SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "DATABASE_URL", "STORAGE_URL")


def redact(text: str) -> str:
    """Remove secret values from text: a provider may echo our request (including the
    workers' environment) in an error, and these errors reach public CI logs."""
    for k, v in os.environ.items():
        if v and len(v) >= 8 and any(m in k.upper() for m in SECRET_MARKERS):
            text = text.replace(v, f"<{k}>")
    return text


def http(method: str, url: str, headers: dict, body: dict | None = None, timeout: int = 60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise ProviderError(redact(f"{method} {url.split('?')[0]} -> {e.code}: "
                                   f"{e.read().decode(errors='replace')[:1500]}")) from None
    except urllib.error.URLError as e:
        raise ProviderError(f"{method} {url.split('?')[0]} -> {e.reason}") from None


@dataclass
class LaunchSpec:
    image: str                 # full image reference, e.g. docker.io/user/mw-worker-train:day1
    gpu_name: str              # the provider's own name for the GPU
    cpu: int
    ram_gb: int
    disk_gb: int
    max_usd_per_hour: float
    env: dict[str, str] = field(default_factory=dict)
    interruptible: bool = False
    priority: str = "batch"    # SaladCloud only


@dataclass
class Machine:
    name: str
    ref: str                   # provider's id for it
    usd_per_hour: float


class Provider(Protocol):
    name: str

    def configured(self) -> bool: ...
    def launch(self, name: str, spec: LaunchSpec) -> Machine: ...
    def list_live(self) -> dict[str, str]: ...     # name -> ref
    def destroy(self, name: str, ref: str | None) -> None: ...


# --- SaladCloud ---------------------------------------------------------------------

class Salad:
    name = "salad"

    def configured(self) -> bool:
        return all(os.environ.get(k) for k in ("SALAD_API_KEY", "SALAD_ORG", "SALAD_PROJECT"))

    def _req(self, method, path, body=None):
        base = f"https://api.salad.com/api/public/organizations/{os.environ['SALAD_ORG']}"
        path = path.replace("{project}", os.environ["SALAD_PROJECT"])
        return http(method, base + path, {"Salad-Api-Key": os.environ["SALAD_API_KEY"]}, body)

    def _gpu_ids(self, gpu_name: str) -> list[str]:
        res = self._req("GET", "/gpu-classes")
        items = res.get("items", res) if isinstance(res, dict) else res
        want = gpu_name.lower().strip()
        # Exact model match: "RTX 3090" must not also pick "RTX 3090 Ti" or laptop parts.
        ids = [g["id"] for g in items if g["name"].lower().split(" (")[0].strip() == want]
        if not ids:
            raise NoCapacity(f"SaladCloud has no GPU class named {gpu_name!r}")
        return ids

    def launch(self, name: str, spec: LaunchSpec) -> Machine:
        container = {"image": spec.image,
                     "resources": {"cpu": spec.cpu, "memory": spec.ram_gb * 1024,
                                   "gpu_classes": self._gpu_ids(spec.gpu_name),
                                   "storage_amount": spec.disk_gb * 1024 ** 3},
                     "environment_variables": spec.env}
        self._req("POST", "/projects/{project}/containers", {
            "name": name, "container": container, "replicas": 1, "restart_policy": "always",
            "autostart_policy": True, "priority": spec.priority})
        return Machine(name, name, spec.max_usd_per_hour)

    def list_live(self) -> dict[str, str]:
        res = self._req("GET", "/projects/{project}/containers") or {}
        items = res.get("items", res) if isinstance(res, dict) else res
        return {g["name"]: g["name"] for g in items if g["name"].startswith("mw-")}

    def destroy(self, name: str, ref: str | None) -> None:
        base = f"/projects/{{project}}/containers/{ref or name}"
        try:
            self._req("POST", f"{base}/stop")
        except ProviderError:
            pass  # already stopped; deleting is what ends billing
        self._req("DELETE", base)


# --- Vast.ai ------------------------------------------------------------------------

class Vast:
    name = "vast"
    API = "https://console.vast.ai/api/v0"
    API_V1 = "https://console.vast.ai/api/v1"

    def __init__(self, min_reliability: float = 0.98, verified_only: bool = True, offers_to_try: int = 3,
                 min_inet_down_mbps: float = 0):
        self.min_reliability = min_reliability
        self.verified_only = verified_only
        self.offers_to_try = offers_to_try
        self.min_inet_down = min_inet_down_mbps

    def configured(self) -> bool:
        return bool(os.environ.get("VAST_API_KEY"))

    def _req(self, method, path, body=None):
        return http(method, self.API + path, {"Authorization": f"Bearer {os.environ['VAST_API_KEY']}"}, body)

    def offers(self, spec: LaunchSpec) -> list[dict]:
        q = {"gpu_name": {"eq": spec.gpu_name}, "num_gpus": {"eq": 1}, "rentable": {"eq": True},
             "reliability": {"gte": self.min_reliability}, "cpu_ram": {"gte": spec.ram_gb * 1024},
             "disk_space": {"gte": spec.disk_gb}, "cpu_cores_effective": {"gte": spec.cpu},
             "dph_total": {"lte": spec.max_usd_per_hour}, "type": "ondemand",
             "order": [["dph_total", "asc"]], "limit": self.offers_to_try}
        if self.verified_only:
            q["verified"] = {"eq": True}
        if self.min_inet_down:
            q["inet_down"] = {"gte": self.min_inet_down}
        return (self._req("POST", "/bundles/", q) or {}).get("offers", [])

    def launch(self, name: str, spec: LaunchSpec) -> Machine:
        offers = self.offers(spec)
        if not offers:
            raise NoCapacity(f"no Vast offer for {spec.gpu_name} under ${spec.max_usd_per_hour}/h")
        last = None
        for o in offers:  # an offer can be taken between search and rent; try the next
            try:
                res = self._req("PUT", f"/asks/{o['id']}/", {
                    "image": spec.image, "label": name, "disk": spec.disk_gb,
                    # 'args' keeps the image's own ENTRYPOINT (mw) and passes CMD.
                    "runtype": "args", "args_str": "worker", "env": spec.env})
                if res and res.get("new_contract"):
                    return Machine(name, str(res["new_contract"]), float(o["dph_total"]))
                last = ProviderError(f"Vast rent of offer {o['id']} returned {res}")
            except ProviderError as e:
                last = e
        raise NoCapacity(f"could not rent any of {len(offers)} Vast offers: {last}")

    def list_live(self) -> dict[str, str]:
        # Listing moved to v1 (v0 returns 410) and is paginated; rent/destroy remain on v0.
        live, token = {}, None
        for _ in range(50):
            path = "/instances/" + (f"?next_token={urllib.parse.quote(token)}" if token else "")
            res = http("GET", self.API_V1 + path,
                       {"Authorization": f"Bearer {os.environ['VAST_API_KEY']}"}) or {}
            live.update({i["label"]: str(i["id"]) for i in res.get("instances", [])
                         if (i.get("label") or "").startswith("mw-")})
            token = res.get("next_token")
            if not token:
                return live
        raise ProviderError("Vast instance listing did not finish paginating")

    def destroy(self, name: str, ref: str | None) -> None:
        if not ref:
            ref = self.list_live().get(name)
            if not ref:
                return
        self._req("DELETE", f"/instances/{ref}/")


# --- RunPod -------------------------------------------------------------------------

class RunPod:
    name = "runpod"
    API = "https://rest.runpod.io/v1"

    def configured(self) -> bool:
        return bool(os.environ.get("RUNPOD_API_KEY"))

    def _req(self, method, path, body=None):
        return http(method, self.API + path, {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}, body)

    def launch(self, name: str, spec: LaunchSpec) -> Machine:
        try:
            pod = self._req("POST", "/pods", {
                "name": name, "imageName": spec.image, "gpuTypeIds": [spec.gpu_name], "gpuCount": 1,
                "cloudType": "COMMUNITY", "interruptible": spec.interruptible,
                "containerDiskInGb": spec.disk_gb, "volumeInGb": 0, "minRAMPerGPU": spec.ram_gb,
                "ports": [], "env": spec.env})
        except ProviderError as e:
            if "no " in str(e).lower() and ("available" in str(e).lower() or "instances" in str(e).lower()):
                raise NoCapacity(str(e)) from None
            raise
        price = float(pod.get("costPerHr") or spec.max_usd_per_hour)
        if price > spec.max_usd_per_hour:
            self.destroy(name, pod["id"])
            raise NoCapacity(f"RunPod charged ${price}/h for {spec.gpu_name}, over the ${spec.max_usd_per_hour} limit")
        return Machine(name, pod["id"], price)

    def list_live(self) -> dict[str, str]:
        pods = self._req("GET", "/pods") or []
        return {p["name"]: p["id"] for p in pods
                if (p.get("name") or "").startswith("mw-") and p.get("desiredStatus") != "TERMINATED"}

    def destroy(self, name: str, ref: str | None) -> None:
        if not ref:
            ref = self.list_live().get(name)
            if not ref:
                return
        self._req("DELETE", f"/pods/{ref}")


def all_providers(policy) -> dict[str, Provider]:
    v = policy.get("autoscale", "vast", default={})
    return {"salad": Salad(),
            "vast": Vast(float(v.get("min_reliability", 0.98)), bool(v.get("verified_only", True)),
                         int(v.get("offers_to_try", 3)), float(v.get("min_inet_down_mbps", 0))),
            "runpod": RunPod()}


# --- Stopping our own machine from inside it ----------------------------------------

def self_destroy() -> str | None:
    """Destroy the machine this process runs on, using the instance-scoped key the
    provider injects. Returns the provider name, or None when it isn't possible here
    (SaladCloud gives containers no such key; the autoscaler covers it)."""
    if os.environ.get("CONTAINER_ID") and os.environ.get("CONTAINER_API_KEY"):
        http("DELETE", f"{Vast.API}/instances/{os.environ['CONTAINER_ID']}/",
             {"Authorization": f"Bearer {os.environ['CONTAINER_API_KEY']}"})
        return "vast"
    if os.environ.get("RUNPOD_POD_ID") and os.environ.get("RUNPOD_API_KEY"):
        http("DELETE", f"{RunPod.API}/pods/{os.environ['RUNPOD_POD_ID']}",
             {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"})
        return "runpod"
    return None
