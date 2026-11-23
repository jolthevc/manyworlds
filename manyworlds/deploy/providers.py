"""GPU providers behind one small interface, so the autoscaler can start and stop
machines on SaladCloud, Vast.ai or RunPod and the policy decides which.

Each provider can: launch one worker machine under a name, list our live machines
(names start with "mw-"), and destroy one. Errors raise ProviderError with the
provider's response body (never request headers, which carry keys).

API shapes follow each provider's public docs as of Nov 2026:
  SaladCloud  https://api.salad.com/api/public   (container groups)
  Vast.ai     https://console.vast.ai/api/v0     (offers -> instances)
  RunPod      https://api.runpod.io/v2           (pods; v1 retired 2026-11-15)
  Fly         http://flaps/v1 over /.fly/api      (Machines API from inside the app: the finisher, started not created)
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
    min_cuda: float | None = None  # the image's CUDA build; hosts with older drivers can't run it
    cloud: str = "COMMUNITY"       # RunPod only: COMMUNITY or SECURE (per pool)
    download_gb: float = 0.0       # what a machine downloads at boot (image + weights): Vast bills it per GB
    # RunPod only: {id, data_center, path}: a network volume holding model weights across machines, so each boot
    # loads them instead of downloading ~82 GB (talk). Machines then run in the volume's data center.
    network_volume: dict | None = None
    process_group: str | None = None   # Fly only: the app's process group whose stopped machine is started


@dataclass
class Machine:
    name: str
    ref: str                   # provider's id for it
    usd_per_hour: float
    upfront_usd: float = 0.0   # expected non-hourly charges (Vast: downloading image and weights)


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

def parse_vast_charges(res: dict) -> list[dict]:
    """Rows of Vast's GET /charges?format=tree: one result per instance and day, items gpu/disk/bwd/bwu with
    positive amounts (credits, if any, keep their sign)."""
    from datetime import datetime as _dt, timezone as _tz
    out = []
    for r in res.get("results") or []:
        if r.get("type") not in (None, "instance"):
            continue
        ref = (r.get("source") or "").removeprefix("instance-")
        label = (r.get("metadata") or {}).get("label")
        day = _dt.fromtimestamp(int(r.get("start") or 0), _tz.utc).date()
        for it in r.get("items") or []:
            out.append({"ref": ref, "label": label, "day": day, "component": it.get("type") or "other",
                        "usd": float(it.get("amount") or 0)})
    return out


class Vast:
    name = "vast"
    API = "https://console.vast.ai/api/v0"
    API_V1 = "https://console.vast.ai/api/v1"

    def __init__(self, min_reliability: float = 0.98, verified_only: bool = True, offers_to_try: int = 3,
                 min_inet_down_mbps: float = 0, exclude_countries: list[str] | None = None,
                 max_inet_down_usd_per_gb: float | None = None, expected_hours: float = 1.0,
                 candidates: int = 50):
        # Vast hosts set their own bandwidth prices (20x apart on 2026-11-20); offers are ranked by
        # expected_hours of rent plus the boot download, never by hourly price alone.
        self.max_inet_down = max_inet_down_usd_per_gb
        self.expected_hours = expected_hours
        self.candidates = candidates
        self.exclude_countries = list(exclude_countries or [])
        self.min_reliability = min_reliability
        self.verified_only = verified_only
        self.offers_to_try = offers_to_try
        self.min_inet_down = min_inet_down_mbps
        self.prices: dict[str, float] = {}  # label -> current $/hour, filled by list_live

    def configured(self) -> bool:
        return bool(os.environ.get("VAST_API_KEY"))

    def _req(self, method, path, body=None):
        return http(method, self.API + path, {"Authorization": f"Bearer {os.environ['VAST_API_KEY']}"}, body)

    def offers(self, spec: LaunchSpec) -> list[dict]:
        q = {"gpu_name": {"eq": spec.gpu_name}, "num_gpus": {"eq": 1}, "rentable": {"eq": True},
             "reliability": {"gte": self.min_reliability}, "cpu_ram": {"gte": spec.ram_gb * 1024},
             "disk_space": {"gte": spec.disk_gb}, "cpu_cores_effective": {"gte": spec.cpu},
             "dph_total": {"lte": spec.max_usd_per_hour}, "type": "ondemand",
             "order": [["dph_total", "asc"]], "limit": self.candidates}
        if self.max_inet_down is not None:
            q["inet_down_cost"] = {"lte": self.max_inet_down}
        if self.verified_only:
            q["verified"] = {"eq": True}
        if self.min_inet_down:
            q["inet_down"] = {"gte": self.min_inet_down}
        if self.exclude_countries:
            q["geolocation"] = {"notin": self.exclude_countries}
        if spec.min_cuda:
            q["cuda_max_good"] = {"gte": spec.min_cuda}
        offers = (self._req("POST", "/bundles/", q) or {}).get("offers", [])
        offers.sort(key=lambda o: self.expected_cost(o, spec))
        return offers[:self.offers_to_try]

    def expected_cost(self, offer: dict, spec: LaunchSpec) -> float:
        return float(offer["dph_total"]) * self.expected_hours + spec.download_gb * float(offer.get("inet_down_cost") or 0)

    def charges(self, since, until) -> list[dict]:
        """Vast's itemised charges per instance and day: gpu, disk, bwd (download), bwu (upload)."""
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        filters = urllib.parse.quote(_json.dumps({"day": {"gte": int(since.timestamp()), "lte": int(until.timestamp())}}))
        sums: dict[tuple, dict] = {}
        token = None
        for _ in range(20):
            path = f"/charges/?select_filters={filters}&format=tree&limit=500" + (
                f"&after_token={urllib.parse.quote(token)}" if token else "")
            res = self._req("GET", path) or {}
            for r in parse_vast_charges(res):
                k = (r["ref"], r["day"], r["component"])   # summed, so split rows add up instead of overwriting
                if k in sums:
                    sums[k]["usd"] += r["usd"]
                else:
                    sums[k] = r
            token = res.get("next_token")
            if not token:
                return list(sums.values())
        raise ProviderError("Vast charges listing did not finish paginating")

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
                    return Machine(name, str(res["new_contract"]), float(o["dph_total"]),
                                   upfront_usd=spec.download_gb * float(o.get("inet_down_cost") or 0))
                last = ProviderError(f"Vast rent of offer {o['id']} returned {res}")
            except ProviderError as e:
                last = e
        raise NoCapacity(f"could not rent any of {len(offers)} Vast offers: {last}")

    def list_live(self) -> dict[str, str]:
        # Listing moved to v1 (v0 returns 410) and is paginated; rent/destroy remain on v0.
        live, prices, token = {}, {}, None
        for _ in range(50):
            path = "/instances/" + (f"?next_token={urllib.parse.quote(token)}" if token else "")
            res = http("GET", self.API_V1 + path,
                       {"Authorization": f"Bearer {os.environ['VAST_API_KEY']}"}) or {}
            for i in res.get("instances", []):
                if (i.get("label") or "").startswith("mw-"):
                    live[i["label"]] = str(i["id"])
                    if i.get("dph_total"):
                        prices[i["label"]] = float(i["dph_total"])
            token = res.get("next_token")
            if not token:
                self.prices = prices
                return live
        raise ProviderError("Vast instance listing did not finish paginating")

    def destroy(self, name: str, ref: str | None) -> None:
        if not ref:
            ref = self.list_live().get(name)
            if not ref:
                return
        self._req("DELETE", f"/instances/{ref}/")


# --- RunPod -------------------------------------------------------------------------

# v2 answers 400 both for no capacity and for rule violations; only these phrases mean "try another pool".
NO_CAPACITY_PHRASES = ("no instances currently available", "no longer any instances available", "no available capacity", "could not be placed",
                       "not enough capacity", "insufficient capacity")


class RunPod:
    """RunPod REST v2 (https://api.runpod.io/v2). v1 at rest.runpod.io was retired on 2026-11-15."""
    MAX_PAGES = 50
    name = "runpod"
    API = "https://api.runpod.io/v2"

    def __init__(self, cuda_versions: list[str] | None = None):
        self.cuda_versions = [str(v) for v in (cuda_versions or [])]

    def configured(self) -> bool:
        return bool(os.environ.get("RUNPOD_API_KEY"))

    def _req(self, method, path, body=None):
        return http(method, self.API + path, {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}, body)

    def launch(self, name: str, spec: LaunchSpec) -> Machine:
        gpu = {"id": spec.gpu_name, "count": 1, "minRamPerGpu": spec.ram_gb, **self._cuda(spec)}
        body = {"name": name, "image": spec.image, "cloud": spec.cloud, "disk": spec.disk_gb, "env": spec.env,
                "ports": [], "gpu": gpu}
        if spec.network_volume:   # REST v2: mounts.network [{volumeId, path}]; the pod runs where the volume is
            v = spec.network_volume
            body["mounts"] = {"network": [{"volumeId": v["id"], "path": v["path"]}]}
            body["dataCenterIds"] = [v["data_center"]]
        try:
            pod = self._req("POST", "/pods", body)
        except ProviderError as e:
            text = str(e).lower()
            if " -> 402" in text:   # prepaid balance used up: the autoscaler skips the provider
                raise ProviderError(f"insufficient funds (402): {e}") from None
            # v2 returns 400 both for no capacity and for rule violations; only the detail tells them apart.
            if " -> 400" in text and any(m in text for m in NO_CAPACITY_PHRASES):
                raise NoCapacity(str(e)) from None
            raise
        price = float(pod.get("cost") or spec.max_usd_per_hour)
        if price > spec.max_usd_per_hour:
            self.destroy(name, pod["id"])
            raise NoCapacity(f"RunPod charged ${price}/h for {spec.gpu_name}, over the ${spec.max_usd_per_hour} limit")
        return Machine(name, pod["id"], price)

    def _cuda(self, spec: LaunchSpec) -> dict:
        """Pods on hosts whose driver predates the image's CUDA fail with 'CUDA unknown error'. A floor, not a
        list: on 2026-11-20 every free host reported CUDA 13.2, which an exact list of 12.8/12.9/13.0 excluded."""
        return {"minCudaVersion": f"{spec.min_cuda:.1f}"} if spec.min_cuda else {}

    def list_live(self) -> dict[str, str]:
        live, cursor, seen = {}, None, set()
        for _ in range(self.MAX_PAGES):
            page = self._req("GET", "/pods" + (f"?cursor={urllib.parse.quote(cursor)}" if cursor else "")) or {}
            for p in page.get("pods") or []:
                if (p.get("name") or "").startswith("mw-") and p.get("status") != "TERMINATED":
                    live[p["name"]] = p["id"]
            nxt = page.get("pagination") or {}
            if not nxt.get("hasNextPage") or not nxt.get("nextCursor") or nxt["nextCursor"] in seen:
                return live
            cursor = nxt["nextCursor"]
            seen.add(cursor)
        raise ProviderError(f"RunPod pod list did not end after {self.MAX_PAGES} pages")

    def destroy(self, name: str, ref: str | None) -> None:
        if not ref:
            ref = self.list_live().get(name)
            if not ref:
                return
        self._req("DELETE", f"/pods/{ref}")


# --- Fly Machines (the finisher) ----------------------------------------------------

FLY_API_SOCKET = "/.fly/api"     # Fly's in-machine proxy to the Machines API, scoped to this app (no token here)
FLY_RUN_KEY = "mw_group"         # machine metadata: the gpu_groups row of the run the autoscaler started it for
FLY_LIVE_STATES = ("starting", "started")
FLY_STOPPED_STATES = ("stopped", "suspended")


def unix_transport(socket_path: str, timeout: float = 60):
    """HTTP over a unix socket (Fly's /.fly/api): call(method, path, body) -> the JSON answer. ProviderError on any
    failure, with the response body (never request headers)."""
    def call(method: str, path: str, body: dict | None = None):
        import http.client
        import socket

        class _Conn(http.client.HTTPConnection):
            def connect(self):
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(self.timeout)
                s.connect(socket_path)
                self.sock = s

        conn = _Conn("flaps", timeout=timeout)
        try:
            conn.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                         headers={"Content-Type": "application/json", "Accept": "application/json",
                                  "User-Agent": USER_AGENT})
            r = conn.getresponse()
            status, raw = r.status, r.read()
        except OSError as e:
            raise ProviderError(f"{method} {path} -> {e}") from None
        finally:
            conn.close()
        if status >= 300:
            raise ProviderError(redact(f"{method} {path} -> {status}: {raw.decode(errors='replace')[:1500]}"))
        return json.loads(raw) if raw.strip() else None
    return call


FLY_MACHINES_API = "api.machines.dev"


def token_transport(token: str, timeout: float = 60, host: str = FLY_MACHINES_API):
    """HTTPS to Fly's public Machines API with an app-scoped token (Fly secret FLY_API_TOKEN, a deploy token for this
    app only): call(method, path, body) -> the JSON answer. The in-machine proxy (/.fly/api) answers 403 to an app
    listing its own machines (2026-11-23), so starting the finisher needs this. Errors never carry the token."""
    def call(method: str, path: str, body: dict | None = None):
        import http.client

        conn = http.client.HTTPSConnection(host, timeout=timeout)
        try:
            conn.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                         headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                                  "Accept": "application/json", "User-Agent": USER_AGENT})
            r = conn.getresponse()
            status, raw = r.status, r.read()
        except OSError as e:
            raise ProviderError(f"{method} {path} -> {type(e).__name__}") from None
        finally:
            conn.close()
        if status >= 300:
            raise ProviderError(redact(f"{method} {path} -> {status}: {raw.decode(errors='replace')[:1500]}"))
        return json.loads(raw) if raw.strip() else None
    return call


class FlyMachines:
    """Machines of this Fly app (the finisher: fly.toml process group `finisher`), through the in-machine Machines API
    proxy (/.fly/api, http://flaps/v1/apps/<app>/machines). `launch` starts a STOPPED machine of the pool's process
    group, after writing the run's group name into its metadata (mw_group), so list_live names it as registered and
    its worker knows its group; it never creates one (fly deploy owns the machines and their size). `list_live`: the
    started machines of those groups (one without a run name gets an mw- name, so the enforcer stops it as
    unregistered). `destroy` stops one. Where there is no socket (the GitHub backstop) it is not configured: the
    enforcer then keeps the registered runs as they are and leaves stopping them to the autoscaler on Fly."""
    name = "fly"

    def __init__(self, process_groups: list[str] | None = None, *, app: str | None = None, transport=None,
                 socket_path: str = FLY_API_SOCKET):
        self.process_groups = [str(g) for g in (process_groups or []) if g]
        self.app = app or os.environ.get("FLY_APP_NAME")
        self.socket_path = socket_path
        self._transport = transport

    def configured(self) -> bool:
        return bool(self.app) and (self._transport is not None or bool(os.environ.get("FLY_API_TOKEN"))
                                   or os.path.exists(self.socket_path))

    def _req(self, method: str, path: str, body: dict | None = None):
        token = os.environ.get("FLY_API_TOKEN")
        call = self._transport or (token_transport(token) if token else unix_transport(self.socket_path))
        return call(method, f"/v1/apps/{self.app}{path}", body)

    @staticmethod
    def _meta(m: dict) -> dict:
        return (m.get("config") or {}).get("metadata") or {}

    def _machines(self) -> list[dict]:
        res = self._req("GET", "/machines") or []
        return res if isinstance(res, list) else list(res.get("machines") or [])

    def launch(self, name: str, spec: LaunchSpec) -> Machine:
        group = spec.process_group or (self.process_groups[0] if len(self.process_groups) == 1 else None)
        if not group:
            raise ProviderError("no Fly process group for this pool (gpu_pools.<pool>.process_group)")
        stopped = [m for m in self._machines()
                   if self._meta(m).get("fly_process_group") == group and m.get("state") in FLY_STOPPED_STATES]
        if not stopped:
            raise NoCapacity(f"no stopped machine in Fly process group {group!r} (fly deploy makes it; never created here)")
        ref = str(stopped[0]["id"])
        self._req("POST", f"/machines/{ref}/metadata/{FLY_RUN_KEY}", {"value": name})
        self._req("POST", f"/machines/{ref}/start")
        return Machine(name, ref, spec.max_usd_per_hour)

    def list_live(self) -> dict[str, str]:
        live = {}
        for m in self._machines():
            group = self._meta(m).get("fly_process_group")
            if group in self.process_groups and m.get("state") in FLY_LIVE_STATES:
                live[str(self._meta(m).get(FLY_RUN_KEY) or f"mw-fly-{group}-{m['id']}")] = str(m["id"])
        return live

    def destroy(self, name: str, ref: str | None) -> None:
        if not ref:
            ref = self.list_live().get(name)
            if not ref:
                return
        self._req("POST", f"/machines/{ref}/stop")

    def own_group(self) -> str | None:
        """The run this machine was started for (its mw_group metadata), or None (not on Fly, or not started by the
        autoscaler): the finisher's worker takes it as MW_GROUP."""
        mid = os.environ.get("FLY_MACHINE_ID")
        if not mid or not self.configured():
            return None
        group = self._meta(self._req("GET", f"/machines/{mid}") or {}).get(FLY_RUN_KEY)
        return str(group) if group else None

    def stop_self(self) -> str | None:
        """Stop the machine this process runs on (the worker's self-stop: deadline, stopped by the autoscaler, or a
        database it can't reach); None when not on a Fly machine."""
        mid = os.environ.get("FLY_MACHINE_ID")
        if not mid or not self.configured():
            return None
        self._req("POST", f"/machines/{mid}/stop")
        return self.name


def fly_process_groups(policy) -> list[str]:
    """The Fly process groups of the policy's fly pools (gpu_pools.<pool>.process_group)."""
    return [str(p["process_group"]) for p in (policy.get("gpu_pools", default={}) or {}).values()
            if p.get("provider") == "fly" and p.get("process_group")]


def all_providers(policy) -> dict[str, Provider]:
    v = policy.get("autoscale", "vast", default={})
    return {"fly": FlyMachines(fly_process_groups(policy)),
            "salad": Salad(),
            "vast": Vast(float(v.get("min_reliability", 0.98)), bool(v.get("verified_only", True)),
                         int(v.get("offers_to_try", 3)), float(v.get("min_inet_down_mbps", 0)),
                         list(v.get("exclude_countries", [])),
                         max_inet_down_usd_per_gb=(float(v["max_inet_down_usd_per_gb"])
                                                   if v.get("max_inet_down_usd_per_gb") is not None else None),
                         expected_hours=float(v.get("expected_hours", 1.0)), candidates=int(v.get("candidates", 50))),
            "runpod": RunPod(policy.get("autoscale", "runpod_cuda_versions", default=[]))}


# --- Stopping our own machine from inside it ----------------------------------------

def image_published(image: str) -> bool:
    """True when `image` (registry/namespace/repo:tag) can be pulled. A machine started
    on an unpublished tag bills while its pull fails, so the autoscaler checks first.
    Only Docker Hub is checked; other registries are assumed published."""
    ref, _, tag = image.rpartition(":")
    parts = ref.split("/")
    if parts[0] not in ("docker.io", "index.docker.io") or len(parts) != 3:
        return True
    url = f"https://hub.docker.com/v2/repositories/{parts[1]}/{parts[2]}/tags/{tag or 'latest'}"
    try:
        http("GET", url, {})
        return True
    except ProviderError as e:
        if " -> 404" in str(e):
            return False
        raise


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
