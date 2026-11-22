"""Hosted talking avatars: the quality ceiling we compare our own lip sync against (plan 2026-11-22 wave 3, item 11).

OFF, and refused unless the owner turns on BOTH `hosted_avatar.enabled` and `hosted_avatar.terms_reviewed` (after
reading docs/research/2026-11-22-hosted-avatar-terms.md and the provider's current terms). Not a production path:
the intended use is a ceiling benchmark (20 clips against InfiniteTalk/LongCat), and every request is priced before
it is sent: one whose estimate exceeds `hosted_avatar.max_usd_per_job` is refused.

  AvatarProvider   the adapter interface: render(image_url, audio_url, prompt, seconds) -> AvatarRender
  KlingAvatarFal   Kling AI Avatar v2 (Standard or Pro) through fal's queue API (fal-ai/kling-video/ai-avatar/v2/*).
                   Input {image_url, audio_url, prompt}; output {video{url}, duration} (fal API page, 2026-11-22).
                   Queue: POST https://queue.fal.run/<endpoint> with "Authorization: Key <FAL_KEY>", then the
                   status_url / response_url the submit returns (fal queue docs, 2026-11-22). fal reports no cost,
                   so the charge is the policy price per second x the returned duration (fal page: $0.0562/s
                   Standard, $0.115/s Pro, 2026-11-22). The key comes from the environment (`key_env`) and is
                   never logged: error text has it replaced.

The worker handler (worker/handlers/hosted_avatar.py, kind `hosted.avatar`, pool `hosted`) records the charge in
the ledger as `hosted_api`. The kind is not in policy `job_kinds`, so nothing claims it until the owner adds it.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .policy import Policy


class NotAllowed(RuntimeError):
    """The hosted avatar is off, its terms are not reviewed, or the request would cost too much."""


class HostedError(RuntimeError):
    """The host refused or failed the request. `retryable` is False for a 4xx that retrying cannot fix."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def cfg(policy: Policy | None) -> dict:
    return (policy.get("hosted_avatar", default={}) or {}) if policy is not None else {}


def check_allowed(policy: Policy | None) -> dict:
    """The block when the owner has switched it on and reviewed the terms; NotAllowed otherwise."""
    c = cfg(policy)
    if c.get("enabled") is not True:
        raise NotAllowed("hosted_avatar is off (policy hosted_avatar.enabled)")
    if c.get("terms_reviewed") is not True:
        raise NotAllowed("hosted_avatar: the provider's terms are not reviewed (policy hosted_avatar.terms_reviewed; "
                         "docs/research/2026-11-22-hosted-avatar-terms.md)")
    return c


def provider_cfg(policy: Policy | None, name: str | None = None) -> tuple[str, dict]:
    c = cfg(policy)
    name = name or c.get("provider") or "kling_fal"
    spec = (c.get("providers") or {}).get(name)
    if not spec:
        raise NotAllowed(f"unknown hosted avatar provider {name!r}")
    return name, spec


def usd_per_second(policy: Policy | None, provider: str | None = None, tier: str | None = None) -> float:
    _, spec = provider_cfg(policy, provider)
    tier = tier or spec.get("default_tier") or "standard"
    tiers = spec.get("tiers") or {}
    if tier not in tiers:
        raise NotAllowed(f"unknown tier {tier!r}; known: {sorted(tiers)}")
    return float(tiers[tier]["usd_per_second"])


def estimate_usd(seconds: float, policy: Policy | None, provider: str | None = None, tier: str | None = None) -> float:
    return round(float(seconds) * usd_per_second(policy, provider, tier), 6)


def check_cost(seconds: float, policy: Policy | None, provider: str | None = None, tier: str | None = None) -> float:
    """The estimate, or NotAllowed when it is over `max_usd_per_job` or the clip is over `max_seconds`."""
    c = cfg(policy)
    if float(seconds) <= 0:
        raise NotAllowed("a hosted avatar needs a positive audio length")
    if float(seconds) > float(c.get("max_seconds", 30)):
        raise NotAllowed(f"{seconds} s is over hosted_avatar.max_seconds ({c.get('max_seconds', 30)})")
    usd = estimate_usd(seconds, policy, provider, tier)
    if usd > float(c.get("max_usd_per_job", 0)):
        raise NotAllowed(f"${usd} is over hosted_avatar.max_usd_per_job (${c.get('max_usd_per_job', 0)})")
    return usd


@dataclass
class AvatarRender:
    video_url: str
    seconds: float
    usd: float
    provider: str
    model: str
    remote_id: str | None = None
    meta: dict = field(default_factory=dict)


class AvatarProvider(Protocol):
    name: str

    def render(self, *, image_url: str, audio_url: str, prompt: str, seconds: float) -> AvatarRender: ...


# --- Kling AI Avatar v2 via fal ----------------------------------------------------------------------------------

def _request(method: str, url: str, key: str, body: dict | None = None, timeout: float = 60) -> dict:
    from .llm import USER_AGENT
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Key {key}", "Content-Type": "application/json",
                                          "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")[:400].replace(key, "<key>")
        raise HostedError(f"{method} {url} -> {e.code}: {text}",
                          retryable=not (400 <= e.code < 500 and e.code != 429)) from None


class KlingAvatarFal:
    name = "kling_fal"

    def __init__(self, policy: Policy | None, *, tier: str | None = None,
                 request: Callable[..., dict] | None = None, sleep: Callable[[float], None] = time.sleep):
        self.policy = policy
        self.spec = provider_cfg(policy, self.name)[1]
        self.tier = tier or self.spec.get("default_tier") or "standard"
        tiers = self.spec.get("tiers") or {}
        if self.tier not in tiers:
            raise NotAllowed(f"unknown tier {self.tier!r}; known: {sorted(tiers)}")
        self.endpoint = str(tiers[self.tier]["endpoint"])
        self.request = request or _request
        self.sleep = sleep

    def _key(self) -> str:
        env = str(self.spec.get("key_env", "FAL_KEY"))
        key = os.environ.get(env)
        if not key:
            raise NotAllowed(f"{env} is not set")
        return key

    def render(self, *, image_url: str, audio_url: str, prompt: str, seconds: float) -> AvatarRender:
        check_allowed(self.policy)
        check_cost(seconds, self.policy, self.name, self.tier)
        key = self._key()
        base = str(self.spec.get("base_url", "https://queue.fal.run")).rstrip("/")
        body = {"image_url": image_url, "audio_url": audio_url, "prompt": prompt or "."}
        sub = self.request("POST", f"{base}/{self.endpoint}", key, body)
        rid = sub.get("request_id")
        status_url, response_url = sub.get("status_url"), sub.get("response_url")
        if not (rid and status_url and response_url):
            raise HostedError(f"fal submit returned no request: {sorted(sub)}")
        deadline = time.monotonic() + float(self.spec.get("timeout_s", 1200))
        while True:
            st = self.request("GET", status_url, key)
            status = str(st.get("status"))
            if status == "COMPLETED":
                break
            if status not in ("IN_QUEUE", "IN_PROGRESS"):
                raise HostedError(f"fal request {rid}: status {status}")
            if time.monotonic() > deadline:
                raise HostedError(f"fal request {rid} not done after {self.spec.get('timeout_s', 1200)} s")
            self.sleep(float(self.spec.get("poll_seconds", 10)))
        out = self.request("GET", response_url, key)
        video = (out.get("video") or {}).get("url")
        if not video:
            raise HostedError(f"fal request {rid} completed without a video: {sorted(out)}")
        got = float(out.get("duration") or seconds)
        usd = estimate_usd(got, self.policy, self.name, self.tier)
        return AvatarRender(video_url=video, seconds=got, usd=usd, provider=self.name, model=self.endpoint,
                            remote_id=rid, meta={"tier": self.tier})


PROVIDERS: dict[str, type] = {"kling_fal": KlingAvatarFal}


def provider(policy: Policy | None, name: str | None = None, **kw) -> AvatarProvider:
    """The configured provider, after the on/terms check (a refused provider is never constructed)."""
    check_allowed(policy)
    name, _ = provider_cfg(policy, name)
    if name not in PROVIDERS:
        raise NotAllowed(f"no adapter for hosted avatar provider {name!r}")
    return PROVIDERS[name](policy, **kw)


def download(url: str, path: Path, timeout: float = 300) -> Path:
    from .llm import USER_AGENT
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(path, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return path
