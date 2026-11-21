"""Image-to-video on a hosted API (Together's video API: Wan 2.7, Seedance, MiniMax H3, ...): no GPU machine.

At our volume a hosted 5 s clip ($0.10 for Wan 2.7 i2v on Together, 2026-11-21) costs about what a rented GPU does
for the render alone, without the boot, the model download and the idle tail that made 95% of our GPU bill
produce nothing. Runs on the control plane (pool `hosted`). The approved photo goes to the host as a
short-lived presigned URL; the clip comes back as a URL that expires, so it is downloaded at once.

Payload: image_uri, prompt, keyframes, seed; params{model, seconds, resolution, ratio, negative_prompt,
poll_seconds, timeout_s}. The host's reported cost goes to the ledger (`hosted_api`) as soon as it is known.
A rerun after a crash renders again (the host keeps no idempotency key): bounded by the job's attempts.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from ... import media
from ...llm import USER_AGENT
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

HOSTS = {"together": {"base_url": "https://api.together.ai/v2", "key_env": "TOGETHER_API_KEY"}}
DONE, FAILED = {"completed"}, {"failed", "cancelled"}


def _request(method: str, url: str, key: str, body: dict | None = None, timeout: float = 60) -> dict:
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")[:400].replace(key, "<key>")
        if 400 <= e.code < 500 and e.code != 429:
            raise NonRetryable(f"{method} {url} -> {e.code}: {text}") from None
        raise RuntimeError(f"{method} {url} -> {e.code}: {text}") from None


def _download(url: str, path: Path, timeout: float = 300) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(path, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return path


def generate(payload: dict, ctx: JobContext, *, sleep=time.sleep) -> HandlerResult:
    if not payload.get("image_uri") or not payload.get("prompt"):
        raise NonRetryable("payload needs image_uri and prompt")
    params = payload.get("params") or {}
    if not params.get("model"):
        raise NonRetryable("params needs model (policy generation.hosted_video)")
    host = HOSTS[params.get("host", "together")]
    key = os.environ.get(host["key_env"])
    if not key:
        raise NonRetryable(f"{host['key_env']} is not set")
    body = {"model": params["model"], "prompt": payload["prompt"],
            "seconds": str(params.get("seconds", 5)),
            "media": {"frame_images": [{"input_image": ctx.store.public_url(payload["image_uri"], 3600),
                                        "frame": "first"}]}}
    for k in ("resolution", "ratio", "negative_prompt"):
        if params.get(k):
            body[k] = params[k]
    if payload.get("seed") is not None:
        body["seed"] = int(payload["seed"]) % 2147483647
    t0 = time.monotonic()
    job = _request("POST", f"{host['base_url']}/videos", key, body)
    deadline = t0 + float(params.get("timeout_s", 900))
    while str(job.get("status")) not in DONE | FAILED:
        if time.monotonic() > deadline:
            raise RuntimeError(f"hosted video {job.get('id')} not done after {params.get('timeout_s', 900)} s")
        sleep(float(params.get("poll_seconds", 10)))
        job = _request("GET", f"{host['base_url']}/videos/{job['id']}", key)
    outputs = job.get("outputs") or {}
    if job.get("status") in FAILED:
        err = job.get("error") or {}
        raise RuntimeError(f"hosted video {job.get('id')} {job.get('status')}: {err.get('code')} {err.get('message')}")
    cost = outputs.get("cost")
    # The host's own figure when it gives one, else the policy price: every render is in the ledger either way.
    usd = float(cost) if cost is not None else float(params.get("usd_per_clip", 0))
    ctx.charge(f"{params.get('host', 'together')}:{params['model']}", usd, quantity=1, unit="clip")
    clip = _download(outputs["video_url"], Path(ctx.workdir) / "clip.mp4")
    render_s = round(time.monotonic() - t0, 1)
    frames = [OutputFile(p, "keyframe", {"index": i, "frame": idx, "of_frames": total})
              for i, (p, idx, total) in enumerate(media.keyframes(clip, Path(ctx.workdir), int(payload.get("keyframes", 4))))]
    recipe = {"model": params["model"], "host": params.get("host", "together"), "prompt": payload["prompt"],
              "image_uri": payload["image_uri"], "seed": body.get("seed"), "params": params, "remote_id": job.get("id")}
    return HandlerResult([OutputFile(clip, "video", {"seconds": float(params.get("seconds", 5)), "hosted": True}),
                          *frames], recipe=recipe, meta={"render_seconds": render_s, "hosted_usd": usd})


HANDLERS = {"hosted.video": generate}
