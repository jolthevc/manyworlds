"""Hosted talking avatars (manyworlds.hosted_avatar, Kling AI Avatar v2 via fal): refused unless the owner turned it
on AND reviewed the terms, priced before it is sent, charged as hosted_api, and the key never shows in errors."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from manyworlds import hosted_avatar as ha, media, policy as policy_mod, queue
from manyworlds.worker import runner
from manyworlds.worker.handlers import hosted_avatar as handler
from manyworlds.worker.runner import JobContext, NonRetryable


def _on(policy, **over):
    body = dict(policy.body)
    body["hosted_avatar"] = {**body["hosted_avatar"], "enabled": True, "terms_reviewed": True, **over}
    return policy_mod.from_dict(body)


class FakeFal:
    def __init__(self, *, polls=2, duration=4.0, status_error=None):
        self.polls, self.duration, self.status_error = polls, duration, status_error
        self.requests = []

    def request(self, method, url, key, body=None, timeout=60):
        self.requests.append((method, url, key, body))
        if method == "POST":
            return {"request_id": "r-1", "status_url": "https://queue.fal.run/fal-ai/kling-video/requests/r-1/status",
                    "response_url": "https://queue.fal.run/fal-ai/kling-video/requests/r-1", "queue_position": 0}
        if url.endswith("/status"):
            if self.status_error:
                raise self.status_error
            self.polls -= 1
            return {"status": "IN_PROGRESS" if self.polls > 0 else "COMPLETED"}
        return {"video": {"url": "https://v3.fal.media/files/x/avatar.mp4"}, "duration": self.duration}


def test_the_shipped_policy_keeps_hosted_avatars_off(raw_policy, policy):
    c = raw_policy.get("hosted_avatar")
    assert c["enabled"] is False and c["terms_reviewed"] is False
    with pytest.raises(ha.NotAllowed, match="off"):
        ha.check_allowed(raw_policy)
    terms_only = policy_mod.from_dict({**policy.body, "hosted_avatar": {**c, "enabled": True}})
    with pytest.raises(ha.NotAllowed, match="terms"):
        ha.check_allowed(terms_only)
    with pytest.raises(ha.NotAllowed):
        ha.provider(raw_policy)


def test_requests_are_priced_from_policy_and_refused_over_the_cap(policy):
    on = _on(policy)
    std = on.get("hosted_avatar", "providers", "kling_fal", "tiers", "standard", "usd_per_second")
    assert ha.estimate_usd(10, on) == pytest.approx(10 * std)
    assert ha.check_cost(10, on) == pytest.approx(10 * std)
    with pytest.raises(ha.NotAllowed, match="max_seconds"):
        ha.check_cost(on.get("hosted_avatar", "max_seconds") + 1, on)
    with pytest.raises(ha.NotAllowed, match="max_usd_per_job"):
        ha.check_cost(25, _on(policy, max_usd_per_job=1.0), tier="pro")
    with pytest.raises(ha.NotAllowed):
        ha.check_cost(0, on)


def test_kling_on_fal_submits_polls_and_returns_the_clip_priced_by_its_duration(policy, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fal-secret")
    on = _on(policy)
    fal = FakeFal(duration=4.2)
    prov = ha.provider(on, request=fal.request, sleep=lambda s: None)
    out = prov.render(image_url="https://r2/x.png", audio_url="https://r2/x.wav", prompt="", seconds=4.0)
    method, url, key, body = fal.requests[0]
    assert method == "POST" and url == "https://queue.fal.run/fal-ai/kling-video/ai-avatar/v2/standard"
    assert key == "fal-secret"
    assert body == {"image_url": "https://r2/x.png", "audio_url": "https://r2/x.wav", "prompt": "."}
    assert [r[0] for r in fal.requests] == ["POST", "GET", "GET", "GET"]
    std = on.get("hosted_avatar", "providers", "kling_fal", "tiers", "standard", "usd_per_second")
    assert out.video_url.endswith("avatar.mp4") and out.seconds == 4.2
    assert out.usd == pytest.approx(round(4.2 * std, 6)) and out.remote_id == "r-1"


def test_without_the_key_nothing_is_sent_and_http_errors_never_show_the_key(policy, monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    fal = FakeFal()
    with pytest.raises(ha.NotAllowed, match="FAL_KEY"):
        ha.provider(_on(policy), request=fal.request).render(image_url="a", audio_url="b", prompt="", seconds=3)
    assert fal.requests == []

    import io
    import urllib.error

    def refuse(req, timeout=60):
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b'{"detail": "bad key fal-secret"}'))
    monkeypatch.setattr(ha.urllib.request, "urlopen", refuse)
    with pytest.raises(ha.HostedError) as e:
        ha._request("POST", "https://queue.fal.run/x", "fal-secret", {})
    assert "fal-secret" not in str(e.value) and e.value.retryable is False


def _video(path: Path, seconds: int = 2) -> Path:
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=160x288:rate=24:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    return path


def _ctx(store, tmp_path):
    job = queue.Job(id="j", kind="hosted.avatar", payload={}, attempts=1, max_attempts=3, creator_id=None, batch=None)
    return JobContext(job=job, workdir=Path(tempfile.mkdtemp(dir=tmp_path)), store=store)


def _payload(store, tmp_path, seconds=4.0):
    img, wav = tmp_path / "face.png", tmp_path / "line.wav"
    img.write_bytes(b"\x89PNG face")
    wav.write_bytes(b"RIFF line")
    return {"image_uri": store.put_file(img, "face.png").uri, "audio_uri": store.put_file(wav, "line.wav").uri,
            "seconds": seconds, "prompt": "", "keyframes": 4}


def test_the_handler_refuses_while_off_and_charges_the_ledger_when_on(store, tmp_path, policy, raw_policy, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fal-secret")
    with pytest.raises(NonRetryable, match="off"):
        handler.generate(_payload(store, tmp_path), _ctx(store, tmp_path), policy=raw_policy)
    on = _on(policy)
    fal = FakeFal(duration=4.0)
    remote = _video(tmp_path / "remote.mp4")
    ctx = _ctx(store, tmp_path)
    res = handler.generate(_payload(store, tmp_path), ctx, policy=on,
                           adapter=ha.KlingAvatarFal(on, request=fal.request, sleep=lambda s: None),
                           download=lambda url, path: shutil.copy(remote, path) and Path(path))
    assert [o.kind for o in res.outputs] == ["video"] + ["keyframe"] * 4
    std = on.get("hosted_avatar", "providers", "kling_fal", "tiers", "standard", "usd_per_second")
    assert ctx.charges == [{"provider": "kling_fal:fal-ai/kling-video/ai-avatar/v2/standard",
                            "usd": pytest.approx(round(4.0 * std, 6)), "quantity": 4.0, "unit": "second"}]
    with pytest.raises(NonRetryable, match="max_usd_per_job"):
        handler.generate(_payload(store, tmp_path, seconds=20), _ctx(store, tmp_path),
                         policy=_on(policy, max_usd_per_job=0.5),
                         adapter=ha.KlingAvatarFal(on, request=fal.request, sleep=lambda s: None))


def test_a_failed_fal_request_is_retried_and_a_rejected_one_is_not(store, tmp_path, policy, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fal-secret")
    on = _on(policy)
    for err, expected in ((ha.HostedError("503", retryable=True), RuntimeError),
                          (ha.HostedError("422", retryable=False), NonRetryable)):
        fal = FakeFal(status_error=err)
        with pytest.raises(expected):
            handler.generate(_payload(store, tmp_path), _ctx(store, tmp_path), policy=on,
                             adapter=ha.KlingAvatarFal(on, request=fal.request, sleep=lambda s: None))


def test_the_avatar_kind_has_its_handler():
    assert runner.handlers_for(["hosted.avatar"])["hosted.avatar"] is handler.generate
    assert runner.handlers_for(["hosted.video"])["hosted.video"].__module__.endswith("hosted_video")
