"""A talking avatar from a hosted API (manyworlds.hosted_avatar: Kling AI Avatar v2 via fal), on the control plane.

OFF: refused unless the policy on this host says `hosted_avatar.enabled: true` and `hosted_avatar.terms_reviewed:
true`; a payload can never switch it on. The approved portrait and the voice line go to the host as short-lived
presigned URLs; the clip comes back as a URL that expires, so it is downloaded at once. The charge (policy price
per second x the returned duration) goes to the ledger as `hosted_api` through ctx.charge as soon as the render is
done. A rerun after a crash renders again (fal keeps no idempotency key): bounded by the job's attempts.

Payload: image_uri, audio_uri, seconds (the voice line's length), prompt, keyframes; params{provider, tier}.
"""

from __future__ import annotations

from pathlib import Path

from ... import hosted_avatar as ha
from ... import media
from ...policy import Policy
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile


def _policy(policy: Policy | None) -> Policy:
    if policy is not None:
        return policy
    from ... import policy as policy_mod
    return policy_mod.load()


def generate(payload: dict, ctx: JobContext, *, policy: Policy | None = None, adapter=None,
             download=None) -> HandlerResult:
    policy = _policy(policy)
    params = payload.get("params") or {}
    if not payload.get("image_uri") or not payload.get("audio_uri"):
        raise NonRetryable("payload needs image_uri and audio_uri")
    try:
        seconds = float(payload.get("seconds") or 0)
        prov = adapter or ha.provider(policy, params.get("provider"), tier=params.get("tier"))
        ha.check_cost(seconds, policy, prov.name, getattr(prov, "tier", None))
    except ha.NotAllowed as e:
        raise NonRetryable(str(e)) from None
    prompt = str(payload.get("prompt") or "")
    try:
        out = prov.render(image_url=ctx.store.public_url(payload["image_uri"], 3600),
                          audio_url=ctx.store.public_url(payload["audio_uri"], 3600), prompt=prompt, seconds=seconds)
    except ha.NotAllowed as e:
        raise NonRetryable(str(e)) from None
    except ha.HostedError as e:
        if not e.retryable:
            raise NonRetryable(str(e)) from None
        raise RuntimeError(str(e)) from None
    ctx.charge(f"{out.provider}:{out.model}", out.usd, quantity=round(out.seconds, 3), unit="second")
    clip = (download or ha.download)(out.video_url, Path(ctx.workdir) / "avatar.mp4")
    frames = [OutputFile(p, "keyframe", {"index": i, "frame": idx, "of_frames": total})
              for i, (p, idx, total) in enumerate(media.keyframes(clip, Path(ctx.workdir),
                                                                  int(payload.get("keyframes", 4))))]
    recipe = {"model": out.model, "host": out.provider, "prompt": prompt, "image_uri": payload["image_uri"],
              "audio_uri": payload["audio_uri"], "params": params, "remote_id": out.remote_id, **out.meta}
    return HandlerResult([OutputFile(clip, "video", {"seconds": out.seconds, "hosted": True, "avatar": True}),
                          *frames], recipe=recipe, meta={"hosted_usd": out.usd})


HANDLERS = {"hosted.avatar": generate}
