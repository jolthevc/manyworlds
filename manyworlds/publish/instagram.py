"""Instagram publisher (Brain M4): Meta's Instagram Graph API, content publishing.

Per Meta's docs (2026-11-20, v27.0): media are published in two steps, a container (POST /<IG_ID>/media) that
Instagram fetches and processes from a public URL, then POST /<IG_ID>/media_publish with its id once its
`status_code` is FINISHED. `is_ai_generated=true` is Meta's self-disclosure of AI use (set on the album, not
on carousel children); every post sets it (the owner's disclosure rule). Images must be JPEG; ours are PNG, so
each is converted and served from our bucket by a short-lived presigned URL.

Non-blocking: the first call creates the container(s) and keeps their ids on the post (`body.ig`); later calls
check the status and publish when it is finished, so the producer loop never waits on Instagram. A container
already PUBLISHED (a crash between publishing and recording it) is found again among the account's recent
media by caption, never published twice. ERROR / EXPIRED containers are dropped and retried next time, at most
`instagram.max_attempts` times. The daily API quota (50 posts / 24 h) is checked before creating anything.

Accounts: `accounts` row (platform instagram, status connected) with the IG user id and `token_env`, the name
of the .env variable holding the long-lived token. Tokens never appear in errors, logs or the database.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from psycopg.types.json import Jsonb

from ..policy import Policy

VIDEO_FORMATS = ("story_clip", "talking_video", "narrated_edit", "reel")


class GraphError(RuntimeError):
    pass


class AccountNotConnected(RuntimeError):
    pass


class Pending(Exception):
    """The container is being processed; try again next tick."""


class QuotaReached(Exception):
    pass


def _cfg(policy: Policy) -> dict:
    return policy.get("instagram", default={}) or {}


_POLICY: dict = {}


def graph(method: str, path: str, params: dict | None = None, token: str | None = None) -> dict:
    """One Graph API call. The token goes in the Authorization header, never in a URL."""
    host = _POLICY.get("host", "graph.instagram.com")
    version = _POLICY.get("version", "v27.0")
    url = f"https://{host}/{version}{path}"
    data = None
    if method == "GET" and params:
        url += "?" + urllib.parse.urlencode(params)
    elif params:
        data = urllib.parse.urlencode({k: (json.dumps(v) if isinstance(v, bool) else v)
                                       for k, v in params.items()}).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise GraphError(f"{method} {path} -> {e.code}: {e.read().decode(errors='replace')[:800]}") from None
    except urllib.error.URLError as e:
        raise GraphError(f"{method} {path} -> {e.reason}") from None


def public_jpeg(store, uri: str, tmp: Path) -> str:
    """A JPEG copy of the image in our bucket, served by a presigned URL Instagram can fetch."""
    from PIL import Image

    src = store.get_file(uri, tmp / Path(uri).name)
    out = src.with_suffix(".jpg")
    Image.open(src).convert("RGB").save(out, "JPEG", quality=92)
    key = uri.split("/", 3)[-1].rsplit(".", 1)[0] + ".jpg"
    stored = store.put_file(out, f"public/{key}")
    return store.public_url(stored.uri, int(_POLICY.get("url_seconds", 3600)))


def _redact(text: str, token: str | None) -> str:
    return text.replace(token, "<token>") if token else text


class InstagramPublisher:
    platform = "instagram"

    def __init__(self, store=None):
        self._store = store

    def store(self):
        if self._store is None:
            from .. import storage
            self._store = storage.from_url()
        return self._store

    def account(self, conn, creator_id) -> tuple[str, str]:
        a = conn.execute("SELECT external_id, token_env FROM accounts WHERE creator_id = %s AND platform = 'instagram' "
                         "AND status = 'connected'", (creator_id,)).fetchone()
        token = os.environ.get(a["token_env"]) if a and a["token_env"] else None
        if not a or not a["external_id"] or not token:
            raise AccountNotConnected("no connected Instagram account with a token for this creator")
        return a["external_id"], token

    def _caption(self, post: dict) -> str:
        tags = " ".join((post.get("body") or {}).get("hashtags") or [])
        return (post.get("caption") or "") + (f"\n\n{tags}" if tags else "")

    def _create(self, ig_id: str, token: str, post: dict) -> dict:
        body, fmt = post.get("body") or {}, post["format"]
        label = bool(body.get("platform_ai_label", True))
        with tempfile.TemporaryDirectory() as tmp:
            urls = [public_jpeg(self.store(), im["uri"], Path(tmp)) for im in body.get("images") or []]
            if fmt in VIDEO_FORMATS and body.get("video"):
                video_url = self.store().public_url(body["video"]["uri"], int(_POLICY.get("url_seconds", 3600)))
                params = {"media_type": "REELS", "video_url": video_url, "caption": self._caption(post),
                          "share_to_feed": True, "is_ai_generated": label}
                if urls:
                    params["cover_url"] = urls[0]
                return {"container": graph("POST", f"/{ig_id}/media", params, token)["id"]}
            if fmt == "story":
                return {"container": graph("POST", f"/{ig_id}/media", {"media_type": "STORIES", "image_url": urls[0],
                                                                       "is_ai_generated": label}, token)["id"]}
            if len(urls) > 1:
                children = [graph("POST", f"/{ig_id}/media", {"image_url": u, "is_carousel_item": True}, token)["id"]
                            for u in urls]
                album = graph("POST", f"/{ig_id}/media", {"media_type": "CAROUSEL", "children": ",".join(children),
                                                          "caption": self._caption(post), "is_ai_generated": label},
                              token)["id"]
                return {"container": album, "children": children}
            return {"container": graph("POST", f"/{ig_id}/media", {"image_url": urls[0], "caption": self._caption(post),
                                                                   "is_ai_generated": label}, token)["id"]}

    def publish(self, conn, policy: Policy, post: dict) -> dict:
        _POLICY.clear()
        _POLICY.update(_cfg(policy))
        ig_id, token = self.account(conn, post["creator_id"])
        body = dict(post.get("body") or {})
        try:
            ig = body.get("ig")
            if not ig:
                q = graph("GET", f"/{ig_id}/content_publishing_limit", {"fields": "quota_usage,config"}, token)["data"][0]
                if int(q.get("quota_usage") or 0) >= int((q.get("config") or {}).get("quota_total") or 50):
                    raise QuotaReached()
                body["ig"] = ig = self._create(ig_id, token, post)
                conn.execute("UPDATE posts SET body = %s WHERE id = %s", (Jsonb(body), post["id"]))
                raise Pending()
            status = graph("GET", f"/{ig['container']}", {"fields": "status_code"}, token).get("status_code")
            if status == "IN_PROGRESS":
                raise Pending()
            if status in ("ERROR", "EXPIRED"):
                raise GraphError(f"container {ig['container']} is {status}")
            if status == "PUBLISHED":   # published before a crash: find it, never publish twice
                recent = graph("GET", f"/{ig_id}/media", {"fields": "id,caption,permalink", "limit": 10}, token)
                media = next((m for m in recent.get("data") or [] if (m.get("caption") or "") == self._caption(post)),
                             None)
                if not media:
                    raise GraphError(f"container {ig['container']} is PUBLISHED but the media was not found")
                media_id = media["id"]
            else:
                media_id = graph("POST", f"/{ig_id}/media_publish", {"creation_id": ig["container"]}, token)["id"]
            permalink = graph("GET", f"/{media_id}", {"fields": "permalink"}, token).get("permalink")
            return {"external_id": media_id, "permalink": permalink, "ai_label": bool(body.get("platform_ai_label", True)),
                    "simulated": False}
        except GraphError as e:
            body.pop("ig", None)
            errors = (body.get("publish_errors") or []) + [_redact(str(e), token)[:500]]
            body["publish_errors"] = errors[-5:]
            conn.execute("UPDATE posts SET body = %s WHERE id = %s", (Jsonb(body), post["id"]))
            raise


def connect(conn, policy: Policy, creator_id, *, token_env: str) -> dict:
    """Link a creator's Instagram professional account: the token (named by `token_env` in .env, never passed
    on the command line) is checked against Meta, the account is recorded as connected, and the creator leaves
    the sandbox (publish_mode instagram). Publishing still follows `autonomy.publish`."""
    _POLICY.clear()
    _POLICY.update(_cfg(policy))
    token = os.environ.get(token_env)
    if not token:
        raise AccountNotConnected(f"{token_env} is not set in the environment (.env)")
    me = graph("GET", "/me", {"fields": "user_id,username,account_type"}, token)
    ig_id = str(me.get("user_id") or me.get("id") or "")
    if not ig_id:
        raise GraphError("Meta returned no Instagram user id for this token")
    conn.execute(
        """INSERT INTO accounts (creator_id, platform, handle, external_id, token_env, status, meta)
           VALUES (%s, 'instagram', %s, %s, %s, 'connected', %s)
           ON CONFLICT (creator_id, platform) DO UPDATE SET handle = excluded.handle, external_id = excluded.external_id,
             token_env = excluded.token_env, status = 'connected', meta = excluded.meta, updated_at = now()""",
        (creator_id, me.get("username"), ig_id, token_env, Jsonb({"account_type": me.get("account_type")})))
    conn.execute("UPDATE creators SET publish_mode = 'instagram' WHERE id = %s", (creator_id,))
    return {"instagram_user_id": ig_id, "username": me.get("username"), "account_type": me.get("account_type")}
