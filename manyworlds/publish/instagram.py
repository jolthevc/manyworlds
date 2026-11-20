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


class Held(Exception):
    """The post needs the owner: it must not be (re)published automatically."""


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
        raise GraphError(_clean(f"{method} {path} -> {e.code}: {e.read().decode(errors='replace')[:800]}", token)) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:   # incl. read timeouts and bad JSON
        raise GraphError(_clean(f"{method} {path} -> {type(e).__name__}: {e}", token)) from None


def _clean(text: str, token: str | None) -> str:
    """No token, env secret or presigned-URL signature in any error we keep or print."""
    import re
    from ..deploy.providers import redact
    text = text.replace(token, "<token>") if token else text
    return re.sub(r"(X-Amz-[A-Za-z]+=)[^&\s\"']+", r"\1<redacted>", redact(text))


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

    def check(self, post: dict) -> str | None:
        """Why this post can't go out as it is (never retried), or None."""
        body, fmt = post.get("body") or {}, post["format"]
        if body.get("platform_ai_label") is False:
            return "post says platform_ai_label false: every post must carry the AI label"
        caption = self._caption(post)
        if len(caption) > 2200:
            return f"caption is {len(caption)} characters (Instagram allows 2200)"
        if caption.count("#") > 30:
            return f"{caption.count('#')} hashtags (Instagram allows 30)"
        if fmt in VIDEO_FORMATS and not body.get("video"):
            return "a video post without a video"
        if fmt not in VIDEO_FORMATS and not body.get("images"):
            return "a post without images"
        if len(body.get("images") or []) > 20:
            return "more than 20 images in a carousel"
        return None

    def _create(self, ig_id: str, token: str, post: dict) -> dict:
        body, fmt = post.get("body") or {}, post["format"]
        label = True   # disclosure is non-negotiable: every post, album and Reel says it is AI-generated
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

    def _find_published(self, ig_id: str, token: str, post: dict, ig: dict) -> str | None:
        """The media a container became (it was published but not recorded): by publish time and product type,
        Stories from /stories. Never by caption alone."""
        from datetime import datetime, timedelta
        since = datetime.fromisoformat(ig["publishing_at"]) - timedelta(minutes=5)
        story = post["format"] == "story"
        listing = graph("GET", f"/{ig_id}/stories" if story else f"/{ig_id}/media",
                        {"fields": "id,timestamp,media_product_type", "limit": 25}, token)
        want = "STORY" if story else ("REELS" if post["format"] in VIDEO_FORMATS else "FEED")
        hits = [m for m in listing.get("data") or []
                if m.get("timestamp") and datetime.fromisoformat(m["timestamp"].replace("+0000", "+00:00")) >= since
                and (m.get("media_product_type") or want) == want]
        return hits[-1]["id"] if len(hits) == 1 else None

    def _save(self, conn, post: dict, body: dict) -> None:
        conn.execute("UPDATE posts SET body = %s WHERE id = %s", (Jsonb(body), post["id"]))

    def _hold(self, conn, post: dict, body: dict, why: str) -> None:
        body["hold_reason"] = why
        conn.execute("UPDATE posts SET status = 'held', body = %s, updated_at = now() WHERE id = %s",
                     (Jsonb(body), post["id"]))
        conn.execute("INSERT INTO events (type, subject, payload, routed_to, creator_id, dedupe_key) VALUES "
                     "('publish_held', %s, %s, 'owner', %s, %s) ON CONFLICT DO NOTHING",
                     (Jsonb({"post_id": str(post["id"])}), Jsonb({"why": why[:500]}), post["creator_id"],
                      f"publish_held:{post['id']}"))
        raise Held(why)

    def publish(self, conn, policy: Policy, post: dict) -> dict:
        from datetime import datetime, timezone
        _POLICY.clear()
        _POLICY.update(_cfg(policy))
        body = dict(post.get("body") or {})
        ig_id, token = self.account(conn, post["creator_id"])   # no account: the post just waits
        if (why := self.check(post)):
            self._hold(conn, post, body, why)
        ig = body.get("ig")
        try:
            if ig and ig.get("publishing_at"):
                # media_publish was called: from here on only reconcile, never create or publish again.
                status = graph("GET", f"/{ig['container']}", {"fields": "status_code"}, token).get("status_code")
                if status != "PUBLISHED":
                    self._hold(conn, post, body, f"publish was sent but the container is {status}: check the account")
                media_id = self._find_published(ig_id, token, post, ig)
                if not media_id:
                    self._hold(conn, post, body, "published, but the media could not be identified: check the account")
            else:
                if not ig:
                    attempts = int(body.get("ig_attempts", 0))
                    if attempts >= int(_POLICY.get("max_attempts", 3)):
                        self._hold(conn, post, body, f"{attempts} attempts failed: " + "; ".join(body.get("publish_errors") or []))
                    q = graph("GET", f"/{ig_id}/content_publishing_limit", {"fields": "quota_usage,config"}, token)
                    q = (q.get("data") or [{}])[0]
                    if int(q.get("quota_usage") or 0) >= int((q.get("config") or {}).get("quota_total") or 50):
                        raise QuotaReached()
                    body["ig_attempts"] = attempts + 1
                    body["ig"] = ig = self._create(ig_id, token, post)
                    self._save(conn, post, body)
                    raise Pending()
                status = graph("GET", f"/{ig['container']}", {"fields": "status_code"}, token).get("status_code")
                if status == "IN_PROGRESS":
                    raise Pending()
                if status in ("ERROR", "EXPIRED"):
                    raise GraphError(f"container {ig['container']} is {status}")
                if status == "PUBLISHED":   # published without our record of calling publish: never again
                    ig["publishing_at"] = datetime.now(timezone.utc).isoformat()
                    self._save(conn, post, body)
                    self._hold(conn, post, body, "the container shows PUBLISHED before we published it: check the account")
                ig["publishing_at"] = datetime.now(timezone.utc).isoformat()
                self._save(conn, post, body)   # recorded before the call: a crash after it can only reconcile
                media_id = graph("POST", f"/{ig_id}/media_publish", {"creation_id": ig["container"]}, token)["id"]
        except GraphError as e:
            errors = (body.get("publish_errors") or []) + [_clean(str(e), token)[:500]]
            body["publish_errors"] = errors[-5:]
            if not (body.get("ig") or {}).get("publishing_at"):
                body.pop("ig", None)   # nothing was published: a fresh container next time
            self._save(conn, post, body)
            raise
        try:
            permalink = graph("GET", f"/{media_id}", {"fields": "permalink"}, token).get("permalink")
        except GraphError:
            permalink = None   # cosmetic: the post is out; never retry the publish for it
        return {"external_id": media_id, "permalink": permalink, "ai_label": True, "simulated": False}

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
