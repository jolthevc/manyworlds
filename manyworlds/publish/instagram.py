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

Accounts (policy `instagram.login`): with `unified` (Meta's recommended Instagram API with Unified Login, on
graph.facebook.com) one Business Integration System User token, in the env variable `instagram.token_env`, covers
every professional account assigned to our app in the business portfolio; an account row carries only its IG user
id. Standard Access is enough because every account is our own (no App Review). The older `instagram` login keeps a
token per account (`accounts.token_env`, on graph.instagram.com). Tokens never appear in errors, logs, reprs or the
database.

Ramp (policy `instagram.ramp`): a newly connected account publishes only a few posts a day at first, more as it
ages; Meta's own 24 h publishing limit is read live from `content_publishing_limit`, never assumed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from psycopg.types.json import Jsonb

from .. import notify
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


class Ramped(QuotaReached):
    """The account is new: today's ramp allowance is used up (policy instagram.ramp)."""


class Held(Exception):
    """The post needs the owner: it must not be (re)published automatically."""


def _cfg(policy: Policy) -> dict:
    return policy.get("instagram", default={}) or {}


_POLICY: dict = {}
HOSTS = {"unified": "graph.facebook.com", "instagram": "graph.instagram.com"}


@dataclass(frozen=True)
class Creds:
    """An account's credentials: its IG user id, the token (never in the repr) and the Graph host for its login.
    Unpacks as (ig_id, creds), and `graph` takes it wherever it takes a token."""
    ig_id: str
    token: str = field(repr=False)
    host: str

    def __iter__(self):
        return iter((self.ig_id, self))


def _host(login: str) -> str:
    return (_POLICY.get("hosts") or {}).get(login) or HOSTS[login]


def graph(method: str, path: str, params: dict | None = None, token=None) -> dict:
    """One Graph API call. The token goes in the Authorization header, never in a URL."""
    host = token.host if isinstance(token, Creds) else _host(_POLICY.get("login", "unified"))
    token = token.token if isinstance(token, Creds) else token
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
    token = token.token if isinstance(token, Creds) else token
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


class InstagramPublisher:
    platform = "instagram"

    def __init__(self, store=None):
        self._store = store

    def store(self):
        if self._store is None:
            from .. import storage
            self._store = storage.from_url()
        return self._store

    def account(self, conn, creator_id) -> Creds:
        return account(conn, creator_id)

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
        notify.owner_event(conn, "publish_held", subject={"post_id": str(post["id"])}, payload={"why": why[:500]},
                           creator_id=post["creator_id"], dedupe_key=f"publish_held:{post['id']}")
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
                    if ramp_left(conn, policy, post["creator_id"]) == 0:
                        raise Ramped()
                    q = graph("GET", f"/{ig_id}/content_publishing_limit", {"fields": "quota_usage,config"}, token)
                    q = (q.get("data") or [{}])[0]
                    total = (q.get("config") or {}).get("quota_total") or _POLICY.get("quota_fallback", 50)
                    if int(q.get("quota_usage") or 0) >= int(total):
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


def _use(policy: Policy) -> None:
    _POLICY.clear()
    _POLICY.update(_cfg(policy))


def account(conn, creator_id) -> Creds:
    """The connected account's credentials: its own token (Instagram Login) or the system user's (Unified Login)."""
    a = conn.execute("SELECT external_id, token_env, meta FROM accounts WHERE creator_id = %s "
                     "AND platform = 'instagram' AND status = 'connected'", (creator_id,)).fetchone()
    if not a or not a["external_id"]:
        raise AccountNotConnected("no connected Instagram account for this creator")
    if a["token_env"]:
        login, token = (a["meta"] or {}).get("login", "instagram"), os.environ.get(a["token_env"])
    else:
        login, token = "unified", os.environ.get(_POLICY.get("token_env", "META_SYSTEM_USER_TOKEN"))
    if not token:
        raise AccountNotConnected("the Instagram token for this account is not set in the environment")
    return Creds(a["external_id"], token, _host(login))


def ramp_allowance(policy: Policy, age_days: float) -> int | None:
    """Posts a day an account of this age may publish (policy instagram.ramp), or None for no ramp limit."""
    allowed = None
    for step in sorted(_cfg(policy).get("ramp") or [], key=lambda s: float(s["from_day"])):
        if age_days >= float(step["from_day"]):
            allowed = step.get("posts_per_day")
    return None if allowed is None else int(allowed)


def _connected(conn, creator_id):
    return conn.execute("SELECT id, coalesce(connected_at, created_at) AS since FROM accounts WHERE creator_id = %s "
                        "AND platform = 'instagram' AND status = 'connected'", (creator_id,)).fetchone()


def ramp_dials(conn, policy: Policy, creator_id, dials: dict, when=None) -> dict:
    """The creator's dials with Instagram posts a day capped by the ramp at `when`, so the planner never makes
    posts a new account may not publish (a post made and never published is GPU time thrown away)."""
    from datetime import datetime, timezone
    a = _connected(conn, creator_id)
    per = dict(dials.get("posts_per_day") or {})
    if not a or "instagram" not in per:
        return dials
    when = when or datetime.now(timezone.utc)
    allowed = ramp_allowance(policy, max(0.0, (when - a["since"]).total_seconds() / 86400))
    if allowed is None or float(per["instagram"]) <= allowed:
        return dials
    return {**dials, "posts_per_day": {**per, "instagram": float(allowed)}}


def ramp_left(conn, policy: Policy, creator_id, now=None) -> int | None:
    """Posts this creator's account may still publish in the last 24 h under the ramp, or None for unlimited."""
    from datetime import datetime, timezone
    now = now or datetime.now(timezone.utc)
    a = _connected(conn, creator_id)
    if not a:
        return None
    allowed = ramp_allowance(policy, max(0.0, (now - a["since"]).total_seconds() / 86400))
    if allowed is None:
        return None
    done = conn.execute("SELECT count(*) AS n FROM publications WHERE account_id = %s AND NOT simulated "
                        "AND published_at > %s - interval '24 hours'", (a["id"], now)).fetchone()["n"]
    return max(0, allowed - int(done))


def _assigned(creds: Creds) -> list[dict]:
    """The Instagram professional accounts assigned to our app's system user (Unified Login), every page."""
    out, after = [], None
    for _ in range(int(_POLICY.get("max_pages", 50))):
        params = {"fields": "id,username", "limit": 100, **({"after": after} if after else {})}
        page = graph("GET", "/me/assigned_instagram_accounts", params, creds)
        out += page.get("data") or []
        paging = page.get("paging") or {}
        after = (paging.get("cursors") or {}).get("after")
        if not after or not paging.get("next"):
            break
    return out


def setup_sheet(conn, policy: Policy, creator_id, store=None) -> dict:
    """What a person needs to set up this creator's Instagram account by hand: names, the bio (with the AI
    disclosure), the profile photo, and the checklist that `connect` then verifies through the API."""
    c = conn.execute("SELECT handle, bible, anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()
    bible, cfg = c["bible"] or {}, _cfg(policy).get("setup", {}) or {}
    disclosure = policy.get("scope", "disclosure_line", default="AI-generated character")
    limit = int(cfg.get("bio_max_chars", 150))
    line = (bible.get("one_line") or "").strip()
    bio = disclosure if not line else f"{line[:max(0, limit - len(disclosure) - 3)].rstrip()} · {disclosure}"
    refs = (c["anchors"] or {}).get("reference_images") or []
    photo = None
    if refs:
        if store is None:
            from .. import storage
            store = storage.from_url()
        photo = store.public_url(refs[0]["uri"], int(cfg.get("photo_url_seconds", 7 * 86400)))
    return {"creator": c["handle"], "username": bible.get("handle") or c["handle"],
            "display_name": bible.get("display_name") or c["handle"], "bio": bio[:limit],
            "profile_photo_url": photo, "checklist": list(cfg.get("checklist") or []),
            "then": f"mw account connect {c['handle']} --username <the username you created>"}


def verify(policy: Policy, creator: dict, profile: dict) -> tuple[list[str], list[str]]:
    """(problems that stop the connection, warnings) for an account's public profile against its setup sheet."""
    disclosure = str(policy.get("scope", "disclosure_line", default="AI-generated character"))
    problems, warnings = [], []
    if disclosure.lower() not in (profile.get("biography") or "").lower():
        problems.append(f"the bio must contain the disclosure '{disclosure}'")
    if not profile.get("profile_picture_url"):
        warnings.append("no profile photo")
    want = (creator.get("bible") or {}).get("display_name")
    if want and (profile.get("name") or "").strip().lower() != want.strip().lower():
        warnings.append(f"display name is '{profile.get('name')}', the sheet says '{want}'")
    return problems, warnings


def connect(conn, policy: Policy, creator_id, *, username: str | None = None, token_env: str | None = None) -> dict:
    """Link a creator's Instagram professional account and take the creator out of the sandbox (publish_mode
    instagram); publishing still follows `autonomy.publish` and the ramp.

    Unified Login (default): the account, found by `username` among those assigned to our app's system user, is
    checked against its setup sheet (the AI disclosure in the bio is required) and for publishing permission.
    `token_env` instead connects with the older Instagram Login, a token per account. Tokens stay in the
    environment, never on the command line or in the database."""
    from datetime import datetime, timezone
    _use(policy)
    creator = conn.execute("SELECT handle, bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    if token_env:
        token = os.environ.get(token_env)
        if not token:
            raise AccountNotConnected(f"{token_env} is not set in the environment (.env)")
        creds = Creds("", token, _host("instagram"))
        me = graph("GET", "/me", {"fields": "user_id,username,account_type,name,biography,profile_picture_url"}, creds)
        ig_id, login, profile = str(me.get("user_id") or me.get("id") or ""), "instagram", me
        if not ig_id:
            raise GraphError("Meta returned no Instagram user id for this token")
    else:
        if not username:
            raise AccountNotConnected("give the account's --username (or --token-env for Instagram Login)")
        env = _POLICY.get("token_env", "META_SYSTEM_USER_TOKEN")
        token = os.environ.get(env)
        if not token:
            raise AccountNotConnected(f"{env} (the system user's token) is not set in the environment")
        creds = Creds("", token, _host("unified"))
        want = username.strip().lstrip("@").lower()
        hit = next((a for a in _assigned(creds) if (a.get("username") or "").lower() == want), None)
        if not hit:
            raise AccountNotConnected(f"@{want} is not assigned to our app: add it to the business portfolio and "
                                      "assign it to the app's system user (Business Settings)")
        ig_id, login = str(hit["id"]), "unified"
        profile = graph("GET", f"/{ig_id}", {"fields": "username,name,biography,profile_picture_url,followers_count,"
                                                       "media_count"}, creds)
    other = conn.execute("SELECT c.handle FROM accounts a JOIN creators c ON c.id = a.creator_id WHERE a.platform = "
                         "'instagram' AND a.external_id = %s AND a.creator_id <> %s", (ig_id, creator_id)).fetchone()
    if other:
        raise AccountNotConnected(f"this Instagram account is already connected to {other['handle']}")
    problems, warnings = verify(policy, creator, profile)
    acct = Creds(ig_id, creds.token, creds.host)
    try:   # publishing permission: Meta answers this only for a token that may publish to the account
        graph("GET", f"/{ig_id}/content_publishing_limit", {"fields": "quota_usage,config"}, acct)
    except GraphError as e:
        problems.append("cannot read the publishing limit (is instagram_content_publish granted?): "
                        + _clean(str(e), acct)[:200])
    if problems:
        raise AccountNotConnected("not connected: " + "; ".join(problems))
    meta = {"login": login, "warnings": warnings, "verified_at": datetime.now(timezone.utc).isoformat(),
            **{k: profile.get(k) for k in ("account_type", "followers_count", "media_count") if k in profile}}
    conn.execute(
        """INSERT INTO accounts (creator_id, platform, handle, external_id, token_env, status, meta, connected_at)
           VALUES (%s, 'instagram', %s, %s, %s, 'connected', %s, now())
           ON CONFLICT (creator_id, platform) DO UPDATE SET handle = excluded.handle, external_id = excluded.external_id,
             token_env = excluded.token_env, status = 'connected', meta = excluded.meta, updated_at = now(),
             connected_at = coalesce(accounts.connected_at, excluded.connected_at)""",
        (creator_id, profile.get("username"), ig_id, token_env, Jsonb(meta)))
    conn.execute("UPDATE creators SET publish_mode = 'instagram' WHERE id = %s", (creator_id,))
    return {"instagram_user_id": ig_id, "username": profile.get("username"), "login": login, "warnings": warnings}
