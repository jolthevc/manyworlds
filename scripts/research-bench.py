"""Small acquisition benchmark (docs/research/2026-11-25-research-providers.md): the same public creators x their 12 most
recent posts through each provider we hold a key for (manyworlds.research.providers), with cost, latency and the fields
present. Spends real money (Apify ~$0.45, ScrapeCreators ~20 credits for 10+10 creators); every call is in the ledger
(agent research:bench). Run on the control plane:

    python scripts/research-bench.py --ig alixearle reba ... --tt alixearle reba ... [--providers apify scrapecreators]

Prints one JSON object. Read-only apart from the ledger."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.parse

from manyworlds import db
from manyworlds.policy import load
from manyworlds.research import providers


def _expiry_hours(url: str | None) -> float | None:
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url or "").query)
    for k in ("oe", "x-expires", "expire"):
        if q.get(k):
            v = q[k][0]
            return round(((int(v, 16) if k == "oe" else int(v)) - time.time()) / 3600, 1)
    return None


def fields(posts: list[dict]) -> dict:
    n = len(posts) or 1
    vids = [p for p in posts if p["media_type"] == "video"]
    cars = [p for p in posts if p["media_type"] == "carousel"]
    share = lambda xs, f: round(sum(1 for p in xs if f(p)) / len(xs), 2) if xs else None   # noqa: E731
    exp = [h for p in posts for h in (_expiry_hours((p.get("images") or [None])[0]), _expiry_hours(p.get("video_url")))
           if h is not None]
    return {"posts": len(posts), "videos": len(vids), "carousels": len(cars),
            "caption": share(posts, lambda p: p.get("caption")), "posted_at": sum(1 for p in posts if p["posted_at"]) / n,
            "likes": share(posts, lambda p: (p.get("metrics") or {}).get("likes") is not None),
            "comments": share(posts, lambda p: (p.get("metrics") or {}).get("comments") is not None),
            "image_url": share(posts, lambda p: p.get("images")),
            "video_views": share(vids, lambda p: (p.get("metrics") or {}).get("views") is not None),
            "video_url": share(vids, lambda p: p.get("video_url")),
            "carousel_children": share(cars, lambda p: (p.get("children") or 0) > 1),
            "url_expiry_hours_median": statistics.median(exp) if exp else None}


def run(conn, policy, name: str, platform: str, handles: list[str]) -> dict:
    adapter = providers.ADAPTERS[name](policy) if name != "graph" else providers.GraphProvider(policy, conn=conn)
    why = adapter.available(conn, platform)
    if why:
        return {"skipped": why}
    router = providers.Router(conn, policy, adapters={name: adapter}, agent="research:bench")
    per, posts = [], []
    for h in handles:
        t0 = time.monotonic()
        got, used, errors = router._run("posts", platform, lambda a, h=h: a.fetch_posts(platform, h, 12, None))
        per.append({"handle": h, "s": round(time.monotonic() - t0, 1), "posts": len(got), "errors": errors[-1:]})
        posts += got
    return {"per_creator": per, **fields(posts), "usd": round(sum(float(x.get("usd") or 0) for x in router.log), 4),
            "units": sum(float(x.get("units") or 0) for x in router.log),
            "s_median": statistics.median([x["s"] for x in per]) if per else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ig", nargs="*", default=[])
    ap.add_argument("--tt", nargs="*", default=[])
    ap.add_argument("--providers", nargs="*", default=["apify", "scrapecreators", "graph"])
    args = ap.parse_args()
    policy = load()
    conn = db.connect()
    conn.autocommit = True
    out = {}
    for name in args.providers:
        for platform, handles in (("instagram", args.ig), ("tiktok", args.tt)):
            if handles:
                out[f"{name}:{platform}"] = run(conn, policy, name, platform, handles)
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
