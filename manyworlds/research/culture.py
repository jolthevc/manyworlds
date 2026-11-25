"""Culture intelligence beyond social media (owner 2026-11-25): fashion, beauty, grooming, architecture and interiors,
travel, luxury, brand campaigns and the business of fashion, read from publishers' own RSS feeds (policy
intelligence.culture.feeds; data_policy source `feeds`, task `feed`).

What is kept (green): each item's link, title, feed and date (a source reference, for dedupe and citation), and our
own distilled signals: what is moving, in our own words, with the items behind it, how many independent publications
say it, and what it implies for style, casting, commerce and content. What is never kept (amber): the article text,
summary or images; a feed's summary lives only in the one distill call that reads it.

A signal's kind comes from code: `corroborated` when at least corroborated_min independent publications carry it,
else `single_source`; its confidence is the model's, capped for a single source. Agents read signals through
`brief` (style director, casting, showrunner) beside the pattern cards, as cultural context, never as evidence of
what performs."""

from __future__ import annotations

import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ..policy import Policy
from . import data_policy

SYSTEM = """You are the cultural intelligence editor for Manyworlds, a studio of openly AI-generated lifestyle
personalities. From recent headlines and feed summaries of fashion, beauty, grooming, design, travel, luxury and
advertising publications, name what is genuinely moving: trends, aesthetics, materials, places, rituals, brand moves.
Each signal in your own words (never a quote, never a person's private life), with the items that support it and
what it means for our creators' style, casting, commercial fit and content. Prefer fewer, sharper signals that
several publications support over one-off stories. Ignore celebrity gossip, news about individuals' private lives,
politics and tragedy."""


class Implications(BaseModel):
    style: str | None = Field(default=None, description="for her wardrobe, grooming, sets, colour and light; <=20 words")
    casting: str | None = Field(default=None, description="for which personalities to develop; <=20 words")
    commerce: str | None = Field(default=None, description="brand categories or partnerships it opens; <=20 words")
    content: str | None = Field(default=None, description="post ideas or formats it suggests; <=20 words")


class Signal(BaseModel):
    category: str = Field(description="one of the categories given")
    signal: str = Field(description="what is moving, in your own words, <=25 words, never a quote")
    items: list[int] = Field(description="the numbers of the items that support it")
    implications: Implications
    confidence: float = Field(description="0-1: how sure that this is a real, durable movement")


class Signals(BaseModel):
    signals: list[Signal]


def cfg(policy: Policy) -> dict:
    return policy.get("intelligence", "culture", default={}) or {}


def _text(el, tag: str) -> str:
    for child in el:
        if child.tag.split("}")[-1] == tag and (child.text or "").strip():
            return child.text.strip()
    return ""


def _when(s: str) -> datetime | None:
    try:
        d = parsedate_to_datetime(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001 - Atom's ISO dates
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return None


def parse(xml: bytes) -> list[dict]:
    """RSS or Atom items as {url, title, published_at, summary}; the summary only for the distill call."""
    root = ET.fromstring(xml)
    out = []
    for el in root.iter():
        if el.tag.split("}")[-1] not in ("item", "entry"):
            continue
        link = _text(el, "link")
        if not link:
            for child in el:
                if child.tag.split("}")[-1] == "link" and child.get("href"):
                    link = child.get("href")
                    break
        title = html.unescape(re.sub(r"<[^>]+>", "", _text(el, "title")))
        summary = html.unescape(re.sub(r"<[^>]+>", " ", _text(el, "description") or _text(el, "summary")))
        if link and title:
            out.append({"url": link.strip(), "title": title[:200],
                        "published_at": _when(_text(el, "pubDate") or _text(el, "published") or _text(el, "updated")),
                        "summary": re.sub(r"\s+", " ", summary).strip()[:300]})
    return out


def fetch(url: str, timeout: float = 20) -> bytes:   # replaced in tests
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; research-feed-reader)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - feed URLs come from policy
        return r.read(5_000_000)


def collect(conn, policy: Policy, *, get=None, now: datetime | None = None) -> tuple[list[dict], dict]:
    """New items from every feed (dedupe by link), stored without their summaries; returns (the new items with their
    summaries, for this run's distill only, {feed: new or error})."""
    data_policy.require(policy, "feeds", "feed")
    get = get or fetch
    now = now or datetime.now(timezone.utc)
    max_age = timedelta(days=float(cfg(policy).get("max_item_age_days", 14)))
    new, report = [], {}
    for f in cfg(policy).get("feeds") or []:
        try:
            items = parse(get(f["url"]))
        except Exception as e:  # noqa: BLE001 - one feed down never stops the others
            report[f["name"]] = f"error: {type(e).__name__}"
            continue
        n = 0
        for it in items:
            if it["published_at"] and now - it["published_at"] > max_age:
                continue
            row = conn.execute(
                """INSERT INTO culture_items (url, feed, category, title, published_at) VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (url) DO NOTHING RETURNING url""",
                (it["url"], f["name"], f["category"], it["title"], it["published_at"])).fetchone()
            if row:
                new.append({**it, "feed": f["name"], "category": f["category"]})
                n += 1
        report[f["name"]] = n
    return new, report


def distill(conn, policy: Policy, llm, items: list[dict], *, role: str | None = None) -> list[dict]:
    """One call per batch of new items: signals in our words, stored with their sources; kind from code."""
    c = cfg(policy)
    if not items:
        return []
    role = role or c.get("role", "culture_distill")
    if not data_policy.processor_ok(policy, llm.model_for(role)):
        return []
    cats = sorted({i["category"] for i in items})
    batch = items[:int(c.get("max_items_per_call", 120))]
    lines = "\n".join(f"{k + 1}. [{i['category']} | {i['feed']}] {i['title']} — {i['summary']}"
                      for k, i in enumerate(batch))
    got = llm.structured(role, system=SYSTEM, content=f"Categories: {cats}\n\nItems:\n{lines}", schema=Signals,
                         effort=str(c.get("effort", "low")))
    need = int(c.get("corroborated_min", 2))
    single_cap = float(c.get("single_source_confidence_max", 0.5))
    expires = datetime.now(timezone.utc) + timedelta(days=float(c.get("signal_days", 120)))
    out = []
    for s in (getattr(got, "signals", None) or [])[:int(c.get("max_signals", 12))]:
        refs = [batch[k - 1] for k in s.items if isinstance(k, int) and 1 <= k <= len(batch)]
        if not refs:
            continue
        feeds = sorted({r["feed"] for r in refs})
        kind = "corroborated" if len(feeds) >= need else "single_source"
        conf = max(0.0, min(1.0, float(s.confidence)))
        if kind == "single_source":
            conf = min(conf, single_cap)
        sources = [{"url": r["url"], "feed": r["feed"],
                    "published": r["published_at"].isoformat() if r["published_at"] else None} for r in refs]
        row = conn.execute(
            """INSERT INTO culture_signals (category, signal, kind, n_sources, sources, implications, confidence,
                                           expires_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (s.category[:40], " ".join(s.signal.split()[:30]), kind, len(feeds), Jsonb(sources),
             Jsonb(s.implications.model_dump()), conf, expires)).fetchone()
        out.append({"id": row["id"], "category": s.category, "kind": kind, "n_sources": len(feeds)})
    return out


def brief(conn, policy: Policy, *, use: str, categories: list[str] | None = None, k: int | None = None) -> str:
    """The freshest, best-supported signals for a decision (`use`: style, casting, commerce or content), as prompt
    lines. Empty when there are none."""
    c = cfg(policy)
    if not c.get("enabled"):
        return ""
    k = int(k if k is not None else (c.get("k") or {}).get(use, 5))
    rows = conn.execute(
        """SELECT category, signal, kind, n_sources, implications, confidence FROM culture_signals
           WHERE expires_at > now() AND (%s::text[] IS NULL OR category = ANY(%s::text[]))
             AND coalesce(implications->>%s, '') <> ''
           ORDER BY (kind = 'corroborated') DESC, confidence DESC, created_at DESC LIMIT %s""",
        (categories, categories, use, k)).fetchall()
    return "\n".join(f"- [{r['category']}; {r['kind'].replace('_', ' ')}, {r['n_sources']} publication(s)] "
                     f"{r['signal']} -> {r['implications'].get(use)}" for r in rows)


def purge(conn, policy: Policy) -> dict:
    days = float(cfg(policy).get("item_days", 180))
    items = conn.execute("DELETE FROM culture_items WHERE fetched_at < now() - %s * interval '1 day' RETURNING url",
                         (days,)).fetchall()
    signals = conn.execute("DELETE FROM culture_signals WHERE expires_at < now() RETURNING id").fetchall()
    return {"culture_items_deleted": len(items), "culture_signals_expired": len(signals)}


def ensure_due(conn, policy: Policy, *, now: datetime | None = None):
    """One culture run a day (a brain.intelligence job, mode culture), from the Brain's cycle, while
    intelligence.culture.enabled and operations are live, or paused with `research_monitor` in
    operations.while_paused (the owner's leave for research during the build). Returns the job id, or None."""
    from .. import operations, queue
    if not cfg(policy).get("enabled"):
        return None
    live = operations.is_live(policy)
    if not live and "research_monitor" not in (policy.get("operations", "while_paused", default=[]) or []):
        return None
    day = (now or datetime.now(timezone.utc)).date().isoformat()
    return queue.enqueue(conn, "brain.intelligence", {"mode": "culture", "day": day, **({} if live else {"test": True})},
                         batch=f"research-culture-{day}", idempotency_key=f"intelligence-culture:{day}")
