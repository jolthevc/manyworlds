"""What research may collect, keep and send where (owner 2026-11-25; policy `data_policy`; docs/superpowers/specs/
2026-11-25-data-policy.md). Maximise permissible intelligence, minimise retained third-party content.

  green  persistent intelligence: factual creator and post data (handles, public professional names, profile links,
         counts, growth, categories, commercial signals), our own observations, findings, pattern cards, our own data.
  amber  transient observation: third-party photos, videos, frames and caption text, held only until they are read
         (and at most `retention.media_max_days` / `caption_days`), then deleted; what was learned stays.
  red    never: biometric templates of real people (face embeddings, voiceprints), private content, comment text or
         follower lists, contact details, sensitive personal data, training on third-party media.

Enforced in code, not prompts: a source/task not approved in `data_policy.sources` is refused (`refused`, used by the
provider router and the direct Apify pulls); stored creator and post records go through `clean_creator` and
`caption_features`; research media gets a short purge date and is purged once read (research.references.purge); only
the `processors` may read research media; `red` lists what no code path may produce (impersonation has no face check).
"""

from __future__ import annotations

import re

from ..policy import Policy


class SourcePaused(RuntimeError):
    """A source or task the data policy does not approve (paused while its rights are unclear, or blocked)."""


def cfg(policy: Policy) -> dict:
    return policy.get("data_policy", default={}) or {}


def source(policy: Policy, name: str) -> dict:
    return (cfg(policy).get("sources") or {}).get(name) or {}


def refused(policy: Policy, name: str, task: str) -> str | None:
    """Why `name` may not do `task` (profiles, posts, posts_media, search, feed), or None. Unlisted sources are
    refused: a new source needs an entry, with its basis, before any code may use it."""
    s = source(policy, name)
    if not s:
        return f"{name}: not in data_policy.sources"
    status = s.get("status")
    if status != "approved":
        return f"{name}: {status or 'unreviewed'} ({s.get('why') or s.get('basis') or 'no basis recorded'})"
    if task not in (s.get("tasks") or []):
        return f"{name}: task {task} not approved"
    return None


def require(policy: Policy, name: str, task: str) -> None:
    why = refused(policy, name, task)
    if why:
        raise SourcePaused(why)


def media_allowed(policy: Policy, name: str) -> bool:
    """Whether media from this source may be held transiently for analysis."""
    return bool(source(policy, name).get("transient_media")) and refused(policy, name, "posts_media") is None


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d")
_CONTACT_WORDS = re.compile(r"\b(?:e-?mail|contact|whats?app|dm for|booking|inquir\w*|enquir\w*)\b\s*[:\-]?", re.I)


def clean_bio(text: str | None, policy: Policy | None = None) -> str | None:
    """A public bio without contact details (red: e-mail addresses, phone numbers) and cut to `bio_chars`."""
    if not text:
        return text
    out = _PHONE.sub("", _EMAIL.sub("", str(text)))
    out = re.sub(r"\s{2,}", " ", out).strip(" |·-:\n")
    n = int((cfg(policy).get("fields") or {}).get("bio_chars", 300)) if policy is not None else 300
    return out[:n] or None


def clean_creator(fields: dict, policy: Policy | None = None) -> dict:
    """A creator record as the registry keeps it: the allowed keys only, the bio without contact details."""
    allowed = set(((cfg(policy).get("fields") or {}).get("creator")) or []) if policy is not None else set()
    out = {k: v for k, v in fields.items() if not allowed or k in allowed}
    if "bio" in out:
        out["bio"] = clean_bio(out["bio"], policy)
    return out


_TAG = re.compile(r"#(\w+)")
_MENTION = re.compile(r"@([\w.]+)")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_CTA = re.compile(r"\b(link in bio|comment|save this|share|follow|tag a|shop|use code|swipe)\b", re.I)


def caption_features(caption: str | None) -> dict:
    """What the learning loops need from a caption, kept after the caption text itself is gone (amber): its size,
    hashtags, @mentions (brands, collaborators: commercial intelligence), emoji count, question, calls to action, and
    whether it discloses a partnership. Never the words themselves."""
    c = caption or ""
    words = c.split()
    first = c.strip().split("\n", 1)[0]
    return {"words": len(words), "chars": len(c), "lines": c.count("\n") + 1 if c else 0,
            "first_line_words": len(first.split()), "hashtags": sorted({t.lower() for t in _TAG.findall(c)})[:30],
            "mentions": sorted({m.lower().rstrip(".") for m in _MENTION.findall(c)})[:20],
            "emoji": len(_EMOJI.findall(c)), "question": "?" in c,
            "ctas": sorted({m.lower() for m in _CTA.findall(c)}),
            "disclosure": bool(re.search(r"#(ad|sponsored|partner|gifted)\b|paid partnership", c, re.I))}


def processor_ok(policy: Policy, model: str | None) -> bool:
    """Whether a model may read third-party research media and text. A model on a third-party open-weight host
    (llm.open_models) needs its prefix in data_policy.processors (its host's terms checked: no training on our inputs,
    bounded retention); the directly contracted models (Claude) are the listed default."""
    if not model:
        return False
    prefixes = cfg(policy).get("processors") or []
    if any(str(model).startswith(p) for p in prefixes):
        return True
    return str(model) not in (policy.get("llm", "open_models", default={}) or {})


def retention(policy: Policy) -> dict:
    return cfg(policy).get("retention") or {}


def audit(conn, policy: Policy, *, apply: bool = False) -> dict:
    """What research holds, by class (`mw research data-audit`), and with `apply` the clean-up of what the policy no
    longer allows: contact details out of bios, caption_features for posts stored before them. Media and caption text
    go by research.references.purge on the housekeeping tick."""
    q = lambda sql, *a: conn.execute(sql, a).fetchone()   # noqa: E731
    green = {"creators": q("SELECT count(*) n FROM reference_creators")["n"],
             "posts": q("SELECT count(*) n FROM reference_posts")["n"],
             "annotations": q("SELECT count(*) n FROM reference_annotations")["n"],
             "pattern_cards": q("SELECT count(*) n FROM pattern_cards")["n"]}
    media = q("""SELECT count(*) AS items, coalesce(sum((m->>'bytes')::bigint), 0) AS bytes,
                        count(DISTINCT p.id) AS posts, min(p.media_purge_after) AS next_purge,
                        max(p.media_purge_after) AS last_purge
                 FROM reference_posts p, jsonb_array_elements(p.media) m""")
    amber = {"media_items": media["items"], "media_mb": round(int(media["bytes"]) / 1e6, 1),
             "posts_with_media": media["posts"], "media_purge_between": [str(media["next_purge"]),
                                                                         str(media["last_purge"])],
             "captions_held": q("SELECT count(*) n FROM reference_posts WHERE caption IS NOT NULL")["n"]}
    bios = conn.execute("SELECT id, bio FROM reference_creators WHERE bio IS NOT NULL").fetchall()
    contact = [b for b in bios if _EMAIL.search(b["bio"] or "") or _PHONE.search(b["bio"] or "")]
    missing = q("SELECT count(*) n FROM reference_posts WHERE caption IS NOT NULL AND caption_features = '{}'::jsonb")["n"]
    red = {"bios_with_contact_details": len(contact),
           "face_or_voice_columns": [r["column_name"] for r in conn.execute(
               """SELECT table_name || '.' || column_name AS column_name FROM information_schema.columns
                  WHERE table_name LIKE 'reference_%%' AND (column_name ILIKE '%%face%%' OR column_name ILIKE '%%embed%%'
                        OR column_name ILIKE '%%voice%%')""").fetchall()]}
    out = {"green": green, "amber": amber, "red": red, "caption_features_missing": missing, "applied": {}}
    if apply:
        for b in contact:
            conn.execute("UPDATE reference_creators SET bio = %s, updated_at = now() WHERE id = %s",
                         (clean_bio(b["bio"], policy), b["id"]))
        from psycopg.types.json import Jsonb
        n = 0
        for r in conn.execute("SELECT id, caption FROM reference_posts WHERE caption IS NOT NULL "
                              "AND caption_features = '{}'::jsonb").fetchall():
            conn.execute("UPDATE reference_posts SET caption_features = %s WHERE id = %s",
                         (Jsonb(caption_features(r["caption"])), r["id"]))
            n += 1
        out["applied"] = {"bios_cleaned": len(contact), "caption_features_backfilled": n}
    return out
