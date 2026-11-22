"""Set and wardrobe bibles: a creator's recurring places and clothes, described once and pictured once.

Stored on the creator (`creators.anchors`, beside the reference images, LoRA and voice), so no table is needed:

    anchors.sets.<set_id>       {set_id, name, description, plate_uri, plate_asset_id, plate_jobs, created_from_post}
    anchors.outfits.<outfit_id> {outfit_id, description, reference_uri, reference_asset_id, reference_jobs,
                                 created_from_post}

  register   a set_id or outfit_id the director names for the first time becomes an entry; its canonical
             description is what that first shot wrote (`set.set_dressing`, `wardrobe.outfit`: the director's
             prompt asks for the fixed look there). A known id is never overwritten.
  plates     with `sets.generate_plates`, a new entry gets one image job (policy `sets.plate_kind`, Z-Image) of the
             empty room (set plate) or of the clothes alone (outfit reference): no person. The image gate reviews
             it like any image, with a "nobody in it, matches the description" scene, but with no identity gate
             (payload `plate`: there is no person to match, so it would always fail and count as a reject); the plate
             is chosen from its realism, safety and scene rows (`refresh`), plate jobs stay out of routing's cost
             per accepted output (manyworlds.routing), and the batch is `post-sets-<creator>` so it waits to share a
             machine with production. The producer refreshes pending plates every tick (`refresh_all`).
  apply      a scene whose shot names a registered set or outfit gets the canonical description in its shot (so the
             direction compiler writes it into the prompt) and the plate and outfit reference as payload
             `references`, which the consistency check (manyworlds.review.detail) compares against.
  edit       with `sets.reference_conditioning` (off until benchmarked) and the edit kind in policy `job_kinds`, the
             scene image is made by `image.qwen_edit` from [creator anchor, set plate, outfit reference] instead of
             text-only Z-Image (manyworlds.worker.handlers.image_qwen_edit).
"""

from __future__ import annotations

import copy
import hashlib
import logging

from psycopg.types.json import Jsonb

from .. import queue
from ..policy import Policy

log = logging.getLogger("manyworlds.sets")

# kind -> (anchors key, id field, image uri field, image asset field, jobs field)
_KINDS = {"set": ("sets", "set_id", "plate_uri", "plate_asset_id", "plate_jobs"),
          "outfit": ("outfits", "outfit_id", "reference_uri", "reference_asset_id", "reference_jobs")}
_PLATE_GATES = ("realism", "safety", "scene")   # identity is left out: a plate has nobody in it


def cfg(policy: Policy) -> dict:
    return policy.get("sets", default={}) or {}


def enabled(policy: Policy) -> bool:
    return bool(cfg(policy).get("enabled"))


def _join(parts, sep: str = ", ") -> str:
    return sep.join(str(p).strip() for p in parts if p and str(p).strip())


def bibles(conn, creator_id) -> dict:
    """{"sets": {set_id: entry}, "outfits": {outfit_id: entry}} for one creator."""
    row = conn.execute("SELECT anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()
    anchors = (row or {}).get("anchors") or {}
    return {"sets": dict(anchors.get("sets") or {}), "outfits": dict(anchors.get("outfits") or {})}


def known(conn, creator_id) -> dict:
    """The registered sets and outfits as the director sees them: {"sets": {id: "name: description"},
    "outfits": {id: description}}."""
    b = bibles(conn, creator_id)
    return {"sets": {k: _join([v.get("name"), v.get("description")], ": ") for k, v in b["sets"].items()},
            "outfits": {k: v.get("description", "") for k, v in b["outfits"].items()}}


def _ids(shot: dict | None) -> tuple[str, str]:
    if not isinstance(shot, dict):
        return "", ""
    return (((shot.get("set") or {}).get("set_id") or "").strip(),
            ((shot.get("wardrobe") or {}).get("outfit_id") or "").strip())


def set_entry(shot: dict, post_id=None) -> dict | None:
    """A new set's entry from the first shot that names it, or None when the shot names no set or says nothing
    about it. The canonical description is the shot's set dressing (the fixed look); without one, its location,
    practical lights and light direction."""
    st = shot.get("set") or {}
    sid = (st.get("set_id") or "").strip()
    if not sid:
        return None
    location = (st.get("location") or "").strip()
    light = (st.get("light_direction") or "").strip()
    description = (st.get("set_dressing") or "").strip() or _join(
        [location, st.get("practical_lights"), f"light from {light}" if light else ""])
    if not description:
        return None
    return {"set_id": sid, "name": location or sid.replace("_", " "), "description": description,
            "plate_uri": None, "plate_asset_id": None, "plate_jobs": [],
            "created_from_post": str(post_id) if post_id else None}


def outfit_entry(shot: dict, post_id=None) -> dict | None:
    """A new outfit's entry from the first shot that names it: the shot's `wardrobe.outfit` is its description."""
    w = shot.get("wardrobe") or {}
    oid, description = (w.get("outfit_id") or "").strip(), (w.get("outfit") or "").strip()
    if not oid or not description:
        return None
    return {"outfit_id": oid, "description": description, "reference_uri": None, "reference_asset_id": None,
            "reference_jobs": [], "created_from_post": str(post_id) if post_id else None}


def _seed(creator_id, kind: str, eid: str, attempt: int) -> int:
    return int(hashlib.sha256(f"{creator_id}:{kind}:{eid}".encode()).hexdigest()[:7], 16) + attempt * 10


def plate_payload(policy: Policy, kind: str, entry: dict, seed: int) -> dict:
    """The image job for a set plate (the empty room) or an outfit reference (the clothes, nobody wearing them)."""
    c = cfg(policy)
    prompts = c.get("prompts") or {}
    img = policy.get("generation", "image", default={}) or {}
    words = {"name": entry.get("name") or "", "description": entry.get("description") or ""}
    return {"prompt": str(prompts.get(kind, "{description}")).format(**words),
            # the image gate scores the image against it (review.gates `scene`)
            "scene": str(prompts.get(f"{kind}_scene", "{description}")).format(**words),
            "n": int(c.get("plate_candidates", 4)), "seed": seed,
            "width": img.get("width", 896), "height": img.get("height", 1152),
            "params": {"steps": img.get("steps", 9), "guidance": img.get("guidance", 0.0)},
            "plate": {"kind": kind, "id": entry[_KINDS[kind][1]]}}


def _enqueue_plate(conn, policy: Policy, creator_id, kind: str, entry: dict, attempt: int):
    eid = entry[_KINDS[kind][1]]
    return queue.enqueue(conn, str(cfg(policy).get("plate_kind", "image.zimage")),
                         plate_payload(policy, kind, entry, _seed(creator_id, kind, eid, attempt)),
                         creator_id=creator_id, batch=f"post-sets-{creator_id}",
                         idempotency_key=f"{kind}:{creator_id}:{eid}:{attempt}")


def _add(conn, creator_id, kind: str, entry: dict) -> bool:
    """Add an entry unless its id is already registered (atomic: a concurrent registration keeps the first)."""
    key, id_field = _KINDS[kind][0], _KINDS[kind][1]
    row = conn.execute(
        f"UPDATE creators SET anchors = jsonb_set(anchors, '{{{key}}}', "
        f"coalesce(anchors->'{key}', '{{}}'::jsonb) || jsonb_build_object(%s::text, %s::jsonb)) "
        f"WHERE id = %s AND NOT (coalesce(anchors->'{key}', '{{}}'::jsonb) ? %s) RETURNING id",
        (entry[id_field], Jsonb(entry), creator_id, entry[id_field])).fetchone()
    return row is not None


def _set_field(conn, creator_id, kind: str, eid: str, field: str, value) -> None:
    conn.execute("UPDATE creators SET anchors = jsonb_set(anchors, %s::text[], %s::jsonb) WHERE id = %s",
                 ([_KINDS[kind][0], eid, field], Jsonb(value), creator_id))


def register(conn, policy: Policy, creator_id, shots: list[dict], *, post_id=None) -> dict:
    """Register the sets and outfits these shots name for the first time; queue their plates. Returns the new ids:
    {"sets": [...], "outfits": [...]}."""
    out: dict = {"sets": [], "outfits": []}
    if not enabled(policy):
        return out
    have = bibles(conn, creator_id)
    plates = bool(cfg(policy).get("generate_plates"))
    for shot in shots or []:
        if not isinstance(shot, dict):
            continue
        for kind, entry in (("set", set_entry(shot, post_id)), ("outfit", outfit_entry(shot, post_id))):
            key, id_field, _, _, jobs_field = _KINDS[kind]
            if entry is None or entry[id_field] in have[key]:
                continue
            if plates:
                entry[jobs_field] = [str(_enqueue_plate(conn, policy, creator_id, kind, entry, 0))]
            if _add(conn, creator_id, kind, entry):
                out[key].append(entry[id_field])
            have[key][entry[id_field]] = entry
    if out["sets"] or out["outfits"]:
        log.info("creator %s: new sets %s, outfits %s", creator_id, out["sets"], out["outfits"])
    return out


def _pick(conn, policy: Policy, job_ids: list[str]) -> tuple[dict | None, bool]:
    """(the best plate among these jobs' images, whether every image has been reviewed). A plate passes the gate's
    realism, safety and scene rows, with the scene and realism scores at policy's bars."""
    c = cfg(policy)
    rows = conn.execute(
        """SELECT a.id, a.uri, count(r.id) AS n,
                  coalesce(bool_and(r.passed) FILTER (WHERE r.gate = ANY(%(gates)s)), false) AS ok,
                  min(r.score) FILTER (WHERE r.gate = 'scene') AS scene,
                  min(r.score) FILTER (WHERE r.gate = 'realism') AS realism
           FROM assets a LEFT JOIN reviews r ON r.asset_id = a.id
           WHERE a.job_id = ANY(%(jobs)s::uuid[]) AND a.kind = 'image'
           GROUP BY a.id, a.uri ORDER BY a.uri""",
        {"gates": list(_PLATE_GATES), "jobs": job_ids}).fetchall()
    good = [r for r in rows if r["n"] and r["ok"] and r["scene"] is not None and r["realism"] is not None
            and float(r["scene"]) >= float(c.get("min_plate_scene", 7))
            and float(r["realism"]) >= float(c.get("min_plate_realism", 6))]
    best = max(good, key=lambda r: float(r["scene"]) + float(r["realism"]), default=None)
    return best, bool(rows) and all(r["n"] for r in rows)


def refresh(conn, policy: Policy, creator_id) -> dict:
    """Choose plates from reviewed plate jobs; retry a set or outfit whose images were all rejected (or whose job
    failed), up to `sets.max_plate_attempts` jobs. Returns {"chosen": [...], "retried": [...], "gave_up": [...]}."""
    out: dict = {"chosen": [], "retried": [], "gave_up": []}
    if not enabled(policy):
        return out
    c = cfg(policy)
    b = bibles(conn, creator_id)
    for kind, (key, _, uri_field, asset_field, jobs_field) in _KINDS.items():
        for eid, entry in b[key].items():
            jobs = [str(j) for j in entry.get(jobs_field) or []]
            if entry.get(uri_field) or entry.get("plate_failed") or not jobs:
                continue
            best, all_reviewed = _pick(conn, policy, jobs)
            if best is not None:
                _set_field(conn, creator_id, kind, eid, uri_field, best["uri"])
                _set_field(conn, creator_id, kind, eid, asset_field, str(best["id"]))
                out["chosen"].append(f"{kind}:{eid}")
                continue
            last = conn.execute("SELECT status FROM jobs WHERE id = %s", (jobs[-1],)).fetchone()
            status = (last or {}).get("status")
            if not ((status == "done" and all_reviewed) or status in ("failed", "dead") or last is None):
                continue   # still being made or reviewed
            if not c.get("generate_plates"):
                continue
            if len(jobs) >= int(c.get("max_plate_attempts", 3)):
                _set_field(conn, creator_id, kind, eid, "plate_failed", True)
                out["gave_up"].append(f"{kind}:{eid}")
                continue
            job = _enqueue_plate(conn, policy, creator_id, kind, entry, len(jobs))
            _set_field(conn, creator_id, kind, eid, jobs_field, [*jobs, str(job)])
            out["retried"].append(f"{kind}:{eid}")
    return out


def refresh_all(conn, policy: Policy) -> dict:
    """`refresh` for every creator with a set or outfit still waiting for its plate (the producer tick), so the
    first post in a new set gets its references once the plate is reviewed, not only when a later scene asks.
    Returns {creator_id: refresh result} for creators where something changed."""
    out: dict = {}
    if not enabled(policy):
        return out
    rows = conn.execute(
        """SELECT DISTINCT c.id FROM creators c,
                  LATERAL (SELECT value FROM jsonb_each(coalesce(c.anchors->'sets', '{}'::jsonb))
                           UNION ALL SELECT value FROM jsonb_each(coalesce(c.anchors->'outfits', '{}'::jsonb))) e(v)
           WHERE jsonb_typeof(e.v) = 'object'
             AND coalesce(e.v->>'plate_uri', e.v->>'reference_uri') IS NULL
             AND coalesce(e.v->>'plate_failed', 'false') <> 'true'
             AND CASE WHEN jsonb_typeof(coalesce(e.v->'plate_jobs', e.v->'reference_jobs')) = 'array'
                      THEN jsonb_array_length(coalesce(e.v->'plate_jobs', e.v->'reference_jobs')) ELSE 0 END > 0
           ORDER BY c.id""").fetchall()
    for r in rows:
        res = refresh(conn, policy, r["id"])
        if any(res.values()):
            out[str(r["id"])] = res
    return out


def _with_canon(own: str, canon: str) -> str:
    own, canon = (own or "").strip(), (canon or "").strip()
    if not own or own.lower() in canon.lower():
        return canon
    if canon.lower() in own.lower():
        return own
    return f"{canon}; {own}"


def apply(conn, policy: Policy, creator_id, shot: dict | None) -> tuple[dict | None, list[dict]]:
    """(the shot with its registered set's and outfit's canonical descriptions, the references to judge it against
    [{kind: set|outfit, uri}]). A shot that names nothing registered is returned as it is, with no references."""
    if not enabled(policy) or not isinstance(shot, dict):
        return shot, []
    sid, oid = _ids(shot)
    if not sid and not oid:
        return shot, []
    b = bibles(conn, creator_id)
    st, ou = b["sets"].get(sid) if sid else None, b["outfits"].get(oid) if oid else None
    if not st and not ou:
        return shot, []
    if (st and not st.get("plate_uri")) or (ou and not ou.get("reference_uri")):
        refresh(conn, policy, creator_id)   # a plate reviewed since the last scene may be ready now
        b = bibles(conn, creator_id)
        st, ou = b["sets"].get(sid) if sid else None, b["outfits"].get(oid) if oid else None
    out = copy.deepcopy(shot)
    refs: list[dict] = []
    if st:
        s = out.setdefault("set", {})
        s["set_dressing"] = _with_canon(s.get("set_dressing", ""), st.get("description", ""))
        if not (s.get("location") or "").strip():
            s["location"] = st.get("name", "")
        if st.get("plate_uri"):
            refs.append({"kind": "set", "uri": st["plate_uri"]})
    if ou:
        w = out.setdefault("wardrobe", {})
        w["outfit"] = ou.get("description") or w.get("outfit", "")
        if ou.get("reference_uri"):
            refs.append({"kind": "outfit", "uri": ou["reference_uri"]})
    return out, refs


# --- reference-conditioned scene images (image.qwen_edit) -------------------------------------------------------

def edit_cfg(policy: Policy) -> dict:
    return cfg(policy).get("edit") or {}


def edit_kind(policy: Policy) -> str:
    return str(edit_cfg(policy).get("kind", "image.qwen_edit"))


def edit_pools(policy: Policy) -> list[str] | None:
    """The edit kind's pools (policy job_kinds): its jobs are pinned there, on 48-80 GB cards only. The kind has its
    own worker family (job_kinds.<kind>.family: edit), so image machines don't list it either."""
    return list(policy.get("job_kinds", edit_kind(policy), "pools", default=[]) or []) or None


def conditioning(conn, policy: Policy, creator_id, references: list[dict]) -> list[dict] | None:
    """The input images for a reference-conditioned scene, [{kind: person|set|outfit, uri}] with the creator's
    anchor first, or None (text-only Z-Image) unless `sets.reference_conditioning` is on, the edit kind is in
    policy job_kinds, and there are an anchor and at least one reference."""
    if not enabled(policy) or not cfg(policy).get("reference_conditioning") or not references:
        return None
    if edit_kind(policy) not in (policy.get("job_kinds", default={}) or {}):
        return None
    row = conn.execute("SELECT anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()
    anchors = ((row or {}).get("anchors") or {}).get("reference_images") or []
    anchor = next((a.get("uri") for a in anchors if isinstance(a, dict) and a.get("uri")), None)
    if not anchor:
        return None
    return [{"kind": "person", "uri": anchor}, *[r for r in references if r.get("kind") in ("set", "outfit")]]


def edit_payload(policy: Policy, payload: dict, images: list[dict]) -> dict:
    """A scene payload turned into an image.qwen_edit payload: the images in order, a prompt that says which picture
    is which, the edit settings, and no LoRA (the creator's LoRA is for Z-Image)."""
    e = edit_cfg(policy)
    lead = []
    for n, im in enumerate(images, start=1):
        words = e.get(im["kind"])
        if words:
            lead.append(str(words).format(n=n))
    out = {k: v for k, v in payload.items() if k not in ("lora_uri", "lora_scale", "params")}
    out.update(prompt=_join(lead, " ") + (" " if lead else "") + payload["prompt"],
               images=[im["uri"] for im in images], n=int(e.get("candidates", payload.get("n", 4))),
               params={"steps": int(e.get("steps", 4)), "true_cfg": float(e.get("true_cfg", 1.0)),
                       "negative_prompt": str(e.get("negative_prompt", " ")),
                       "lightning": bool(e.get("lightning", True))})
    return out
