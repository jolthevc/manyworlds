"""The director: an accepted idea's scenes become shot sheets (manyworlds.direction), one per scene.

Called where an accepted idea becomes a post (manyworlds.brain.ideas.schedule_ready), for the formats in policy
`direction.direct_formats` while `direction.enabled`. One structured call to the LLM role `direction.role`
(`director`), charged to the agent `director` (its own daily budget). It is asked with a strict schema
(`DirectedShots`): ShotSheet inside an ordinary structured output is too large a grammar (HTTP 400, measured
2026-11-22), which is also why the planner and the showrunner no longer write shots themselves.

The director composes from a toolbox, not a format template (owner 2026-11-23): its brief is the capability catalogue
(policy `capabilities`, only what the pipeline can render now; the rest is named as unavailable), the creator's
archetype (bible archetype, niche, one-liner) with how that kind of creator communicates
(`direction.director.archetypes`: talk-led, label-led, voiceover-led), the recipes (`moves`) for the post's format it
may draw on, the craft rules and the "reads as AI" tells (`direction.director.craft`, `.avoid`), lengths by intent,
and the names it may use: caption styles, on-screen text kinds, styles and positions, transitions, room-tone words
and the licensed motion references.

What comes back is checked before it is used: one shot per scene, a move's shot defaults (policy
`moves.library.<move>.shot`) fill what the director left empty in the shot that performs the move, the shots'
lines and overlays keep the voice rules, a motion reference not in `dance.references` is cleared, and a talking
video's lines must be its script's words (otherwise the lines are cleared and production splits the script as it
does without shots). Anything that fails raises; the caller then makes the post undirected, never drops it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .. import direction
from ..direction import TEXT_KINDS, TRANSITION_KINDS, DirectedShots, ShotSheet, spoken
from . import budget, memory, planner

log = logging.getLogger("manyworlds.director")
AGENT = "director"

SYSTEM = """You are the director for an original fictional social media creator at Manyworlds, a studio
whose creators are all openly labeled as AI. An idea for a post has been accepted. Turn each of its scenes
into a shot sheet, one per scene, in the same order: never add, drop or merge scenes, and keep what each
scene says happens. Direct the set and its light, wardrobe and look, framing and camera, blocking beat by
beat, performance and lines, sound, the edit and continuity, so every shot looks like a real phone photo or
video of this person's life.

Rules:
- The creator's appearance is fixed and added to every prompt separately: never describe their face, body,
  age or ethnicity, only what changes per scene (outfit, hair state, makeup, skin state).
- Recurring sets and outfits are named (`set.set_id`, `wardrobe.outfit_id`, short snake_case) and reused with
  the same name and description; reuse the known sets and outfits you are given when a scene is in one.
- A new named set's `set.set_dressing` is its canonical look, kept for every later post: the fixed layout,
  materials, colours, furniture and fixed props, and its light sources (windows, lamps), written as an empty
  room: objects and surfaces only. A new named outfit's `wardrobe.outfit` is its canonical description:
  garments, colours, fabrics, cut. For a known set the canonical look is added for you: leave `set_dressing`
  empty or name only what differs in this scene.
- Write every visual field (set, wardrobe, look, framing, blocking) as what the camera sees, in positive
  words: these fields become image and video prompts, where a negation ("no mirror", "without people")
  summons the thing. Say "plain wall", "an empty street", "bare hands" instead.
- Compose every shot from what the studio can render (the capability catalogue in the brief; its `write`
  says which fields ask for it), never from a capability listed as unavailable. The post's format is only
  the container; recipes are examples you may draw on or ignore. First decide how this creator communicates
  (talk-led, label-led or voiceover-led, how much they talk, which devices, what text) from their archetype,
  then direct every shot that way.
- Hands are relaxed or out of frame unless the action needs them.
- Text belongs in `edit.on_screen_text` (added in post), never on signs, screens, packaging or other surfaces:
  a list of items, each a kind (label, hook, timestamp, question), its words, when it shows (seconds from
  the shot's start; end 0 = to the end of the shot), and optionally a style, a position and a number. Video
  only: photos carry none. The hook item goes in the first shot, at 0.
- `edit.caption_style` is a caption style name for talk-led posts, "none" when the post is label-led (no
  word captions), or "" for the creator's default.
- `edit.transition_in` says how the shot starts after the previous one: cut, match (a cut inside a motion
  both shots share), whip, blur or flash.
- `sound.room_tone` names the place's own quiet sound in plain words, in every shot.
- `motion_reference` is one of the listed motion reference ids, or empty; never invent one.
- `framing.camera_move` is one of the allowed camera moves, or empty.
- In a talking video each shot is one take: its `performance.lines` are the script's words, verbatim, split
  across the shots in order, with nothing added or reworded; `delivery` says how a line is said and is never
  spoken. A shot that shows something else while the voice continues is `cutaway: true`. In every other
  format `performance.lines` is empty.
- When the idea has a move, the shot that performs it starts from the move's `shot` defaults and follows its
  note; give that shot's index (from 0) in `move_shot`, else -1.
- Every field must be present: use "" for text you leave unset, 0 for numbers you leave unset, [] for empty
  lists and false for flags. Leave a field unset when it doesn't matter."""


def system_prompt(policy) -> str:
    return SYSTEM + craft_rules(policy) + planner.voice_rules(policy)


def _cfg(policy) -> dict:
    return policy.get("direction", default={}) or {}


def _brief(policy) -> dict:
    return _cfg(policy).get("director") or {}


def craft_rules(policy) -> str:
    """The craft rules and the "reads as AI" tells (policy `direction.director.craft`, `.avoid`) as a prompt section."""
    b = _brief(policy)
    craft = [f"- {r}" for r in b.get("craft") or [] if str(r).strip()]
    avoid = [str(a).strip() for a in b.get("avoid") or [] if str(a).strip()]
    parts = []
    if craft:
        parts.append("\n\nCRAFT (what the top human creators do):\n" + "\n".join(craft))
    if avoid:
        parts.append("\n\nWhat reads as AI, never direct it: " + "; ".join(avoid) + ".")
    return "".join(parts)


def communication(policy, bible: dict) -> list[str]:
    """How this kind of creator communicates, from the archetype library (manyworlds.brain.archetypes): the
    profile's named archetypes, else every archetype whose words start a word in the bible's archetype, niche or
    one-liner; none matching gives the library default."""
    from . import archetypes
    return archetypes.communicates(policy, bible)


def catalogue(policy) -> tuple[dict, list[str]]:
    """The capability catalogue for the brief: ({group: {name: {description, how, limits, write}}} of what the
    pipeline can render now, ['group.name' that it can't])."""
    usable: dict = {}
    missing: list[str] = []
    for group, entries in direction.capabilities(policy).items():
        for name, e in entries.items():
            if not e["available"]:
                missing.append(f"{group}.{name}")
                continue
            usable.setdefault(group, {})[name] = {k: e[k] for k in ("description", "how", "limits", "write")
                                                  if e.get(k)}
    return usable, missing


def recipes(policy, fmt: str) -> dict:
    """The recipes (policy `moves`) for a format, whose readiness the models carry: {name: {note, suits, uses,
    length_s}}. Examples the director may draw on, never a template."""
    offered = set(_cfg(policy).get("offer_readiness") or [])
    out = {}
    for name, m in (policy.get("moves", "library", default={}) or {}).items():
        if fmt in (m.get("formats") or []) and m.get("readiness") in offered:
            out[name] = {k: m[k] for k in ("note", "suits", "uses", "length_s") if m.get(k)}
    return out


def _names(policy) -> dict:
    """The names a shot may use, for the brief: caption styles, on-screen text, transitions, room tone, motion refs."""
    room = policy.get("sound", "room_tone", "keywords", default=[]) or []
    refs = {str(r["id"]): r for r in (policy.get("dance", "references", default=[]) or [])
            if isinstance(r, dict) and r.get("id")}
    return {
        "caption_styles": [*list((policy.get("burned_captions", "styles", default={}) or {}).keys()), "none"],
        "on_screen_text": {"kinds": list(TEXT_KINDS),
                           "styles": list((policy.get("on_screen_text", "styles", default={}) or {}).keys()),
                           "positions": list((policy.get("on_screen_text", "positions", default={}) or {}).keys())},
        "transitions": list(TRANSITION_KINDS),
        "room_tone_words": {str(r.get("preset")): list(r.get("words") or []) for r in room if isinstance(r, dict)},
        "motion_references": [{"id": i, **{k: refs[i][k] for k in ("move", "notes") if refs[i].get(k)}}
                              for i in motion_references(policy)],
    }


def motion_references(policy) -> list[str]:
    """The motion reference ids a shot may name now: `dance.references` while dance is on (direction.
    motion_reference_ids), and only while production renders them (the `capabilities.actions.motion_reference`
    entry's switches; no such entry, no extra condition)."""
    entry = policy.get("capabilities", "actions", "motion_reference", default=None)
    if entry is not None and not direction.capability_available(entry, policy):
        return []
    return direction.motion_reference_ids(policy)


def known_sets(conn, creator_id, n: int) -> dict:
    """Named sets and outfits from the creator's recent directed posts, newest description first:
    {"sets": {set_id: location}, "outfits": {outfit_id: outfit}}."""
    rows = conn.execute(
        "SELECT body->'shots' AS shots FROM posts WHERE creator_id = %s AND status <> 'dropped' "
        "AND jsonb_typeof(body->'shots') = 'array' ORDER BY created_at DESC LIMIT %s", (creator_id, n)).fetchall()
    sets: dict = {}
    outfits: dict = {}
    for r in rows:
        for s in r["shots"] or []:
            if not isinstance(s, dict):
                continue
            st, w = s.get("set") or {}, s.get("wardrobe") or {}
            if st.get("set_id") and st.get("location"):
                sets.setdefault(st["set_id"], st["location"])
            if w.get("outfit_id") and w.get("outfit"):
                outfits.setdefault(w["outfit_id"], w["outfit"])
    return {"sets": sets, "outfits": outfits}


def _known(conn, policy, creator_id, n: int) -> dict:
    """Named sets and outfits for reuse: those of recent posts, with the registered canon (creator.sets) winning."""
    from ..creator import sets
    out = known_sets(conn, creator_id, n)
    if sets.enabled(policy):
        canon = sets.known(conn, creator_id)
        out = {k: {**out[k], **canon[k]} for k in ("sets", "outfits")}
    return out


def _prompt(conn, policy, body: dict, creator_id, fmt: str, scenes: list[str], move_name: str,
            move: dict | None) -> str:
    cfg = _cfg(policy)
    row = conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    bible = (row or {}).get("bible") or {}
    creator = {k: bible[k] for k in ("display_name", "archetype", "niche", "one_line", "appearance", "voice_notes",
                                     "style_and_wardrobe", "home_and_places") if bible.get(k)}
    facts = [f"{m['kind']}: {m['text']}"
             for m in memory.facts(conn, creator_id, limit=int(cfg.get("memory_facts", 40)))]
    notes = policy.get("planner", "format_notes", default={}) or {}
    idea = {"format": fmt, "format_note": notes.get(fmt, ""), "scenes": scenes,
            **{k: body[k] for k in ("script", "on_screen_text", "caption", "staging", "flavor") if body.get(k)}}
    if move:
        idea["move"] = {"name": move_name, "note": move.get("note", ""), "shot": move.get("shot") or {}}
    usable, missing = catalogue(policy)
    names = _names(policy)
    refs = names.pop("motion_references")
    return (f"The creator (appearance for reference only): {json.dumps(creator, default=str)}\n"
            f"How this kind of creator communicates (from their archetype): {json.dumps(communication(policy, bible))}\n"
            f"What is known about their life: {json.dumps(facts)}\n"
            f"Known sets and outfits (reuse the names): "
            f"{json.dumps(_known(conn, policy, creator_id, int(cfg.get('recent_posts', 20))))}\n\n"
            f"What the studio can render now (compose every shot from these): {json.dumps(usable)}\n"
            f"Not available now (never use): {json.dumps(missing)}\n"
            f"Allowed camera moves: {json.dumps(list(cfg.get('camera_moves') or []))}\n"
            f"Recipes for a {fmt} you may draw on (examples, chosen by what suits this creator; not rules): "
            f"{json.dumps(recipes(policy, fmt))}\n"
            f"Length by intent: {json.dumps(_brief(policy).get('lengths') or {})}\n"
            f"Names you may use: {json.dumps(names)}\n"
            + (f"Motion references (motion_reference, only these ids): {json.dumps(refs)}\n" if refs else
               "Motion references: none now (leave motion_reference empty).\n")
            + f"\nThe accepted idea: {json.dumps(idea, default=str)}\n\n"
            f"Write exactly {len(scenes)} shot sheet(s), one per scene, in order.")


def _empty(value) -> bool:
    return value is None or value is False or (isinstance(value, (str, list, dict)) and not value)


def fill(shot: dict, defaults: dict) -> dict:
    """The shot with its empty fields ("" / None / [] / {} / false) taken from `defaults`, recursively; what the
    director set wins."""
    out = dict(shot)
    for k, dv in (defaults or {}).items():
        sv = out.get(k)
        if isinstance(dv, dict) and isinstance(sv, dict):
            out[k] = fill(sv, dv)
        elif _empty(sv):
            out[k] = dv
    return out


_WORDS = re.compile(r"[\w']+")


def _words(text: str) -> list[str]:
    return _WORDS.findall((text or "").lower().replace("’", "'"))


def direct(conn, policy, llm, body: dict, creator_id, *, fmt: str | None = None, post_id=None) -> list[dict]:
    """One shot sheet per scene of a post or idea body, as plain dicts (validated through ShotSheet). Raises when
    there is nothing usable; the caller keeps the post undirected. Sets and outfits named for the first time are
    registered in the creator's set and wardrobe bibles (creator.sets), with their plates queued."""
    fmt = fmt or body.get("format") or ""
    scenes = [s.strip() for s in body.get("scenes") or [] if isinstance(s, str) and s.strip()]
    if not scenes:
        raise ValueError("nothing to direct: no scenes")
    cfg = _cfg(policy)
    move_name = body.get("move") or ""
    move = (policy.get("moves", "library", default={}) or {}).get(move_name) if move_name else None
    content = _prompt(conn, policy, body, creator_id, fmt, scenes, move_name, move)
    agent = budget.tag_for(conn, AGENT, creator_id)
    with llm.tagged(agent, None, before_call=lambda: budget.check(conn, policy, agent)):
        out = llm.structured(str(cfg.get("role", AGENT)), system=system_prompt(policy), content=content,
                             schema=DirectedShots, effort=str(cfg.get("effort", "medium")), creator_id=creator_id)
    shots: list[dict[str, Any]] = [s.model_dump(mode="json") for s in (getattr(out, "shots", None) or [])]
    if not shots:
        raise ValueError("the director answered no shots")
    shots = shots[:len(scenes)] + [ShotSheet().model_dump(mode="json") for _ in range(len(scenes) - len(shots))]
    if move and move.get("shot"):
        k = int(getattr(out, "move_shot", -1))
        k = k if 0 <= k < len(shots) else (0 if len(shots) == 1 else -1)
        if k >= 0:
            shots[k] = fill(shots[k], move["shot"])
    for s in shots:
        for line in (s.get("performance") or {}).get("lines") or []:
            if not line.get("pause_after_s"):
                line["pause_after_s"] = None   # 0 from the strict schema means "the default pause"
    shots = [ShotSheet.model_validate(s).model_dump(mode="json") for s in shots]
    usable_refs = motion_references(policy)
    for s in shots:
        if s.get("motion_reference") and s["motion_reference"] not in usable_refs:
            log.warning("director named motion reference %r, which is not usable now; cleared", s["motion_reference"])
            s["motion_reference"] = ""
    issues = planner.shot_voice_issues(policy, shots)
    if issues:
        raise ValueError("the shots read as AI: " + ", ".join(issues))
    if fmt == "talking_video":
        said = _words(" ".join(spoken(s) for s in shots))
        if said and said != _words(body.get("script") or ""):
            log.warning("director's lines are not the script's words; lines cleared, the script is split as before")
            for s in shots:
                s["performance"]["lines"] = []
    _register(conn, policy, creator_id, shots, post_id)
    return shots


def _register(conn, policy, creator_id, shots: list[dict], post_id) -> None:
    from ..creator import sets
    if not sets.enabled(policy):
        return
    try:
        sets.register(conn, policy, creator_id, shots, post_id=post_id)
    except Exception as e:  # noqa: BLE001 - a set not registered now is registered by its next post
        log.warning("set and wardrobe registration failed for creator %s (%s: %s)", creator_id,
                    type(e).__name__, str(e)[:300])
