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
and the licensed motion references. That static part (`static_brief`, policy and format only) leads the message as a
block marked for the prompt cache; the creator's part follows (`creator_brief`): her persona, dossier planning view,
archetype hooks, edit grammar and pitfalls, her talent manager's focus and stop, style, recent outfits and shot types,
known sets, and the idea with its reasoning, attributes, series (premise, hook), pillar, tone and canon. The edit pass
sends the same blocks plus the directed shots, so it reads the static part from the cache too.

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
from collections import Counter
from typing import Any

from .. import direction
from ..direction import TEXT_KINDS, TRANSITION_KINDS, DirectedEdits, DirectedShots, ShotSheet, spoken
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
- When the brief gives her style: only her anchor sets are recurring rooms (`set.set_id` = that id, reused
  whenever a scene is in one: where she talks, gets ready, records). Every other place is a one-off: leave
  `set_id` empty and make it specific and varied (cafes, streets, friends' places, trips; she has a life beyond
  her flat). Outfits don't recur: leave `wardrobe.outfit_id` empty and dress her from her capsule and wardrobe
  style, different from her recent outfits; her signature items do recur. Without a style, recurring sets and
  outfits are named (`set.set_id`, `wardrobe.outfit_id`, short snake_case) and reused with the same name.
- `shot_type`: when the brief gives her shot vocabulary (her style's `shot_types`), EVERY shot names one of
  those types, never empty: a shot of her face uses the vocabulary's face type that fits (a portrait, a talking
  anchor, a mirror shot, a candid). Most of an aspirational feed is her world (her back on a street at sunset, a
  ringed hand on a glass, a table from above, a window, her shoes), not her face to camera. Frame and light each
  shot as the vocabulary says; when its face is not "yes", direct no face, gaze or expression. Vary the types
  against her recent posts' shot types in the brief. Follow the brief's carousel rules (variety, one imperfect
  frame, the dump order). Only without a vocabulary is `shot_type` empty.
- The idea's reasoning, series, pillar and tone say what the post is for; her persona, dossier and talent
  manager's focus say who it is for. Direct every shot toward that: the hook, the performance and the on-screen
  text in her voice, using her archetype's hooks and edit grammar and avoiding its pitfalls.
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


def _style_view(conn, policy, creator_id) -> dict:
    from ..creator import profile as profile_mod
    from .agents import style_director
    cur = profile_mod.current(conn, creator_id)
    return style_director.planning_view(policy, cur[1] if cur else None)


def recent_outfits(conn, creator_id, n: int) -> list[str]:
    """What she wore in her recent directed posts, newest first (the director dresses her differently)."""
    rows = conn.execute(
        "SELECT body->'shots' AS shots FROM posts WHERE creator_id = %s AND status <> 'dropped' "
        "AND jsonb_typeof(body->'shots') = 'array' ORDER BY created_at DESC LIMIT %s", (creator_id, n)).fetchall()
    seen: list[str] = []
    for r in rows:
        for s in r["shots"] or []:
            o = ((s or {}).get("wardrobe") or {}).get("outfit") if isinstance(s, dict) else ""
            if o and o not in seen:
                seen.append(o)
    return seen[:n]


def styled(policy, shots: list[dict], look: dict, prof_style: dict) -> list[dict]:
    """Code's part of an aesthetic shot: its subject from the shot vocabulary, her grade's prompt words (the night
    grade for night and flash shots), and, for a creator with a style, no recurring outfits and no recurring set
    that isn't one of her anchor sets (manyworlds.creator.sets registers only what is left)."""
    from . import aesthetics
    anchors = set((look or {}).get("anchor_sets") or {})
    day, night = prof_style.get("grade") or "", prof_style.get("night_grade") or prof_style.get("grade") or ""
    for s in shots:
        s["subject"] = aesthetics.subject_of(policy, s.get("shot_type") or "") if s.get("shot_type") else ""
        if s.get("shot_type") and not aesthetics.shot(policy, s["shot_type"]):
            s["shot_type"] = ""   # not in the vocabulary: a plain shot
        tod = str((s.get("set") or {}).get("time_of_day") or "").lower()
        late = any(w in tod for w in ("night", "late evening", "midnight")) or s.get("shot_type") == "flash_night"
        s["grade"] = aesthetics.grade_words(policy, night if late else day) if prof_style else ""
        if look:
            if (s.get("wardrobe") or {}).get("outfit_id"):
                s["wardrobe"]["outfit_id"] = ""
            sid = (s.get("set") or {}).get("set_id") or ""
            if sid and sid not in anchors:
                s["set"]["set_id"] = ""
    return shots


def recent_shot_types(conn, creator_id, n: int) -> list[str]:
    """The shot types of her recent directed posts, newest first, one entry per shot (repeats kept, so the director
    sees which dominate and varies them)."""
    rows = conn.execute(
        "SELECT body->'shots' AS shots FROM posts WHERE creator_id = %s AND status <> 'dropped' "
        "AND jsonb_typeof(body->'shots') = 'array' ORDER BY created_at DESC LIMIT %s", (creator_id, n)).fetchall()
    seen: list[str] = []
    for r in rows:
        for s in r["shots"] or []:
            t = (s or {}).get("shot_type") if isinstance(s, dict) else ""
            if t:
                seen.append(str(t))
    return seen


def _capped(value, n: int):
    """Lists cut to `n` items, all the way down (policy direction.context.list_items): context, not a dump."""
    if isinstance(value, list):
        return [_capped(v, n) for v in value[:n]]
    if isinstance(value, dict):
        return {k: _capped(v, n) for k, v in value.items()}
    return value


def static_brief(policy, fmt: str) -> str:
    """The part of the director's brief that depends only on the policy and the format (the capability catalogue, what
    is unavailable, camera moves, the format's recipes, lengths, the names a shot may use, motion references, what the
    format is). It leads the user message as its own block, marked for the prompt cache, so every post of a format
    and both director calls (shots, then edit) read it from the cache after the first."""
    cfg = _cfg(policy)
    usable, missing = catalogue(policy)
    names = _names(policy)
    refs = names.pop("motion_references")
    notes = policy.get("planner", "format_notes", default={}) or {}
    return (f"THE STUDIO'S BRIEF (the same for every {fmt} post)\n"
            f"What the studio can render now (compose every shot from these): {json.dumps(usable)}\n"
            f"Not available now (never use): {json.dumps(missing)}\n"
            f"Allowed camera moves: {json.dumps(list(cfg.get('camera_moves') or []))}\n"
            f"Recipes for a {fmt} you may draw on (examples, chosen by what suits this creator; not rules): "
            f"{json.dumps(recipes(policy, fmt))}\n"
            f"Length by intent: {json.dumps(_brief(policy).get('lengths') or {})}\n"
            f"Names you may use: {json.dumps(names)}\n"
            + (f"Motion references (motion_reference, only these ids): {json.dumps(refs)}\n" if refs else
               "Motion references: none now (leave motion_reference empty).\n")
            + f"What a {fmt} is: {json.dumps(notes.get(fmt, ''))}\n")


def _idea_extras(conn, body: dict) -> dict:
    """The idea's reasoning and attributes: on the body when directing an idea, else on the idea a post came from
    (body.idea_id; a post carries no reasoning)."""
    out = {k: body[k] for k in ("reasoning", "attributes") if body.get(k)}
    if len(out) < 2 and body.get("idea_id"):
        try:
            row = conn.execute("SELECT body->>'reasoning' AS reasoning, attributes FROM ideas WHERE id = %s::uuid",
                               (str(body["idea_id"]),)).fetchone()
        except Exception as e:  # noqa: BLE001 - context, not a condition for directing
            log.warning("idea %s unreadable for the director (%s)", body.get("idea_id"), type(e).__name__)
            row = None
        for k in ("reasoning", "attributes"):
            if row and row[k] and k not in out:
                out[k] = row[k]
    return out


def creator_brief(conn, policy, body: dict, creator_id, fmt: str, scenes: list[str], move_name: str,
                  move: dict | None) -> str:
    """The part of the brief about this creator and this post: who she is (persona, dossier, archetype playbook), what
    her talent manager wants now, her style and what she has worn and been shot as lately, and the idea in full
    (why it was planned, its series, pillar, tone and canon)."""
    from ..creator import dossier, profile as profile_mod
    from . import archetypes
    from .agents import talent_manager
    cfg = _cfg(policy)
    ctx_cfg = cfg.get("context") or {}
    items = int(ctx_cfg.get("list_items", 8))
    row = conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    bible = (row or {}).get("bible") or {}
    cur = profile_mod.current(conn, creator_id)
    prof = cur[1] if cur else None
    creator = {k: bible[k] for k in ("display_name", "archetype", "niche", "one_line", "appearance", "voice_notes",
                                     "style_and_wardrobe", "home_and_places") if bible.get(k)}
    persona = {k: bible[k] for k in ("personality", "interests", "caption_voice", "never",
                                     "opinions_and_running_jokes") if bible.get(k)}
    facts = [f"{m['kind']}: {m['text']}"
             for m in memory.facts(conn, creator_id, limit=int(cfg.get("memory_facts", 40)))]
    plays = [{"archetype": p.get("archetype"), **{k: p[k] for k in ("hooks", "edit_grammar", "pitfalls") if p.get(k)}}
             for p in archetypes.playbooks(policy, prof or bible)]
    dossier_view = dossier.planning_view(prof)
    brief = talent_manager.latest_brief(conn, creator_id)
    manager = {k: (brief["output"] or {}).get(k) for k in ("focus", "stop") if (brief["output"] or {}).get(k)} \
        if brief else {}
    idea = {"format": fmt, "scenes": scenes,
            **{k: body[k] for k in ("script", "on_screen_text", "caption", "staging", "flavor", "pillar", "tone",
                                    "canon", "series") if body.get(k)},
            **_idea_extras(conn, body)}
    if body.get("series"):
        about = planner.series_catalogue(prof).get(body["series"])
        if about:
            idea["series_is"] = about
    if move:
        idea["move"] = {"name": move_name, "note": move.get("note", ""), "shot": move.get("shot") or {}}
    look = _style_view(conn, policy, creator_id)
    cut_rules = ""
    if fmt == "talking_video":   # how this talking post is cut (manyworlds.cuts): its grammar's rules
        from .. import cuts
        cut_rules = cuts.director_rules(policy, cuts.choose(policy, body, prof or bible))
    styled = ""
    if look:
        styled = (f"Her style, world and this month (keep the style, vary outfits and places): {json.dumps(look, default=str)}\n"
                  f"Her recent outfits (dress her differently): {json.dumps(recent_outfits(conn, creator_id, int(cfg.get('recent_posts', 20))))}\n")
    shot_types = Counter(recent_shot_types(conn, creator_id, int(ctx_cfg.get("recent_shot_type_posts", 12))))
    known = ({"sets": look.get("anchor_sets") or {}} if look else
             _known(conn, policy, creator_id, int(cfg.get('recent_posts', 20))))
    return (f"The creator (appearance for reference only): {json.dumps(creator, default=str)}\n"
            + (f"Her persona (how she thinks, talks and what she never does; direct performance, lines and on-screen "
               f"text in it): {json.dumps(persona, default=str)}\n" if persona else "")
            + f"How this kind of creator communicates (from their archetype): {json.dumps(communication(policy, bible))}\n"
            + (f"Her archetype's craft (hooks that open, edit grammar, pitfalls to avoid): "
               f"{json.dumps(_capped(plays, items), default=str)}\n" if plays else "")
            + (f"Her dossier (positioning, audience, story, content system; the post serves these): "
               f"{json.dumps(_capped(dossier_view, items), default=str)}\n" if dossier_view else "")
            + (f"Her talent manager's current focus and what to stop: {json.dumps(_capped(manager, items), default=str)}\n"
               if manager else "")
            + cut_rules
            + f"What is known about their life: {json.dumps(facts)}\n"
            + styled
            + (f"Shot types in her recent posts, with how often (vary them: don't lean on the most used; reach for her "
               f"other shot families): {json.dumps(dict(shot_types.most_common()))}\n" if shot_types else "")
            + f"Known sets{'' if look else ' and outfits'} (reuse the names): {json.dumps(known, default=str)}\n\n"
            f"The accepted idea (its reasoning says why it was planned; direct toward that): "
            f"{json.dumps(idea, default=str)}\n\n"
            f"Write exactly {len(scenes)} shot sheet(s), one per scene, in order.")


def _prompt(conn, policy, body: dict, creator_id, fmt: str, scenes: list[str], move_name: str,
            move: dict | None) -> list[dict]:
    """The director's message: the static brief first, as a cached block, then this creator and this post."""
    return [{"type": "text", "text": static_brief(policy, fmt), "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": creator_brief(conn, policy, body, creator_id, fmt, scenes, move_name, move)}]


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
    _edit(conn, policy, llm, content, shots, creator_id, agent, cfg)
    if move and move.get("shot"):
        k = int(getattr(out, "move_shot", -1))
        k = k if 0 <= k < len(shots) else (0 if len(shots) == 1 else -1)
        if k >= 0:
            shots[k] = fill(shots[k], move["shot"])
    for s in shots:
        for line in (s.get("performance") or {}).get("lines") or []:
            if not line.get("pause_after_s"):
                line["pause_after_s"] = None   # 0 from the strict schema means "the default pause"
    look = _style_view(conn, policy, creator_id)
    if look:
        from ..creator import profile as profile_mod
        shots = styled(policy, shots, look, (profile_mod.current(conn, creator_id)[1] or {}).get("style") or {})
    else:
        shots = styled(policy, shots, {}, {})
    shots = [ShotSheet.model_validate(s).model_dump(mode="json") for s in shots]
    usable_refs = motion_references(policy)
    for s in shots:
        if s.get("motion_reference") and s["motion_reference"] not in usable_refs:
            log.warning("director named motion reference %r, which is not usable now; cleared", s["motion_reference"])
            s["motion_reference"] = ""
    issues = planner.shot_voice_issues(policy, shots)
    if issues:
        # The edit call's text (labels, hooks) can break a voice rule on its own: the shots keep their direction
        # without it. Only shots that still fail make the post undirected.
        blank = ShotSheet().model_dump(mode="json")["edit"]
        log.warning("the edit reads as AI (%s); edits cleared, shots kept", ", ".join(issues))
        shots = [{**s, "edit": blank} for s in shots]
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


EDIT_ASK = ("THE SHOTS ARE DIRECTED (below). Now write each shot's edit, one per shot in order: the on-screen text "
            "(hook, labels, timestamps, question bubbles, in her voice), how the shot is cut into, the caption style, "
            "and whether its frame is the cover. Follow the brief above; most shots need little or no text.")


def _edit(conn, policy, llm, content, shots: list[dict], creator_id, agent: str, cfg: dict) -> None:
    """The edit pass (DirectedEdits): a second, small call that sees the brief and the directed shots and fills each
    shot's `edit`. If it fails, the shots keep an empty edit (no text, cuts by default) rather than losing direction."""
    view = [{"shot_type": s.get("shot_type") or "", "location": (s.get("set") or {}).get("location") or "",
             **{k: s.get(k) for k in ("framing", "blocking", "performance", "sound", "cutaway", "motion_reference")}}
            for s in shots]
    ask = list(content) if isinstance(content, list) else [{"type": "text", "text": str(content)}]
    ask = ask + [{"type": "text", "text": f"{EDIT_ASK}\n{json.dumps(view, default=str)}"}]
    try:
        with llm.tagged(agent, None, before_call=lambda: budget.check(conn, policy, agent)):
            out = llm.structured(str(cfg.get("role", AGENT)), system=system_prompt(policy), content=ask,
                                 schema=DirectedEdits, effort=str(cfg.get("edit_effort", "low")),
                                 creator_id=creator_id)
    except Exception as e:  # noqa: BLE001 - direction without an edit beats no direction
        log.warning("the edit pass failed (%s: %s); shots keep an empty edit", type(e).__name__, str(e)[:300])
        return
    for s, e in zip(shots, getattr(out, "edits", None) or []):
        s["edit"] = e.model_dump(mode="json")


def _register(conn, policy, creator_id, shots: list[dict], post_id) -> None:
    from ..creator import sets
    if not sets.enabled(policy):
        return
    try:
        sets.register(conn, policy, creator_id, shots, post_id=post_id)
    except Exception as e:  # noqa: BLE001 - a set not registered now is registered by its next post
        log.warning("set and wardrobe registration failed for creator %s (%s: %s)", creator_id,
                    type(e).__name__, str(e)[:300])
