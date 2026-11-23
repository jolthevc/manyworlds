"""The style layer (owner 2026-11-24): consistent face, appearance and style; varied outfits and places; an
aspirational feed made mostly of her world (aesthetic shots without her face); branded carousel slides; and the
style director who owns it all per creator."""

from __future__ import annotations

from datetime import date, datetime, timezone

from psycopg.types.json import Jsonb

from manyworlds import direction, slides
from manyworlds.brain import aesthetics, director
from manyworlds.brain.agents import showrunner, style_director
from manyworlds.creator import profile as profile_mod
from manyworlds.review import gates

from brain_helpers import StubLLM, creator, make_ctx

STYLE = {"vibe": "quiet mornings, warm light", "palette": [{"name": "oat", "hex": "#e8dcc8"}, {"name": "rust", "hex": "nope"}],
         "grade": "warm_film", "night_grade": "not_a_grade", "signature_items": ["a thin gold ring", "a camel coat", "x", "y"],
         "wardrobe": {"silhouettes": ["wide-leg trousers"], "fabrics": ["linen"], "colours": ["oat"], "footwear": ["loafers"],
                      "never": ["neon"], "rewear_after_days": 2},
         "motifs": ["fresh flowers"], "tastes": {"foods": ["sourdough"], "drinks": ["flat white"], "places": ["bakeries"]},
         "shot_families": ["portrait_off_camera", "invented_shot"],
         "face_share": {"face": 2, "partial": 1, "none": 1}}
WORLD = {"anchor_sets": [{"set_id": "Living Room Corner", "purpose": "talks to camera", "description": "a linen sofa"}],
         "home_base": "Leith, Edinburgh", "hemisphere": "north",
         "travel": {"trips_per_quarter": 9, "destinations": ["Lisbon"]},
         "carousel": {"headline_font": "DMSerifDisplay", "body_font": "Poppins", "background": "#f4efe8",
                      "text": "bad", "accent": "#b5654a", "layout": "editorial"}}
SEASON = {"capsule": ["oat knit, wide-leg linen trousers, loafers"], "notes": ["short days"],
          "trips": [{"where": "home", "first_day": 0, "days": 0}, {"where": "Lisbon", "first_day": 10, "days": 4}]}


class StyleLLM(StubLLM):
    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        return schema.model_validate({"StylePart": STYLE, "WorldPart": WORLD, "SeasonPart": SEASON}[schema.__name__])


def _creator_with_profile(conn, policy):
    cid = creator(conn)
    conn.execute("UPDATE creators SET status = 'active', bible = %s WHERE id = %s",
                 (Jsonb({"display_name": "Isla", "appearance": {"ethnicity_and_skin": "fair skin",
                                                                "signature_details": "a thin gold ring"}}), cid))
    body = {"identity": {"apparent_age": 27}, "look": {"hair": "bob", "wardrobe": "knits", "accessories": "a watch",
                                                      "makeup": "none", "signature_item": "a thin gold ring"},
            "persona": {"traits": ["warm"], "vibe": "quiet", "ethos": "e", "humour": "h", "caption_voice": "c",
                        "speaking_voice": "s", "never": []}, "dials": {}}
    profile_mod.save(conn, policy, cid, body, author="test", reason="seed")
    return cid


def test_the_library_is_complete():
    lib = aesthetics.library()
    assert len(lib["shots"]) >= 40 and lib["grades"] and lib["templates"]
    for name, s in lib["shots"].items():
        assert s["face"] in ("yes", "partial", "no") and s.get("framing") and s.get("category"), name
        assert s["subject"] in lib["subjects"] or s["subject"] == "face", name
        assert (s["face"] == "yes") == (s["subject"] == "face"), name   # only a face shot draws her face
    assert sum(1 for s in lib["shots"].values() if s["face"] != "yes") > len(lib["shots"]) / 2   # mostly her world


def test_code_checks_what_the_style_director_writes(policy):
    st = style_director.clean(policy, "style", style_director.StylePart.model_validate(STYLE))["style"]
    assert st["grade"] == "warm_film" and st["night_grade"] == "warm_film"           # unknown grade -> her day grade
    assert [c["name"] for c in st["palette"]] == ["oat"] and len(st["signature_items"]) == 3
    assert "invented_shot" not in st["shot_families"]
    lib = aesthetics.library(policy)["shots"]
    assert sum(1 for s in st["shot_families"] if lib[s]["face"] != "yes") >= 4        # filled up with her world
    assert abs(sum(st["face_share"].values()) - 1) < 0.01 and st["wardrobe"]["rewear_after_days"] == 5
    w = style_director.clean(policy, "world", style_director.WorldPart.model_validate(WORLD))["world"]
    assert w["anchor_sets"][0]["set_id"] == "living_room_corner" and w["carousel"]["text"] == "#2b2622"
    assert w["travel"]["trips_per_quarter"] == 3
    s = style_director.clean(policy, "season", style_director.SeasonPart.model_validate(SEASON))["season"]
    assert [t["where"] for t in s["trips"]] == ["Lisbon"] and s["started"]


def test_the_style_director_builds_style_world_and_season_once(conn, policy):
    from manyworlds import queue
    cid = _creator_with_profile(conn, policy)
    assert style_director.ensure(conn, policy) != []                                  # no style yet: queued
    job = queue.enqueue(conn, "brain.style_director", {"creator_id": str(cid)}, creator_id=cid)
    ctx = make_ctx(conn, policy, job)
    ctx.llm = StyleLLM(conn=conn)
    assert style_director.StyleDirector().run(ctx)["built"] == ["style", "world", "season"]
    prof = profile_mod.current(conn, cid)[1]
    assert prof["style"]["grade"] == "warm_film" and prof["world"]["home_base"] and prof["season"]["capsule"]
    assert style_director.StyleDirector().run(ctx)["built"] == []                     # nothing due again
    view = style_director.planning_view(policy, prof, today=date.fromisoformat(prof["season"]["started"]))
    assert view["where_she_is_now"] == "home" and "living_room_corner" in view["anchor_sets"]
    assert "walk_away_sunset" in view["shot_types"] or len(view["shot_types"]) >= 4


def test_an_aesthetic_shot_shows_no_face_and_is_not_judged_on_identity(policy):
    shot = {"shot_type": "walk_away_sunset", "wardrobe": {"outfit_id": "camel_coat", "outfit": "a camel coat"},
            "set": {"set_id": "old_town_street", "location": "an old town street", "time_of_day": "golden hour"}}
    look = {"anchor_sets": {"living_room_corner": "talks: a linen sofa"}}
    [s] = director.styled(policy, [shot], look, {"grade": "warm_film", "night_grade": "flash_digicam"})
    assert s["subject"] == "back" and s["grade"] == aesthetics.grade_words(policy, "warm_film")
    assert s["wardrobe"]["outfit_id"] == "" and s["set"]["set_id"] == ""              # no recurring outfit or set
    assert direction.faceless(s) and "identity" not in gates.gates_for("a street", s)
    bible = {"appearance": {"face": "freckles and green eyes", "ethnicity_and_skin": "fair skin"}}
    prompt = direction.compile_prompt(s, bible, "photo", policy)
    assert "seen from behind" in prompt and "green eyes" not in prompt and "warm film" in prompt
    [p] = director.styled(policy, [{"shot_type": "portrait_off_camera"}], look, {"grade": "warm_film"})
    assert not direction.faceless(p) and "identity" in gates.gates_for("a cafe", p)
    [a] = director.styled(policy, [{"set": {"set_id": "living_room_corner"}}], look, {})
    assert a["set"]["set_id"] == "living_room_corner"                                 # anchor sets do recur


def test_every_slide_template_renders_in_the_grid_shape(policy):
    from PIL import Image
    brand = slides.brand_of({"world": {"carousel": {"headline_font": "Anton", "accent": "#336699"}}})
    photo = Image.new("RGB", (800, 1000), (120, 90, 60))
    for t in slides.TEMPLATES:
        img = slides.render({"template": t, "heading": "the 10-minute sunday reset", "text": "left side | right side",
                             "items": ["one", "two"]}, brand, photo if t in ("cover", "step") else None,
                            policy=policy, number=1, of=3, handle="isla")
        assert img.size == (1080, 1350), t


def test_a_carousel_with_slides_ships_them_around_her_photos(conn, policy, store, tmp_path):
    from PIL import Image
    cid = _creator_with_profile(conn, policy)
    photos = []
    for i in range(2):
        p = tmp_path / f"p{i}.png"
        Image.new("RGB", (400, 500), (i * 80, 60, 40)).save(p)
        obj = store.put_file(p, f"t/p{i}.png")
        photos.append(conn.execute("INSERT INTO assets (creator_id, kind, uri, sha256, bytes) VALUES (%s, 'image', %s, "
                                   "%s, %s) RETURNING id, uri", (cid, obj.uri, obj.sha256, obj.bytes)).fetchone())
    post = conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, 'carousel') RETURNING *", (cid,)).fetchone()
    body = {"slides": [{"template": "cover", "heading": "my slow sunday", "photo": 0},
                       {"template": "cta", "heading": "save this"}]}
    out = slides.render_post(conn, policy, store, post, body, photos)
    assert len(out) == 3 and out[1]["uri"] == photos[1]["uri"]                       # cover, her other photo, the ask
    again = slides.render_post(conn, policy, store, post, body, photos)
    assert [o["uri"] for o in again] == [o["uri"] for o in out]                      # idempotent


def test_the_showrunner_knows_the_season_where_she_lives():
    assert showrunner._season(date(2026, 11, 24), "north") == "late autumn"
    assert showrunner._season(date(2026, 11, 24), "south") == "late spring"
    assert showrunner._season(date(2026, 1, 10), "north") == "mid winter"
