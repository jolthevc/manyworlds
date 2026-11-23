"""What the Brain's planners see and keep (2026-11-23): carousel slides reach the post, a decision names the model
that made it, a dossier series is a series, one source of truth for the look, and outcomes and scored predictions
close the loop for the showrunner and the talent manager."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds import queue
from manyworlds.brain import evidence, ideas, planner, predictions
from manyworlds.brain.agents import showrunner
from manyworlds.creator import profile

from brain_helpers import creator, make_ctx
from test_brain_showrunner import PlanLLM, _maren, _run, idea
from test_talent_manager import DossierLLM, _creator, _run as _run_manager

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
TARGET = 259200


def test_carousel_slides_reach_the_idea_and_the_post_and_the_decision_names_the_showrunners_model(conn, policy):
    cid = _maren(conn)
    slides = [{"template": "cover", "heading": "three teas for a cold night", "text": "", "items": [], "photo": 0},
              {"template": "list_item", "heading": "rooibos", "text": "no caffeine, still cosy", "items": [], "photo": 1}]
    _run(conn, policy, cid, PlanLLM([idea(0, "carousel", scenes=("the kettle", "the mug"), slides=slides),
                                     idea(1, "carousel", scenes=("a", "b"), caption="no slides here", slides=[])],
                                    conn=conn))
    rows = conn.execute("SELECT id, body FROM ideas ORDER BY body->>'day'").fetchall()
    assert rows[0]["body"]["slides"] == slides and "slides" not in rows[1]["body"]
    role = policy.get("brain", "showrunner", "role")
    assert conn.execute("SELECT model FROM decisions WHERE kind = 'content_batch'").fetchone()["model"] == f"stub-{role}"
    for r in rows:
        conn.execute("INSERT INTO evaluations (item_kind, idea_id, creator_id, rubric, rubric_version, evaluator_model, "
                     "prompt_version, scores, samples, role) VALUES ('idea', %s, %s, 'idea', 1, 'm', 'p', '{}', 3, "
                     "'evaluator')", (r["id"], cid))
    ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
    post = conn.execute("SELECT body FROM posts WHERE body->>'idea_id' = %s", (str(rows[0]["id"]),)).fetchone()
    assert post["body"]["slides"] == slides                         # production reads body.slides (manyworlds.slides)


def _post(fmt="story_clip", series=""):
    return planner.PlannedPost(day=0, platform="instagram", format=fmt, series=series, scenes=["steam off the kettle"],
                               caption="kettle number four", hashtags=[], canon="she owns four kettles")


def test_a_dossier_series_is_one_of_her_series(policy):
    prof = {"content": {"video_series": [{"name": "kettle ratings", "premise": "she rates kettles"}]},
            "content_system": {"series": [{"name": "2am tea", "premise": "the same ritual each night",
                                           "hook": "a time stamp"},
                                          {"name": "kettle ratings", "premise": "other words", "hook": "a score"}]}}
    cat = planner.series_catalogue(prof)
    assert set(cat) == {"kettle ratings", "2am tea"}
    assert cat["kettle ratings"]["premise"] == "she rates kettles" and cat["kettle ratings"]["hook"] == "a score"
    kw = dict(platforms=["instagram"], days=3, prof=prof, hashtags_max={})
    assert planner.validate(policy, _post(series="2am tea"), **kw)[0] is not None        # the dossier's series
    assert planner.validate(policy, _post(series="kettle ratings"), **kw)[0] is not None  # the profile's
    kept, why = planner.validate(policy, _post(series="vlogmas"), **kw)
    assert kept is None and "series" in why


def test_the_showrunner_is_shown_every_series_by_name(conn, policy):
    cid = _creator(conn, policy, content_system={"series": [{"name": "2am tea", "premise": "the same ritual each night",
                                                             "hook": "a time stamp"}]})
    job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(cid), "start": "2026-11-23"}, creator_id=cid)
    prompt, _ = showrunner._prompt(make_ctx(conn, policy, job), cid, date(2026, 11, 23), 3)
    assert "Her series" in prompt and "2am tea" in prompt and "the same ritual each night" in prompt


STYLE = {"signature_items": ["a thin gold ring", "a camel coat"], "face_share": {"face": 0.3, "partial": 0.2, "none": 0.5},
         "wardrobe": {"silhouettes": ["wide-leg trousers"], "fabrics": ["wool"], "colours": ["camel", "cream"],
                      "footwear": ["loafers"], "never": ["logos"], "rewear_after_days": 10}}
BASE = {"identity": {"apparent_age": 27},
        "look": {"hair": "bob", "wardrobe": "black knits", "accessories": "a watch", "makeup": "none",
                 "signature_item": "round glasses"},
        "persona": {"traits": ["dry"], "vibe": "quiet", "ethos": "e", "humour": "h", "caption_voice": "c",
                    "speaking_voice": "s", "never": ["glam"]}}


def test_the_bible_takes_the_look_from_her_style_when_she_has_one(policy):
    plain = profile.render_bible(BASE)
    assert plain["appearance"]["signature_details"] == "round glasses; a watch"
    assert plain["style_and_wardrobe"].startswith("black knits")
    styled = profile.render_bible({**BASE, "style": STYLE})
    assert styled["appearance"]["signature_details"] == "round glasses; a watch"     # every photo: wearable only
    wardrobe = styled["style_and_wardrobe"]
    assert "wide-leg trousers" in wardrobe and "Never wears: logos" in wardrobe and "black knits" not in wardrobe


def test_the_face_rule_defers_to_her_style(policy):
    assert "signature item" in planner.face_rule(None)
    rule = planner.face_rule({"style": STYLE})
    assert "30%" in rule and "50%" in rule and "most images" not in rule


def _published(conn, cid, *, simulated: bool, reach=None, age=TARGET, status="published"):
    pid = conn.execute("INSERT INTO posts (creator_id, format, status, caption) VALUES (%s, 'photo', %s, 'tea') "
                       "RETURNING id", (cid, status)).fetchone()["id"]
    conn.execute("INSERT INTO publications (post_id, platform, simulated, published_at) VALUES (%s, %s, %s, %s)",
                 (pid, "simulated" if simulated else "instagram", simulated, NOW))
    if reach is not None and simulated:
        evidence.record_simulated_metrics(conn, pid, [{"post_age_s": age, "metrics": {"reach": reach, "saves": 3}}],
                                          model="m", panel_version="p")
    elif reach is not None:
        conn.execute("INSERT INTO metric_snapshots (post_id, metrics, post_age_s, captured_at) VALUES (%s, %s, %s, %s)",
                     (pid, Jsonb({"reach": reach, "saves": 3}), age, NOW + timedelta(seconds=age)))
    return pid


def test_post_outcomes_read_reach_at_the_target_age_and_never_mix_modes(conn, policy):
    real = creator(conn, publish_mode="instagram")
    old = _published(conn, real, simulated=False, reach=900)
    young = _published(conn, real, simulated=False, reach=40, age=3600)            # too young to judge
    out = evidence.post_outcomes(conn, real, 10, "real", policy=policy)
    assert set(out) == {str(old)} and out[str(old)]["reach"] == 900 and out[str(old)]["saves"] == 3
    assert "label" not in out[str(old)] and str(young) not in out
    assert evidence.post_outcomes(conn, real, 10, "sandbox", policy=policy) == {}   # no simulated rows of a real post
    sandbox = creator(conn)
    sim = _published(conn, sandbox, simulated=True, reach=777)
    got = evidence.post_outcomes(conn, sandbox, 10, "sandbox", policy=policy)
    assert got[str(sim)]["reach"] == 777 and got[str(sim)]["label"] == evidence.LABEL
    assert evidence.post_outcomes(conn, sandbox, 10, "real", policy=policy) == {}


def test_the_showrunner_sees_how_recent_posts_did(conn, policy):
    cid = _maren(conn)
    _published(conn, cid, simulated=True, reach=777)
    job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(cid), "start": "2026-11-23"}, creator_id=cid)
    prompt, _ = showrunner._prompt(make_ctx(conn, policy, job), cid, date(2026, 11, 23), 3)
    assert '"reach": 777' in prompt and "SIMULATED" in prompt


def _resolved(conn, cid, claim, *, held: bool, role="talent_manager"):
    pid = predictions.record(conn, role=role, creator_id=cid, claim=claim, metric="median_reach", comparator="above",
                             threshold=500, probability=0.7, horizon_days=14, now=NOW)
    conn.execute("UPDATE predictions SET resolved_at = %s, observed = %s, outcome = %s, brier = %s WHERE id = %s",
                 (NOW + timedelta(days=20), 650 if held else 300, held, 0.09 if held else 0.49, pid))
    return pid


def test_resolved_predictions_are_read_per_creator_and_role(conn, policy):
    cid = creator(conn, publish_mode="instagram")
    _resolved(conn, cid, "reach above 500 by december", held=True)
    _resolved(conn, cid, "the CEO's call", held=False, role="strategist")
    predictions.record(conn, role="talent_manager", creator_id=cid, claim="still open", metric="median_reach",
                       comparator="above", threshold=1, probability=0.5, horizon_days=14, now=NOW)
    [mine] = predictions.resolved_for(conn, cid, role="talent_manager")
    assert mine["claim"] == "reach above 500 by december" and mine["held"] is True and mine["brier"] == 0.09
    assert mine["observed"] == 650 and "label" not in mine
    assert len(predictions.resolved_for(conn, cid)) == 2


def test_the_talent_manager_sees_outcomes_and_its_own_scored_predictions(conn, policy):
    cid = _creator(conn, policy)                                     # a sandbox creator: everything labelled
    _published(conn, cid, simulated=True, reach=777)
    _resolved(conn, cid, "reach above 500 by december", held=True)
    llm = DossierLLM(conn=conn)
    _run_manager(conn, policy, cid, "review", llm)
    review = next(str(c["content"]) for c in llm.calls if "Your past predictions" in str(c["content"]))
    assert "reach above 500 by december" in review and "SIMULATED" in review
    assert '"reach": 777' in review                                  # how her recent post did
