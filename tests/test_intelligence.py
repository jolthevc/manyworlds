"""Influencer intelligence, first PR: research references kept apart from production, forget and retention, the
pattern library, video measurement on the finisher, the annotation schema and the cheap impersonation checks.
Everything offline: no Apify, no LLM, no GPU."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import psycopg
import pytest
from psycopg.types.json import Jsonb

from manyworlds import dance, queue
from manyworlds.brain import aesthetics, impersonation, patterns
from manyworlds.research import annotation, references, video_measure
from manyworlds.storage import LocalStore

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)
NEW_TABLES = ("reference_creators", "reference_posts", "reference_annotations", "pattern_cards", "auditions",
              "research_forget")


@pytest.fixture()
def rstore(tmp_path, monkeypatch):
    """The research store: a separate place (RESEARCH_STORAGE_URL), every key under research-refs/."""
    root = tmp_path / "research"
    monkeypatch.setenv(references.STORE_ENV, root.resolve().as_uri())
    return LocalStore(root)


def _ref_creator(conn, handle="mira.lou", name="Mira Lou", platform="instagram", last_seen=None):
    return conn.execute(
        """INSERT INTO reference_creators (platform, handle, display_name, profile_url, last_seen)
           VALUES (%s, %s, %s, %s, coalesce(%s, now())) RETURNING id""",
        (platform, handle, name, f"https://www.instagram.com/{handle}/", last_seen)).fetchone()["id"]


def _ref_post(conn, rstore, creator_id, tmp_path, ext="p1", purge_after=None):
    f = tmp_path / f"{ext}.jpg"
    f.write_bytes(b"jpeg")
    post_id = conn.execute(
        """INSERT INTO reference_posts (creator_id, platform, external_id, url, media_type, media_purge_after)
           VALUES (%s, 'instagram', %s, %s, 'video', %s) RETURNING id""",
        (creator_id, ext, f"https://www.instagram.com/reel/{ext}/", purge_after)).fetchone()["id"]
    o = rstore.put_file(f, references.media_key(post_id, f"{ext}.jpg"))
    conn.execute("UPDATE reference_posts SET media = %s WHERE id = %s",
                 (Jsonb([{"uri": o.uri, "kind": "image", "bytes": o.bytes}]), post_id))
    return post_id, o.uri


# --- the schema ---------------------------------------------------------------------------------------------------
def test_research_tables_exist_and_rented_machines_cannot_read_them(conn):
    for t in NEW_TABLES:
        assert conn.execute("SELECT to_regclass(%s) AS r", (t,)).fetchone()["r"] is not None
        assert conn.execute("SELECT has_table_privilege('mw_worker', %s, 'SELECT') AS p", (t,)).fetchone()["p"] is False


def test_theses_carry_hypotheses_with_their_new_states(conn):
    tid = conn.execute("""INSERT INTO theses (summary, kind, status, version, owner_role, budget_usd, body)
                          VALUES ('aspirational yet relatable', 'creator_hypothesis', 'researching', 1, 'market_lead',
                                  16, %s) RETURNING id""",
                       (Jsonb({"disconfirming": ["crowded niche"]}),)).fetchone()["id"]
    conn.execute("UPDATE theses SET status = 'archived', archive_reason = 'no traction' WHERE id = %s", (tid,))
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("UPDATE theses SET status = 'whatever' WHERE id = %s", (tid,))


def test_no_asset_row_may_point_into_the_research_store(conn, rstore, tmp_path):
    f = tmp_path / "x.jpg"
    f.write_bytes(b"x")
    uri = rstore.put_file(f, references.media_key("p", "x.jpg")).uri
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("INSERT INTO assets (kind, uri, sha256, bytes) VALUES ('image', %s, 'h', 1)", (uri,))


# --- the guard: research media is never a production input ----------------------------------------------------------
def test_research_uris_are_recognised_by_prefix_and_by_the_research_bucket(monkeypatch):
    assert references.is_research_uri("s3://manyworlds/research-refs/posts/1/v.mp4")
    assert references.is_research_uri("file:///tmp/r/research-refs/posts/1/v.mp4")
    assert not references.is_research_uri("s3://manyworlds/creators/maren/ref.png")
    assert not references.is_research_uri("research-refs/not-a-uri")
    monkeypatch.setenv(references.STORE_ENV, "s3://manyworlds-research")
    assert references.is_research_uri("s3://manyworlds-research/anything.mp4")


@pytest.mark.parametrize("kind", ["image.zimage", "image.qwen_edit", "comfy.wan_i2v", "comfy.wan_animate2",
                                  "talk.infinitetalk", "voice.speak", "voice.design", "train.lora", "finish.video",
                                  "music.song", "hosted.video"])
def test_no_production_job_may_carry_research_media(conn, kind):
    payload = {"scene": "x", "refs": [{"uri": "s3://manyworlds/research-refs/posts/1/frames/f01.jpg"}]}
    with pytest.raises(references.ResearchMediaRefused):
        queue.enqueue(conn, kind, payload)
    assert conn.execute("SELECT count(*) AS n FROM jobs").fetchone()["n"] == 0
    assert queue.enqueue(conn, kind, {"scene": "x", "refs": [{"uri": "s3://manyworlds/creators/a.png"}]})


def test_the_measurement_and_the_research_agent_may(conn):
    uri = "s3://manyworlds/research-refs/posts/1/video.mp4"
    assert queue.enqueue(conn, "finish.measure_video", {"post_id": "1", "media_uri": uri})
    assert queue.enqueue(conn, "brain.intelligence", {"frames": [uri]})


def test_a_research_clip_is_never_a_motion_reference(policy):
    entry = {"id": "x", "uri": "s3://manyworlds/research-refs/posts/1/video.mp4", "owner": "licensed",
             "licence": "Pexels License", "source": "https://www.pexels.com/video/1/", "move": "dance"}
    with pytest.raises(dance.RefusedReference, match="research media"):
        dance.reference("x", policy, sent=entry)


# --- deletion on request and retention ------------------------------------------------------------------------------
def test_forget_deletes_a_creator_everywhere_and_blocklists_them(conn, rstore, tmp_path):
    cid = _ref_creator(conn)
    other = _ref_creator(conn, handle="someone.else", name="Someone Else")
    pid, uri = _ref_post(conn, rstore, cid, tmp_path)
    opid, _ = _ref_post(conn, rstore, other, tmp_path, ext="p2")
    conn.execute("INSERT INTO reference_annotations (post_id, schema) VALUES (%s, 'iia_v1')", (pid,))
    conn.execute("""INSERT INTO pattern_cards (key, medium, status, sources, evidence) VALUES ('k', 'video', 'draft', %s,
                    '{"n_posts": 2}')""",
                 (Jsonb([{"post_id": str(pid), "creator_id": str(cid), "url": "u1"},
                         {"post_id": str(opid), "creator_id": str(other), "url": "u2"}]),))
    tid = conn.execute("INSERT INTO theses (summary) VALUES ('h') RETURNING id").fetchone()["id"]
    conn.execute("""INSERT INTO auditions (thesis_id, arm, "references") VALUES (%s, 'C', %s)""",
                 (tid, Jsonb([str(cid), str(other)])))
    out = references.forget(conn, rstore, "@Mira.Lou")
    assert out["creators"] == 1 and out["posts"] == 1 and out["media_deleted"] == 1
    assert out["cards"] == 1 and out["auditions"] == 1
    assert not Path(url2pathname(urlparse(uri).path)).exists()
    assert conn.execute("SELECT count(*) AS n FROM reference_annotations").fetchone()["n"] == 0
    card = conn.execute("SELECT sources, evidence FROM pattern_cards").fetchone()
    assert [s["creator_id"] for s in card["sources"]] == [str(other)] and card["evidence"]["n_posts"] == 1
    aud = conn.execute("""SELECT "references", checks FROM auditions""").fetchone()
    assert aud["references"] == [str(other)] and aud["checks"]["reference_forgotten"] is True
    assert references.forgotten(conn, "instagram", "mira.lou")
    assert conn.execute("SELECT count(*) AS n FROM events WHERE type = 'research_forget'").fetchone()["n"] == 1
    again = references.forget(conn, rstore, "mira.lou")                 # idempotent
    assert again["creators"] == 0
    assert conn.execute("SELECT count(*) AS n FROM reference_creators").fetchone()["n"] == 1


def test_forget_someone_we_hold_nothing_on_still_blocks_collection(conn, rstore):
    out = references.forget(conn, rstore, "@not.collected.yet")
    assert out["creators"] == 0 and out["blocklisted"]
    assert references.forgotten(conn, "tiktok", "not.collected.yet")


def test_retention_deletes_old_media_and_uncited_stale_creators(conn, rstore, tmp_path, policy):
    days = policy.get("intelligence", "retention", "records_days")
    assert policy.get("intelligence", "retention", "media_days") == 14 and days == 730   # data_policy amber
    fresh = _ref_creator(conn, handle="fresh")
    fp, furi = _ref_post(conn, rstore, fresh, tmp_path, ext="f1", purge_after=NOW - timedelta(days=1))
    keep, _ = _ref_post(conn, rstore, fresh, tmp_path, ext="f2", purge_after=NOW + timedelta(days=10))
    stale = _ref_creator(conn, handle="stale", last_seen=NOW - timedelta(days=days + 1))
    _ref_post(conn, rstore, stale, tmp_path, ext="s1")
    cited = _ref_creator(conn, handle="cited", last_seen=NOW - timedelta(days=days + 1))
    tid = conn.execute("INSERT INTO theses (summary, status) VALUES ('h', 'researching') RETURNING id").fetchone()["id"]
    conn.execute("""INSERT INTO auditions (thesis_id, arm, "references") VALUES (%s, 'C', %s)""",
                 (tid, Jsonb([str(cited)])))
    out = references.purge(conn, policy, rstore, now=NOW)
    assert out == {"media_deleted": 2, "media_failed": 0, "creators_deleted": 1}
    assert conn.execute("SELECT media FROM reference_posts WHERE id = %s", (fp,)).fetchone()["media"] == []
    assert len(conn.execute("SELECT media FROM reference_posts WHERE id = %s", (keep,)).fetchone()["media"]) == 1
    handles = {r["handle"] for r in conn.execute("SELECT handle FROM reference_creators").fetchall()}
    assert handles == {"fresh", "cited"}


def test_housekeeping_runs_research_retention(conn, rstore, tmp_path, policy):
    from manyworlds.deploy import autoscale
    cid = _ref_creator(conn)
    _ref_post(conn, rstore, cid, tmp_path, purge_after=datetime.now(timezone.utc) - timedelta(days=1))
    acts = autoscale.housekeeping(conn, policy)
    assert any(a["action"] == "purged_research_references" and a["media_deleted"] == 1 for a in acts)


def test_mw_research_cli(conn, rstore, tmp_path, capsys, database_url, monkeypatch):
    from manyworlds import cli
    monkeypatch.setenv("DATABASE_URL", database_url)
    _ref_creator(conn, handle="cli.person")
    assert cli.main(["research", "forget", "cli.person"]) == 0
    assert '"creators": 1' in capsys.readouterr().out
    assert cli.main(["research", "patterns", "--seed"]) == 0
    assert conn.execute("SELECT count(*) AS n FROM pattern_cards WHERE status = 'illustrative'").fetchone()["n"] == 9


# --- the pattern library --------------------------------------------------------------------------------------------
def test_the_seed_cards_load_and_use_only_names_the_pipeline_renders(policy):
    lib = patterns.library(policy)
    assert len(lib["cards"]) == 9
    media = {c["medium"] for c in lib["cards"].values()}
    assert {"photo", "carousel", "video", "sequence"} <= media                  # photo and video in one library
    for k in ("dinner_flash_friend", "street_style_crossing", "grwm_confession", "mirror_phone_cover", "poolside_edge",
              "storytime_selfie_walk", "day_in_life_montage", "outfit_check_spin"):
        assert k in lib["cards"]
    out = patterns.validate(policy)
    assert out["problems"] == []
    assert out["proposed_shot_types"] == {"poolside_edge": ["pool_edge_legs"]}  # a new type, proposed, not used
    assert all(c["status"] == "illustrative" and not c.get("sources") for c in lib["cards"].values())
    assert not any(patterns.counts_as_evidence(c) for c in lib["cards"].values())


def test_unknown_names_are_problems(policy):
    bad = {"cards": {"x": {"medium": "video", "status": "illustrative", "situation": "s", "mechanism": "m",
                           "dims": {"shot_types": ["not_a_shot"], **{d: "x" for d in patterns.DIMENSIONS}},
                           "audience_role": ["fame"], "best_fit": ["no_such_archetype"],
                           "recipe": {"grammar": "no_grammar", "deliveries": ["shouting"], "motion_moves": ["moonwalk"],
                                      "device": "drone"},
                           "sources": [{"url": "u"}]}}}
    probs = " | ".join(patterns.validate(policy, bad)["problems"])
    for word in ("not_a_shot", "fame", "no_such_archetype", "no_grammar", "shouting", "moonwalk", "drone",
                 "illustrative card has no sources"):
        assert word in probs


def test_seeding_is_once_per_key(conn, policy):
    assert patterns.seed(conn, policy) == 9 and patterns.seed(conn, policy) == 0
    row = conn.execute("SELECT signature_ok, body FROM pattern_cards WHERE key = 'outfit_check_spin'").fetchone()
    assert row["signature_ok"] is True and row["body"]["recipe"]["move"] == "outfit_change_spin"


def test_signature_elements_may_recur(policy):
    rules = aesthetics.brief(policy)["rules"]
    assert "signature" in rules and "do not apply" in rules["signature"]
    cards = [{"key": k, **c} for k, c in patterns.library(policy)["cards"].items()]
    assert set(patterns.signature_elements(cards)) == {"street_style_crossing", "outfit_check_spin"}


# --- video measurement --------------------------------------------------------------------------------------------
def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Three solid shots (1.5 s, 2 s, 1.5 s: hard cuts at 1.5 and 3.5 s), 25 fps, with a silent track."""
    out = tmp_path_factory.mktemp("vid") / "v.mp4"
    parts = [("black", 1.5), ("white", 2.0), ("black", 1.5)]
    cmd = [_ffmpeg(), "-v", "error", "-y"]
    for colour, d in parts:
        cmd += ["-f", "lavfi", "-i", f"color=c={colour}:s=180x320:r=25:d={d}"]
    cmd += ["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]", "-map", "3:a", "-t", "5", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-c:a", "aac", str(out)]
    subprocess.run(cmd, check=True, timeout=120)
    return out


def test_probe_and_cuts_on_a_synthetic_video(synthetic_video, policy):
    c = video_measure.cfg(policy)
    info = video_measure.probe(str(synthetic_video))
    assert 4.9 <= info["duration_s"] <= 5.1 and info["width"] == 180 and info["height"] == 320 and info["has_audio"]
    hard = video_measure.cuts_at(video_measure.scene_scores(str(synthetic_video), c), c["hard_cut"], c["min_gap_s"])
    assert len(hard) == 2 and abs(hard[0] - 1.5) < 0.1 and abs(hard[1] - 3.5) < 0.1
    shots = video_measure.shot_lengths(hard, info["duration_s"])
    assert all(abs(s - e) < 0.15 for s, e in zip(shots, [1.5, 2.0, 1.5])) and len(shots) == 3 and abs(sum(shots) - info["duration_s"]) < 0.05


def test_pace_from_words_and_from_subtitles():
    words = [{"start": 0.2 + i * 0.25, "end": 0.4 + i * 0.25, "word": "w"} for i in range(20)]
    s = video_measure.speech_from_words(words)
    assert s["words"] == 20 and s["first_word_s"] == 0.2 and s["words_per_s"] == round(20 / 4.95, 2)
    assert s["words_in_first_2s"] == 8
    vtt = "WEBVTT\n\n00:00:00.500 --> 00:00:02.500\nfour words right here\n\n00:00:03.000 --> 00:00:04.000\ntwo more\n"
    v = video_measure.speech_from_vtt(vtt)
    assert v["words"] == 6 and v["speech_s"] == 3.0 and v["words_per_s"] == 2.0 and v["words_in_first_2s"] == 4
    assert video_measure.speech_from_words([]) is None and video_measure.speech_from_vtt("nothing") is None


def test_measure_video_runs_on_the_finisher_against_the_research_store_only(conn, rstore, synthetic_video, policy,
                                                                             raw_policy, tmp_path):
    from manyworlds.worker import runner
    from manyworlds.worker.handlers import finish_video
    assert raw_policy.get("job_kinds", video_measure.KIND, "pools") == ["fly_finisher"]
    assert raw_policy.family(video_measure.KIND) == "finish"
    fly = (ROOT / "fly.toml").read_text(encoding="utf-8")
    assert re.search(r'finisher = "worker --pool fly_finisher --kinds [^"]*finish\.measure_video', fly)
    handler = runner.handlers_for([video_measure.KIND])[video_measure.KIND]
    assert handler is finish_video.measure_video

    cid = _ref_creator(conn)
    pid = conn.execute("""INSERT INTO reference_posts (creator_id, platform, external_id, media_type)
                          VALUES (%s, 'instagram', 'v1', 'video') RETURNING id""", (cid,)).fetchone()["id"]
    o = rstore.put_file(synthetic_video, references.media_key(pid, "video.mp4"))
    conn.execute("UPDATE reference_posts SET media = %s WHERE id = %s",
                 (Jsonb([{"uri": o.uri, "kind": "video", "bytes": o.bytes}]), pid))
    post = conn.execute("SELECT * FROM reference_posts WHERE id = %s", (pid,)).fetchone()
    words = [{"start": 0.5 + i * 0.3, "end": 0.7 + i * 0.3, "word": "w"} for i in range(10)]
    jid = video_measure.enqueue(conn, policy, post, words=words)
    j = conn.execute("SELECT id, kind, payload, attempts, max_attempts, creator_id, batch FROM jobs WHERE id = %s",
                     (jid,)).fetchone()
    ctx = runner.JobContext(job=queue.Job(**j), workdir=tmp_path / "work", store=LocalStore(tmp_path / "prod"))
    res = handler(j["payload"], ctx, policy=policy)
    m = res.meta["video_measure"]
    assert m["hard"]["cuts"] == 2 and m["speech"]["words"] == 10 and m["speech_share"] is not None
    frames = [x for x in m["media"] if x["kind"] == "frame"]
    assert len(frames) == policy.get("intelligence", "measure", "frames")
    assert all(references.is_research_uri(x["uri"]) for x in m["media"])        # frames stay research media
    assert not list((tmp_path / "work").rglob("*.mp4"))                         # the download is gone
    conn.execute("UPDATE jobs SET status = 'running', worker_id = 'fin', attempts = 1 WHERE id = %s", (jid,))
    queue.complete(conn, jid, "fin", {"asset_ids": [], **res.meta})
    assert video_measure.collect(conn, policy) == {"measured": 1, "failed": 0}
    assert video_measure.collect(conn, policy) == {"measured": 0, "failed": 0}  # each job once
    ann = conn.execute("SELECT schema, measured FROM reference_annotations WHERE post_id = %s", (pid,)).fetchone()
    assert ann["schema"] == f"measure_{video_measure.VERSION}" and ann["measured"]["hard"]["cuts"] == 2
    row = conn.execute("SELECT media, media_purge_after FROM reference_posts WHERE id = %s", (pid,)).fetchone()
    assert len(row["media"]) == 1 + len(frames) + 1 and row["media_purge_after"] is not None


def test_measure_video_refuses_production_media(policy, rstore, tmp_path):
    from manyworlds.worker.handlers import finish_video
    from manyworlds.worker.runner import NonRetryable
    ctx = type("C", (), {"workdir": tmp_path})()
    with pytest.raises(NonRetryable, match="research references only"):
        finish_video.measure_video({"post_id": "1", "media_uri": "s3://manyworlds/creators/a.mp4"}, ctx, policy=policy)


# --- annotation schema and impersonation checks ---------------------------------------------------------------------
def test_the_annotation_schema_keeps_observed_mechanism_and_speculation_apart():
    a = annotation.Annotation()
    assert a.observed.visual.situation == "unknown" and a.observed.performance is None     # unknown is a value
    dumped = annotation.Annotation.model_json_schema()
    assert {"observed", "mechanism", "speculation", "confidence"} <= set(dumped["properties"])
    fields = set(annotation.Personality.model_fields) | set(annotation.Visual.model_fields)
    assert not fields & {"health", "relationship_status", "family", "religion", "politics", "address", "children"}
    assert len(annotation.clip_note(" ".join(["word"] * 50)).split()) == annotation.NOTE_WORDS
    with pytest.raises(ValueError):
        annotation.Annotation(mechanism=[{"text": "a"}] * 4)


def test_names_block_an_impersonation_and_warn_on_a_resemblance(policy):
    refs = [{"id": "r1", "display_name": "Mira Lou", "handle": "mira.lou"}]
    same = impersonation.check({"name": "Mira Lou", "handle": "miralou_"}, refs, policy)
    assert same["blocked"] and same["ran"] == ["names"]
    near = impersonation.check({"name": "Mara Lou", "handle": "maralou.daily"}, refs, policy)
    assert not near["blocked"] and near["warnings"]
    apart = impersonation.check({"name": "Ines Okoro", "handle": "ines.after.six"}, refs, policy)
    assert not apart["blocked"] and not apart["warnings"]


def test_no_face_comparison_of_real_people(policy):
    # data_policy red (owner 2026-11-25): no face-identity templates of real creators, so no face check at all
    assert impersonation.check({"name": "A"}, [], policy)["ran"] == ["names"]
    assert not hasattr(impersonation, "faces")


def test_near_duplicates_by_difference_hash(policy, tmp_path):
    from PIL import Image, ImageDraw

    def img(name, shift=0, flip=False):
        im = Image.new("L", (90, 160), 0)
        d = ImageDraw.Draw(im)
        for i in range(0, 90, 15):
            d.rectangle([i + shift, 0, i + shift + 6, 160], fill=255 if not flip else 40)
        d.ellipse([20, 40 + shift, 70, 110 + shift], fill=180)
        p = tmp_path / name
        im.convert("RGB").save(p)
        return p
    from PIL import ImageEnhance
    a, other = img("a.png"), img("b.png", shift=7, flip=True)
    a2 = tmp_path / "a2.jpg"                                  # the same picture, re-encoded a little brighter
    ImageEnhance.Brightness(Image.open(a)).enhance(1.05).save(a2, quality=70)
    Image.open(other).rotate(90, expand=True).save(other)
    assert impersonation.near_duplicates(a2, {"cited": a}, policy)
    assert not impersonation.near_duplicates(other, {"cited": a}, policy)


def test_an_annotation_read_back_from_the_strict_shape_validates():
    # 2026-11-25: the annotation's own schema was too large for structured outputs; the strict shape answers every
    # field, with "" for unset enums and strings, which must fall back to the defaults
    from manyworlds.brain.agents.intelligence import ReadBatch
    from manyworlds.llm import strict_json_schema
    assert ReadBatch.strict_required and strict_json_schema(ReadBatch)["properties"]["reads"]
    got = ReadBatch.model_validate({"reads": [{"post": 1, "annotation": {
        "observed": {"visual": {"situation": "", "gaze": "", "framing": "close-up", "face_share": 0.4},
                     "performance": {"format": "", "movement": ""}, "craft": {"hook": "", "sound": ""},
                     "personality": {"disclosure": "", "relationship": ""}, "economics": {"sponsored": ""}},
        "mechanism": [{"text": "direct gaze holds the scroll", "audience_role": ""}], "speculation": [],
        "confidence": 0.6}}]})
    a = got.reads[0].annotation
    assert a.observed.visual.situation == "unknown" and a.observed.visual.framing == "close-up"
    assert a.mechanism[0].audience_role is None and a.observed.economics.sponsored == "unknown"


def test_the_strict_shape_drops_constraints_structured_outputs_reject():
    import json
    from manyworlds.brain.agents.intelligence import ReadBatch
    from manyworlds.llm import strict_json_schema
    text = json.dumps(strict_json_schema(ReadBatch))
    assert all(k not in text for k in ('"maxItems"', '"minimum"', '"maximum"'))
