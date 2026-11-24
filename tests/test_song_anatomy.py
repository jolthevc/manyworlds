"""Song anatomy and the music department's learning loop (spec 2026-11-23-music-department.md, sections 1, 2, 4):
CPU measurements on synthetic audio, songs and their versioned features, production's and the bench's hooks, the
music analysis over real-labelled outcomes read through evidence (simulated kept out), bands, nearest songs, song
predictions, the listening adapters and the critic with a stub LLM."""

from __future__ import annotations

import math
import random
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from psycopg.types.json import Jsonb

from manyworlds import llm as llm_mod
from manyworlds import music, music_listening, song_measure, songs
from manyworlds import policy as policy_mod
from manyworlds.brain import evidence, music_analysis, predictions
from manyworlds.brain.agents import music_critic
from manyworlds.production import song as song_mod

from brain_helpers import StubLLM, creator, make_ctx

SR = 22050
SECTION_S = 8.0
VERSE, CHORUS, BRIDGE = (261.63, 329.63, 392.0), (220.0, 261.63, 329.63, 440.0), (174.61, 220.0, 261.63)
FORM = [("verse", VERSE, 0.08), ("chorus", CHORUS, 0.3), ("verse", VERSE, 0.08), ("chorus", CHORUS, 0.3),
        ("bridge", BRIDGE, 0.12), ("chorus", CHORUS, 0.3)]
VOCAL_ENTRY_S = 3.0


def _section(chord, amp, rng, *, drums: float) -> np.ndarray:
    """A held chord at `amp` with a noise hit every half second (120 bpm) at `drums` (peaks stay under 1)."""
    t = np.arange(int(SR * SECTION_S)) / SR
    x = amp * sum(np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) for f in chord) / len(chord)
    env = np.zeros_like(t)
    for start in np.arange(0, SECTION_S, 0.5):
        i, n = int(start * SR), int(0.06 * SR)
        env[i:i + n] = np.exp(-np.linspace(0, 6, n))
    return (x + drums * env * rng.uniform(-1, 1, len(t))).astype("float32")


def _song(tmp: Path, *, scale: float = 1.0, stereo: bool = True, clip: bool = False) -> tuple[Path, Path]:
    """A 48 s synthetic song in FORM (the chorus three times, loudest, with drums) and its vocal stem (a tone that
    enters at VOCAL_ENTRY_S), written as wav files."""
    import soundfile as sf
    rng = np.random.default_rng(0)
    inst = np.concatenate([_section(c, a, rng, drums=0.4 if name == "chorus" else 0.1) for name, c, a in FORM])
    t = np.arange(len(inst)) / SR
    vocal = (0.06 * np.sin(2 * np.pi * 523.25 * t + 3 * np.sin(2 * np.pi * 5 * t))).astype("float32")
    vocal[t < VOCAL_ENTRY_S] = 0
    mix = (inst + vocal) * scale
    if clip:
        mix = np.clip(mix * 4, -1, 1)
    data = np.stack([mix, 0.9 * mix + 0.02 * rng.standard_normal(len(mix)).astype("float32")], axis=1) if stereo \
        else mix
    mp, vp = tmp / f"mix-{scale}-{stereo}-{clip}.wav", tmp / "vocals.wav"
    sf.write(mp, data, SR)
    sf.write(vp, vocal, SR)
    return mp, vp


LYRICS = ("[Verse]\nThe kettle sings before I do\nSteam on the window, writing you\n"
          "[Chorus]\nWindow seat, the night is ours\nCounting the streetlights instead of stars\n"
          "[Verse]\nSocks on the tiles, a borrowed sweater\nEvery small hour feels a little better\n"
          "[Chorus]\nWindow seat, the night is ours\nCounting the streetlights instead of stars")


@pytest.fixture(scope="module")
def measured(tmp_path_factory, raw_policy):
    pytest.importorskip("librosa")
    tmp = tmp_path_factory.mktemp("song")
    mix, vocal = _song(tmp)
    return {"tmp": tmp, "mix": mix, "vocal": vocal,
            "out": song_measure.measure(str(mix), str(vocal), LYRICS, policy=raw_policy, title="Window Seat")}


# --- measurements -------------------------------------------------------------------------------------------------

def test_the_repeated_loud_section_is_found_as_the_chorus(measured):
    f = measured["out"]["features"]
    assert f["chorus_repeats"] == 3
    assert abs(f["time_to_first_chorus_s"] - SECTION_S) <= 1.5
    assert f["chorus_energy_lift_db"] > 3                          # the chorus is the loud part
    assert f["repetition_share"] > 0.5 and f["self_similarity"] > 0.5
    names = [s["name"] for s in measured["out"]["detail"]["sections"]]
    assert names.count("chorus") == 3


def test_loudness_and_sound_numbers_are_sane(measured, tmp_path, raw_policy):
    f = measured["out"]["features"]
    assert -35 < f["lufs_integrated"] < -5
    assert f["true_peak_dbtp"] >= f["sample_peak_dbfs"] - 0.05      # oversampling never lowers the peak
    assert f["clipping_share"] == 0 and f["crest_factor_db"] > 3
    assert abs(f["duration_s"] - SECTION_S * len(FORM)) < 0.1
    assert f["loudness_range_lu"] > 3                               # quiet verses, loud choruses
    tempo = f["tempo_bpm"]
    assert any(abs(tempo - b) < 8 for b in (60, 120, 240))         # the drums' 120 (or an octave of it)
    assert f["mode"] in ("major", "minor") and re.match(r"^[A-G][#b]? (major|minor)$", f["key"])
    assert 0 < f["stereo_width"] < 0.5 and f["stereo_correlation"] > 0.9
    shares = [f[f"spectral_{b}_share"] for b in ("low", "low_mid", "high_mid", "high")]
    assert all(0 <= s <= 1 for s in shares) and 0.9 < sum(shares) <= 1.0001
    assert f["onset_density"] > 0.5
    half, _ = _song(tmp_path, scale=0.5)
    g = song_measure.measure(str(half), policy=raw_policy)["features"]
    assert g["lufs_integrated"] == pytest.approx(f["lufs_integrated"] - 6.02, abs=0.6)   # half the level: -6 dB
    assert "time_to_first_vocal_s" not in g                         # no vocal stem, no vocal features
    mono, _ = _song(tmp_path, stereo=False)
    h = song_measure.measure(str(mono), policy=raw_policy)["features"]
    assert h["stereo_width"] == 0 and h["stereo_correlation"] == 1 and h["mono_loss_db"] == 0
    hot, _ = _song(tmp_path, clip=True)
    assert song_measure.measure(str(hot), policy=raw_policy)["features"]["clipping_share"] > 0.001


def test_the_vocal_stem_gives_entry_clarity_and_singability(measured):
    f = measured["out"]["features"]
    assert abs(f["time_to_first_vocal_s"] - VOCAL_ENTRY_S) <= 0.3
    assert 0.8 < f["vocal_presence_share"] <= 1
    assert f["vocal_clarity_db"] < 0                                # a quiet voice under the band
    assert f["syllables_per_beat"] > 0


def test_lyric_features_from_the_text():
    f = song_measure.lyric_features(LYRICS, "Window Seat")
    assert f["line_count"] == 8 and f["repetition_ratio"] == 0.5    # the chorus's two lines, twice
    assert f["rhyme_density"] == 0.25          # orthographic: sweater/better rhyme; do/you and ours/stars don't
    clear = song_measure.lyric_features("[Verse]\nI walk alone at night\nUnder the yellow light\nHolding it tight")
    assert clear["rhyme_density"] == 1.0 and clear["title_in_hook"] is None
    assert f["title_in_hook"] == 1 and f["written_choruses"] == 2 and f["written_sections"] == 4
    assert song_measure.lyric_features(LYRICS, "Kettle Song")["title_in_hook"] == 0
    assert song_measure.lyric_features("[Instrumental]") == {}
    assert song_measure.syllables("streetlights") == 2 and song_measure.syllables("window") == 2


def test_every_measured_feature_is_in_the_registry(measured, raw_policy):
    names = {f["name"] for f in song_measure.registry(raw_policy)}
    extra = set(measured["out"]["features"]) - names
    assert extra == set(), f"measured but not in music.features: {sorted(extra)}"
    assert all(f["source"] and f["kind"] in ("number", "flag", "category") for f in song_measure.registry(raw_policy))


# --- songs as rows ------------------------------------------------------------------------------------------------

def _song_job(conn, cid, *, uris=("s3://b/mix.wav", "s3://b/vocals.wav", "s3://b/inst.wav"), batch=None,
              title="Window Seat"):
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, batch, payload) VALUES ('music.song', 'done', %s, "
                       "%s, %s) RETURNING id",
                       (cid, batch, Jsonb({"title": title, "lyrics": LYRICS, "tags": "indie pop", "bpm": 100,
                                           "key": "A minor", "engine": {"engine": "ace_step_1_5"}}))).fetchone()["id"]
    for role, uri in zip(music.ROLES, uris):
        meta = {"role": role, "seconds": 48}
        if role == "mix":
            meta["check"] = {"passed": True, "line_coverage": 0.9}
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) VALUES "
                     "(%s, %s, 'audio', %s, 'x', 1, %s)", (job, cid, uri, Jsonb(meta)))
    return job


def test_production_registers_the_chosen_song_and_links_the_post(conn, policy):
    cid = creator(conn, publish_mode="instagram")
    post = dict(conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, 'music_video') RETURNING *",
                             (cid,)).fetchone())
    spec = {"title": "Window Seat", "tags": ["indie pop", "soft female vocal"], "bpm": 100, "key": "a minor",
            "duration_s": 25, "lyrics": LYRICS, "genre": "indie"}
    body = {"scenes": ["a bedroom"], "song": spec}
    assert song_mod.advance_song(conn, policy, post, body)[0] == "waiting"
    job = body["song_jobs"][0]
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job,))
    for i, role in enumerate(music.ROLES):
        conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes, meta) VALUES (%s, 'audio', %s, 'x', 1, %s)",
                     (job, f"s3://b/{i}.wav", Jsonb({"role": role, "check": {"passed": True}})))
    assert song_mod.advance_song(conn, policy, post, body)[0] == "chosen"
    sid = body["song_id"]
    row = conn.execute("SELECT * FROM songs WHERE id = %s", (sid,)).fetchone()
    assert row["creator_id"] == cid and str(row["job_id"]) == str(job) and row["genre"] == "indie"
    assert row["version"] == 1
    assert row["mix_uri"] == "s3://b/0.wav" and row["vocal_uri"] == "s3://b/1.wav" and "lyrics" not in row["brief"]
    assert conn.execute("SELECT song_id FROM posts WHERE id = %s", (post["id"],)).fetchone()["song_id"] == row["id"]
    song_mod.advance_song(conn, policy, post, body)                 # later ticks: the same row
    assert conn.execute("SELECT count(*) n FROM songs").fetchone()["n"] == 1


def test_revisions_releases_and_versioned_features(conn):
    cid = creator(conn, publish_mode="instagram")
    first = songs.from_job(conn, _song_job(conn, cid), genre="pop")
    assert songs.from_job(conn, conn.execute("SELECT job_id FROM songs WHERE id = %s", (first,)).fetchone()["job_id"]) \
        == first                                                     # one row per job
    second = songs.revise(conn, first, _song_job(conn, cid, uris=("s3://b/m2.wav", "s3://b/v2.wav", "s3://b/i2.wav")))
    r = songs.get(conn, second)
    assert r["parent_id"] == first and r["version"] == 2 and r["genre"] == "pop"
    rel = songs.new_release(conn, cid, "ep", "Small Hours")
    songs.place(conn, first, rel, 1)
    songs.place(conn, second, rel, 2)
    assert songs.get(conn, second)["track_number"] == 2
    songs.store_features(conn, first, {"tempo_bpm": 90, "duration_s": 30}, version="m0")
    assert songs.latest_features(conn, [first])[str(first)]["tempo_bpm"] == 90
    songs.store_features(conn, first, {"tempo_bpm": 100, "duration_s": 30})
    songs.add_features(conn, first, {"aesthetics_pq_mix": 7.0})
    assert songs.latest_features(conn, [first])[str(first)] == {"tempo_bpm": 100, "duration_s": 30,
                                                                "aesthetics_pq_mix": 7.0}
    assert conn.execute("SELECT count(*) n FROM song_features WHERE song_id = %s", (first,)).fetchone()["n"] == 2
    songs.store_features(conn, first, {"tempo_bpm": 101, "duration_s": 30})   # a re-measure keeps the merged scores
    assert songs.latest_features(conn, [first])[str(first)]["aesthetics_pq_mix"] == 7.0


def test_the_producer_tick_measures_new_songs_on_the_control_plane(conn, policy, store, measured):
    cid = creator(conn, publish_mode="instagram")
    mix = store.put_file(measured["mix"], "songs/mix.wav").uri
    vocal = store.put_file(measured["vocal"], "songs/vocals.wav").uri
    sid = songs.from_job(conn, _song_job(conn, cid, uris=(mix, vocal, mix)))
    broken = songs.from_job(conn, _song_job(conn, cid, uris=("file:///nowhere/x.wav",) * 3))
    from manyworlds import production
    out = production.advance(conn, policy, store=store)
    assert out["songs_measured"] == {"measured": 1, "failed": 1}
    f = songs.latest_features(conn, [sid])[str(sid)]
    assert f["chorus_repeats"] == 3 and f["line_coverage"] == 0.9       # the worker's coverage kept with them
    assert conn.execute("SELECT version, policy_version FROM song_features WHERE song_id = %s",
                        (sid,)).fetchone() == {"version": song_measure.VERSION, "policy_version": policy.version}
    assert songs.get(conn, broken)["measure_errors"] == 1 and songs.get(conn, broken)["measure_error"]
    assert songs.pending(conn, policy) == [broken]                      # measured once; the broken one is retried


def test_the_bench_report_lists_its_songs(conn, policy):
    from manyworlds import bench
    job = _song_job(conn, None, batch="bench-music.song-x")
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, gpu_type, gpu_seconds, outcome) "
                 "VALUES (%s, 'w', 'vast_4090', 'rtx4090', 60, 'done')", (job,))
    r = bench.report(conn, policy, "bench-music.song-x")
    assert len(r["song_features"]) == 1 and r["song_features"][0]["title"] == "Window Seat"
    assert r["song_features"][0]["features"] is None                    # not measured yet


# --- the learning loop --------------------------------------------------------------------------------------------

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
TARGET = 259200


def _history(conn, policy, n, *, real: bool, effect: float, seed=0, genre="pop", cid=None):
    """An artist with n posts, each using its own song; reach = 1000 * exp(effect * z(chorus_repeats) + noise). Real
    artists get real publications and metric snapshots; sandbox ones simulated ones (through evidence)."""
    rng = random.Random(seed)
    cid = cid or creator(conn, publish_mode="instagram" if real else "simulated")
    reps = [rng.choice([1, 2, 3, 4]) for _ in range(n)]
    m, s = statistics_mean_sd(reps)
    ids = []
    for i, rep in enumerate(reps):
        when = T0 + timedelta(hours=12 * i)
        sid = conn.execute("INSERT INTO songs (creator_id, title, genre) VALUES (%s, %s, %s) RETURNING id",
                           (cid, f"song {i}", genre)).fetchone()["id"]
        songs.store_features(conn, sid, {"duration_s": 30, "chorus_repeats": rep, "tempo_bpm": rng.uniform(80, 140),
                                         "lufs_integrated": rng.uniform(-14, -8), "mode": rng.choice(["major", "minor"]),
                                         "onset_density": rng.uniform(1, 4), "vocal_clarity_db": rng.uniform(-6, 3),
                                         "time_to_first_chorus_s": rng.uniform(5, 30)})
        post = conn.execute("INSERT INTO posts (creator_id, format, status, published_at, song_id) VALUES "
                            "(%s, 'music_video', %s, %s, %s) RETURNING id",
                            (cid, "published" if real else "simulated", when, sid)).fetchone()["id"]
        conn.execute("INSERT INTO publications (post_id, platform, simulated, published_at) VALUES (%s, %s, %s, %s)",
                     (post, "instagram" if real else "simulated", not real, when))
        reach = round(1000 * math.exp(effect * (rep - m) / s + rng.gauss(0, 0.1)))
        if real:
            conn.execute("INSERT INTO metric_snapshots (post_id, metrics, post_age_s, captured_at) VALUES (%s, %s, %s, %s)",
                         (post, Jsonb({"reach": reach, "saves": reach // 50}), TARGET, when + timedelta(seconds=TARGET)))
        else:
            evidence.record_simulated_metrics(conn, post, [{"post_age_s": TARGET, "metrics": {"reach": reach}}],
                                              model="test", panel_version="v1")
        ids.append(str(sid))
    return cid, ids


def statistics_mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def test_the_music_analysis_finds_a_planted_effect_in_real_outcomes(conn, policy):
    cid, _ = _history(conn, policy, 40, real=True, effect=0.5)
    _history(conn, policy, 40, real=False, effect=-0.8, seed=1)          # a sandbox artist saying the opposite
    result = music_analysis.analyse(conn, policy, scope="creator", creator_id=cid)
    assert result["mode"] == "real" and not result["insufficient"]
    top = {(e["kind"], e["key"]): e for e in result["effects"]}
    chorus = top[("feature", "chorus_repeats")]
    assert chorus["grade"] == "tentative" and chorus["effect"] > 0 and chorus["low"] > 0
    ids = music_analysis.record(conn, policy, result)
    found = music_analysis.findings(conn, mode="real", creator_id=cid)
    assert ids and any("chorus_repeats" in f["statement"] and f["grade"] == "tentative" for f in found)
    assert all(f["domain"] == "music" for f in found)
    assert evidence.findings(conn, mode="real", creator_id=cid, domain="content") == []
    # the portfolio in real mode sees only the real artist: the sandbox's opposite effect never reaches it
    port = music_analysis.analyse(conn, policy, scope="portfolio")
    assert port["creators"] == [str(cid)]
    assert {(e["kind"], e["key"]): e for e in port["effects"]}[("feature", "chorus_repeats")]["effect"] > 0
    # a new analysis replaces the old findings; the analyst's own supersede leaves music findings alone
    evidence.supersede(conn, creator_id=cid)
    assert music_analysis.findings(conn, mode="real", creator_id=cid)
    music_analysis.record(conn, policy, result)
    assert len(music_analysis.findings(conn, mode="real", creator_id=cid)) == len(ids)


def test_sandbox_music_findings_are_simulated_and_stay_out_of_real_reads(conn, policy):
    sid, _ = _history(conn, policy, 40, real=False, effect=0.6, seed=2)
    result = music_analysis.analyse(conn, policy, scope="creator", creator_id=sid)
    assert result["mode"] == "sandbox" and not result["insufficient"]
    assert music_analysis.record(conn, policy, result)
    assert music_analysis.findings(conn, mode="real", creator_id=sid) == []
    labels = {f["label"] for f in music_analysis.findings(conn, mode="sandbox", creator_id=sid)}
    assert labels == {evidence.LABEL}
    assert music_analysis.analyse(conn, policy, scope="creator", creator_id=sid, mode="real")["n"] == 0


def test_genre_findings_and_too_little_data(conn, policy):
    _history(conn, policy, 40, real=True, effect=0.5, genre="country")
    g = music_analysis.analyse(conn, policy, scope="genre", genre="country")
    assert music_analysis.record(conn, policy, g)
    assert all(f["genre"] == "country" and f["scope"] == "genre"
               for f in music_analysis.findings(conn, mode="real", genre="country"))
    few, _ = _history(conn, policy, 5, real=True, effect=0.5, genre="rock")
    r = music_analysis.analyse(conn, policy, scope="creator", creator_id=few)
    assert r["insufficient"] and music_analysis.record(conn, policy, r) == []


def test_bands_come_from_the_artists_best_songs_else_the_genre(conn, policy):
    cid, _ = _history(conn, policy, 30, real=True, effect=0.5)
    b = music_analysis.bands(conn, policy, creator_id=cid, genre="pop")
    assert b["source"] == "artist" and b["n"] >= policy.get("music", "analysis", "bands", "min_songs")
    lo, hi = b["bands"]["chorus_repeats"]
    assert lo >= 2 and hi <= 4                                          # the best songs repeat their chorus
    pos = {p["feature"]: p["position"] for p in music_analysis.position({"chorus_repeats": 1}, b)}
    assert pos == {"chorus_repeats": "below"}
    new = creator(conn, publish_mode="instagram")
    pb = music_analysis.bands(conn, policy, creator_id=new, genre="indie")
    assert pb["source"] == "genre_playbook" and pb["bands"]["tempo_bpm"] == policy.get("music", "genre_bands", "indie",
                                                                                       "tempo_bpm")
    assert music_analysis.bands(conn, policy, creator_id=new, genre="polka")["source"] is None


def test_nearest_songs_by_normalised_features_with_outcomes(conn, policy):
    cid, ids = _history(conn, policy, 12, real=True, effect=0.5)
    target = conn.execute("INSERT INTO songs (creator_id, title, genre) VALUES (%s, 'new', 'pop') RETURNING id",
                          (cid,)).fetchone()["id"]
    twin = songs.latest_features(conn, [ids[5]])[ids[5]]
    songs.store_features(conn, target, {**twin, "tempo_bpm": twin["tempo_bpm"] + 0.5})
    near = music_analysis.nearest(conn, policy, target, k=3)
    assert near[0]["song_id"] == ids[5] and near[0]["distance"] < near[1]["distance"]
    assert near[0]["outcome"]["n_posts"] == 1 and near[0]["outcome"]["reach_median"] > 0
    assert len(near[0]["alike_on"]) == 3 and near[0]["shared"] >= policy.get("music", "analysis", "similarity",
                                                                              "min_shared")
    assert all(n["song_id"] != str(target) for n in near)


def test_a_song_prediction_resolves_against_the_posts_using_the_song(conn, policy):
    cid = creator(conn, publish_mode="instagram")
    sid = conn.execute("INSERT INTO songs (creator_id, title) VALUES (%s, 'x') RETURNING id", (cid,)).fetchone()["id"]
    pid = music_analysis.file_prediction(conn, policy, role="music_critic", song_id=sid, claim="reach above 500",
                                         metric="song_median_reach", comparator="above", threshold=500, probability=0.7)
    assert pid and predictions.record(conn, role="x", claim="c", metric="song_median_reach", comparator="above",
                                      threshold=1, probability=0.5, horizon_days=14, creator_id=cid) is None
    pred = conn.execute("SELECT * FROM predictions WHERE id = %s", (pid,)).fetchone()
    assert pred["song_id"] == sid
    when = pred["created_at"] + timedelta(days=1)
    for song, reach in ((sid, 900), (None, 100)):
        post = conn.execute("INSERT INTO posts (creator_id, format, song_id) VALUES (%s, 'music_video', %s) RETURNING id",
                            (cid, song)).fetchone()["id"]
        conn.execute("INSERT INTO publications (post_id, platform, simulated, published_at) VALUES (%s, 'instagram', "
                     "false, %s)", (post, when))
        conn.execute("INSERT INTO metric_snapshots (post_id, metrics, post_age_s) VALUES (%s, %s, %s)",
                     (post, Jsonb({"reach": reach}), TARGET))
    assert predictions.measure(conn, pred) == 900                       # only the song's post counts
    out = predictions.resolve_due(conn, policy, now=pred["resolve_at"] + timedelta(days=5))
    assert out["resolved"] == 1
    assert conn.execute("SELECT outcome FROM predictions WHERE id = %s", (pid,)).fetchone()["outcome"] is True


# --- listening adapters -------------------------------------------------------------------------------------------

def _with(policy, **critic):
    body = dict(policy.body)
    body["music"] = {**body["music"], "critic": {**body["music"]["critic"], **critic}}
    return policy_mod.from_dict(body)


def test_listening_adapters_are_chosen_by_config(conn, policy):
    assert isinstance(music_listening.listener(policy), music_listening.StubListener)
    assert music_listening.listener(policy).describe({}) is None
    assert music_listening.aesthetics(policy).scores({}) is None
    pw = _with(policy, listener={**policy.get("music", "critic", "listener"), "adapter": "worker"},
               aesthetics={"adapter": "worker"})
    cid = creator(conn)
    song = songs.get(conn, songs.from_job(conn, _song_job(conn, cid)))
    lw, aw = music_listening.listener(pw, conn=conn), music_listening.aesthetics(pw, conn=conn)
    assert lw.describe(song) is None and aw.scores(song) is None        # asked once, nothing yet
    jobs = conn.execute("SELECT id, kind, payload FROM jobs WHERE kind = 'music.listen'").fetchall()
    assert len(jobs) == 1 and jobs[0]["payload"]["mix_uri"] == song["mix_uri"]
    conn.execute("UPDATE jobs SET status = 'done', result = %s WHERE id = %s",
                 (Jsonb({"listener": {"genre": "indie pop", "mix_problems": ["buried vocal"]},
                         "aesthetics": {"mix": {"PQ": 6.5}}}), jobs[0]["id"]))
    assert lw.describe(song).mix_problems == ["buried vocal"] and aw.scores(song) == {"mix": {"PQ": 6.5}}
    ph = _with(policy, listener={**policy.get("music", "critic", "listener"), "adapter": "hosted"})
    assert music_listening.listener(ph, llm=StubLLM(), store=object()).describe(song) is None   # no role yet
    with pytest.raises(ValueError):
        music_listening.listener(_with(policy, listener={"adapter": "telepathy"}))


def test_audio_goes_only_to_models_that_hear():
    block = llm_mod.audio_block(b"RIFF", "wav")
    out = llm_mod._openai_content([block, {"type": "text", "text": "listen"}], vision=False, audio=True)
    assert out[0] == {"type": "input_audio", "input_audio": {"data": block["source"]["data"], "format": "wav"}}
    with pytest.raises(llm_mod.OpenModelError):
        llm_mod._openai_content([block], vision=True)


# --- the critic ---------------------------------------------------------------------------------------------------

class CriticLLM(StubLLM):
    def __init__(self, answer, conn=None):
        super().__init__(conn=conn)
        self.answer = answer

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        self.calls[-1]["system"] = system
        return schema.model_validate(self.answer)


ANSWER = {
    "dimensions": [
        {"dimension": "structure", "finding": "the chorus arrives late", "evidence": ["time_to_first_chorus_s",
                                                                                    "band.time_to_first_chorus_s"],
         "severity": "major"},
        {"dimension": "mix", "finding": "sounds great", "evidence": ["aesthetics.mix.PQ"], "severity": "strength"},
        {"dimension": "hook", "finding": "a vibe", "evidence": [], "severity": "minor"}],
    "fixes": [
        {"role": "songwriter", "problem": "chorus at 28 s", "change": "open with the chorus",
         "evidence": ["time_to_first_chorus_s"]},
        {"role": "dj", "problem": "x", "change": "y", "evidence": ["lufs_integrated"]},
        {"role": "mix_master", "problem": "quiet", "change": "master to -9 LUFS", "evidence": ["lufs_integrated"]}],
    "verdict": "revise",
    "prediction": {"metric": "song_reach_lift", "comparator": "above", "threshold": 0.0, "probability": 0.4,
                   "claim": "the song's posts beat her usual reach"},
    "summary": "fix the structure first"}


def _measured_song(conn, cid):
    sid = songs.from_job(conn, _song_job(conn, cid), genre="pop")
    songs.store_features(conn, sid, {"duration_s": 30, "time_to_first_chorus_s": 28.0, "lufs_integrated": -16.0,
                                     "chorus_repeats": 2, "tempo_bpm": 100})
    return sid


def test_the_critic_reasons_over_evidence_routes_fixes_and_files_a_prediction(conn, policy):
    from manyworlds import queue
    cid = creator(conn, publish_mode="instagram")
    sid = _measured_song(conn, cid)
    job = queue.enqueue(conn, "brain.music_critic", {"song_id": str(sid)}, creator_id=cid)
    ctx = make_ctx(conn, policy, job)
    ctx.llm = CriticLLM(ANSWER, conn=conn)
    out = music_critic.critique(ctx, sid, aesthetics=music_listening.StubAesthetics({"mix": {"PQ": 8.0, "CE": 7.0}}),
                                listener=music_listening.StubListener())
    assert out["verdict"] == "revise" and out["fixes"] == 2 and out["routed"] == ["songwriter", "mix_master"]
    assert out["dropped"] == 3                         # the aesthetics-only finding, the uncited one, the unknown role
    call = ctx.llm.calls[0]
    assert call["role"] == policy.get("music", "critic", "role") and "nothing heard" in call["content"]
    assert "band.time_to_first_chorus_s" in call["content"]             # the genre's playbook band was shown
    d = conn.execute("SELECT kind, output, creator_id FROM decisions WHERE id = %s", (out["decision_id"],)).fetchone()
    assert d["kind"] == "music_critique" and d["creator_id"] == cid
    assert [x["dimension"] for x in d["output"]["dimensions"]] == ["structure"]
    a = conn.execute("SELECT verdict FROM song_assessments WHERE song_id = %s", (sid,)).fetchall()
    assert [r["verdict"] for r in a] == ["revise"]
    fb = songs.feedback_for(conn, "songwriter", song_id=sid)
    assert len(fb) == 1 and fb[0]["change"] == "open with the chorus" and fb[0]["evidence"] == ["time_to_first_chorus_s"]
    p = conn.execute("SELECT metric, song_id, role FROM predictions WHERE id = %s", (out["prediction"],)).fetchone()
    assert p == {"metric": "song_reach_lift", "song_id": sid, "role": "music_critic"}
    assert songs.latest_features(conn, [sid])[str(sid)]["aesthetics_pq_mix"] == 8.0
    again = music_critic.critique(ctx, sid, aesthetics=music_listening.StubAesthetics(),
                                  listener=music_listening.StubListener())   # a rerun: no new call, no duplicates
    assert len(ctx.llm.calls) == 1 and again["routed"] == [] and again["prediction"] is None
    assert conn.execute("SELECT count(*) n FROM song_feedback").fetchone()["n"] == 2
    assert conn.execute("SELECT count(*) n FROM predictions").fetchone()["n"] == 1


def test_the_critic_runs_as_a_brain_job_and_refuses_an_unmeasured_song(conn, policy):
    from manyworlds import queue
    from manyworlds.brain import agent
    from manyworlds.worker.runner import NonRetryable
    cid = creator(conn, publish_mode="instagram")
    sid = _measured_song(conn, cid)
    ctx = make_ctx(conn, policy, queue.enqueue(conn, "brain.music_critic", {"song_id": str(sid)}, creator_id=cid))
    out = agent.get("music_critic").run(ctx)                  # the stub LLM: an empty but valid assessment
    assert out["verdict"] == "revise" and out["fixes"] == 0 and out["prediction"] is None
    bare = songs.from_job(conn, _song_job(conn, cid))
    ctx2 = make_ctx(conn, policy, queue.enqueue(conn, "brain.music_critic", {"song_id": str(bare)}, creator_id=cid))
    with pytest.raises(NonRetryable, match="not measured"):
        agent.get("music_critic").run(ctx2)
    assert agent.get("music_critic").name in policy.get("brain", "chief_of_staff", "decision_agents")
    assert policy.get("brain", "llm_usd_per_day", "music_critic") > 0


def test_findings_accept_the_genre_scope_and_carry_a_domain(conn):
    fid = evidence.record_finding(conn, scope="genre", statement="s", evidence="observational", grade="speculative",
                                  domain="music", genre="rap")
    row = evidence.findings(conn, mode="real", scope="genre", genre="rap")
    assert [r["id"] for r in row] == [fid] and row[0]["domain"] == "music"
    plain = evidence.record_finding(conn, scope="creator", statement="t", evidence="observational", grade="speculative")
    assert conn.execute("SELECT domain FROM findings WHERE id = %s", (plain,)).fetchone()["domain"] == "content"
    assert uuid.UUID(str(fid))
