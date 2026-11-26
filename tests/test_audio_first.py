"""Audio-first talking video (performance direction 3.7): a take's beats re-timed to its real voice line before the
talk prompt (production.beats), punch-ins on meaning (manyworlds.cuts), and J- and L-cuts at shot changes in the
joiner (media.concat audio_cuts; production.talk.join_audio_cuts / fit_audio_cuts). Everything stubbed; ffmpeg from
imageio-ffmpeg for the real joins."""

from __future__ import annotations

import subprocess

import pytest

from manyworlds import cuts, media, production
from manyworlds import policy as policy_mod
from manyworlds.creator import lab
from manyworlds.production import beats, talk

from test_production import creator

# --- the re-timing (pure) --------------------------------------------------------------------------------------

SHOT = {"performance": {"lines": [{"text": "Okay so I *finally* tried it.", "delivery": "dry"},
                                  {"text": "Was it worth it?", "delivery": "teasing"}]},
        "blocking": [{"at_s": 0, "action": "looks up from her phone", "eyeline": "lens"},
                     {"at_s": 1.5, "action": "points at the camera"},
                     {"at_s": 3.0, "action": "tilts her head"},
                     {"at_s": 4.5, "action": "laughs"}]}
META = {"seconds": 3.6,
        "lines": [{"index": 0, "text": "Okay so I finally tried it.", "start_s": 0.0, "end_s": 1.6},
                  {"index": 1, "text": "Was it worth it?", "start_s": 2.2, "end_s": 3.4}]}
WORDS = [{"word": w, "start": a, "end": b} for w, a, b in [
    ("okay", 0.0, 0.2), ("so", 0.25, 0.4), ("I", 0.45, 0.5), ("finally", 0.6, 1.1), ("tried", 1.15, 1.35),
    ("it", 1.4, 1.6), ("was", 2.2, 2.4), ("it", 2.45, 2.55), ("worth", 2.6, 3.0), ("it", 3.05, 3.4)]]
ROLLS = (0.3, 0.6)


def _cfg(policy):
    return policy.get("production", "beats")


def _wps(policy):
    return policy.get("generation", "talk_long", "words_per_second")


def _times(shot):
    return [b["at_s"] for b in shot["blocking"]]


def test_the_rolls_are_the_talk_workers_own(policy):
    from manyworlds.worker.handlers import talk_comfy
    params = {"lead_in_s": 0.3, "end_hold_s": 0.6}
    for seg, moving in [(None, False), ({"index": 0, "of": 3}, False), ({"index": 1, "of": 3}, False),
                        ({"index": 2, "of": 3}, False), (None, True)]:
        payload = {**({"segment": seg} if seg else {}), **({"video_uri": "s3://b/m.mp4"} if moving else {})}
        mine = beats.take_rolls(params, segment=(seg or {}).get("index"), of=(seg or {}).get("of", 1), moving=moving)
        assert mine == talk_comfy.rolls(payload, params)


def test_beats_move_onto_the_voice(policy):
    made = beats.retime(SHOT, META, WORDS, ROLLS, _cfg(policy), words_per_second=_wps(policy))
    # first in the lead-in; the point on *finally* (0.6 s + 0.3 s lead-in); the head tilt on the stressed word of
    # the rising line ("worth", stretched by the voice); the laugh in the end hold, just after the last word
    assert _times(made["shot"]) == [0.0, 0.9, 2.9, 3.8]
    assert [m["beat"] for m in made["moves"]] == [1, 2, 3]
    assert {m["kind"] for m in made["moves"]} == {"gesture", "head", "reaction"}
    assert made["shot"]["performance"] == SHOT["performance"] and SHOT["blocking"][1]["at_s"] == 1.5   # a copy
    assert [b["action"] for b in made["shot"]["blocking"]] == [b["action"] for b in SHOT["blocking"]]


def test_a_reaction_lands_in_a_pause_and_a_named_line_pulls_its_beat(policy):
    shot = {**SHOT, "blocking": [{"at_s": 0, "action": "looks up"}, {"at_s": 1.8, "action": "smiles"},
                                 {"at_s": 3.2, "action": "pours the coffee"}]}
    meta = {**META, "lines": [META["lines"][0], {**META["lines"][1]}]}
    made = beats.retime(shot, meta, WORDS, (0.3, 0.0), _cfg(policy), words_per_second=_wps(policy))
    # the smile at the start of the silence between the lines (1.6 s + lead-in); no hold: the last beat stays on
    # the voice (at the nearest line start, "pours" is neither gesture nor reaction)
    assert _times(made["shot"])[:2] == [0.0, 1.9]
    named = {**SHOT, "performance": {"lines": [{"text": "Right."}, {"text": "This *coffee* changed my mornings."}]},
             "blocking": [{"at_s": 0, "action": "looks up"}, {"at_s": 0.5, "action": "holds up the coffee"}]}
    meta2 = {"seconds": 3.0, "lines": [{"text": "Right.", "start_s": 0.0, "end_s": 0.4},
                                       {"text": "This coffee changed my mornings.", "start_s": 0.9, "end_s": 2.8}]}
    made = beats.retime(named, meta2, None, (0.3, 0.0), _cfg(policy), words_per_second=_wps(policy))
    said = beats.timed_words(named, meta2, None, 0.3, policy)
    at = next(w.start for w in said if w.text.startswith("coffee"))
    assert _times(made["shot"])[1] == round(round(at / 0.1) * 0.1, 3)   # the gesture on the emphasised word it names
    assert "names" in made["moves"][0]["why"]


def test_without_word_timestamps_the_marks_and_line_spans_are_enough(policy):
    made = beats.retime(SHOT, META, None, ROLLS, _cfg(policy), words_per_second=_wps(policy))
    t = _times(made["shot"])
    assert t[0] == 0.0 and 0.3 <= t[1] <= 0.3 + 1.6        # the point lands inside the first line, on *finally*
    assert t[-1] == 3.8                                    # the end hold still holds the last beat
    assert t == sorted(t) and all(b - a >= _cfg(policy)["min_gap_s"] - 1e-9 for a, b in zip(t, t[1:]))


def test_nothing_known_nothing_moves(policy):
    assert beats.retime(SHOT, {"lines": [{"text": "x"}]}, None, ROLLS, _cfg(policy), words_per_second=2.5) == \
        {"shot": SHOT, "moves": []}
    bare = {**SHOT, "blocking": []}
    assert beats.retime(bare, META, WORDS, ROLLS, _cfg(policy), words_per_second=2.5)["moves"] == []


def test_only_a_voice_line_that_says_the_shot_retimes_it():
    assert beats.says_its_shot(SHOT, META)
    assert not beats.says_its_shot(SHOT, {"lines": [{**META["lines"][0], "text": "Okay so I finally tried it."}]})
    assert not beats.says_its_shot(None, META)


def test_the_talk_job_carries_the_retimed_beats(conn, policy):
    cid = creator(conn)
    post = dict(conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, 'talking_video') RETURNING *",
                             (cid,)).fetchone())
    body = {"scenes": ["a cafe"], "script": "Okay so I *finally* tried it. Was it worth it?", "shots": [SHOT]}
    audio = {"id": None, "uri": "s3://b/voice.wav", "meta": {**META, "words": WORDS}}
    job = production._talk_job(conn, policy, post, body, [{"uri": "s3://b/a.png"}], 0, audio=audio)
    p = conn.execute("SELECT payload FROM jobs WHERE id = %s", (job,)).fetchone()["payload"]
    assert _times(p["shot"]) == [0.0, 0.9, 2.9, 3.8]                      # the judge sees the voice's beats too
    bible = conn.execute("SELECT bible FROM creators WHERE id = %s", (cid,)).fetchone()["bible"]
    assert p["prompt"] == lab.prompts("talk", bible, "a cafe", shot=p["shot"], policy=policy)["prompt"]
    kept = body["talk_beats"]["take"]
    assert kept["blocking"] == p["shot"]["blocking"] and kept["words"] == "transcript" and kept["rolls"] == [0.3, 0.6]
    # off: the director's times, as before
    off = policy_mod.from_dict({**policy.body, "production": {**policy.body["production"],
                                                              "beats": {**_cfg(policy), "enabled": False}}})
    body2 = {"scenes": ["a cafe"], "script": body["script"], "shots": [SHOT]}
    job = production._talk_job(conn, off, post, body2, [{"uri": "s3://b/a.png"}], 1, audio=audio)
    p = conn.execute("SELECT payload FROM jobs WHERE id = %s", (job,)).fetchone()["payload"]
    assert _times(p["shot"]) == [0, 1.5, 3.0, 4.5] and "talk_beats" not in body2


# --- punch-ins on meaning ------------------------------------------------------------------------------------------

def _even_words(n=21, emphasis=()):
    return [{"text": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4, "emphasis": i in emphasis} for i in range(n)]


PUNCH = {"every_s": [2.0, 2.0], "scales": [1.0, 1.2], "min_shot_s": 1.0, "center": [0.5, 0.4]}


def _cuts(plan):
    return [(s["start"], s["scale"]) for s in plan]


def test_a_punch_in_lands_on_the_emphasised_word_and_a_punch_out_never_does(policy):
    emphasis = policy.get("generation", "talk_edit", "emphasis")
    plain = cuts.punch_plan(_even_words(), [(0.0, 10.4)], 10.4, PUNCH)
    assert _cuts(plain)[:3] == [(0.0, 1.0), (1.95, 1.2), (3.95, 1.0)]           # the rhythm alone
    meant = cuts.punch_plan(_even_words(emphasis={5, 10}), [(0.0, 10.4)], 10.4, {**PUNCH, "emphasis": emphasis})
    # in just before w5 (its gap at 2.45), out at the rhythm's own gap (w10's emphasis is not let go of)
    assert _cuts(meant)[:3] == [(0.0, 1.0), (2.45, 1.2), (4.45, 1.0)]
    off = cuts.punch_plan(_even_words(emphasis={5}), [(0.0, 10.4)], 10.4,
                          {**PUNCH, "emphasis": {**emphasis, "enabled": False}})
    assert off == plain


def test_a_cut_prefers_a_line_start_after_a_pause(policy):
    emphasis = policy.get("generation", "talk_edit", "emphasis")
    words = _even_words(9) + [{"text": f"x{i}", "start": 5.0 + i * 0.5, "end": 5.4 + i * 0.5} for i in range(10)]
    # w8 ends at 4.4, x0 starts at 5.0: a 0.6 s pause, its gap at 4.7; the rhythm's first target is 2.0 -> 1.95,
    # the next 3.95: the pause's gap (4.7) is within the window and wins
    plan = cuts.punch_plan(words, [(0.0, 9.9)], 9.9, {**PUNCH, "emphasis": emphasis})
    assert (4.7, 1.0) in _cuts(plan)


def test_stressed_words_are_the_ones_the_voice_stretched():
    words = [{"text": "it", "start": 0.0, "end": 0.1}, {"text": "was", "start": 0.12, "end": 0.25},
             {"text": "huge", "start": 0.27, "end": 0.8}, {"text": "today", "start": 0.82, "end": 1.1}]
    assert cuts.stressed_indices(words, ratio=1.3, min_letters=4, pause_s=0.3) == {2}
    assert cuts.stressed_indices(words, ratio=5.0, min_letters=4, pause_s=0.3) == set()


def test_the_finish_hands_the_punch_ins_their_emphasis(conn, policy):
    post = {"id": None, "creator_id": creator(conn), "format": "talking_video"}
    _, punch = production._talk_edit(conn, policy, post, {})
    assert punch["emphasis"] == policy.get("generation", "talk_edit", "emphasis") and punch["scales"]


# --- J- and L-cuts ---------------------------------------------------------------------------------------------

def _shot(transition=""):
    return {"performance": {"lines": [{"text": "Hi."}]}, "edit": {"transition_in": transition}}


def _body(**kw):
    return {"shots": [_shot(), _shot(kw.get("t1", "")), _shot()], "talk_segments": ["a", "b", "c", "d"],
            "talk_shots": [0, 1, 1, 2], "talk_modes": ["talk", "talk", "talk", "narrated"]}


def test_the_join_plans_a_j_into_a_new_shot_and_an_l_into_a_cutaway(policy):
    c = policy.get("finishing", "jl_cuts")
    want = talk.join_audio_cuts(policy, _body(), 4, None)
    assert want == [{"cut": "j", "max_s": c["lead_ms"] / 1000}, None, {"cut": "l", "max_s": c["lag_ms"] / 1000}]
    # the next shot's edit asks for one; a transition that isn't a cut takes none
    assert talk.join_audio_cuts(policy, _body(t1="L-cut on her laugh"), 4, None)[0]["cut"] == "l"
    assert talk.join_audio_cuts(policy, _body(t1="a full cut"), 4, None)[0]["cut"] == "j"     # "full cut" is no L
    assert talk.join_audio_cuts(policy, _body(), 4, ["whip", "cut", "cut"])[0] is None
    off = policy_mod.from_dict({**policy.body, "finishing": {**policy.body["finishing"],
                                                             "jl_cuts": {**c, "enabled": False}}})
    assert talk.join_audio_cuts(off, _body(), 4, None) is None
    assert talk.join_audio_cuts(policy, {"talk_segments": ["a", "b"]}, 2, None) is None     # undirected: plain


def test_a_jl_cut_only_trims_silence_in_whole_frames(policy):
    c = policy.get("finishing", "jl_cuts")
    wanted = [{"cut": "j", "max_s": 0.16}, None, {"cut": "l", "max_s": 0.2}]
    edges = [(0.3, 0.5), (0.0, 0.0), (0.0, 0.0), (0.6, 0.6)]
    made = talk.fit_audio_cuts(wanted, edges, [3.0, 2.0, 2.0, 3.0], 30.0, c)
    assert made == [{"cut": "j", "s": round(4 / 30, 4)}, None, {"cut": "l", "s": 0.2}]
    assert talk.shifted_segment_seconds([3.0, 2.0, 2.0, 3.0], made) == [round(3.0 - 4 / 30, 4), 2.0, 1.8, 3.0]
    # no silence to trim: a plain cut
    assert talk.fit_audio_cuts(wanted, [(0, 0.01), (0, 0), (0, 0), (0.0, 0)], [3, 2, 2, 3], 30.0, c) is None


def test_join_times_follow_a_j_cuts_picture():
    class Conn:
        def execute(self, *a):
            class R:
                def fetchone(self):
                    return {"meta": {"segment_seconds": [2.8, 2.0, 3.0],
                                     "audio_cuts": [{"cut": "j", "s": 0.2}, {"cut": "l", "s": 0.1}]}}
            return R()
    assert production.join_times(Conn(), {"id": 1}) == [3.0, 4.8]


def _voiced(path, *, tone_s, silence_s, silence_first=False, seconds=1.5):
    tone = f"sine=frequency=440:duration={tone_s}"
    af = f"adelay={int(silence_s * 1000)}:all=1,apad=whole_dur={seconds}" if silence_first else \
        f"apad=whole_dur={seconds}"
    cmd = [media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=160x288:rate=30:duration={seconds}",
           "-f", "lavfi", "-i", tone, "-af", af, "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", str(path)]
    subprocess.run(cmd, check=True)
    return path


def _audio_seconds(path, tmp_path):
    wav = tmp_path / f"{path.stem}.wav"
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-i", str(path), "-map", "0:a", str(wav)], check=True)
    return media.probe(wav)["seconds"]


def test_edge_silence_reads_the_quiet_ends(tmp_path, raw_policy):
    c = raw_policy.get("finishing", "jl_cuts")
    a = _voiced(tmp_path / "a.mp4", tone_s=1.0, silence_s=0.5)
    b = _voiced(tmp_path / "b.mp4", tone_s=1.1, silence_s=0.4, silence_first=True)
    head, tail = media.edge_silence(a, noise_db=c["silence_db"], min_s=c["silence_min_s"])
    assert head == 0.0 and tail == pytest.approx(0.5, abs=0.08)
    head, tail = media.edge_silence(b, noise_db=c["silence_db"], min_s=c["silence_min_s"])
    assert head == pytest.approx(0.4, abs=0.08) and tail < 0.08


def test_a_j_cut_keeps_picture_and_sound_the_same_length(tmp_path, raw_policy):
    import imageio_ffmpeg
    a = _voiced(tmp_path / "a.mp4", tone_s=1.0, silence_s=0.5)
    b = _voiced(tmp_path / "b.mp4", tone_s=1.1, silence_s=0.4, silence_first=True)
    plain = media.concat([a, b], tmp_path / "plain.mp4", policy=raw_policy)
    for cut in ("j", "l"):
        out = media.concat([a, b], tmp_path / f"{cut}.mp4", policy=raw_policy, audio_cuts=[{"cut": cut, "s": 0.2}])
        frames, _ = imageio_ffmpeg.count_frames_and_secs(str(out))
        assert abs(frames - (90 - 6)) <= 2                                       # 6 frames at 30 fps left out
        assert _audio_seconds(out, tmp_path) == pytest.approx(frames / 30, abs=0.06)   # the sound lost the same
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(plain))
    assert abs(frames - 90) <= 2
    with pytest.raises(ValueError, match="audio cuts"):
        media.concat([a, b], tmp_path / "x.mp4", policy=raw_policy, audio_cuts=[None, None])


def test_the_graph_trims_the_incoming_picture_for_a_j_and_the_outgoing_one_for_an_l(raw_policy):
    fin = raw_policy.get("finishing")
    segs = [{"seconds": 2.0, "fps": 30.0, "audio": True}] * 2
    j = media.with_audio_cuts([{"type": "cut"}], [{"cut": "j", "s": 0.2}])
    graph, _, _ = media.transition_graph(segs, j, width=160, height=288, fps=30, fin=fin)
    parts = graph.split(";")
    assert "trim=" not in parts[0] and "atrim=end=1.8," in parts[1]           # the outgoing sound ends early
    assert "trim=start=0.2," in parts[2] and "[1:a]" in parts[3] and "atrim=start" not in parts[3]
    l_ = media.with_audio_cuts([{"type": "cut"}], [{"cut": "l", "s": 0.2}])
    graph, _, _ = media.transition_graph(segs, l_, width=160, height=288, fps=30, fin=fin)
    parts = graph.split(";")
    assert "trim=end=1.8," in parts[0] and "atrim=end" not in parts[1]       # the outgoing picture ends early
    assert "trim=" not in parts[2] and "atrim=start=0.2," in parts[3]
    # on a transition that isn't a plain cut, the audio cut is ignored
    assert media.with_audio_cuts([{"type": "whip"}], [{"cut": "j", "s": 0.2}]) == [{"type": "whip"}]
