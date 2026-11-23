"""On-screen text (manyworlds.overlays): labels, the hook, timestamps, question bubbles; wrap, safe zone, face, ASS."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from manyworlds import captions as cap
from manyworlds import overlays as ov
from manyworlds import policy as policy_mod

W, H = 1080, 1920


def _dialogues(ass: str, layer: int | None = None) -> list[list[str]]:
    rows = [line[len("Dialogue: "):].split(",", 9) for line in ass.splitlines() if line.startswith("Dialogue: ")]
    return [r for r in rows if layer is None or int(r[0]) == layer]


def _with(raw_policy, **block):
    return policy_mod.from_dict({**raw_policy.body, "on_screen_text": {**raw_policy.body["on_screen_text"], **block}})


# --- text -----------------------------------------------------------------------------------------------------

def test_wrap_keeps_short_labels_on_one_line_and_balances_longer_ones():
    assert ov.wrap("lazy ootd", 24, 3) == ["lazy ootd"]
    lines = ov.wrap("beach day, but don't know what to wear?", 24, 3)
    assert len(lines) == 2 and all(len(s) <= 24 for s in lines)
    assert " ".join(lines) == "beach day, but don't know what to wear?"
    assert abs(len(lines[0]) - len(lines[1])) <= 8                          # balanced, not greedy
    assert ov.wrap("Supercalifragilisticexpialidocious", 10, 3) == ["Supercalifragilisticexpialidocious"]
    assert ov.wrap("", 20, 3) == []


def test_wrap_never_leaves_one_word_alone_on_the_last_line():
    for text in ("tight on baggy always works", "what's your skin type and routine", "one two three"):
        for lines in (ov.wrap(text, 14, 3), ov.wrap(text, 9, 3)):
            assert " ".join(lines) == text
            assert len(lines) == 1 or " " in lines[-1], lines
    long = " ".join(["word"] * 20)
    lines = ov.wrap(long, 16, 3)                                             # past the search: greedy, no orphan
    assert " ".join(lines) == long and " " in lines[-1]


@pytest.mark.parametrize("text,time,rest", [("7:30 AM Hygiene", "7:30 AM", "Hygiene"),
                                            ("7am - coffee run", "7am", "coffee run"),
                                            ("19:45 dinner", "19:45", "dinner"),
                                            ("7:30 a.m.", "7:30 a.m.", ""),
                                            ("10 things I do", None, "10 things I do")])
def test_split_time(text, time, rest):
    assert ov.split_time(text) == (time, rest)


def test_emoji_are_dropped_unless_kept(raw_policy):
    (o,) = ov.overlay_events([{"kind": "label", "text": "Tight on baggy ✅", "number": 2}], (W, H), raw_policy,
                             duration=5)
    assert [ln.text for ln in o.lines] == ["2. Tight on baggy"] and o.style_name == "numbered"
    (kept,) = ov.overlay_events([{"kind": "label", "text": "Tight on baggy ✅"}], (W, H),
                                _with(raw_policy, missing_glyphs="keep"), duration=5)
    assert "✅" in kept.lines[0].text
    assert ov.overlay_events([{"kind": "label", "text": "✅✨"}], (W, H), raw_policy, duration=5) == []


# --- items and times ------------------------------------------------------------------------------------------

def test_items_for_shot_shift_into_the_video():
    assert ov.items_for_shot({"on_screen_text": "lazy ootd"}, 4.0, 7.0) == [
        {"kind": "label", "text": "lazy ootd", "start_s": 4.0, "end_s": 7.0}]
    items = ov.items_for_shot({"on_screen_text": [{"kind": "hook", "text": "wait for 3", "start_s": 0},
                                                  {"kind": "label", "text": "1.", "start_s": 0.5, "end_s": 9}]}, 2.0, 6.0)
    assert [(i["start_s"], i["end_s"]) for i in items] == [(2.0, 6.0), (2.5, 6.0)]
    assert ov.items_for_shot({}, 0, 1) == [] and ov.items_for_shot({"on_screen_text": " "}, 0, 1) == []


def test_times_default_per_kind_and_clip_to_the_video(raw_policy):
    laid = ov.overlay_events([{"kind": "hook", "text": "you need this jacket"},
                              {"kind": "label", "text": "lazy ootd", "start_s": 1.0},
                              {"kind": "label", "text": "too late", "start_s": 9.98}],     # 0.02 s: left out
                             (W, H), raw_policy, duration=10.0)
    assert [o.kind for o in laid] == ["hook", "label"]
    hook, label = laid
    k = raw_policy.get("on_screen_text", "kinds", "hook")
    assert hook.start == 0.0 and hook.end == pytest.approx(k["duration_s"]) and hook.end <= k["max_duration_s"]
    assert label.start == 1.0 and label.end == 10.0
    assert len(ov.overlay_events([{"kind": "label", "text": "too late", "start_s": 9.99}], (W, H), raw_policy,
                                 duration=10.0)) == 0
    long_hook = ov.overlay_events([{"kind": "hook", "text": "hi", "end_s": 6}], (W, H), raw_policy, duration=10)[0]
    assert long_hook.end == pytest.approx(k["max_duration_s"])
    assert ov.overlay_events([{"text": "x"}], (W, H), _with(raw_policy, enabled=False), duration=5) == []


# --- layout ---------------------------------------------------------------------------------------------------

ITEMS = [
    {"kind": "label", "text": "beach day, but don't know what to wear?"},
    {"kind": "label", "text": "lazy ootd", "style": "tiktok_box"},
    {"kind": "label", "text": "tight on baggy", "number": 1},
    {"kind": "hook", "text": "the jacket everyone keeps asking about"},
    {"kind": "timestamp", "text": "7:30 AM Hygiene"},
    {"kind": "question", "text": "What's your skin type? And how do you keep it from going shiny by noon?"},
]


@pytest.mark.parametrize("size", [(1080, 1920), (720, 1280)])
def test_every_style_and_position_stays_inside_the_safe_zone(raw_policy, size):
    w, h = size
    x0, y0, x1, y1 = ov.safe_area(w, h, raw_policy)
    if size == (1080, 1920):   # research: top ~250 px, bottom ~320 px, the right-side rail
        assert y0 == pytest.approx(0.13 * h) and h - y1 >= 320 and y0 >= 245 and w - x1 >= 120
    for style in ov.styles(raw_policy):
        for position in raw_policy.get("on_screen_text", "positions"):
            items = [{**it, "style": style, "position": position, "start_s": 10.0 * i, "end_s": 10.0 * i + 2}
                     for i, it in enumerate(ITEMS + [{"kind": "label", "text": "Supercalifragilisticexpialidocious!"}])]
            for o in ov.overlay_events(items, (w, h), raw_policy):
                ex0, ey0, ex1, ey1 = o.extent
                assert x0 - 0.5 <= ex0 and ex1 <= x1 + 0.5, (style, position, o.text, o.extent)
                assert y0 - 0.5 <= ey0 and ey1 <= y1 + 0.5, (style, position, o.text, o.extent)
                assert o.position == position                                 # no face, no clash: it stays put
                for line in o.lines:
                    if o.align == "center":
                        assert (line.x0 + line.x1) / 2 == pytest.approx(w / 2)
                    else:
                        assert line.x0 >= x0


def test_positions_are_shares_of_the_frame(raw_policy):
    item = [{"kind": "label", "text": "lazy ootd"}]
    big = ov.overlay_events(item, (1080, 1920), raw_policy, duration=3)[0]
    small = ov.overlay_events(item, (720, 1280), raw_policy, duration=3)[0]
    assert big.lines[0].baseline / 1920 == pytest.approx(small.lines[0].baseline / 1280, abs=0.005)
    mid = (big.extent[1] + big.extent[3]) / 2 / 1920
    assert mid == pytest.approx(raw_policy.get("on_screen_text", "positions", "upper_third", "y"), abs=0.005)
    q = ov.overlay_events([{"kind": "question", "text": "What's your skin type?"}], (W, H), raw_policy, duration=3)[0]
    assert q.position == "top_bubble" and q.extent[1] == pytest.approx(0.14 * H, abs=1)   # bubble at the top
    assert q.tail is not None and len(q.boxes) == 1


def test_a_label_over_the_face_moves_to_a_clear_position(raw_policy):
    face = (0.4, 0.24, 0.6, 0.36)            # a full-body shot: the head in the upper third
    (o,) = ov.overlay_events([{"kind": "label", "text": "lazy ootd"}], (W, H), raw_policy, duration=3, face_box=face)
    assert o.position != "upper_third"
    _, ey0, _, ey1 = o.extent
    assert not cap.covers_face(ey0 / H, ey1 / H, face)
    (o,) = ov.overlay_events([{"kind": "label", "text": "lazy ootd"}], (W, H), raw_policy, duration=3,
                             face_box=(0.4, 0.6, 0.6, 0.7))                  # face elsewhere: stays
    assert o.position == "upper_third"


def test_items_showing_together_do_not_overlap(raw_policy):
    a, b = ov.overlay_events([{"kind": "label", "text": "1.", "start_s": 0, "end_s": 3},
                              {"kind": "label", "text": "the cardigan", "start_s": 1, "end_s": 3}], (W, H),
                             raw_policy)
    assert a.position == "upper_third" and b.position != "upper_third"
    c, d = ov.overlay_events([{"kind": "label", "text": "1.", "start_s": 0, "end_s": 1},
                              {"kind": "label", "text": "2.", "start_s": 1, "end_s": 2}], (W, H), raw_policy)
    assert c.position == d.position == "upper_third"                          # one after the other: same place


# --- ASS ------------------------------------------------------------------------------------------------------

def test_each_kind_renders_as_ass(raw_policy):
    # One after another (each at its kind's position); the hook keeps its own 0-1.5 s.
    items = [it if it["kind"] == "hook" else {**it, "start_s": 2.0 + i, "end_s": 2.9 + i} for i, it in enumerate(ITEMS)]
    overlays = ov.overlay_events(items, (W, H), raw_policy, duration=8.0)
    assert [o.position for o in overlays] == ["upper_third", "upper_third", "upper_third", "upper_third",
                                              "upper_left", "top_bubble"]
    ass = ov.make_script(None, overlays, raw_policy)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    for name in ("tiktok_plain", "tiktok_box", "numbered", "hook", "serif_small", "question_bubble"):
        assert re.search(rf"^Style: ost_{name},", ass, re.M), name
    assert re.search(r"^Style: ost_tiktok_plain,TikTok Sans SemiBold,", ass, re.M)
    assert re.search(r"^Style: ost_tiktok_box,TikTok Sans SemiBold,[\d.]+,&H00000000,", ass, re.M)  # black text
    assert re.search(r"^Style: ost_hook,TikTok Sans,[\d.]+,[^,]+,[^,]+,[^,]+,[^,]+,-1,", ass, re.M)  # bold
    text, boxes = _dialogues(ass, layer=3), _dialogues(ass, layer=2)
    by_style = {}
    for r in text + boxes:
        by_style.setdefault(r[3], []).append(r)
    hook = [r for r in text if r[3] == "ost_hook"]
    assert hook and all(r[1] == "0:00:00.00" and r[2] == "0:00:01.50" and "\\t(0," in r[9] for r in hook)
    plain = [r for r in text if r[3] == "ost_tiktok_plain"]
    assert len(plain) == 2 and all("\\an2\\pos(" in r[9] and "\\fad(" in r[9] and "\\blur" in r[9] for r in plain)
    assert [r[9].split("}", 1)[1] for r in plain] == ov.wrap("beach day, but don't know what to wear?", 24, 3)
    pills = [r for r in boxes if r[3] == "ost_tiktok_box"]
    assert len(pills) == 1 and "\\1c&HFFFFFF&" in pills[0][9] and "\\p1}m " in pills[0][9]
    assert any(r[9].endswith("}1. tight on baggy") for r in text if r[3] == "ost_numbered")
    stamp = [r for r in text if r[3] == "ost_serif_small"]
    assert [r[9].split("}", 1)[1] for r in stamp] == ["7:30 AM", "Hygiene"]
    assert "\\an1" in stamp[0][9] and "\\fs" not in stamp[0][9] and "\\fs" in stamp[1][9]   # caption smaller
    (bubble,) = [r for r in boxes if r[3] == "ost_question_bubble"]
    shape = bubble[9].split("\\p1}", 1)[1]
    assert shape.count(" l ") + shape.startswith("l ") == 7                   # 4 sides + 3 for the tail
    q = [r for r in text if r[3] == "ost_question_bubble"]
    assert len(q) >= 2 and "Reply" not in ass
    for r in text + boxes:
        assert r[1] < r[2]


def test_bubble_path_without_a_tail_is_the_rounded_rect():
    assert ov.bubble_path(200, 60, 20) == cap._rounded_rect(200, 60, 20)
    tailed = ov.bubble_path(200, 60, 20, (5, 30, 15))
    assert tailed.count(" l ") == cap._rounded_rect(200, 60, 20).count(" l ") + 3
    assert "75" in tailed                                                     # the tip hangs 15 px under 60


def test_one_script_carries_captions_and_overlays(raw_policy):
    words = cap.estimate_timings([{"text": "So I finally tried the viral pasta.", "start_s": 0.3, "end_s": 2.4}],
                                 policy=raw_policy)
    plan = cap.make_plan(words, raw_policy, width=W, height=H, style="clean")
    overlays = ov.overlay_events([{"kind": "hook", "text": "the pasta"}], plan, raw_policy, duration=3)
    ass = ov.make_script(plan, overlays, raw_policy)
    assert re.search(r"^Style: clean,", ass, re.M) and re.search(r"^Style: ost_hook,", ass, re.M)
    assert len(_dialogues(ass, 1)) == len(_dialogues(cap.to_ass(plan), 1)) and _dialogues(ass, 3)
    starts = [(r[1], int(r[0])) for r in _dialogues(ass)]
    assert starts == sorted(starts)
    assert ov.make_script(plan, [], raw_policy).count("Dialogue:") == cap.to_ass(plan).count("Dialogue:")
    with pytest.raises(ValueError):
        ov.make_script(None, [], raw_policy)


def test_overlay_fonts_are_bundled_with_their_licence(raw_policy):
    fonts = cap.fonts_dir(raw_policy)
    for name in ov.styles(raw_policy):
        style = ov.resolve_style(raw_policy, name)[1]
        assert (fonts / style["file"]).exists(), style["file"]
        assert (fonts / f"{style['file'].split('-')[0]}-OFL.txt").exists()
        chars = cap.font_chars(str(fonts / style["file"]))
        assert chars and all(ord(c) in chars for c in "AZaz09:?.,'")


# --- burning --------------------------------------------------------------------------------------------------

def _video(path: Path, w: int, h: int, seconds: float) -> Path:
    from manyworlds import media
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=25:d={seconds}",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", str(seconds), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True, capture_output=True)
    return path


def _bright_pixels(video: Path, t: float) -> int:
    from manyworlds import media
    r = subprocess.run([media.ffmpeg(), "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    return sum(1 for b in r.stdout if b > 200)


@pytest.mark.parametrize("use_ass", [None, False])   # None: libass when this ffmpeg has it; False: Pillow drawing
def test_burn_draws_labels_only_while_they_show(raw_policy, tmp_path, use_ass):
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("PIL")
    src = _video(tmp_path / "in.mp4", 360, 640, 2.5)
    overlays = ov.overlay_events([{"kind": "label", "text": "lazy ootd", "start_s": 0.5, "end_s": 1.6},
                                  {"kind": "question", "text": "What's your skin type?", "start_s": 0.5,
                                   "end_s": 1.6}], (360, 640), raw_policy)
    out = ov.burn(src, tmp_path / "out.mp4", workdir=tmp_path / "work", overlays=overlays, policy=raw_policy,
                  use_ass=use_ass)
    assert out.exists() and out.stat().st_size > 0
    assert _bright_pixels(out, 1.1) > 200                                     # the white bubble and the label
    assert _bright_pixels(out, 0.1) == 0 and _bright_pixels(out, 2.3) == 0
    if use_ass is False:
        assert (tmp_path / "work" / "overlay" / "overlay.ffconcat").exists()


def test_a_label_moves_off_the_burned_captions_band(policy):
    """A label placed where the captions are (review 2026-11-22: lower labels sat on the lower caption band)."""
    items = [{"kind": "label", "text": "Soup day", "position": "lower"}]
    free = ov.overlay_events(items, (1080, 1920), policy, duration=5.0)
    assert free and free[0].position == "lower"
    top, bottom = free[0].extent[1] / 1920, free[0].extent[3] / 1920
    moved = ov.overlay_events(items, (1080, 1920), policy, duration=5.0, avoid_bands=[(top, bottom)])
    assert moved[0].position != "lower"
