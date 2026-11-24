"""Burned-in captions (manyworlds.captions): word timings, chunking, safe-zone layout, styles, ASS output, burn."""

from __future__ import annotations

import re
import struct
import subprocess
from pathlib import Path

import pytest

from manyworlds import captions as cap
from manyworlds import policy as policy_mod

SCRIPT = ("So I finally tried the viral pasta everyone keeps posting about. Honestly, it was fine, but the sauce was "
          "way too sweet for me. I went back to the bodega on Fifth and bought the good parmesan, and that fixed "
          "almost everything.")
TIME = re.compile(r"^\d+:\d{2}:\d{2}\.\d{2}$")


def _lines(text: str, wps: float = 2.6, gap: float = 0.4, start: float = 0.3) -> list[dict]:
    out, t = [], start
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        d = len(s.split()) / wps
        out.append({"text": s, "start_s": round(t, 3), "end_s": round(t + d, 3)})
        t += d + gap
    return out


def _dialogues(ass: str, layer: int | None = None) -> list[list[str]]:
    rows = [line[len("Dialogue: "):].split(",", 9) for line in ass.splitlines() if line.startswith("Dialogue: ")]
    return [r for r in rows if layer is None or int(r[0]) == layer]


# --- word timings ---------------------------------------------------------------------------------------------

def _check_monotonic(words):
    for a, b in zip(words, words[1:]):
        assert a.start <= a.end <= b.start <= b.end


def test_parse_emphasis():
    assert cap.parse_emphasis("this is *so* good") == [("this", False), ("is", False), ("so", True), ("good", False)]
    assert cap.parse_emphasis("*this whole thing*, honestly") == [("this", True), ("whole", True),
                                                                  ("thing,", True), ("honestly", False)]
    assert cap.parse_emphasis("wait — what") == [("wait —", False), ("what", False)]


def test_estimate_timings_monotonic_and_cover_each_line(raw_policy):
    lines = [{"text": "So I finally tried the *viral* pasta.", "start_s": 0.0, "end_s": 2.1},
             {"text": "Honestly? It was fine.", "start_s": 2.5, "end_s": 3.9}]
    words = cap.estimate_timings(lines, policy=raw_policy, emphasis=["fine"])
    assert [w.text for w in words] == ["So", "I", "finally", "tried", "the", "viral", "pasta.", "Honestly?", "It",
                                       "was", "fine."]
    _check_monotonic(words)
    assert words[0].start == 0.0 and words[6].end == 2.1 and words[7].start == 2.5 and words[-1].end == 3.9
    assert all(0.0 <= w.start and w.end <= 2.1 for w in words[:7])
    assert {w.text for w in words if w.emphasis} == {"viral", "fine."}
    # A longer word gets more of its line than a short one.
    assert (words[2].end - words[2].start) > (words[1].end - words[1].start)
    shifted = cap.estimate_timings(lines, policy=raw_policy, offset=0.4)
    assert shifted[0].start == 0.4 and shifted[-1].end == pytest.approx(4.3)


def test_estimate_timings_reads_voice_meta_lines_out_of_order(raw_policy):
    lines = [{"text": "second line here", "start_s": 2.0, "end_s": 3.0, "index": 1},
             {"text": "first line", "start_s": 0.2, "end_s": 1.5, "index": 0}, {"text": "", "start_s": 4, "end_s": 5},
             {"text": "no timing"}]
    words = cap.estimate_timings(lines, policy=raw_policy)
    assert [w.text for w in words] == ["first", "line", "second", "line", "here"]
    _check_monotonic(words)


def test_word_timings_from_transcript_cleans_and_takes_script_spelling():
    transcript = [{"word": " so", "start": 0.10, "end": 0.32}, {"word": "i", "start": 0.30, "end": 0.41},
                  {"word": "tried", "start": 0.45, "end": 0.80}, {"word": ",", "start": 0.80, "end": 0.81},
                  {"word": "the", "start": 0.95, "end": 1.0}, {"word": "viral", "start": 1.0, "end": 1.4},
                  {"word": "pasta", "start": 1.38, "end": 1.9}, {"text": "today", "start_s": 2.0, "end_s": 2.3}]
    words = cap.word_timings(list(reversed(transcript)), script="So I tried, the *viral* pasta today.", offset=1.0)
    assert [w.text for w in words] == ["So", "I", "tried,", "the", "viral", "pasta", "today."]
    _check_monotonic(words)
    assert words[0].start == pytest.approx(1.10) and words[-1].end == pytest.approx(3.3)
    assert [w.text for w in words if w.emphasis] == ["viral"]


# --- chunks ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["clean", "karaoke", "boxed", "editorial", "bold_pop", "minimal"])
def test_chunks_respect_limits_and_never_orphan_a_word(raw_policy, name):
    words = cap.estimate_timings(_lines(SCRIPT), policy=raw_policy)
    _, style = cap.resolve_style(raw_policy, name)
    chunks = cap.chunk_words(words, policy=raw_policy, style=style)
    limits = raw_policy.get("burned_captions", "chunking")
    max_words = min(int(style["max_words"]), int(limits["max_words"]))
    assert [w for c in chunks for w in c.words] == words          # every word once, in order
    for c in chunks:
        assert 2 <= len(c.words) <= max_words, c.text               # no orphan: the script has no one-word sentence
        assert 1 <= len(c.lines) <= 2
        for line in c.lines:
            text = " ".join(w.text for w in line)
            assert len(line) == 1 or len(text) <= style["max_chars_per_line"], text
            assert len(c.lines) == 1 or len(line) >= 2 or len(c.words) < 4   # 4+ words: two a line at least
        assert c.start <= c.words[0].start and c.end >= c.words[-1].end
        assert c.words[-1].end - c.words[0].start <= float(limits["max_duration_s"]) * 1.5
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start


def test_chunks_break_at_punctuation_not_after_articles(raw_policy):
    words = cap.estimate_timings(_lines(SCRIPT), policy=raw_policy)
    _, style = cap.resolve_style(raw_policy, "clean")
    chunks = cap.chunk_words(words, policy=raw_policy, style=style)
    ends = [c.words[-1].text for c in chunks]
    assert "about." in ends and "me." in ends
    assert not {cap.key(e) for e in ends} & {"the", "a", "an", "to", "of", "my", "i"}


def test_a_word_alone_between_pauses_may_stand_alone(raw_policy):
    lines = [{"text": "Okay.", "start_s": 0.0, "end_s": 0.4}, {"text": "So this is the plan.", "start_s": 1.4,
                                                                  "end_s": 2.8}]
    words = cap.estimate_timings(lines, policy=raw_policy)
    chunks = cap.chunk_words(words, policy=raw_policy, style=cap.resolve_style(raw_policy, "clean")[1])
    assert chunks[0].text == "Okay." and all(len(c.words) >= 2 for c in chunks[1:])


# --- layout ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(1080, 1920), (720, 1280)])
@pytest.mark.parametrize("placement", ["lower", "mid_lower", "upper", "center"])
def test_layout_stays_inside_the_safe_zone(raw_policy, size, placement):
    w, h = size
    x0, y0, x1, y1 = cap.safe_area(w, h, raw_policy)
    assert (x0, y0, x1, y1) == pytest.approx((0.12 * w, 0.14 * h, 0.88 * w, 0.75 * h))
    words = cap.estimate_timings(_lines(SCRIPT + " Supercalifragilisticexpialidocious, honestly."), policy=raw_policy)
    for name in cap.styles(raw_policy):
        plan = cap.make_plan(words, raw_policy, width=w, height=h, style=name, placement=placement)
        assert plan.placement == placement
        for cl in cap.layout(plan):
            for line in cl.lines:
                assert x0 - 0.5 <= line.x0 and line.x1 <= x1 + 0.5, (name, line)
                assert y0 - 0.5 <= line.y0 and line.y1 <= y1 + 0.5, (name, line)
                assert (line.x0 + line.x1) / 2 == pytest.approx(w / 2)
            bottom, top = max(l.y1 for l in cl.lines), min(l.y0 for l in cl.lines)
            if placement == "lower":
                assert bottom > 0.62 * h                              # lower third, above the bottom UI
            elif placement == "upper":
                assert top < 0.4 * h
            elif placement == "mid_lower":
                assert top < 0.62 * h < bottom                        # over the body, under the face
            else:
                assert top < 0.5 * h < bottom


def test_layout_scales_with_the_frame(raw_policy):
    words = cap.estimate_timings(_lines(SCRIPT), policy=raw_policy)
    big = cap.layout(cap.make_plan(words, raw_policy, width=1080, height=1920))
    small = cap.layout(cap.make_plan(words, raw_policy, width=720, height=1280))
    assert len(big) == len(small)
    for a, b in zip(big, small):
        assert a.lines[-1].baseline / 1920 == pytest.approx(b.lines[-1].baseline / 1280, abs=0.01)


def test_a_too_long_word_is_scaled_down_not_let_out(raw_policy):
    words = cap.estimate_timings([{"text": "Supercalifragilisticexpialidocious", "start_s": 0, "end_s": 1}],
                                 policy=raw_policy)
    plan = cap.make_plan(words, raw_policy, width=1080, height=1920, style="bold_pop")
    (cl,) = cap.layout(plan)
    assert cl.scale < 1.0
    assert "\\fscx" in cap.to_ass(plan)


def test_placement_for_a_low_face(raw_policy):
    assert cap.placement_for(None, raw_policy) == "lower"
    assert cap.placement_for((0.3, 0.2, 0.7, 0.5), raw_policy) == "lower"   # face high: captions below it
    assert cap.placement_for((0.3, 0.5, 0.7, 0.85), raw_policy) == "upper"  # face low: captions above it


def test_face_safe_band_choice(raw_policy):
    assert cap.placement_for((0.3, 0.3, 0.7, 0.58), raw_policy, preferred="mid_lower") == "lower"   # under the chin
    assert cap.placement_for((0.3, 0.45, 0.7, 0.7), raw_policy, preferred="mid_lower") == "upper"
    assert cap.placement_for((0.0, 0.0, 1.0, 1.0), raw_policy, preferred="mid_lower") == "mid_lower"  # none clears
    assert cap.placement_for(None, raw_policy, preferred="mid_lower") == "mid_lower"
    assert cap.placement_for(None, raw_policy, preferred="nope") == raw_policy.get("burned_captions",
                                                                                   "default_placement")
    assert cap.covers_face(0.6, 0.7, (0.3, 0.2, 0.7, 0.59), 0.02) and not cap.covers_face(0.6, 0.7, (0.3, 0.2, 0.7, 0.5))
    assert cap.face_box_union([None, (0.3, 0.2, 0.6, 0.5), (0.35, 0.25, 0.7, 0.55)]) == (0.3, 0.2, 0.7, 0.55)
    assert cap.face_box_union([None]) is None


def test_placement_follows_the_shot_size_then_the_face(raw_policy):
    def shot(size):
        return {"framing": {"shot_size": size}}
    assert cap.placement_for_shot(raw_policy, shot("close-up")) == "lower"
    assert cap.placement_for_shot(raw_policy, shot("medium close-up")) == "lower"
    assert cap.placement_for_shot(raw_policy, shot("Full body")) == "mid_lower"
    assert cap.placement_for_shot(raw_policy, shot("wide")) == "mid_lower"
    assert cap.placement_for_shot(raw_policy, None) == "lower"
    # Full body, but she sits: the face is mid-frame, so the captions go above it.
    assert cap.placement_for_shot(raw_policy, shot("full body"), (0.4, 0.5, 0.6, 0.62)) == "upper"


def test_caption_style_resolution_order(raw_policy):
    def shot(text):
        return {"edit": {"caption_style": text}}
    f = cap.caption_style_for
    assert f(raw_policy, creator="editorial", fmt="talking_video", shot=shot("karaoke")) == "karaoke"   # shot first
    assert f(raw_policy, creator="editorial", fmt="talking_video", shot=shot("bold pop, big and loud")) == "bold_pop"
    # Free text naming no style falls through: to the format's, then the creator's, then the default.
    assert f(raw_policy, creator="editorial", fmt="talking_video", shot=shot("small lowercase, bottom third")) == "editorial"
    assert f(raw_policy, creator="editorial", fmt="talking_video") == "editorial"
    assert f(raw_policy, fmt="talking_video") == raw_policy.get("burned_captions", "default_style")
    # A format can carry labels, not captions (story_clip: none); a shot can still ask for them.
    assert f(raw_policy, creator="karaoke", fmt="story_clip") is None
    assert f(raw_policy, fmt="story_clip", shot=shot("minimal")) == "minimal"
    assert f(raw_policy, fmt="talking_video", shot=shot("no captions, labels only")) is None
    assert f(raw_policy, fmt=["story_clip", "talking_video"]) is None                       # the first key listed wins
    creator_first = policy_mod.from_dict({**raw_policy.body, "burned_captions": {
        **raw_policy.body["burned_captions"], "style_precedence": ["creator", "shot", "format"]}})
    assert f(creator_first, creator="minimal", fmt="talking_video", shot=shot("karaoke")) == "minimal"


def test_word_timestamps_per_format(raw_policy):
    assert cap.word_timestamps_for(raw_policy, "talking_video")
    assert cap.word_timestamps_for(raw_policy, "narrated_edit")
    assert not cap.word_timestamps_for(raw_policy, "story_clip")

    def bc(**kw):
        return policy_mod.from_dict({**raw_policy.body, "burned_captions": {**raw_policy.body["burned_captions"], **kw}})
    assert cap.word_timestamps_for(bc(word_timestamps=True, word_timestamps_formats=[]), "narrated_edit")
    assert not cap.word_timestamps_for(bc(word_timestamps_formats=[]), "talking_video")
    assert cap.word_timestamps_for(bc(word_timestamps=["narrated_edit"]), "narrated_edit")


def test_block_timings_share_a_block_by_syllables_with_room_at_pauses(raw_policy):
    words = cap.block_timings("I went to the supermarket. Then home.", 0.0, 3.0, policy=raw_policy)
    assert [w.text for w in words] == ["I", "went", "to", "the", "supermarket.", "Then", "home."]
    assert words[0].start == 0.0 and words[-1].end == 3.0
    span = {w.text: w.end - w.start for w in words}
    assert span["supermarket."] > 3 * span["to"]                              # not shared evenly
    assert words[5].start - words[4].end > 0.1                                 # the full stop leaves room
    _check_monotonic(words)


def test_emoji_and_undrawable_characters_are_dropped(raw_policy):
    poppins = str(cap.fonts_dir(raw_policy) / "Poppins-Bold.ttf")
    assert cap.drawable("so good 🔥🔥", poppins) == "so good"
    assert cap.drawable("Tight on baggy ✅", poppins) == "Tight on baggy"
    assert cap.drawable("so good 🔥", poppins, "keep") == "so good 🔥"
    assert cap.drawable("so good 🔥", None) == "so good"                       # no character map: by pattern
    assert cap.drawable("plain words, fine.", poppins) == "plain words, fine."
    words = cap.estimate_timings([{"text": "this is so good 🔥 honestly", "start_s": 0, "end_s": 2}], policy=raw_policy)
    plan = cap.make_plan(words, raw_policy, width=1080, height=1920, style="clean")
    assert [w.text for c in plan.chunks for w in c.words] == ["this", "is", "so", "good", "honestly"]


# --- styles and ASS -------------------------------------------------------------------------------------------

def test_style_by_name_with_fallback(raw_policy):
    assert {"clean", "karaoke", "boxed", "editorial", "bold_pop", "minimal"} <= set(cap.styles(raw_policy))
    assert cap.resolve_style(raw_policy, "nope")[0] == raw_policy.get("burned_captions", "default_style")
    name, style = cap.resolve_style(raw_policy, "karaoke")
    assert name == "karaoke" and style["mode"] == "karaoke" and style["box"]["enabled"] is False  # defaults merged


def _ttf_names(path: Path) -> set[str]:
    data = path.read_bytes()
    tables = {}
    for i in range(struct.unpack_from(">H", data, 4)[0]):
        tag, _, off, _ = struct.unpack_from(">4sIII", data, 12 + 16 * i)
        tables[tag] = off
    base = tables[b"name"]
    count, strings = struct.unpack_from(">HH", data, base + 2)
    names = set()
    for i in range(count):
        pid, _, _, nid, length, off = struct.unpack_from(">6H", data, base + 6 + 12 * i)
        if pid == 3 and nid in (1, 4, 16):
            names.add(data[base + strings + off:base + strings + off + length].decode("utf-16-be"))
    return names


def test_every_style_font_is_bundled_with_its_licence(raw_policy):
    fonts = cap.fonts_dir(raw_policy)
    assert fonts is not None and (fonts / "README.md").exists()
    for name in cap.styles(raw_policy):
        style = cap.resolve_style(raw_policy, name)[1]
        ttf = fonts / style["file"]
        assert ttf.exists(), style["file"]
        assert style["font"] in _ttf_names(ttf), (name, style["font"], _ttf_names(ttf))   # libass finds it by name
        family = style["file"].split("-")[0]
        assert "Open Font License" in (fonts / f"{family}-OFL.txt").read_text(encoding="utf-8")
        m = cap.font_metrics(str(ttf))
        assert m.upem > 0 and m.win_ascent > 0 and 0 < m.x_height < m.cap_height < m.upem
        chars = cap.font_chars(str(ttf))                                    # its character map reads
        assert chars and all(ord(c) in chars for c in "AZaz09.,'?!") and 0x1F525 not in chars


def test_ass_time_format():
    assert cap.ass_time(0) == "0:00:00.00"
    assert cap.ass_time(3723.456) == "1:02:03.46"
    assert cap.ass_time(59.999) == "0:01:00.00"


def test_ass_has_style_and_one_event_per_chunk_line(raw_policy):
    words = cap.estimate_timings(_lines("So I finally tried the *viral* pasta. Honestly, it was fine."),
                                 policy=raw_policy)
    plan = cap.make_plan(words, raw_policy, width=1080, height=1920, style="clean")
    ass = cap.to_ass(plan)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass and "[V4+ Styles]" in ass and "[Events]" in ass
    assert re.search(r"^Style: clean,Poppins,[\d.]+,&H00FFFFFF,", ass, re.M)
    text = _dialogues(ass, layer=1)
    assert len(text) == sum(len(c.lines) for c in plan.chunks)
    for row in text:
        assert TIME.match(row[1]) and TIME.match(row[2]) and row[1] < row[2] and row[3] == "clean"
        assert "\\an2\\pos(" in row[9] and "\\fad(" in row[9]
    assert text[0][1] == cap.ass_time(plan.chunks[0].start)
    assert text[-1][2] == cap.ass_time(plan.chunks[-1].end)
    accent = cap.ass_colour(plan.style["accent"])
    viral = [r[9] for r in text if "viral" in r[9]]
    white = cap.ass_colour(plan.style["color"])
    assert viral and all(f"{{\\1c{accent}}}viral{{\\1c{white}}}" in t for t in viral)   # emphasis takes the accent
    assert sum(r[9].count(accent) for r in text) == len(viral)                       # and nothing else does


def test_karaoke_has_one_event_per_word_with_the_word_highlighted(raw_policy):
    words = cap.estimate_timings(_lines("I went back to the bodega on Fifth and bought the good parmesan."),
                                 policy=raw_policy)
    plan = cap.make_plan(words, raw_policy, width=1080, height=1920, style="karaoke")
    text = _dialogues(cap.to_ass(plan), layer=1)
    assert len(text) == sum(len(c.lines) * len(c.words) for c in plan.chunks)
    hl = cap.ass_colour(plan.style["highlight"])
    assert sum(f"\\1c{hl}" in r[9] for r in text) == len(words)                  # each word lit once, as said
    assert all("\\t(0," in r[9] for r in text if f"\\1c{hl}" in r[9])              # with its pop
    starts = [r[1] for r in text]
    assert starts == sorted(starts)


def test_reveal_boxed_bold_pop_and_small_caps(raw_policy):
    words = cap.estimate_timings(_lines("Honestly, it was fine, but the sauce was way too sweet."), policy=raw_policy)
    reveal = cap.to_ass(cap.make_plan(words, raw_policy, width=1080, height=1920, style="minimal"))
    assert "\\1a&HFF&" in reveal and "honestly," in reveal and "Honestly" not in reveal
    boxed_plan = cap.make_plan(words, raw_policy, width=1080, height=1920, style="boxed")
    boxed = cap.to_ass(boxed_plan)
    pills = _dialogues(boxed, layer=0)
    assert len(pills) == sum(len(c.lines) for c in boxed_plan.chunks) and all("\\p1}m " in r[9] for r in pills)
    pop = cap.to_ass(cap.make_plan(words, raw_policy, width=1080, height=1920, style="bold_pop"))
    assert "HONESTLY," in pop and "\\t(0," in pop and re.search(r"^Style: bold_pop,Anton,", pop, re.M)
    editorial = cap.to_ass(cap.make_plan(words, raw_policy, width=720, height=1280, style="editorial"))
    assert "\\fs" in editorial and "ONESTLY" in editorial and "PlayResY: 1280" in editorial
    assert re.search(r"^Style: editorial,DM Serif Display,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,0,-1,", editorial, re.M)


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


@pytest.mark.parametrize("use_ass", [None, False])   # None: libass when this ffmpeg has it; False: Pillow overlay
def test_burn_draws_captions_only_while_spoken(raw_policy, tmp_path, use_ass):
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("PIL")
    src = _video(tmp_path / "in.mp4", 360, 640, 2.5)
    words = cap.estimate_timings([{"text": "Hello there, my friend.", "start_s": 0.5, "end_s": 1.6}],
                                 policy=raw_policy)
    plan = cap.make_plan(words, raw_policy, width=360, height=640, style="clean")
    out = cap.burn(src, plan, tmp_path / "out.mp4", workdir=tmp_path / "work", policy=raw_policy, use_ass=use_ass)
    assert out.exists() and out.stat().st_size > 0
    assert _bright_pixels(out, 1.2) > 30
    assert _bright_pixels(out, 0.1) == 0 and _bright_pixels(out, 2.3) == 0
    if use_ass is False:
        assert (tmp_path / "work" / "overlay" / "overlay.ffconcat").exists()
