"""A shot with no face of hers in it (her point of view, or a cutaway): its prompt describes her hands, not her face
or eyes, and no gate judges it on identity (review 2026-11-22: POV B-roll either drew her face or failed identity)."""

from __future__ import annotations

from manyworlds import direction
from manyworlds.review import gates, video

from test_review import BIBLE

POV = {"framing": {"device": "pov", "shot_size": "close-up"},
       "blocking": [{"at_s": 0, "action": "pours tea into a mug", "eyeline": "prop"}]}
FACE = {"framing": {"device": "propped", "shot_size": "medium shot"},
        "blocking": [{"at_s": 0, "action": "pours tea into a mug", "eyeline": "lens"}]}


def test_faceless_shots_are_pov_and_cutaways():
    assert direction.faceless(POV) and direction.faceless({**FACE, "cutaway": True})
    assert not direction.faceless(FACE) and not direction.faceless(None)


def test_a_pov_prompt_names_her_hands_not_her_face(policy):
    for target in ("photo", "i2v"):
        prompt = direction.compile_prompt(POV, BIBLE, target, policy)
        assert "hands and forearms" in prompt and "olive skin" in prompt
        assert "oval face" not in prompt and "brown eyes" not in prompt
    face = direction.compile_prompt(FACE, BIBLE, "photo", policy)
    assert "oval face" in face


def test_no_identity_gate_for_a_faceless_shot():
    assert "identity" not in gates.gates_for("a kitchen", POV)
    assert "identity" in gates.gates_for("a kitchen", FACE)
    assert "identity" not in video.gates_for("a kitchen", POV)
    assert "identity" in video.gates_for("a kitchen", FACE)
