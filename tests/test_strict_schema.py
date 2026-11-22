"""Strict structured outputs (manyworlds.llm.strict_json_schema): ShotSheet's ordinary schema is too large a
grammar for Claude (HTTP 400, measured 2026-11-22); the strict shape (all required, no nulls, "" for unset enums,
refs inlined, additionalProperties false everywhere) works, and what comes back validates into the models."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel

from manyworlds.direction import DirectedShots, ShotSheet
from manyworlds.llm import LLM, strict_json_schema


class Inner(BaseModel):
    mood: Literal["calm", "loud"] | None = None
    size: float | None = None


class WithDict(BaseModel):
    tags: dict[str, str] = {}
    inner: Inner | None = None
    items: list[Inner] = []


def _walk(node, path="$"):
    """Every (path, dict) in a schema."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def _blank(schema):
    """What a model answering the strict schema writes when it leaves everything unset."""
    if "enum" in schema:
        return ""
    t = schema.get("type")
    if t == "object":
        return {k: _blank(v) for k, v in schema["properties"].items()}
    if t == "array":
        return []
    if t in ("number", "integer"):
        return 0
    if t == "boolean":
        return False
    return ""


def test_every_object_is_closed_and_fully_required():
    for model in (DirectedShots, ShotSheet, WithDict):
        schema = strict_json_schema(model)
        objects = [(p, n) for p, n in _walk(schema) if isinstance(n, dict) and n.get("type") == "object"]
        assert objects and objects[0][0] == "$"                                     # the root too
        for path, node in objects:
            assert node["additionalProperties"] is False, path
            assert node["required"] == list(node["properties"]), path
        text = json.dumps(schema)
        for key in ('"$ref"', '"$defs"', '"anyOf"', '"allOf"', '"default"', '"title"', '"null"'):
            assert key not in text, key


def test_refs_are_inlined_nulls_collapsed_and_enums_take_an_empty_string():
    schema = strict_json_schema(WithDict)
    assert schema["properties"]["tags"] == {"type": "object", "properties": {}, "required": [],
                                            "additionalProperties": False}               # a dict field: closed
    inner = schema["properties"]["inner"]
    assert inner["type"] == "object" and set(inner["properties"]) == {"mood", "size"}
    assert inner["properties"]["mood"]["enum"] == ["calm", "loud", ""]
    assert inner["properties"]["size"]["type"] == "number"
    assert schema["properties"]["items"]["items"]["properties"]["mood"]["enum"] == ["calm", "loud", ""]
    shot = strict_json_schema(DirectedShots)["properties"]["shots"]["items"]
    framing = shot["properties"]["framing"]
    assert framing["properties"]["handheld"]["enum"] == ["none", "light", "held", ""]
    assert framing["properties"]["lens_mm"]["type"] == "number"
    assert "description" in framing and "description" in framing["properties"]["camera_move"]   # kept for the model
    assert shot["properties"]["blocking"]["items"]["properties"]["eyeline"]["enum"][-1] == ""


def test_a_strict_answer_validates_back_with_empty_strings_unset():
    shot = _blank(strict_json_schema(DirectedShots)["properties"]["shots"]["items"])
    shot["set"]["location"] = "her galley kitchen"
    shot["framing"]["lens_mm"] = ""                    # a number left as "": ignored, the default stays
    shot["blocking"] = [{"at_s": "", "action": "pours coffee", "eyeline": "", "hands": ""}]
    shot["performance"]["lines"] = [{"text": "Morning.", "delivery": "", "pause_after_s": ""}]
    out = DirectedShots.model_validate_json(json.dumps({"shots": [shot], "move_shot": -1}))
    s = out.shots[0]
    assert s.set.location == "her galley kitchen" and s.set.background_people == 0
    assert s.framing.handheld is None and s.framing.device is None and s.framing.lens_mm is None
    assert s.performance.energy is None and s.sound.mic is None
    assert s.blocking[0].eyeline is None and s.blocking[0].at_s == 0.0
    assert s.performance.lines[0].pause_after_s is None and s.cutaway is False
    assert ShotSheet.model_validate(s.model_dump(mode="json")) == s                  # stored dicts read back the same


class _StrictClient:
    """Answers messages.create with a thinking block, then the JSON text."""

    def __init__(self, text: str):
        self.calls, outer = [], self
        usage = SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)

        class messages:
            @staticmethod
            def create(**kw):
                outer.calls.append(kw)
                return SimpleNamespace(model=kw["model"], usage=usage, stop_reason="end_turn",
                                       content=[SimpleNamespace(type="thinking", thinking="hm"),
                                                SimpleNamespace(type="text", text=text)])

            @staticmethod
            def parse(**kw):
                raise AssertionError("a strict schema is never sent through messages.parse")
        self.messages = messages


def test_the_claude_path_sends_the_strict_schema_and_validates_the_text(conn, policy):
    answer = {"shots": [{"set": {"location": "the canal path"}, "framing": {"handheld": ""}}], "move_shot": -1}
    client = _StrictClient(json.dumps(answer))
    out = LLM(conn, policy, client=client).structured("director", system="s", content="c", schema=DirectedShots,
                                                      effort="medium")
    assert isinstance(out, DirectedShots) and out.shots[0].set.location == "the canal path"
    assert out.shots[0].framing.handheld is None
    kw = client.calls[0]
    assert kw["model"] == policy.get("llm", "roles", "director", "model")
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": strict_json_schema(DirectedShots)}
    assert kw["output_config"]["effort"] == "medium" and "output_format" not in kw
    charged = conn.execute("SELECT count(*) n, sum(usd) u FROM cost_entries WHERE provider = %s",
                           (kw["model"],)).fetchone()
    assert charged["n"] == 1 and float(charged["u"]) > 0
