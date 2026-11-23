"""The negative channel of the Wan graphs (worker/comfy.py) and of Z-Image (worker/handlers/image_zimage.py): a
directed job's own negative (payload `negative_prompt`, from manyworlds.direction.compile_prompts) reaches the
negative text encode; an undirected job keeps Wan's standard list; Z-Image at guidance 0 gets none."""

from __future__ import annotations

from manyworlds.worker import comfy
from manyworlds.worker.handlers import image_zimage

SIZE = {"width": 480, "height": 832}


def _graphs(negative=None, **params):
    p = {**SIZE, "steps": 6, **params}
    return {
        "talk": comfy.graph(image="a.png", audio="a.wav", prompt="p", frames=150, seed=1, params=p, blocks_to_swap=0,
                            negative=negative),
        "v2v": comfy.graph_v2v(video="m.mp4", audio="a.wav", prompt="p", frames=150, seed=1,
                               params={**p, "steps": 4, "start_step": 2}, blocks_to_swap=0, negative=negative),
        "i2v": comfy.graph_i2v(image="a.png", prompt="p", seed=1, params=p, blocks_to_swap=0, negative=negative),
        "longcat": comfy.graph_longcat(image="a.png", audio="a.wav", prompt="p", frames=100, seed=1,
                                       params={**p, "steps": 10}, blocks_to_swap=0, negative=negative),
    }


def test_a_jobs_negative_reaches_every_graphs_negative_text_encode():
    for name, g in _graphs("waxy plastic skin, logos").items():
        text = g["text"]["inputs"]
        assert text["negative_prompt"] == "waxy plastic skin, logos", name
        assert text["positive_prompt"] == "p", name


def test_without_one_the_graphs_keep_wans_standard_negative():
    for negative in (None, "", "   "):
        for name, g in _graphs(negative).items():
            assert g["text"]["inputs"]["negative_prompt"] == comfy.NEGATIVE, (name, negative)


def test_params_negative_is_the_fallback_and_the_jobs_own_wins():
    assert comfy.negative_text({"negative_prompt": "from params"}) == "from params"
    assert comfy.negative_text({"negative_prompt": "from params"}, "from the shot") == "from the shot"
    assert comfy.negative_text({"negative_prompt": ""}) == comfy.NEGATIVE
    assert _graphs(negative_prompt="from params")["i2v"]["text"]["inputs"]["negative_prompt"] == "from params"


def test_zimage_reads_a_negative_only_with_guidance_on():
    assert image_zimage.negative_kwargs({"negative_prompt": "logos"}, 0.0) == {}       # Turbo at guidance 0
    assert image_zimage.negative_kwargs({"negative_prompt": "logos"}, 3.5) == {"negative_prompt": "logos"}
    assert image_zimage.negative_kwargs({"negative_prompt": "  "}, 3.5) == {}
    assert image_zimage.negative_kwargs({}, 3.5) == {}
