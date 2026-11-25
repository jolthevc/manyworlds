"""The commercial operating model (owner 2026-11-25): Manyworlds is a talent, media and consumer business. The thesis,
the CEO, casting, the content mix and the brand bible reason about demand, commercial adjacency, optionality and
production economics, without new agents."""

from __future__ import annotations

from manyworlds.brain import casting
from manyworlds.brain.agents import strategist
from manyworlds.creator import dossier


def test_every_strategy_prompt_carries_the_value_chain_and_economics(raw_policy):
    text = dossier.strategy_text(raw_policy)
    for words in ("consumer brands", "distribution platform", "marginal return", "photo-led",
                  "Repeating a commercially proven archetype", "never in every image"):
        assert words in text


def test_the_ceo_thinks_commercially_and_may_repeat_proven_archetypes():
    assert "Growth and virality come first" not in strategist.SYSTEM
    assert "distinct from the existing portfolio" not in strategist.SYSTEM
    assert "expected return per dollar" in strategist.SYSTEM and "owned products" in strategist.SYSTEM


def test_casting_allows_a_shared_archetype_and_follows_the_policy_restrictions(raw_policy):
    ask = raw_policy.get("strategy", "rubric", "criteria", "distinctiveness", "ask")
    assert "own an audience" in ask and "proven archetype" in ask
    assert "implied-nude" not in casting.JUDGE
    assert "signature associations" in raw_policy.get("strategy", "rubric", "criteria", "commercial", "ask")


def test_the_brand_bible_has_a_commercial_thesis_and_optional_paths_to_owned_products():
    bc = dossier.BrandCommerce.model_validate({
        "categories": ["accessories"], "why_categories": "x", "product_integration": "y", "will_do": [],
        "wont_do": [], "will_wear": [], "wont_wear": [], "will_promote": [], "wont_promote": [],
        "commercial_thesis": "a cool menswear audience with high accessory spend",
        "signature_associations": [{"element": "vintage baseball hats", "category": "accessories",
                                    "how_it_recurs": "street and travel looks; never at the pool or formal events"}],
        "owned_brand_options": [{"category": "accessories", "why": "hats are his association",
                                 "evidence_needed": "affiliate clicks on hats and saves on hat posts"}]})
    assert bc.signature_associations[0].element == "vintage baseball hats"
    old = dossier.BrandCommerce.model_validate({"categories": ["x"], "why_categories": "", "product_integration": "",
                                                "will_do": [], "wont_do": [], "will_wear": [], "wont_wear": [],
                                                "will_promote": [], "wont_promote": []})
    assert old.commercial_thesis == "" and old.owned_brand_options == []      # existing bibles still load


def test_the_content_mix_is_allocated_by_return_per_dollar():
    assert "expected return per dollar" in dossier.ASK["content_system"]
    assert "signature associations" in dossier.ASK["brand_commerce"]
