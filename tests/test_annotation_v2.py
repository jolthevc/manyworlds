"""Annotation iia_v2 (docs/research/2026-11-25-annotation-v2.md): a coded section of controlled vocabularies from
policy (out-of-list values are "unknown", never a failed read), v1 records still loading, distill testing every coded
label with an honest n_tests, and the vision benchmark (agreement math, its spend cap, the Gemini adapter point).
Everything offline: stub LLMs, a local store."""

from __future__ import annotations

import contextlib
import uuid

import pytest

from manyworlds import ledger, policy as policy_mod
from manyworlds.brain.agents import intelligence
from manyworlds.research import annotation as ann, vision_bench as vb
from manyworlds.storage import LocalStore


# --- the coded section ----------------------------------------------------------------------------------------------
def test_the_policy_holds_a_vocabulary_for_every_coded_field(policy):
    voc = ann.vocab(policy)
    for f in ann.ENUM_FIELDS + ann.MULTI_FIELDS:
        assert voc.get(f), f"no intelligence.annotation.vocab.{f}"
    assert voc["opener_has_face"] == ["yes", "no"] and voc["renderable_now"] == ["yes", "partly", "no"]
    assert voc["candid_level"] == ["candid", "semi_candid", "posed", "editorial"]
    assert set(ann.bands(policy)) == set(ann.NUMBER_FIELDS)
    text = ann.vocab_prompt(policy)
    assert "styling_category (one): casual" in text and "slide_roles (ordered list, one per slide) (carousel only)" \
        in text and "never brands" in text


def test_values_outside_the_vocabulary_become_unknown_and_are_reported(policy):
    voc = ann.vocab(policy)
    coded, misses = ann.coerce_coded({
        "styling_category": "Quiet Luxury", "pose": "floating", "garments": ["jeans", "toga", "jeans"],
        "accessories": "sunglasses", "environment": ["moon_base"], "slide_roles": ["face", "selfie", "outfit"],
        "aspiration": 9, "opener_has_face": True, "setting_changes": "2", "made_up_field": "x"}, voc)
    assert coded.styling_category == "quiet_luxury"                       # normalised, in the list
    assert coded.pose == "unknown" and coded.garments == ["jeans"] and coded.accessories == ["sunglasses"]
    assert coded.environment == ["unknown"]                               # nothing listed: unknown, not empty
    assert coded.slide_roles == ["face", "unknown", "outfit"]             # slides keep their positions
    assert coded.aspiration is None and coded.opener_has_face == "yes" and coded.setting_changes == 2
    assert {"pose=floating", "garments=toga", "environment=moon_base", "slide_roles=selfie"} <= set(misses)
    # the same through validation with the vocabulary in the context: never a failed read
    a = ann.Annotation.model_validate({"observed": {"coded": {"pose": "floating", "hair": "ponytail"}}},
                                      context={"vocab": voc})
    assert a.observed.coded.pose == "unknown" and a.observed.coded.hair == "ponytail"


def test_a_strict_shape_read_with_blanks_validates():
    got = intelligence.ReadBatch.model_validate({"reads": [{"post": 1, "annotation": {"observed": {
        "coded": {"styling_category": "", "garments": [], "aspiration": 0, "setting_changes": 0, "sound": ""},
        "visual": {"situation": "dinner"}}}}]})
    c = got.reads[0].annotation.observed.coded
    assert c.styling_category == "unknown" and c.aspiration is None and c.setting_changes == 0 and c.sound == "unknown"


def test_a_v1_record_still_loads_with_every_code_unknown(policy):
    v1 = {"visual": {"situation": "dinner", "pose": "seated, leaning in", "gaze": "lens", "wardrobe": "black dress"},
          "personality": {"relationship": "friend"}, "economics": {"post_type": "photo"}}
    o = ann.load(v1, ann.vocab(policy))
    assert o.visual.situation == "dinner" and o.visual.pose == "seated, leaning in"
    assert o.coded.styling_category == "unknown" and o.coded.garments == [] and o.coded.aspiration is None
    assert ann.Annotation.model_validate({"observed": v1, "confidence": 0.6}).observed.coded.pose == "unknown"
    assert ann.ANNOTATION_SCHEMA == "iia_v2" and "iia_v1" in ann.READ_SCHEMAS


def test_the_read_prompt_carries_the_vocabularies_and_stays_blind_to_performance(policy):
    system = intelligence.read_system(policy)
    assert "garments (list):" in system and "jewelry_level (one): none, minimal, moderate, statement" in system
    assert "not told how a post" in system


# --- distill: every coded label tested, n_tests honest ----------------------------------------------------------------
def _row(i, coded, ratio, medium="photo"):
    return {"post_id": str(uuid.uuid4()), "creator_id": f"c{i % 3}", "media_type": medium, "url": f"u{i}",
            "observed": {"visual": {"situation": "dinner"}, "coded": coded},
            "evidence": {"over_own_median": ratio}, "mechanism": [], "measured": {}}


def test_cluster_tests_every_coded_label_and_counts_each_test(conn, policy):
    acfg = policy.get("intelligence", "annotation")
    high = [_row(i, {"garments": ["jeans", "blazer"], "accessories": ["sunglasses"] + (["hat"] if i == 0 else []),
                     "aspiration": 5, "candid_level": "candid", "sound": "voice"}, 2.0) for i in range(3)]
    low = [_row(i + 3, {"garments": ["jeans"], "aspiration": 2, "candid_level": "posed"}, 0.5) for i in range(3)]
    (c,) = intelligence.cluster(high + low, {"min_posts": 6, "min_creators": 3, "high": 1.5, "low": 0.8}, acfg)
    attrs = {x["attribute"]: (x["high"], x["low"]) for x in c["lift"]}
    assert attrs["garments=blazer"] == (1.0, 0.0) and attrs["accessories=sunglasses"] == (1.0, 0.0)
    assert attrs["aspiration=aspirational"] == (1.0, 0.0) and attrs["candid_level=candid"] == (1.0, 0.0)
    assert attrs["situation=dinner"] == (1.0, 1.0) and attrs["garments=jeans"] == (1.0, 1.0)   # v1 fields still in
    assert not any(a.startswith("sound=") for a in attrs)                 # a video field is not a photo's attribute
    assert "accessories=hat" not in attrs                                 # one post: under cluster_min_posts
    # tested: situation, jeans, blazer, sunglasses, aspirational, relatable, candid, posed; hat skipped before testing
    assert c["n_tests"] == 8 and c["n_skipped_rare"] == 1
    out = intelligence.store_card(conn, {"key": "dinner_blazer", "medium": "photo", "signature_ok": False}, c,
                                  "00000000-0000-0000-0000-000000000000")
    ev = conn.execute("SELECT evidence FROM pattern_cards WHERE id = %s", (out["id"],)).fetchone()["evidence"]
    t = ev["tests"][0]
    assert t["q"] == min(1.0, round(t["p"] * 8, 4))                       # Bonferroni over every attribute tested


def test_a_rare_label_floor_comes_from_policy(policy):
    rows = [_row(i, {"garments": ["jeans"] + (["blazer"] if i == 0 else [])}, 2.0 if i < 3 else 0.5)
            for i in range(6)]
    dcfg = {"min_posts": 6, "min_creators": 3, "high": 1.5, "low": 0.8}
    loose = intelligence.cluster(rows, dcfg, {"cluster_min_share": 0, "cluster_min_posts": 0})[0]
    strict = intelligence.cluster(rows, dcfg, policy.get("intelligence", "annotation"))[0]
    assert loose["n_tests"] == strict["n_tests"] + 1 and strict["n_skipped_rare"] == 1


# --- the vision benchmark ---------------------------------------------------------------------------------------------
def test_agreement_math():
    bands = {"aspiration": {"relatable": [1, 2], "middle": [3, 3], "aspirational": [4, 5]}}
    fa = vb.field_agreement
    assert fa("pose", {"pose": "walking"}, {"pose": "walking"}, bands) == 1.0
    assert fa("pose", {"pose": "walking"}, {"pose": "sitting"}, bands) == 0.0
    assert fa("pose", {}, {"pose": "unknown"}, bands) == 1.0                       # unknown is a value
    assert fa("garments", {"garments": ["jeans", "blazer"]}, {"garments": ["jeans"]}, bands) == 0.5
    assert fa("garments", {"garments": []}, {"garments": ["unknown"]}, bands) == 1.0   # both say nothing
    assert fa("slide_roles", {"slide_roles": ["face", "outfit", "detail"]},
              {"slide_roles": ["face", "detail"]}, bands) == pytest.approx(1 / 3)  # positions, not sets
    assert fa("aspiration", {"aspiration": 5}, {"aspiration": 4}, bands) == 1.0     # same band
    assert fa("aspiration", {"aspiration": 5}, {"aspiration": 2}, bands) == 0.0
    photo = vb.agreement({}, {}, "photo", bands)
    assert "sound" not in photo and "slide_roles" not in photo and "garments" in photo
    assert "slide_roles" in vb.agreement({}, {}, "carousel", bands) and "opening" in vb.agreement({}, {}, "video", bands)


class BenchLLM:
    """Stands in for manyworlds.llm on the bench: canned coded answers per candidate, each call charged to the
    ledger under the active tag (as llm.py does), some calls failing as an invalid answer would."""
    MODELS = {"vision_bench_haiku": "claude-haiku-5-5", "vision_bench_sonnet": "claude-sonnet-5-5",
              "vision_bench_opus": "claude-opus-5-5"}

    def __init__(self, conn, answers, cost=0.01, fail=()):
        self.conn, self.answers, self.cost, self.fail, self._tag, self.calls = conn, answers, cost, set(fail), {}, []

    def model_for(self, role):
        return self.MODELS[role]

    @contextlib.contextmanager
    def tagged(self, agent, job_id, before_call=None, attempt=None):
        before, self._tag = self._tag, {"agent": agent, "job_id": job_id}
        try:
            yield
        finally:
            self._tag = before

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        self.calls.append({"role": role, "system": system, "content": content})
        ledger.record(self.conn, category="llm", provider=self.model_for(role), quantity=1, unit="token",
                      usd=self.cost, agent=self._tag.get("agent"))
        name = role.removeprefix(vb.ROLE_PREFIX)
        if (name, sum(1 for c in self.calls if c["role"] == role)) in self.fail:
            raise RuntimeError(f"{role} returned no valid CodedRead")
        return schema.model_validate({"coded": self.answers[name]})


KEY = {"styling_category": "casual", "garments": ["jeans", "blazer"], "pose": "walking", "aspiration": 4}
ANSWERS = {"opus": KEY, "sonnet": KEY,
           "haiku": {"styling_category": "casual", "garments": ["jeans"], "pose": "sitting", "aspiration": 5,
                     "palette": "rainbow"}}


def _images(tmp_path, n):
    from PIL import Image
    store = LocalStore(tmp_path / "objects")
    uris = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (1600, 2000), (40 * i % 255, 90, 120)).save(p)
        uris.append(store.put_file(p, f"creators/sandbox/img{i}.png").uri)
    return store, uris


def _quiet_estimate(policy, **est):
    vbc = {**policy.body["intelligence"]["vision_bench"],
           "estimate": est or {"image_tokens": 0, "text_tokens": 0, "output_tokens": 0}}
    return policy_mod.from_dict({**policy.body, "intelligence": {**policy.body["intelligence"], "vision_bench": vbc}})


def test_the_bench_scores_each_model_against_the_reference(conn, policy, tmp_path):
    store, uris = _images(tmp_path, 2)
    llm = BenchLLM(conn, ANSWERS, fail={("sonnet", 1)})
    out = vb.run(conn, policy, llm, items=uris, models=["haiku", "sonnet", "opus"], usd_cap=1.0, store=store)
    assert out["reference"] == "opus" and out["models"][0] == "opus" and out["items_read"] == 2
    assert [c["role"] for c in llm.calls[:3]] == ["vision_bench_opus", "vision_bench_haiku", "vision_bench_sonnet"]
    assert "styling_category (one):" in llm.calls[0]["system"]
    assert any(b.get("type") == "image" for b in llm.calls[0]["content"])
    s = out["summary"]
    assert s["opus"]["reference"] and s["opus"]["agreement"]["overall"] == 1.0
    h = s["haiku"]
    # a photo is asked 24 fields: pose disagrees, garments half (Jaccard), the rest agree (rainbow -> unknown on both)
    assert h["agreement"]["fields"]["pose"] == 0.0 and h["agreement"]["fields"]["garments"] == 0.5
    assert h["agreement"]["fields"]["aspiration"] == 1.0 and h["agreement"]["fields"]["palette"] == 1.0
    assert h["agreement"]["overall"] == pytest.approx(22.5 / 24, abs=1e-4)
    assert h["in_vocab_rate"] == 0.75                                      # palette=rainbow of 4 values given
    assert h["usd_per_item"] == pytest.approx(0.01) and h["validity_rate"] == 1.0
    assert h["useful_agreement_per_usd"] == pytest.approx(22.5 / 24 / 0.01, rel=1e-3)
    assert set(h["agreement"]["families"]) >= {"fashion", "grooming", "composition"}
    assert s["sonnet"]["validity_rate"] == 0.5 and s["sonnet"]["agreement"]["overall"] == 1.0
    assert s["sonnet"]["errors"] == ["RuntimeError"]
    n = conn.execute("SELECT count(*) n, sum(usd) u FROM cost_entries WHERE agent = %s", (vb.AGENT,)).fetchone()
    assert n["n"] == 6 and float(n["u"]) == pytest.approx(out["spent_usd"])   # failed calls charged too


def test_the_bench_stops_before_its_cap(conn, policy, tmp_path):
    store, uris = _images(tmp_path, 5)
    llm = BenchLLM(conn, ANSWERS, cost=0.01)
    out = vb.run(conn, _quiet_estimate(policy), llm, items=uris, models=["haiku", "sonnet", "opus"], usd_cap=0.1,
                 store=store)
    # 0.03 an item once measured: three items fit (0.09), a fourth would cross 0.10
    assert out["items_read"] == 3 and out["stopped"] == "usd_cap" and out["spent_usd"] <= 0.1
    assert len(llm.calls) == 9                                             # every item read by every model or none
    # before anything is measured, the policy estimate decides: a cap below one item's estimate calls nothing
    est = vb.estimate_call(policy, "claude-opus-5-5", 1)
    assert est > 0
    none = vb.run(conn, policy, BenchLLM(conn, ANSWERS), items=uris, models=["opus"], usd_cap=est / 2, store=store)
    assert none["items_read"] == 0 and none["stopped"] == "usd_cap"


def test_the_bench_refuses_research_media(conn, policy, tmp_path):
    store, _ = _images(tmp_path, 1)
    llm = BenchLLM(conn, ANSWERS)
    out = vb.run(conn, policy, llm, items=["s3://manyworlds/research-refs/posts/1/f01.jpg"], models=["opus"],
                 usd_cap=1.0, store=store)
    assert out["items_read"] == 0 and "research media" in out["items_skipped"][0]["why"] and not llm.calls


def test_the_gemini_adapter_point_refuses(conn, policy, tmp_path, monkeypatch):
    monkeypatch.delenv(vb.GEMINI_KEY_ENV, raising=False)
    with pytest.raises(vb.AdapterUnavailable, match="GEMINI_API_KEY is not set.*llm.hosts.gemini is missing"):
        vb.GeminiModel().check(policy)
    monkeypatch.setenv(vb.GEMINI_KEY_ENV, "test-key")
    hosted = policy_mod.from_dict({**policy.body, "llm": {**policy.body["llm"], "hosts": {
        **policy.body["llm"]["hosts"], "gemini": {"base_url": "https://example.invalid", "key_env": vb.GEMINI_KEY_ENV}}}})
    with pytest.raises(vb.AdapterUnavailable, match="not implemented"):
        vb.GeminiModel().check(hosted)
    store, uris = _images(tmp_path, 1)
    out = vb.run(conn, hosted, BenchLLM(conn, ANSWERS), items=uris, models=["gemini", "haiku"], usd_cap=1.0,
                 store=store)
    assert "gemini" in out["refused"] and out["models"] == ["haiku"] and out["items_read"] == 1


def test_the_dry_run_estimate_and_the_accepted_assets(conn, policy):
    est = vb.estimate(policy, items=10, models=["haiku", "sonnet", "opus"])
    per = {k: v["usd_per_item"] for k, v in est["models"].items()}
    assert per["haiku"] < per["sonnet"] < per["opus"]
    assert est["total_usd"] == pytest.approx(10 * sum(per.values()), abs=1e-3)
    cid = conn.execute("INSERT INTO creators (handle, status) VALUES ('sandy', 'active') RETURNING id").fetchone()["id"]

    def asset(uri, kind="image", passed=(True,)):
        aid = conn.execute("INSERT INTO assets (creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, %s, 'h', 1) "
                           "RETURNING id", (cid, kind, uri)).fetchone()["id"]
        for p in passed:
            conn.execute("INSERT INTO reviews (asset_id, gate, passed) VALUES (%s, 'g', %s)", (aid, p))
        return str(aid)
    good = asset("s3://manyworlds/creators/sandy/a.png", passed=(True, True))
    asset("s3://manyworlds/creators/sandy/b.png", passed=(True, False))
    asset("s3://manyworlds/creators/sandy/c.mp4", kind="video")
    asset("s3://manyworlds/creators/sandy/d.png", passed=())
    items = vb.accepted_image_assets(conn, limit=10)
    assert [i["id"] for i in items] == [good] and items[0]["uris"] == ["s3://manyworlds/creators/sandy/a.png"]
    assert vb.accepted_image_assets(conn, limit=10, creator="someone_else") == []
