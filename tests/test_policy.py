import pytest

from manyworlds import policy as policy_mod


def test_version_changes_with_any_value(policy):
    body = dict(policy.body)
    body["budgets"] = {**body["budgets"], "daily_usd": body["budgets"]["daily_usd"] + 1}
    assert policy_mod.from_dict(body).version != policy.version
    assert policy_mod.from_dict(dict(policy.body)).version == policy.version


def test_pool_pricing(policy):
    assert policy.usd_per_gpu_second("salad_batch_4090") == pytest.approx(0.16 / 3600)
    with pytest.raises(KeyError):
        policy.pool("nope")


def test_every_job_kind_names_known_pools(policy):
    pools = set(policy.get("gpu_pools"))
    for kind, cfg in policy.get("job_kinds").items():
        assert set(cfg["pools"]) <= pools, kind
