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


def test_container_disk_weights_are_not_under_a_volume_mount(policy):
    """LongCat's weights are meant for the container disk (longcat16.weights_on_volume: false); a directory under a
    pool's volume mount fills the shared volume instead (2026-11-24: 47 GB filled the 150 GB volume)."""
    mounts = [p["network_volume"]["path"].rstrip("/") + "/" for p in policy.get("gpu_pools").values()
              if p.get("network_volume")]
    for family, img in policy.get("worker_images").items():
        d = str((img.get("env") or {}).get("MW_LONGCAT_DIR", ""))
        if family == "talk":
            assert d, "the talk family must set MW_LONGCAT_DIR (the image default is under /models)"
        assert not any((d.rstrip("/") + "/").startswith(m) for m in mounts if d), (family, d)


def test_every_job_kind_names_known_pools(policy):
    pools = set(policy.get("gpu_pools"))
    for kind, cfg in policy.get("job_kinds").items():
        assert set(cfg["pools"]) <= pools, kind
