"""Presigned worker storage (security.worker_storage: presigned, manyworlds.deploy.presign): rented machines hold no
storage key; every job carries presigned links for what it reads and writes."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest

from manyworlds import policy as policy_mod, queue
from manyworlds.deploy import autoscale, credentials, presign
from manyworlds.storage import LocalStore, MissingLink, ObjectStore, S3Store, UrlStore
from manyworlds.worker import models
from manyworlds.worker.runner import HandlerResult, OutputFile, Worker

PNG = b"\x89PNG\r\n\x1a\n" + b"stand-in image bytes " * 50


def presigned(policy):
    return policy_mod.from_dict({**policy.body, "security": {**policy.body["security"], "worker_storage": "presigned"}})


class SigningStore(LocalStore):
    """A local store that 'signs' links (no S3 needed): each link names its key, method and a serial number."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.serial = 0

    def uri_for(self, key: str) -> str:
        return f"s3://bucket/{key}"

    def presign(self, key: str, method: str, seconds: int) -> str:
        self.serial += 1
        return f"https://signed.example/{method}/{key}?n={self.serial}&ttl={seconds}"

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def read_bytes(self, key: str) -> bytes:
        return (self.root / key).read_bytes()


class Keyless(ObjectStore):
    """The store of a machine with no storage key: any direct use is a bug."""

    def __getattribute__(self, name):
        if name in ("put_file", "get_file", "exists", "size", "uri_for", "public_url", "delete"):
            raise AssertionError(f"the keyless store was used: {name}")
        return object.__getattribute__(self, name)


def _payload(conn, job_id):
    return conn.execute("SELECT payload FROM jobs WHERE id = %s", (job_id,)).fetchone()["payload"]


# --- mode off -------------------------------------------------------------------------------------------------------

def test_mode_off_changes_nothing(conn, policy, tmp_path, monkeypatch):
    monkeypatch.setenv("WORKER_AWS_ACCESS_KEY_ID", "scoped-key")
    monkeypatch.setenv("WORKER_AWS_SECRET_ACCESS_KEY", "scoped-secret")
    monkeypatch.setenv("WORKER_DATABASE_URL", "postgresql://w:w@localhost/mw")
    pol = policy_mod.from_dict({**policy.body, "security": {**policy.body["security"], "worker_database": "required",
                                                                 "worker_storage": "required"}})
    env = credentials.worker_env(pol)
    assert env["AWS_ACCESS_KEY_ID"] == "scoped-key" and presign.MODE_ENV not in env
    job = queue.enqueue(conn, "fake.image", {"image_uri": "s3://bucket/refs/a.png"})
    assert presign.refresh(conn, pol, SigningStore(tmp_path)) == 0
    assert autoscale.refresh_links(conn, pol, SigningStore(tmp_path)) == []
    assert presign.FIELD not in _payload(conn, job)
    assert queue.claim(conn, ["fake.image"], "w1", 60).id == job   # a keyed worker claims as before


# --- worker env -----------------------------------------------------------------------------------------------------

def test_presigned_worker_env_has_no_storage_key(policy, monkeypatch):
    for k, v in {"AWS_ACCESS_KEY_ID": "cp-key", "AWS_SECRET_ACCESS_KEY": "cp-secret",
                 "WORKER_DATABASE_URL": "postgresql://w:w@db/mw", "STORAGE_URL": "s3://bucket"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("WORKER_AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("WORKER_AWS_SECRET_ACCESS_KEY", raising=False)
    pol = presigned(policy)
    assert credentials.missing(pol) == []        # presigned satisfies storage scoping with no credential
    env = credentials.worker_env(pol)
    assert not [k for k in env if "AWS" in k and "REGION" not in k]
    assert "cp-secret" not in env.values() and "cp-key" not in env.values()
    assert env[presign.MODE_ENV] == presign.MODE and env["STORAGE_URL"] == "s3://bucket"
    assert presign.needs_links({presign.MODE_ENV: presign.MODE}) and not presign.needs_links({})


# --- signing --------------------------------------------------------------------------------------------------------

def test_links_cover_inputs_outputs_and_mirrors(conn, policy, tmp_path):
    pol, store = presigned(policy), SigningStore(tmp_path)
    repo = pol.get("models", "mirror")[0]["repo"]
    manifest = tmp_path / "m.json"
    manifest.write_text('{"repo": "x", "files": [{"path": "model.safetensors", "bytes": 3}]}')
    store.put_file(manifest, models.manifest_key(repo))
    job = queue.enqueue(conn, "image.zimage", {"lora_uri": "s3://bucket/loras/c.safetensors",
                                               "refs": [{"uri": "s3://bucket/refs/a.png"}], "other": "s3://elsewhere/x"})
    assert presign.refresh(conn, pol, store) == 1
    links = _payload(conn, job)[presign.FIELD]
    assert set(links["get"]) == {"loras/c.safetensors", "refs/a.png", models.manifest_key(repo),
                                 f"models/{repo}/files/model.safetensors"}
    assert f"assets/unassigned/{job}/00.png" in links["put"] and f"assets/unassigned/{job}/03.mp4" in links["put"]
    assert links["claim_by"] < links["expires"] and links["prefix"] == "s3://bucket/"
    # a family with no mirror: nothing signed for models; a keyed kind: no links at all
    voice = queue.enqueue(conn, "voice.speak.qwen3", {"text": "hi"})
    fin = queue.enqueue(conn, "finish.video", {"video_uri": "s3://bucket/v.mp4"})
    presign.refresh(conn, pol, store)
    assert not [k for k in _payload(conn, voice)[presign.FIELD]["get"] if k.startswith("models/")]
    assert presign.FIELD not in _payload(conn, fin)


def test_unmirrored_repo_is_absent_and_train_log_is_writable(conn, policy, tmp_path):
    pol, store = presigned(policy), SigningStore(tmp_path)
    img = queue.enqueue(conn, "image.zimage", {})
    lora = queue.enqueue(conn, "train.lora", {})
    presign.refresh(conn, pol, store)
    links = _payload(conn, img)[presign.FIELD]
    repo = pol.get("models", "mirror")[0]["repo"]
    assert links["absent"] == [models.manifest_key(repo)]
    assert UrlStore(links).exists(models.manifest_key(repo)) is False   # falls back to Hugging Face
    assert f"logs/{lora}/train.log" in _payload(conn, lora)[presign.FIELD]["put"]


def test_retried_and_reclaimed_jobs_get_fresh_links(conn, policy, tmp_path):
    pol, store = presigned(policy), SigningStore(tmp_path)
    ahead = float(pol.get("security", "presigned", "refresh_ahead_seconds"))
    job = queue.enqueue(conn, "fake.image", {"image_uri": "s3://bucket/refs/a.png"}, max_attempts=3)
    # unsigned: a keyless worker never takes it
    assert queue.claim(conn, ["fake.image"], "w1", 60, need_links=True) is None
    # signed long ago: links that would die mid-attempt are not claimable
    presign.refresh(conn, pol, store, now=time.time() - 86400 * 2)
    old = _payload(conn, job)[presign.FIELD]
    assert old["claim_by"] < time.time()
    assert queue.claim(conn, ["fake.image"], "w1", 60, need_links=True) is None
    assert presign.refresh(conn, pol, store) == 1
    fresh = _payload(conn, job)[presign.FIELD]
    assert fresh["claim_by"] > time.time() + ahead and fresh["get"] != old["get"]
    # links still fresh: not re-signed every tick
    assert presign.refresh(conn, pol, store) == 0
    # a retry (failed attempt, back in the queue) whose links are near their claim_by is re-signed
    got = queue.claim(conn, ["fake.image"], "w1", 60, need_links=True)
    assert got.id == job and got.payload[presign.FIELD] == fresh
    assert queue.fail(conn, job, "w1", "boom") == "queued"
    later = fresh["claim_by"] - ahead / 2
    assert presign.refresh(conn, pol, store, now=later) == 1
    retried = _payload(conn, job)[presign.FIELD]
    assert retried["put"] != fresh["put"] and retried["claim_by"] > fresh["claim_by"]
    # a running job with a lapsed lease (a preempted machine) is re-signed for the worker that reclaims it
    assert queue.claim(conn, ["fake.image"], "w2", 60, need_links=True).id == job
    due = retried["claim_by"] - ahead / 2
    assert presign.refresh(conn, pol, store, now=due) == 0   # its lease holds: left alone
    conn.execute("UPDATE jobs SET lease_until = now() - interval '1 second' WHERE id = %s", (job,))
    assert presign.refresh(conn, pol, store, now=due) == 1


def test_a_job_whose_links_miss_an_object_fails_without_retry(conn, database_url, policy, tmp_path):
    def handler(payload, ctx):
        ctx.fetch("s3://bucket/refs/not-signed.png", "x.png")

    job = queue.enqueue(conn, "fake.image", {presign.FIELD: {"prefix": "s3://bucket/", "get": {}, "put": {},
                                                              "claim_by": time.time() + 3600}}, max_attempts=3)
    w = Worker(handlers={"fake.image": handler}, store=Keyless(), policy=presigned(policy), pool="local",
               database_url=database_url, need_links=True)
    assert w.run(max_jobs=1) == 1
    row = conn.execute("SELECT status, attempts, error FROM jobs WHERE id = %s", (job,)).fetchone()
    assert row["status"] == "failed" and row["attempts"] == 1 and "MissingLink" in row["error"]


def test_url_store_answers_only_from_its_links():
    s = UrlStore({"prefix": "s3://b/", "get": {"in.png": "https://x/g"}, "put": {"out.png": "https://x/p"},
                  "absent": ["gone.json"]})
    assert s.exists("in.png") and not s.exists("gone.json") and not s.exists("out.png")
    assert s.uri_for("out.png") == "s3://b/out.png" and s.public_url("s3://b/in.png") == "https://x/g"
    for call in (lambda: s.exists("other"), lambda: s.get_file("s3://b/other", Path("x")),
                 lambda: s.put_file(Path(__file__), "other"), lambda: s.get_file("s3://c/in.png", Path("x"))):
        with pytest.raises(MissingLink):
            call()


# --- round trip through the local S3 stand-in, URLs only ---------------------------------------------------------------

@pytest.fixture()
def s3(monkeypatch):
    endpoint = os.environ.get("MW_TEST_S3_ENDPOINT")
    if not endpoint:
        pytest.skip("MW_TEST_S3_ENDPOINT not set (the local S3 stand-in)")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "stand-in")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "stand-in")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    store = S3Store(os.environ.get("MW_TEST_S3_BUCKET", "manyworlds"), endpoint_url=endpoint)
    for _ in range(30):   # the stand-in may still be starting
        try:
            store.ensure_bucket()
            break
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return store


def test_a_job_round_trips_through_s3_using_only_links(conn, database_url, policy, tmp_path, s3, monkeypatch):
    pol = presigned(policy)
    ref_key = f"refs/{uuid.uuid4().hex}.png"
    src = tmp_path / "ref.png"
    src.write_bytes(PNG)
    s3.put_file(src, ref_key)
    seen = {}

    def handler(payload, ctx):
        seen["payload"] = dict(payload)
        got = ctx.fetch(payload["image_uri"], "in.png")
        out = ctx.workdir / "out.png"
        out.write_bytes(got.read_bytes()[::-1])
        return HandlerResult(outputs=[OutputFile(out, "image")], recipe={"model": "stub"})

    job = queue.enqueue(conn, "fake.image", {"image_uri": s3.uri_for(ref_key)})
    assert autoscale.refresh_links(conn, pol, s3) == [{"action": "links_signed", "jobs": 1}]
    # the worker: no storage key in its environment, a store that fails on any use, links only
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(k)
    w = Worker(handlers={"fake.image": handler}, store=Keyless(), policy=pol, pool="local",
               database_url=database_url, need_links=True)
    assert w.run(max_jobs=1) == 1
    row = conn.execute("SELECT status, error FROM jobs WHERE id = %s", (job,)).fetchone()
    assert row["status"] == "done", row["error"]
    assert presign.FIELD not in seen["payload"]           # the handler never sees the links
    asset = conn.execute("SELECT uri, bytes FROM assets WHERE job_id = %s", (job,)).fetchone()
    assert asset["uri"] == s3.uri_for(f"assets/unassigned/{job}/00.png") and asset["bytes"] == len(PNG)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "stand-in")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "stand-in")
    assert s3.read_bytes(f"assets/unassigned/{job}/00.png") == PNG[::-1]
