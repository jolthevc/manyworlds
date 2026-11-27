"""Object storage behind one small interface.

`s3://bucket` works with Cloudflare R2, MinIO or AWS (set S3_ENDPOINT_URL for R2/MinIO).
`file:///path` stores on local disk, for tests and offline runs.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname


@dataclass(frozen=True)
class StoredObject:
    uri: str
    sha256: str
    bytes: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ObjectStore:
    def put_file(self, local: Path, key: str) -> StoredObject:
        raise NotImplementedError

    def get_file(self, uri: str, local: Path) -> Path:
        raise NotImplementedError

    def uri_for(self, key: str) -> str:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def size(self, key: str) -> int | None:
        raise NotImplementedError

    def public_url(self, uri: str, seconds: int = 3600) -> str:
        """A time-limited URL anyone can fetch (e.g. Instagram fetching media to publish)."""
        raise NotImplementedError

    def delete(self, uri: str) -> None:
        """Remove an object (storage retention); a missing one is fine."""
        raise NotImplementedError


class LocalStore(ObjectStore):
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def delete(self, uri: str) -> None:
        from urllib.request import url2pathname
        Path(url2pathname(urlparse(uri).path)).unlink(missing_ok=True)

    def put_file(self, local: Path, key: str) -> StoredObject:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
        return StoredObject(uri=dest.resolve().as_uri(), sha256=_sha256(dest), bytes=dest.stat().st_size)

    def get_file(self, uri: str, local: Path) -> Path:
        src = Path(url2pathname(urlparse(uri).path))
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, local)
        return local

    def uri_for(self, key: str) -> str:
        return (self.root / key).resolve().as_uri()

    def public_url(self, uri: str, seconds: int = 3600) -> str:
        return uri   # local development only: nothing outside this machine can fetch it

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def size(self, key: str) -> int | None:
        p = self.root / key
        return p.stat().st_size if p.exists() else None


class S3Store(ObjectStore):
    def __init__(self, bucket: str, endpoint_url: str | None = None):
        import boto3

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or os.environ.get("S3_ENDPOINT_URL") or None,
            region_name=os.environ.get("AWS_REGION", "auto"),
        )

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def put_file(self, local: Path, key: str) -> StoredObject:
        self.client.upload_file(str(local), self.bucket, key)
        return StoredObject(uri=f"s3://{self.bucket}/{key}", sha256=_sha256(local), bytes=local.stat().st_size)

    def get_file(self, uri: str, local: Path) -> Path:
        parsed = urlparse(uri)
        local.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(local))
        return local

    def uri_for(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def delete(self, uri: str) -> None:
        parsed = urlparse(uri)
        self.client.delete_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))

    def exists(self, key: str) -> bool:
        """False only when the store says the object is not there (404 / NoSuchKey). Any other error (no access, a
        network blip, throttling) is raised: treating it as missing would redo expensive work (a model mirror, a
        finished video) or hide a broken credential."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as e:  # noqa: BLE001 - botocore's ClientError, read without importing botocore
            if _not_found(e):
                return False
            raise

    def size(self, key: str) -> int | None:
        try:
            return int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"])
        except Exception:  # noqa: BLE001
            return None

    def presign(self, key: str, method: str, seconds: int) -> str:
        """A presigned GET or PUT URL for `key` in this bucket (manyworlds.deploy.presign: links for rented machines,
        which hold no storage key). Path-style addressing, so it works against any S3 endpoint (R2, a local stand-in)."""
        if getattr(self, "_signer", None) is None:
            import boto3
            from botocore.config import Config

            self._signer = boto3.client(
                "s3", endpoint_url=self.client.meta.endpoint_url, region_name=self.client.meta.region_name,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
        op = {"GET": "get_object", "PUT": "put_object"}[method]
        return self._signer.generate_presigned_url(op, ExpiresIn=int(seconds),
                                                   Params={"Bucket": self.bucket, "Key": key})

    def read_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def public_url(self, uri: str, seconds: int = 3600) -> str:
        """A presigned GET URL, so a platform can fetch the object without access to the bucket."""
        parsed = urlparse(uri)
        return self.client.generate_presigned_url("get_object", ExpiresIn=int(seconds),
                                                  Params={"Bucket": parsed.netloc, "Key": parsed.path.lstrip("/")})


NOT_FOUND_CODES = ("404", "NoSuchKey", "NotFound")


def _not_found(error: Exception) -> bool:
    """Whether an S3 client error means the object does not exist (a ClientError's response code or HTTP 404)."""
    response = getattr(error, "response", None) or {}
    code = str((response.get("Error") or {}).get("Code", ""))
    status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    return code in NOT_FOUND_CODES or status == 404


class MissingLink(RuntimeError):
    """A job asked the link store for an object it holds no link for: the job's links (manyworlds.deploy.presign)
    don't cover what the handler reads or writes. Retrying elsewhere can't help; the worker fails the job."""


MODEL_PREFIX = "models/"   # model mirrors (worker.models): a mirror with no link is treated as not mirrored


class UrlStore(ObjectStore):
    """The object store of a job on a rented machine that holds no storage key (policy
    `security.worker_storage: presigned`): every object the job may read or write comes with it as a presigned link
    (`payload._storage`, made on the control plane by manyworlds.deploy.presign). Handlers call the same interface
    as on any other store.

    links: {"prefix": "s3://bucket/", "get": {key: url}, "put": {key: url}, "absent": [key, ...], "expires": epoch}"""

    def __init__(self, links: dict, *, timeout: float = 600, chunk: int = 1 << 20, resumes: int = 8,
                 opener=None, pause: float = 2.0):
        self.prefix = str(links["prefix"])
        self.gets = dict(links.get("get") or {})
        self.puts = dict(links.get("put") or {})
        self.absent = set(links.get("absent") or [])
        self.timeout = timeout
        self.chunk = chunk
        self.resumes = resumes      # a cut transfer resumes from the bytes it has (HTTP Range), this many times
        self.pause = pause
        self._open = opener

    def _key(self, uri: str) -> str:
        if not uri.startswith(self.prefix):
            raise MissingLink(f"not an object of this store: {uri[:200]}")
        return uri[len(self.prefix):]

    def uri_for(self, key: str) -> str:
        return self.prefix + key

    def get_file(self, uri: str, local: Path) -> Path:
        """Download through the read link. A transfer cut short (it ends early, or the connection drops) resumes
        from the bytes already on disk with an HTTP Range request, up to `resumes` times; the file is complete only
        when it has the size the server announced. 2026-11-27: one plain GET per file returned 4-10 GB weights cut
        short on Salad and Vast nodes, and every job that loaded them failed."""
        import http.client
        import time
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        key = self._key(uri)
        url = self.gets.get(key)
        if url is None:
            raise MissingLink(f"no read link for {key}")
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(local.name + ".dl")
        part.unlink(missing_ok=True)
        opener = self._open or (lambda req: urlopen(req, timeout=self.timeout))
        total: int | None = None
        for attempt in range(self.resumes + 1):
            have = part.stat().st_size if part.exists() else 0
            req = Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            try:
                with opener(req) as r:
                    status = getattr(r, "status", 200)
                    if have and status != 206:            # the server ignored the range: start again
                        have = 0
                    crange = r.headers.get("Content-Range") if hasattr(r, "headers") else None
                    length = r.headers.get("Content-Length") if hasattr(r, "headers") else None
                    if crange and "/" in crange and crange.rsplit("/", 1)[1].isdigit():
                        total = int(crange.rsplit("/", 1)[1])
                    elif length is not None and total is None:
                        total = have + int(length)
                    with part.open("ab" if have else "wb") as f:
                        shutil.copyfileobj(r, f, self.chunk)
            except HTTPError as e:
                if e.code in (401, 403, 404, 416) or attempt == self.resumes:   # a dead link: no resume helps
                    raise
            except (OSError, http.client.HTTPException):
                if attempt == self.resumes:
                    raise
            size = part.stat().st_size if part.exists() else 0
            if total is None or size >= total:
                break
            time.sleep(min(30.0, self.pause * (2 ** attempt)))
        size = part.stat().st_size if part.exists() else 0
        if total is not None and size < total:
            raise RuntimeError(f"download incomplete: {key} is {size} bytes, expected {total}")
        part.replace(local)
        return local

    def put_file(self, local: Path, key: str) -> StoredObject:
        from urllib.request import Request, urlopen

        url = self.puts.get(key)
        if url is None:
            raise MissingLink(f"no write link for {key}")
        size = local.stat().st_size
        with local.open("rb") as f:
            req = Request(url, data=f, method="PUT",
                          headers={"Content-Length": str(size), "Content-Type": "application/octet-stream"})
            with urlopen(req, timeout=self.timeout) as r:
                r.read()
        return StoredObject(uri=self.uri_for(key), sha256=_sha256(local), bytes=size)

    def exists(self, key: str) -> bool:
        """Inputs were checked when the links were made: a read link means it was there, `absent` that it was not.
        A key the job only writes is reported missing (the work is redone, which handlers allow); a model mirror
        with no link is not mirrored (the worker loads from Hugging Face). Anything else is a gap in the links."""
        if key in self.gets:
            return True
        if key in self.absent or key in self.puts or key.startswith(MODEL_PREFIX):
            return False
        raise MissingLink(f"no link for {key}")

    def size(self, key: str) -> int | None:
        return None   # unknown without a HEAD link: callers treat it as missing and redo the copy

    def public_url(self, uri: str, seconds: int = 3600) -> str:
        url = self.gets.get(self._key(uri))
        if url is None:
            raise MissingLink(f"no read link for {uri[:200]}")
        return url

    def delete(self, uri: str) -> None:
        raise MissingLink("a link store cannot delete")


def from_url(url: str | None = None) -> ObjectStore:
    url = url or os.environ.get("STORAGE_URL", "file://./.data/objects")
    parsed = urlparse(url)
    if parsed.scheme == "s3":
        return S3Store(parsed.netloc)
    if parsed.scheme == "file":
        return LocalStore(Path((parsed.netloc + parsed.path) or ".data/objects"))
    raise ValueError(f"unsupported STORAGE_URL scheme: {url}")
