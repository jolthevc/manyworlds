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


class LocalStore(ObjectStore):
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

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

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()


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

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:  # noqa: BLE001 - 404 or no access: treat as missing
            return False


def from_url(url: str | None = None) -> ObjectStore:
    url = url or os.environ.get("STORAGE_URL", "file://./.data/objects")
    parsed = urlparse(url)
    if parsed.scheme == "s3":
        return S3Store(parsed.netloc)
    if parsed.scheme == "file":
        return LocalStore(Path((parsed.netloc + parsed.path) or ".data/objects"))
    raise ValueError(f"unsupported STORAGE_URL scheme: {url}")
