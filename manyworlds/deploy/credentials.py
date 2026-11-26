"""What a rented GPU machine is allowed to know.

GPU machines run on hosts we don't control (SaladCloud consumer PCs, Vast and RunPod community hosts), so
they never get the control plane's own credentials:

  database  WORKER_DATABASE_URL, a login in the `mw_worker` role (migration 022): jobs, attempts, assets,
            GPU cost entries and its own machine's deadline, nothing else. Made by `mw worker-credentials`.
  storage   WORKER_AWS_ACCESS_KEY_ID / WORKER_AWS_SECRET_ACCESS_KEY, a bucket-scoped R2 token, when set;
            with `security.worker_storage: presigned`, no storage key at all: each job carries presigned links
            for exactly the objects it reads and writes (manyworlds.deploy.presign).

Policy `security.worker_database` / `security.worker_storage`: `required` refuses to launch a machine without
the scoped credential; `presigned` (storage only) satisfies the scoping with no credential. There is no fallback to
the control plane's own database URL or storage key (owner 2026-11-26: unrestricted credentials never reach a rented
machine again); the old `shared` mode is refused like a missing credential. Local development never calls this: its
worker containers get their environment from docker compose.
"""

from __future__ import annotations

import os
import secrets
from urllib.parse import quote, urlparse, urlunparse

from ..policy import Policy
from . import presign

WORKER_ROLE = "mw_worker"            # migration 022
WORKER_LOGIN = "mw_worker_login"     # the one login role in it; `mw worker-credentials` rotates its password

# Passed to every machine unchanged: not credentials for anything we own, or read-only model settings.
SHARED = ["STORAGE_URL", "S3_ENDPOINT_URL", "AWS_REGION", "HF_TOKEN", "MW_IMAGE_MODEL",
          "MW_VIDEO_MODEL", "MW_VIDEO_DISTILL_LORA"]


class MissingWorkerCredentials(RuntimeError):
    pass


def missing(policy: Policy) -> list[str]:
    """Scoped credentials this environment lacks for launching a rented machine."""
    out = []
    if not os.environ.get("WORKER_DATABASE_URL"):
        out.append("WORKER_DATABASE_URL")
    if not presign.enabled(policy) and not (os.environ.get("WORKER_AWS_ACCESS_KEY_ID")
                                            and os.environ.get("WORKER_AWS_SECRET_ACCESS_KEY")):
        out.append("WORKER_AWS_ACCESS_KEY_ID/WORKER_AWS_SECRET_ACCESS_KEY (or security.worker_storage: presigned)")
    return out


def worker_env(policy: Policy) -> dict[str, str]:
    """The environment a GPU machine gets for our services. Raises MissingWorkerCredentials."""
    gaps = missing(policy)
    if gaps:
        raise MissingWorkerCredentials("scoped worker credentials not configured: " + ", ".join(gaps))
    env = {k: os.environ[k] for k in SHARED if os.environ.get(k)}
    env["DATABASE_URL"] = os.environ["WORKER_DATABASE_URL"]
    if presign.enabled(policy):   # no storage key: the machine reads and writes only through its jobs' links
        env[presign.MODE_ENV] = presign.MODE
        return env
    # a bucket-scoped token only, never the control plane's AWS_ACCESS_KEY_ID
    env["AWS_ACCESS_KEY_ID"] = os.environ["WORKER_AWS_ACCESS_KEY_ID"]
    env["AWS_SECRET_ACCESS_KEY"] = os.environ["WORKER_AWS_SECRET_ACCESS_KEY"]
    return env


def login_url(base_url: str, user: str, password: str) -> str:
    """`base_url` with its user and password replaced (same host, database and options)."""
    p = urlparse(base_url)
    host = p.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{host}" + (f":{p.port}" if p.port else "")
    return urlunparse(p._replace(netloc=netloc))


def rotate_login(conn, base_url: str) -> str:
    """Create the worker login (or give it a new password) and return its database URL. Machines started
    with the old password lose access, so run it only while no GPU machine is up."""
    from psycopg import sql

    password = secrets.token_urlsafe(32)
    exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (WORKER_LOGIN,)).fetchone()
    verb = "ALTER" if exists else "CREATE"
    conn.execute(sql.SQL(verb + " ROLE {} WITH LOGIN PASSWORD {}").format(
        sql.Identifier(WORKER_LOGIN), sql.Literal(password)))
    conn.execute(sql.SQL("GRANT {} TO {}").format(sql.Identifier(WORKER_ROLE), sql.Identifier(WORKER_LOGIN)))
    return login_url(base_url, WORKER_LOGIN, password)
