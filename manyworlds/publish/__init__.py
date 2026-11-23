"""Publishing: one interface, a simulated implementation (sandbox creators) and Instagram (M4).

A publisher returns {external_id, permalink, ai_label, simulated}. The scheduler writes one
`publications` row per post either way; outcomes later land in `simulated_metrics` or
`metric_snapshots` with the same metric keys, so learning reads both the same way (through
manyworlds.brain.evidence, which keeps them apart).
"""

from __future__ import annotations

from typing import Protocol


class Publisher(Protocol):
    platform: str

    def publish(self, conn, policy, post: dict) -> dict: ...


class NotForPublishing(Exception):
    """A test post (the breadth test's, or one made in a test batch): it is never published, scheduled or not."""


def batch_prefixes_for_tests(policy) -> list[str]:
    """Batch prefixes that mark test work (operations.test_batch_prefixes, e.g. bench-, test-)."""
    return [str(p) for p in (policy.get("operations", "test_batch_prefixes", default=["bench-", "test-"]) or [])]


def unpublishable(conn, policy, post: dict) -> str | None:
    """Why `post` is a test post that must never be published, or None: it carries body.breadth_run (the breadth
    test), names a test batch in body.batch, or any of its jobs ran in a test batch."""
    body = post.get("body") or {}
    if body.get("breadth_run"):
        return f"breadth test post (run {body['breadth_run']})"
    prefixes = tuple(batch_prefixes_for_tests(policy))
    if prefixes and str(body.get("batch") or "").startswith(prefixes):
        return f"test batch {body['batch']}"
    if prefixes and post.get("id") is not None:
        row = conn.execute("SELECT batch FROM jobs WHERE post_id = %s AND batch LIKE ANY(%s) LIMIT 1",
                           (post["id"], [f"{p}%" for p in prefixes])).fetchone()
        if row:
            return f"made in test batch {row['batch']}"
    return None
