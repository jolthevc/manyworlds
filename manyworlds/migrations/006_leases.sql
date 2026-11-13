-- Named leases. Session advisory locks don't survive a transaction-mode pooler (Neon's
-- pgbouncer runs each statement on any backend, so the unlock can miss the lock forever).
CREATE TABLE leases (
    name       text PRIMARY KEY,
    holder     text NOT NULL,
    until      timestamptz NOT NULL
);
