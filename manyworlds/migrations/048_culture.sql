-- Culture intelligence beyond social media (manyworlds.research.culture, owner 2026-11-25): publishers' feed items
-- kept as source references only (link, title, feed, date; never article text or images), and our own distilled
-- signals with the items behind them. Not granted to mw_worker.
CREATE TABLE culture_items (
    url          text PRIMARY KEY,
    feed         text NOT NULL,
    category     text NOT NULL,
    title        text NOT NULL,
    published_at timestamptz,
    fetched_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX culture_items_fetched_idx ON culture_items (fetched_at);

CREATE TABLE culture_signals (
    id           bigserial PRIMARY KEY,
    category     text NOT NULL,
    signal       text NOT NULL,
    kind         text NOT NULL CHECK (kind IN ('corroborated', 'single_source')),
    n_sources    int NOT NULL,
    sources      jsonb NOT NULL DEFAULT '[]',
    implications jsonb NOT NULL DEFAULT '{}',
    confidence   numeric,
    created_at   timestamptz NOT NULL DEFAULT now(),
    expires_at   timestamptz NOT NULL
);
CREATE INDEX culture_signals_live_idx ON culture_signals (expires_at, category);
