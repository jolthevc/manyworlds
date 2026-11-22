-- Trend watcher (manyworlds.research.trends): continuous public-post collection per niche, velocity, derived
-- features of what performs, and daily briefs. Raw items keep their 30-day purge; features and briefs are derived
-- (no handles, no captions) and outlive them.

ALTER TABLE research_items ADD COLUMN saves bigint;
ALTER TABLE research_items ADD COLUMN sound_id text;
ALTER TABLE research_items ADD COLUMN sound_title text;
ALTER TABLE research_items ADD COLUMN cover_url text;
ALTER TABLE research_items ADD COLUMN url text;
ALTER TABLE research_items ADD COLUMN market text;          -- the niche it was collected for
ALTER TABLE research_items ADD COLUMN source text;          -- scrapecreators | apify
CREATE INDEX research_items_market_idx ON research_items (market, platform, posted_at);

-- Counts each time a post is seen again: how fast it is growing (velocity), not just how big it is.
CREATE TABLE research_snapshots (
    item_id     bigint NOT NULL REFERENCES research_items(id) ON DELETE CASCADE,
    captured_at timestamptz NOT NULL DEFAULT now(),
    views       bigint,
    likes       bigint,
    comments    bigint,
    shares      bigint,
    saves       bigint,
    PRIMARY KEY (item_id, captured_at)
);

-- What a post is made of (hook, format, pacing...), from its cover, caption opening and first seconds of speech,
-- with the numbers needed to judge it. Kept after the raw item is purged.
CREATE TABLE research_features (
    id          bigserial PRIMARY KEY,
    item_id     bigint UNIQUE REFERENCES research_items(id) ON DELETE SET NULL,
    platform    text NOT NULL,
    market      text,
    sample      text NOT NULL CHECK (sample IN ('outlier', 'control')),
    posted_at   timestamptz,
    duration_s  numeric,
    views       bigint,
    outlier     numeric,           -- log10 views over the niche's median views that week
    velocity    numeric,           -- views gained per hour between the first two sightings, when seen twice
    sound_id    text,
    features    jsonb NOT NULL,
    model       text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX research_features_market_idx ON research_features (market, platform, created_at);

-- The daily brief per niche: what is working, rising, cooling, and the hooks to try.
CREATE TABLE trend_briefs (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    market      text NOT NULL,
    day         date NOT NULL,
    stats       jsonb NOT NULL,
    brief       jsonb NOT NULL,
    model       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (market, day)
);
