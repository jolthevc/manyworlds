-- Research: what was pulled, from where, and the derived features the Brain reads.
-- Privacy guardrails: logged-out public data only; author handles are stored as a
-- salted hash, never in clear; captions and raw items are purged after 30 days.

CREATE TABLE research_runs (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source     text NOT NULL,                 -- apify:instagram-hashtag, apify:tiktok, ...
    query      jsonb NOT NULL,
    items      int NOT NULL DEFAULT 0,
    usd        numeric NOT NULL DEFAULT 0,
    error      text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE research_items (
    id               bigserial PRIMARY KEY,
    run_id           uuid REFERENCES research_runs(id),
    platform         text NOT NULL,           -- instagram | tiktok
    external_id      text NOT NULL,
    query_tag        text,
    author_hash      text,
    author_followers bigint,
    media_type       text,                    -- photo | carousel | video
    caption          text,
    hashtags         text[] NOT NULL DEFAULT '{}',
    likes            bigint,
    comments         bigint,
    views            bigint,
    shares           bigint,
    duration_s       numeric,
    posted_at        timestamptz,
    fetched_at       timestamptz NOT NULL DEFAULT now(),
    purge_after      timestamptz NOT NULL DEFAULT now() + interval '30 days',
    UNIQUE (platform, external_id)
);
CREATE INDEX research_items_tag_idx ON research_items (platform, query_tag);

ALTER TABLE theses ADD COLUMN status text NOT NULL DEFAULT 'proposed'
    CHECK (status IN ('proposed', 'chosen', 'launched', 'rejected'));
ALTER TABLE theses ADD COLUMN score numeric;
ALTER TABLE theses ADD COLUMN body jsonb NOT NULL DEFAULT '{}';

-- Every Brain decision, with its inputs and reasoning, for audit and for learning.
CREATE TABLE decisions (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind           text NOT NULL,             -- research_plan | thesis_choice | creator_launch | ...
    creator_id     uuid REFERENCES creators(id),
    inputs         jsonb NOT NULL DEFAULT '{}',
    output         jsonb NOT NULL DEFAULT '{}',
    model          text,
    policy_version text,
    created_at     timestamptz NOT NULL DEFAULT now()
);
