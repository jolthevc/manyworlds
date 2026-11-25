-- Influencer intelligence (docs/superpowers/specs/2026-11-25-influencer-intelligence.md): a durable, searchable research
-- base about named public creators, kept apart from production. Owner 2026-11-25: research records may identify public
-- creators (handles, profile and post links, annotations); downloaded third-party media is a research reference only
-- (its own storage, a retention limit, never a generation input, LoRA or training data, never published) unless a
-- licence on the post says more. The trend watcher's research_items stay anonymous and are unchanged.
--
-- None of these tables is granted to mw_worker: rented GPU machines never read research.

-- Hypotheses are theses with more to them (kind creator_hypothesis): competing explanations, disconfirming
-- evidence, commercial pathways and the rest in body; versions chain through parent_id.
ALTER TABLE theses ADD COLUMN kind text NOT NULL DEFAULT 'market' CHECK (kind IN ('market', 'creator_hypothesis'));
ALTER TABLE theses ADD COLUMN version int NOT NULL DEFAULT 1;
ALTER TABLE theses ADD COLUMN parent_id uuid REFERENCES theses(id);
ALTER TABLE theses ADD COLUMN owner_role text;
ALTER TABLE theses ADD COLUMN budget_usd numeric;
ALTER TABLE theses ADD COLUMN archive_reason text;
ALTER TABLE theses DROP CONSTRAINT theses_status_check;
ALTER TABLE theses ADD CONSTRAINT theses_status_check CHECK (status IN
    ('proposed', 'chosen', 'launched', 'rejected', 'researching', 'auditioning', 'archived'));

-- A public creator we study, by name. One row per platform account.
CREATE TABLE reference_creators (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    platform         text NOT NULL CHECK (platform IN ('instagram', 'tiktok', 'youtube', 'other')),
    handle           text NOT NULL,
    display_name     text,
    profile_url      text,
    bio              text,
    followers        bigint,
    follower_history jsonb NOT NULL DEFAULT '[]',     -- [{at, followers}]
    role             text NOT NULL DEFAULT 'other'
                     CHECK (role IN ('leader', 'riser', 'mid', 'adjacent', 'low_comparator', 'other')),
    cluster          text,
    selected_because text,
    scores           jsonb NOT NULL DEFAULT '{}',     -- relevance, traction vs baseline, consistency, commercial fit...
    coverage         jsonb NOT NULL DEFAULT '{}',     -- posts annotated, share with views, span in days
    caveats          text,                            -- selection bias and what the data can't show
    found_via        jsonb NOT NULL DEFAULT '{}',     -- source, query, run
    first_seen       timestamptz NOT NULL DEFAULT now(),
    last_seen        timestamptz NOT NULL DEFAULT now(),
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX reference_creators_handle_idx ON reference_creators (platform, lower(handle));

-- A public post we study. media: the research copies [{uri, kind: video|image|frame, bytes}] in the research store
-- (manyworlds.research.references), deleted at media_purge_after. licence: what else the post's licence permits
-- ({name, source, uses: [research, motion_reference, generation_input]}); absent = research only.
CREATE TABLE reference_posts (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id        uuid NOT NULL REFERENCES reference_creators(id) ON DELETE CASCADE,
    platform          text NOT NULL,
    external_id       text NOT NULL,
    url               text,
    media_type        text CHECK (media_type IN ('photo', 'carousel', 'video')),
    posted_at         timestamptz,
    caption           text,
    metrics           jsonb NOT NULL DEFAULT '{}',    -- the latest public counts
    snapshots         jsonb NOT NULL DEFAULT '[]',    -- [{at, views, likes, comments, shares, saves}]
    media             jsonb NOT NULL DEFAULT '[]',
    media_purge_after timestamptz,
    licence           jsonb NOT NULL DEFAULT '{}',
    fetched_at        timestamptz NOT NULL DEFAULT now(),
    created_at        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (platform, external_id)
);
CREATE INDEX reference_posts_creator_idx ON reference_posts (creator_id, posted_at);
CREATE INDEX reference_posts_purge_idx ON reference_posts (media_purge_after) WHERE media_purge_after IS NOT NULL;

-- What a post is made of, three layers (manyworlds.research.annotation): measured by code, observed in frames and
-- words, plausible mechanism, speculation, kept apart; evidence and economics with confidence.
CREATE TABLE reference_annotations (
    id          bigserial PRIMARY KEY,
    post_id     uuid NOT NULL REFERENCES reference_posts(id) ON DELETE CASCADE,
    schema      text NOT NULL,
    model       text,
    measured    jsonb NOT NULL DEFAULT '{}',
    observed    jsonb NOT NULL DEFAULT '{}',
    mechanism   jsonb NOT NULL DEFAULT '[]',
    speculation jsonb NOT NULL DEFAULT '[]',
    evidence    jsonb NOT NULL DEFAULT '{}',
    confidence  numeric,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (post_id, schema)
);

-- The creative pattern library, photo and video in one (manyworlds.brain.patterns): seeded from policy/patterns,
-- then filled and versioned by research. sources: [{post_id, creator_id, url}]; a card may cite one creator.
CREATE TABLE pattern_cards (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    key           text NOT NULL,
    version       int NOT NULL DEFAULT 1,
    status        text NOT NULL DEFAULT 'draft' CHECK (status IN ('illustrative', 'draft', 'active', 'retired')),
    medium        text NOT NULL CHECK (medium IN ('photo', 'carousel', 'video', 'sequence')),
    signature_ok  boolean NOT NULL DEFAULT false,
    body          jsonb NOT NULL DEFAULT '{}',
    sources       jsonb NOT NULL DEFAULT '[]',
    evidence      jsonb NOT NULL DEFAULT '{}',
    acceptance    jsonb NOT NULL DEFAULT '{}',
    outcomes      jsonb NOT NULL DEFAULT '{}',
    superseded_by uuid REFERENCES pattern_cards(id),
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (key, version)
);

-- A candidate creator auditioned before casting: its design, the named references and cards behind it, the cheap
-- checks against those references, its nine posts (creator_id + batch), cost, blind scores and the decision.
CREATE TABLE auditions (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    thesis_id    uuid REFERENCES theses(id),
    arm          text NOT NULL CHECK (arm IN ('A', 'B', 'C')),
    candidate    jsonb NOT NULL DEFAULT '{}',
    source_cards jsonb NOT NULL DEFAULT '[]',
    "references" jsonb NOT NULL DEFAULT '[]',     -- reference_creators ids the brief named
    checks       jsonb NOT NULL DEFAULT '{}',     -- manyworlds.brain.impersonation: warnings and blocks
    creator_id   uuid REFERENCES creators(id),
    batch        text,
    est_usd      numeric,
    actual_usd   numeric,
    scores       jsonb NOT NULL DEFAULT '{}',
    decision     text NOT NULL DEFAULT 'pending' CHECK (decision IN ('pending', 'select', 'revise', 'reject')),
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

-- People who asked to be forgotten (mw research forget): never collected again. The only identifier kept is a salted
-- hash of the handle.
CREATE TABLE research_forget (
    handle_hash  text NOT NULL,
    platform     text NOT NULL,
    requested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (platform, handle_hash)
);

-- Research media is never a production asset: no asset row may point into the research store.
ALTER TABLE assets ADD CONSTRAINT assets_not_research CHECK (uri NOT LIKE '%/research-refs/%');
