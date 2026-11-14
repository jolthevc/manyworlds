-- The Brain (docs/superpowers/specs/2026-11-14-brain-design.md): agents, their shared memory,
-- creative evaluations, findings, the strategist's state and the portfolio layer. Real and
-- simulated results live in different tables so they can never be mixed by accident.

-- Markets and clusters first: creators point at them.
CREATE TABLE markets (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name            text UNIQUE NOT NULL,          -- category or sub-niche, e.g. fitness / home workouts
    parent_id       uuid REFERENCES markets(id),
    dossier         jsonb NOT NULL DEFAULT '{}',   -- scouts' research: audience, archetypes, formats, saturation ...
    monetization    jsonb NOT NULL DEFAULT '{}',   -- long-term commercial attractiveness; not in the growth objective
    conviction_mean numeric,
    conviction_low  numeric,
    conviction_high numeric,
    sources         jsonb NOT NULL DEFAULT '[]',
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE clusters (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    market_id    uuid NOT NULL REFERENCES markets(id),
    name         text NOT NULL,
    strategy     text NOT NULL DEFAULT '',
    intelligence jsonb NOT NULL DEFAULT '{}',      -- shared learnings: audience, formats, styles, production
    metrics      jsonb NOT NULL DEFAULT '{}',      -- concentration, overlap, correlated outcomes
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

-- A creator publishes either in the sandbox (simulated) or for real; new creators start in the sandbox.
ALTER TABLE creators ADD COLUMN publish_mode text NOT NULL DEFAULT 'simulated'
    CHECK (publish_mode IN ('simulated', 'instagram'));
ALTER TABLE creators ADD COLUMN market_id uuid REFERENCES markets(id);
ALTER TABLE creators ADD COLUMN cluster_id uuid REFERENCES clusters(id);

-- Per platform account. The token is never stored: token_env names the .env key that holds it.
CREATE TABLE accounts (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id  uuid NOT NULL REFERENCES creators(id),
    platform    text NOT NULL,
    handle      text,
    external_id text,
    token_env   text,
    status      text NOT NULL DEFAULT 'requested'
                CHECK (status IN ('requested', 'connected', 'suspended', 'closed')),
    meta        jsonb NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (creator_id, platform)
);

-- The strategist's view of each creator: stage (observed) and quality tier; conviction is separate.
CREATE TABLE creator_states (
    creator_id  uuid PRIMARY KEY REFERENCES creators(id),
    stage       text NOT NULL DEFAULT 'exploring',   -- labels come from policy, not a fixed list
    entered_at  timestamptz NOT NULL DEFAULT now(),
    tier        text,
    adjustments int NOT NULL DEFAULT 0,
    evidence_window jsonb NOT NULL DEFAULT '{}',      -- counters for the current evidence window
    history     jsonb NOT NULL DEFAULT '[]',
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- Persona memory: versioned facts (tastes, opinions, places, people, routines, canon).
CREATE TABLE creator_memory (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id  uuid NOT NULL REFERENCES creators(id),
    kind        text NOT NULL,
    text        text NOT NULL,
    source_post uuid REFERENCES posts(id),
    valid_from  timestamptz NOT NULL DEFAULT now(),
    valid_until timestamptz,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX creator_memory_creator_idx ON creator_memory (creator_id) WHERE valid_until IS NULL;

-- One future post or Story, as an experiment with measurable creative characteristics.
CREATE TABLE ideas (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id    uuid NOT NULL REFERENCES creators(id),
    batch         text,
    concept       text NOT NULL,
    format        text NOT NULL,
    body          jsonb NOT NULL DEFAULT '{}',        -- scenes, script, caption draft
    attributes    jsonb NOT NULL DEFAULT '{}',        -- setting, outfit, activity, mood, trend ...
    experiment_id uuid REFERENCES experiments(id),
    arm           text,
    status        text NOT NULL DEFAULT 'proposed',
    post_id       uuid REFERENCES posts(id),
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ideas_creator_idx ON ideas (creator_id, status);

CREATE TABLE rubric_versions (
    rubric     text NOT NULL,                     -- idea | execution | feed
    version    int NOT NULL,
    file_hash  text NOT NULL,
    body       jsonb NOT NULL,
    reference  jsonb NOT NULL DEFAULT '{}',       -- reference-set scores under this version
    adopted_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (rubric, version)
);

CREATE TABLE evaluations (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    item_kind       text NOT NULL CHECK (item_kind IN ('idea', 'asset', 'feed')),
    idea_id         uuid REFERENCES ideas(id),
    asset_id        uuid REFERENCES assets(id),
    creator_id      uuid REFERENCES creators(id),
    rubric          text NOT NULL,
    rubric_version  int NOT NULL,
    evaluator_model text NOT NULL,
    prompt_version  text NOT NULL,
    scores          jsonb NOT NULL,               -- dimension -> median
    spreads         jsonb NOT NULL DEFAULT '{}',  -- dimension -> spread across samples
    rationale       jsonb NOT NULL DEFAULT '{}',
    samples         int NOT NULL,
    profile_version int,
    format          text,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX evaluations_idea_idx ON evaluations (idea_id);

CREATE TABLE findings (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    scope         text NOT NULL CHECK (scope IN ('post', 'creator', 'cluster', 'market', 'portfolio')),
    creator_id    uuid REFERENCES creators(id),
    cluster_id    uuid REFERENCES clusters(id),
    market_id     uuid REFERENCES markets(id),
    statement     text NOT NULL,
    evidence      text NOT NULL CHECK (evidence IN ('observational', 'experimental', 'simulated', 'research')),
    grade         text NOT NULL CHECK (grade IN ('proven', 'tentative', 'speculative')),
    effect        numeric,
    interval_low  numeric,
    interval_high numeric,
    n             int,
    status        text NOT NULL DEFAULT 'open',
    sources       jsonb NOT NULL DEFAULT '[]',
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX findings_scope_idx ON findings (scope, creator_id);

ALTER TABLE experiments ADD COLUMN arms jsonb NOT NULL DEFAULT '[]';
ALTER TABLE experiments ADD COLUMN metric text;
ALTER TABLE experiments ADD COLUMN min_sample int;
ALTER TABLE experiments ADD COLUMN result jsonb;
ALTER TABLE experiments ADD COLUMN updated_at timestamptz NOT NULL DEFAULT now();

-- The strategist's money per creator (or cluster, market, bucket) per period.
CREATE TABLE allocations (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    period     text NOT NULL,
    level      text NOT NULL,                     -- bucket | market | cluster | creator | format
    subject    jsonb NOT NULL,
    usd        numeric NOT NULL,
    tier       text,
    plan       jsonb NOT NULL DEFAULT '{}',       -- format mix, cadence, experiments
    decision_id uuid REFERENCES decisions(id),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE budgets (
    period      text PRIMARY KEY,                 -- e.g. 2026-11
    master_usd  numeric NOT NULL,
    reserve_usd numeric NOT NULL,
    pacing      jsonb NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE convictions (
    id         bigserial PRIMARY KEY,
    subject    text NOT NULL CHECK (subject IN ('creator', 'market', 'cluster')),
    subject_id uuid NOT NULL,
    mean       numeric NOT NULL,
    low        numeric NOT NULL,
    high       numeric NOT NULL,
    inputs     jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX convictions_subject_idx ON convictions (subject, subject_id, created_at DESC);

CREATE TABLE publications (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id      uuid NOT NULL REFERENCES posts(id),
    platform     text NOT NULL,                   -- instagram | simulated
    account_id   uuid REFERENCES accounts(id),
    external_id  text,
    permalink    text,
    ai_label     boolean,
    simulated    boolean NOT NULL,
    published_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (post_id, platform)
);

-- Real metrics only from now on; simulated ones have their own table.
ALTER TABLE metric_snapshots ADD COLUMN post_age_s int;
ALTER TABLE metric_snapshots ADD CONSTRAINT metric_snapshots_real_only CHECK (simulated = false);

CREATE TABLE simulated_metrics (
    id            bigserial PRIMARY KEY,
    post_id       uuid NOT NULL REFERENCES posts(id),
    captured_at   timestamptz NOT NULL DEFAULT now(),
    post_age_s    int,
    model         text NOT NULL,
    panel_version text NOT NULL,
    metrics       jsonb NOT NULL
);

CREATE TABLE approvals (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind         text NOT NULL,
    subject      jsonb NOT NULL,
    reason       text NOT NULL,
    amount_usd   numeric,
    status       text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    requested_by text NOT NULL,
    decided_by   text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    decided_at   timestamptz
);

CREATE TABLE events (
    id         bigserial PRIMARY KEY,
    type       text NOT NULL,
    subject    jsonb NOT NULL DEFAULT '{}',
    payload    jsonb NOT NULL DEFAULT '{}',
    routed_to  text,
    job_id     uuid REFERENCES jobs(id),
    created_at timestamptz NOT NULL DEFAULT now(),
    handled_at timestamptz
);

-- Settings the Brain changes at runtime, within policy authority; latest version per path wins.
CREATE TABLE policy_overrides (
    id          bigserial PRIMARY KEY,
    path        text NOT NULL,
    value       jsonb NOT NULL,
    version     int NOT NULL,
    author      text NOT NULL,
    reason      text NOT NULL,
    decision_id uuid REFERENCES decisions(id),
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- Decisions name the agent and job that made them; a rerun of the same job step finds its own.
ALTER TABLE decisions ADD COLUMN agent text;
ALTER TABLE decisions ADD COLUMN job_id uuid REFERENCES jobs(id);
ALTER TABLE decisions ADD COLUMN step text;
ALTER TABLE decisions ADD COLUMN rationale text;
ALTER TABLE decisions ADD COLUMN override_version int;
CREATE UNIQUE INDEX decisions_job_step_uq ON decisions (job_id, step) WHERE job_id IS NOT NULL;

ALTER TABLE cost_entries ADD COLUMN agent text;
CREATE INDEX cost_entries_agent_idx ON cost_entries (agent, created_at) WHERE agent IS NOT NULL;
