-- Manyworlds core schema. Every cost and decision is traceable to a policy version.

CREATE TABLE policies (
    version_hash text PRIMARY KEY,
    body         jsonb NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE creators (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    handle     text UNIQUE NOT NULL,
    status     text NOT NULL DEFAULT 'draft'
               CHECK (status IN ('draft', 'active', 'paused', 'retired')),
    bible      jsonb NOT NULL DEFAULT '{}',   -- persona: tastes, voice rules, disclosure line
    anchors    jsonb NOT NULL DEFAULT '{}',   -- reference images, LoRA uri, voice id, face centroid
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE theses (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id uuid REFERENCES creators(id),
    summary    text NOT NULL,
    evidence   jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE experiments (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id uuid REFERENCES creators(id),
    hypothesis text NOT NULL,
    arm        jsonb NOT NULL DEFAULT '{}',
    status     text NOT NULL DEFAULT 'open',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE plans (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id   uuid NOT NULL REFERENCES creators(id),
    period_start date NOT NULL,
    body         jsonb NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- Work queue. Workers on interruptible GPUs claim jobs with a lease; a worker that
-- disappears stops heartbeating, the lease expires and another worker reclaims the job.
CREATE TABLE jobs (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind            text NOT NULL,
    payload         jsonb NOT NULL DEFAULT '{}',
    status          text NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued', 'running', 'done', 'failed', 'dead')),
    priority        int NOT NULL DEFAULT 0,
    attempts        int NOT NULL DEFAULT 0,
    max_attempts    int NOT NULL DEFAULT 3,
    lease_until     timestamptz,
    worker_id       text,
    idempotency_key text UNIQUE,
    creator_id      uuid REFERENCES creators(id),
    parent_job_id   uuid REFERENCES jobs(id),
    batch           text,                    -- groups jobs, e.g. a benchmark run
    result          jsonb,
    error           text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX jobs_claim_idx ON jobs (kind, status, priority DESC, created_at);
CREATE INDEX jobs_batch_idx ON jobs (batch);

CREATE TABLE attempts (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id      uuid NOT NULL REFERENCES jobs(id),
    worker_id   text NOT NULL,
    pool        text NOT NULL,
    gpu_type    text,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    gpu_seconds numeric,
    outcome     text,                         -- done | failed | lost
    error       text
);
CREATE INDEX attempts_job_idx ON attempts (job_id);

CREATE TABLE assets (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id      uuid REFERENCES jobs(id),
    creator_id  uuid REFERENCES creators(id),
    kind        text NOT NULL,                -- image | video | audio | lora | ...
    uri         text NOT NULL,
    sha256      text NOT NULL,
    bytes       bigint NOT NULL,
    meta        jsonb NOT NULL DEFAULT '{}',  -- width, height, duration, seed
    recipe      jsonb NOT NULL DEFAULT '{}',  -- model, prompt, params: full provenance
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX assets_job_idx ON assets (job_id);

CREATE TABLE reviews (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id   uuid NOT NULL REFERENCES assets(id),
    gate       text NOT NULL,
    passed     boolean NOT NULL,
    score      numeric,
    details    jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE posts (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id   uuid NOT NULL REFERENCES creators(id),
    plan_id      uuid REFERENCES plans(id),
    platform     text,
    format       text NOT NULL,
    status       text NOT NULL DEFAULT 'planned',
    caption      text,
    asset_ids    uuid[] NOT NULL DEFAULT '{}',
    qa_status    text,
    external_id  text,
    published_at timestamptz,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE metric_snapshots (
    id          bigserial PRIMARY KEY,
    post_id     uuid NOT NULL REFERENCES posts(id),
    captured_at timestamptz NOT NULL DEFAULT now(),
    simulated   boolean NOT NULL DEFAULT false,
    metrics     jsonb NOT NULL
);

-- Every dollar, GPU-second and API call. Cost per accepted output is computed from here.
CREATE TABLE cost_entries (
    id             bigserial PRIMARY KEY,
    job_id         uuid REFERENCES jobs(id),
    attempt_id     uuid REFERENCES attempts(id),
    creator_id     uuid REFERENCES creators(id),
    post_id        uuid REFERENCES posts(id),
    category       text NOT NULL,             -- gpu | llm | data | storage | hosted_api
    provider       text NOT NULL,
    quantity       numeric NOT NULL,
    unit           text NOT NULL,             -- gpu_second | token | item | ...
    usd            numeric NOT NULL,
    policy_version text,
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX cost_entries_job_idx ON cost_entries (job_id);

CREATE TABLE capabilities (
    name      text PRIMARY KEY,
    interface text NOT NULL,
    config    jsonb NOT NULL DEFAULT '{}',
    stats     jsonb NOT NULL DEFAULT '{}',     -- observed acceptance rate, latency, cost
    enabled   boolean NOT NULL DEFAULT true
);

CREATE TABLE playbook (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    hypothesis text NOT NULL,
    evidence   jsonb NOT NULL DEFAULT '[]',
    confidence numeric,
    updated_at timestamptz NOT NULL DEFAULT now()
);
