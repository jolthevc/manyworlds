-- The brain's call journal and Message Batches (manyworlds.llm_batch, 2026-11-23).
--
-- llm_journal: every model call a brain job makes while batching is on, keyed by job, attempt, role, model, schema and
-- prompt. A rerun of the job replays its finished calls from here for nothing; a role in policy llm.batch.roles is
-- answered through Anthropic's Message Batches API at half price, the job stepping aside (queued again) until its
-- answer lands. `request` holds a batched call's parameters until it is sent; `content_text` the prompt's text (and
-- image digests) for matching a slightly changed prompt on a rerun. Rows go after llm.batch.keep_days.
CREATE TABLE IF NOT EXISTS llm_journal (
    id             bigserial PRIMARY KEY,
    job_id         uuid NOT NULL,
    attempt        int NOT NULL,
    role           text NOT NULL,
    model          text NOT NULL,
    schema_name    text NOT NULL,
    system_hash    text NOT NULL,
    content_hash   text NOT NULL,
    content_text   text NOT NULL DEFAULT '',
    mode           text NOT NULL,                     -- sync | batch | batch_fallback
    status         text NOT NULL,                     -- queued | submitted | done | failed
    request        jsonb,
    batch_id       text,
    output         jsonb,                             -- a sync answer (the schema's fields)
    result         jsonb,                             -- a batched answer: {text, stop_reason, model}
    error          text,
    usd            numeric NOT NULL DEFAULT 0,
    shadowed       boolean NOT NULL DEFAULT false,
    agent          text,
    creator_id     uuid,
    created_at     timestamptz NOT NULL DEFAULT now(),
    submitted_at   timestamptz,
    done_at        timestamptz
);
CREATE INDEX IF NOT EXISTS llm_journal_job_idx ON llm_journal (job_id, attempt, role, schema_name);
CREATE INDEX IF NOT EXISTS llm_journal_open_idx ON llm_journal (status) WHERE status IN ('queued', 'submitted');
CREATE INDEX IF NOT EXISTS llm_journal_created_idx ON llm_journal (created_at);

CREATE TABLE IF NOT EXISTS llm_batches (
    id             text PRIMARY KEY,                  -- Anthropic's batch id
    status         text NOT NULL DEFAULT 'submitted', -- submitted | collected
    requests       int NOT NULL,
    counts         jsonb,
    submitted_at   timestamptz NOT NULL DEFAULT now(),
    checked_at     timestamptz,
    collected_at   timestamptz
);

-- The request a shadow test compared (system prompt and prompt text, cut to llm.shadow_request_chars), so a judge can
-- say which of two open-ended answers does the job better (model selection's preference mode).
ALTER TABLE llm_shadow ADD COLUMN IF NOT EXISTS request text;
