-- The research desk (manyworlds.research.desk, owner 2026-11-25): findings from Claude's web search/fetch, reusable
-- across agents. Kept: question, our synthesis, classified claims with source URLs, sources (URL, title, cited
-- snippet <= 150 chars). Never page content. The GDELT/RSS culture reader it replaces is removed with its tables.
CREATE TABLE research_findings (
    id             bigserial PRIMARY KEY,
    key            text NOT NULL,
    question       text NOT NULL,
    depth          text NOT NULL CHECK (depth IN ('quick', 'deep')),
    requester      text NOT NULL,
    tags           text[] NOT NULL DEFAULT '{}',
    summary        text NOT NULL,
    claims         jsonb NOT NULL DEFAULT '[]',
    implications   jsonb NOT NULL DEFAULT '{}',
    open_questions jsonb NOT NULL DEFAULT '[]',
    sources        jsonb NOT NULL DEFAULT '[]',
    model          text,
    searches       int NOT NULL DEFAULT 0,
    fetches        int NOT NULL DEFAULT 0,
    created_at     timestamptz NOT NULL DEFAULT now(),
    fresh_until    timestamptz NOT NULL
);
CREATE INDEX research_findings_key_idx ON research_findings (key, created_at DESC);
CREATE INDEX research_findings_fresh_idx ON research_findings (fresh_until);

DROP TABLE IF EXISTS culture_signals;
DROP TABLE IF EXISTS culture_items;
