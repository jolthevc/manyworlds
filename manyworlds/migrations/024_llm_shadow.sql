-- Shadow tests of candidate models (manyworlds.llm, manyworlds.shadow): the same call answered by the role's
-- model and a candidate; the candidate never changes the answer.
CREATE TABLE llm_shadow (
    id                bigserial PRIMARY KEY,
    role              text NOT NULL,
    primary_model     text NOT NULL,
    candidate_model   text NOT NULL,
    primary_output    jsonb NOT NULL,
    candidate_output  jsonb,
    candidate_error   text,
    primary_usd       numeric NOT NULL DEFAULT 0,
    candidate_usd     numeric NOT NULL DEFAULT 0,
    candidate_seconds numeric,
    agent             text,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX llm_shadow_role ON llm_shadow (role, candidate_model, created_at DESC);
