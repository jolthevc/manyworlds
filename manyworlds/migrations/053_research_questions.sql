-- Research questions the Brain asks (manyworlds.brain.research_questions; docs/research/2026-11-26-recipes.md).
-- Pre-registered: `spec` (the tests, filters and whether a screen runs) is stored before anything is tested, so the
-- family of tests a result is corrected over is known in advance. kind: data (the recipe analysis on annotated
-- research posts), web (the research desk) or both. status: queued (waiting, or held by the daily budget),
-- answered, unanswerable (no testable spec), failed (an error, in result). policy_version: the policy it was filed,
-- then answered, under.
CREATE TABLE research_questions (
    id             bigserial PRIMARY KEY,
    requester      text NOT NULL,
    question       text NOT NULL,
    spec           jsonb,
    kind           text NOT NULL DEFAULT 'data' CHECK (kind IN ('data', 'web', 'both')),
    status         text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'answered', 'unanswerable', 'failed')),
    result         jsonb,
    policy_version text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    answered_at    timestamptz
);
CREATE INDEX research_questions_status_idx ON research_questions (status, created_at);
CREATE INDEX research_questions_requester_idx ON research_questions (requester, answered_at DESC);
