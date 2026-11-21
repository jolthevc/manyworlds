-- Model selection (manyworlds.selection): which creator a shadow call was for (so generative outputs can be judged
-- against that creator), and the evaluator's blind scores of both answers (primary and candidate).
ALTER TABLE llm_shadow ADD COLUMN creator_id uuid REFERENCES creators(id);
ALTER TABLE llm_shadow ADD COLUMN judged jsonb;   -- {"primary": {...}, "candidate": {...}} mean scores and rule breaks
