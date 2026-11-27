-- The task an LLM call performed (manyworlds.llm as_task; owner 2026-11-27: the agent is a responsibility, the
-- model a replaceable resource per task). With provider (the model) and job_id, it joins every call to what came
-- of it downstream: acceptance, rerenders, cost per accepted post and, once published, audience outcomes. Also
-- `route`: how the model was chosen (role, task, override, split), so split-routed evidence is told apart.
ALTER TABLE cost_entries ADD COLUMN task text;
ALTER TABLE cost_entries ADD COLUMN route text;
CREATE INDEX cost_entries_task_idx ON cost_entries (task, provider) WHERE task IS NOT NULL;
