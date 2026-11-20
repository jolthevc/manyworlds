-- A machine's limit for one job attempt, recorded when it is registered (like deadline and max_usd), so
-- every enforcer applies the limit it was launched with, even one running older policy (the GitHub
-- backstop runs main's). NULL for machines registered before this column: enforcers fall back to policy.
ALTER TABLE gpu_groups ADD COLUMN max_attempt_minutes numeric;
