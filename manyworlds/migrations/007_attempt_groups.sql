-- Which machine ran each attempt, so cost per accepted output and machine failures can be
-- scored per pool (manyworlds.deploy.scoring) from what each machine billed and produced.
ALTER TABLE attempts ADD COLUMN gpu_group text;
CREATE INDEX attempts_group_idx ON attempts (gpu_group);

-- Earlier attempts: the machine in the same pool that was up when the attempt started.
-- One machine per pool per kind ran at a time, so this match is unambiguous.
UPDATE attempts a SET gpu_group = g.name
FROM gpu_groups g
WHERE a.gpu_group IS NULL AND a.pool = g.pool
  AND a.started_at >= g.started_at AND a.started_at <= coalesce(g.stopped_at, now());
