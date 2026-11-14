-- Day 3: planned posts carry their scenes and schedule, and the jobs that make them.

-- Machines registered before gpu_groups.family was set (SaladCloud's manual launcher):
-- their family is the family of the job kinds they ran, so pool scoring can count them.
UPDATE gpu_groups SET family = split_part(job_kinds[1], '.', 1)
WHERE family IS NULL AND cardinality(job_kinds) > 0;

-- posts.status: planned -> producing -> ready (awaiting approval) -> approved -> published,
-- or dropped when repairs run out. body: scenes, hashtags, canon, chosen images per scene.
ALTER TABLE posts ADD COLUMN body jsonb NOT NULL DEFAULT '{}';
ALTER TABLE posts ADD COLUMN scheduled_for timestamptz;
ALTER TABLE posts ADD COLUMN updated_at timestamptz NOT NULL DEFAULT now();
CREATE INDEX posts_status_idx ON posts (status, scheduled_for);

ALTER TABLE jobs ADD COLUMN post_id uuid REFERENCES posts(id);
CREATE INDEX jobs_post_idx ON jobs (post_id);
