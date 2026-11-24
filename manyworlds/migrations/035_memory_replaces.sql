-- Canon retracted with its post (manyworlds.brain.memory.retract_post, 2026-11-23).
--
-- A post's canon enters creator memory when the post is scheduled; when the post is later dropped, that canon is
-- closed again (valid_until). A keyed canon closed the fact it replaced (memory.add): `replaces` names that fact,
-- so retracting the canon reopens it instead of leaving the key with no open fact at all.
ALTER TABLE creator_memory ADD COLUMN IF NOT EXISTS replaces uuid REFERENCES creator_memory(id);
CREATE INDEX IF NOT EXISTS creator_memory_source_post_idx ON creator_memory (source_post) WHERE source_post IS NOT NULL;
