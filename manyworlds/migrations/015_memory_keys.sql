-- Single-valued creator facts (home, partner, job) carry a key: a newer fact with the same key
-- closes the older one, so the showrunner never sees two homes.
ALTER TABLE creator_memory ADD COLUMN key text;
CREATE INDEX creator_memory_key_idx ON creator_memory (creator_id, key) WHERE valid_until IS NULL AND key IS NOT NULL;
