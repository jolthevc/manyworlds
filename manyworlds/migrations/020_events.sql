-- Events (Brain M8): who an event is about, and a key so a detector raises each event once.
ALTER TABLE events ADD COLUMN creator_id uuid REFERENCES creators(id);
ALTER TABLE events ADD COLUMN dedupe_key text;
CREATE UNIQUE INDEX events_dedupe ON events (dedupe_key) WHERE dedupe_key IS NOT NULL;
CREATE INDEX events_creator_type ON events (creator_id, type, created_at DESC);
