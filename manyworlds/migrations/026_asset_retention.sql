-- Storage retention (manyworlds.retention): a rejected asset's file is deleted after the retention window; the row,
-- its recipe and its reviews stay (provenance), marked with when the file went.
ALTER TABLE assets ADD COLUMN deleted_at timestamptz;
CREATE INDEX assets_retention_idx ON assets (created_at) WHERE deleted_at IS NULL;
