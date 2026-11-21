-- Owner notifications (manyworlds/notify.py): each owner event and pending approval is pushed once.
ALTER TABLE events ADD COLUMN notified_at timestamptz;
ALTER TABLE approvals ADD COLUMN notified_at timestamptz;
CREATE INDEX events_unnotified ON events (created_at) WHERE routed_to = 'owner' AND notified_at IS NULL;
CREATE INDEX approvals_unnotified ON approvals (created_at) WHERE status = 'pending' AND notified_at IS NULL;
