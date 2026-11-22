-- Instagram intake and comments (manyworlds.publish.instagram, comments).
-- connected_at: when the account was first connected; the posting ramp counts its age from here.
ALTER TABLE accounts ADD COLUMN connected_at timestamptz;
UPDATE accounts SET connected_at = updated_at WHERE status = 'connected';

-- When a publication's comments were last read (polling is spread out per policy comments.poll_minutes).
ALTER TABLE publications ADD COLUMN comments_polled_at timestamptz;

-- Top-level comments on our own posts and what the creator did about them. Commenters are kept only as salted
-- hashes; comment text is cleared after comments.keep_days. reply_sent_at is set before the reply is sent, so a
-- crash after sending only reconciles (the reply is looked up), never replies twice.
CREATE TABLE comments (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    platform          text NOT NULL,
    external_id       text NOT NULL,
    publication_id    uuid NOT NULL REFERENCES publications(id),
    creator_id        uuid NOT NULL REFERENCES creators(id),
    author_hash       text,
    text              text,
    commented_at      timestamptz,
    status            text NOT NULL DEFAULT 'new'
                      CHECK (status IN ('new', 'drafted', 'replying', 'replied', 'skipped', 'failed')),
    reply_text        text,
    reply_external_id text,
    reason            text,
    reply_sent_at     timestamptz,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (platform, external_id)
);
CREATE INDEX comments_creator_status ON comments (creator_id, status);
