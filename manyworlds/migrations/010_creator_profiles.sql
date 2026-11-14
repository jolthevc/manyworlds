-- Every creator is a versioned profile: identity (fixed), look, persona, audience,
-- content (pillars, video series, arc), commercialization and dials. Each change is a
-- new version with who made it and why; posts record the version they were made under,
-- so the learning loop can tell which settings performed.
CREATE TABLE creator_profiles (
    creator_id uuid NOT NULL REFERENCES creators(id),
    version    int NOT NULL,
    body       jsonb NOT NULL,
    author     text NOT NULL,              -- brain | human | migration
    reason     text NOT NULL DEFAULT '',
    policy_version text,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (creator_id, version)
);
ALTER TABLE creators ADD COLUMN profile_version int;
ALTER TABLE posts ADD COLUMN profile_version int;
