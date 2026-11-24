-- Song anatomy (spec docs/superpowers/specs/2026-11-23-music-department.md, sections 1, 2 and 4).
--
-- songs: one row per made song (a music.song job's mix, its stems and the spec it was made from). A revision is a
-- new row whose parent_id names the song it revises; `version` counts along that lineage (1 = the first take of an
-- idea). A song may belong to a release (single, EP, album) at a track number. creator_id is null for a benchmark
-- song (no artist). Asset ids and uris are copied, not referenced: storage retention owns the assets table.
CREATE TABLE releases (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id  uuid NOT NULL REFERENCES creators(id),
    kind        text NOT NULL CHECK (kind IN ('single', 'ep', 'album')),
    title       text NOT NULL,
    status      text NOT NULL DEFAULT 'planned' CHECK (status IN ('planned', 'released', 'dropped')),
    released_at timestamptz,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE songs (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id    uuid REFERENCES creators(id),
    title         text NOT NULL DEFAULT '',
    version       int  NOT NULL DEFAULT 1,
    parent_id     uuid REFERENCES songs(id),
    release_id    uuid REFERENCES releases(id),
    track_number  int,
    genre         text,
    subgenre      text,
    brief         jsonb NOT NULL DEFAULT '{}',   -- concept, theme, pov, mood, reference pattern, target listener
    spec          jsonb NOT NULL DEFAULT '{}',   -- what the engine was asked (manyworlds.music: tags, bpm, key...)
    lyrics        text NOT NULL DEFAULT '',
    job_id        uuid UNIQUE REFERENCES jobs(id),
    batch         text,
    mix_asset_id  uuid,
    mix_uri       text,
    vocal_uri     text,
    instrumental_uri text,
    worker_check  jsonb NOT NULL DEFAULT '{}',   -- the worker's objective check (lyric line coverage, length, LUFS)
    status        text NOT NULL DEFAULT 'made' CHECK (status IN ('made', 'shipped', 'dropped')),
    measure_error text,
    measure_errors int NOT NULL DEFAULT 0,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (release_id, track_number)
);
CREATE INDEX songs_creator_idx ON songs (creator_id, created_at);
CREATE INDEX songs_genre_idx ON songs (genre);

-- The measured anatomy, versioned: `version` is the measurement code's (manyworlds.song_measure.VERSION), so a
-- change to how a feature is measured adds a row instead of rewriting history. features: flat {name: value} for
-- the names in policy music.features; detail: sections, energy curve, the key's candidates.
CREATE TABLE song_features (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    song_id        uuid NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    version        text NOT NULL,
    policy_version text,
    features       jsonb NOT NULL DEFAULT '{}',
    detail         jsonb NOT NULL DEFAULT '{}',
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (song_id, version)
);

-- The critic's structured assessment (brain/agents/music_critic.py), and its fixes addressed to the roles.
CREATE TABLE song_assessments (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    song_id     uuid NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    decision_id uuid REFERENCES decisions(id),
    verdict     text NOT NULL CHECK (verdict IN ('ship', 'revise', 'rethink')),
    assessment  jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (decision_id)
);
CREATE INDEX song_assessments_song_idx ON song_assessments (song_id, created_at);

CREATE TABLE song_feedback (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    song_id       uuid NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    assessment_id uuid NOT NULL REFERENCES song_assessments(id) ON DELETE CASCADE,
    role          text NOT NULL,
    problem       text NOT NULL,
    change        text NOT NULL,
    evidence      jsonb NOT NULL DEFAULT '[]',
    status        text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done', 'declined')),
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX song_feedback_role_idx ON song_feedback (role, status);

-- Posts that use a song (a music video, a post with the song under it).
ALTER TABLE posts ADD COLUMN song_id uuid REFERENCES songs(id);
CREATE INDEX posts_song_idx ON posts (song_id) WHERE song_id IS NOT NULL;

-- Findings carry their domain (content | music) and, for genre-wide findings (scope 'genre'), the genre.
ALTER TABLE findings ADD COLUMN domain text NOT NULL DEFAULT 'content' CHECK (domain IN ('content', 'music'));
ALTER TABLE findings ADD COLUMN genre text;
ALTER TABLE simulated_findings ADD COLUMN domain text NOT NULL DEFAULT 'content' CHECK (domain IN ('content', 'music'));
ALTER TABLE simulated_findings ADD COLUMN genre text;
DO $$
DECLARE c record;
BEGIN
    FOR c IN SELECT conrelid::regclass AS t, conname FROM pg_constraint
             WHERE conrelid IN ('findings'::regclass, 'simulated_findings'::regclass) AND contype = 'c'
               AND pg_get_constraintdef(oid) LIKE '%scope%'
    LOOP
        EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', c.t, c.conname);
    END LOOP;
END $$;
ALTER TABLE findings ADD CONSTRAINT findings_scope_check
    CHECK (scope IN ('post', 'creator', 'cluster', 'market', 'portfolio', 'genre'));
ALTER TABLE simulated_findings ADD CONSTRAINT simulated_findings_scope_check
    CHECK (scope IN ('post', 'creator', 'cluster', 'market', 'portfolio', 'genre'));

-- Predictions about one song (metrics song_median_reach, song_reach_lift: manyworlds.brain.predictions).
ALTER TABLE predictions ADD COLUMN song_id uuid REFERENCES songs(id);
