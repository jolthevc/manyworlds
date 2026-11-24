-- One singing voice per artist (manyworlds.creator.music_voice, spec docs/superpowers/specs/2026-11-23-music-department.md
-- section 5). Each row is one version of an artist's voice asset: the converter it is made for (soulx_svc: a prompt
-- clip, zero-shot; rvc: a model trained on the artist's own sung clips), the centroid of its speaker embeddings over
-- voiced windows (what every song's converted vocal is scored against), the prompt clip, and for rvc the model and its
-- retrieval index. Versions only grow; the creator's current one is copied into creators.anchors.music_voice (with its
-- version), as anchors.voice holds the spoken voice. Original voices only: every asset starts from one of our own
-- ACE-Step vocals (source.seed_job), never from a real singer's recording.
CREATE TABLE music_voices (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id      uuid NOT NULL REFERENCES creators(id),
    version         int  NOT NULL,
    adapter         text NOT NULL,                 -- soulx_svc | rvc (music_voice.ADAPTERS)
    centroid        jsonb NOT NULL DEFAULT '[]',   -- mean speaker embedding over voiced windows
    embedding_model text NOT NULL,
    prompt_uri      text NOT NULL,                 -- the reference clip (sung, voiced, a few seconds)
    model_uri       text,                          -- rvc: the trained model (.pth)
    index_uri       text,                          -- rvc: its feature index (.index)
    calibration     jsonb NOT NULL DEFAULT '{}',   -- same-artist vs other-artist spread when it was stored
    source          jsonb NOT NULL DEFAULT '{}',   -- seed job and candidate, dataset jobs, train job
    policy_version  text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (creator_id, version)
);
CREATE INDEX music_voices_creator_idx ON music_voices (creator_id, version DESC);
