-- Series continuity (manyworlds.brain.series, 2026-11-23): each series' episodes and its open threads.
--
-- An episode enters when its post is published (or simulated-published), never when it is scheduled: a dropped post
-- never becomes an episode. `number` is the episode's place among the series' published episodes. Threads are what an
-- episode leaves open (a promised part 2, a running joke, a question asked to the audience); a later published episode
-- closes them by id. A thread with series '' belongs to the creator, not one series.
CREATE TABLE series_episodes (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id   uuid NOT NULL REFERENCES creators(id),
    series       text NOT NULL,
    number       int  NOT NULL,
    post_id      uuid NOT NULL UNIQUE REFERENCES posts(id),
    premise      text NOT NULL DEFAULT '',
    simulated    boolean NOT NULL DEFAULT false,
    published_at timestamptz NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (creator_id, series, number)
);

CREATE TABLE series_threads (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id  uuid NOT NULL REFERENCES creators(id),
    series      text NOT NULL DEFAULT '',
    text        text NOT NULL,
    opened_post uuid NOT NULL REFERENCES posts(id),
    closed_post uuid REFERENCES posts(id),
    opened_at   timestamptz NOT NULL,
    closed_at   timestamptz,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (opened_post, text)
);
CREATE INDEX series_threads_open_idx ON series_threads (creator_id) WHERE closed_at IS NULL;
