-- The music department's roles and portfolio (spec 2026-11-23-music-department.md, sections 3 and 6).
--
-- creators.vertical: a creator is a content creator or a music artist (an artist's posts are mostly songs). Same
-- tables, same portfolio; the strategist sees the vertical and head_of_music runs the music share.
ALTER TABLE creators ADD COLUMN vertical text NOT NULL DEFAULT 'content' CHECK (vertical IN ('content', 'music'));

-- A song made as one of several candidate takes (the melody role picks one; the rest are dropped), and the
-- experiment arm it was assigned (brain.experiments: music experiments are slotted per release batch).
ALTER TABLE songs ADD COLUMN take int;
ALTER TABLE songs ADD COLUMN pipeline_key text;          -- the department run and song it belongs to
ALTER TABLE songs ADD COLUMN experiment_id uuid REFERENCES experiments(id);
ALTER TABLE songs ADD COLUMN arm text;
CREATE INDEX songs_experiment_idx ON songs (experiment_id) WHERE experiment_id IS NOT NULL;
