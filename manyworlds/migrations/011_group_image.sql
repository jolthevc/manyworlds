-- The worker image (name:tag) each machine ran. A pool is paused after machines in a row
-- produce nothing; when that was our software's fault, a new image should get a fresh
-- chance, so the pause only counts machines on the current image.
ALTER TABLE gpu_groups ADD COLUMN image text;
