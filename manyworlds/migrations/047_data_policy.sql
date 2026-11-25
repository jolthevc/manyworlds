-- Data policy (owner 2026-11-25; manyworlds.research.data_policy). Caption text is amber: what the learning loops need
-- from it (size, hashtags, @mentions, calls to action, disclosure) is kept as caption_features, and the text itself
-- goes once read. Third-party media is held only until read, 14 days at most (was 180): existing purge dates shortened.
ALTER TABLE reference_posts ADD COLUMN caption_features jsonb NOT NULL DEFAULT '{}';
UPDATE reference_posts SET media_purge_after = least(media_purge_after, now() + interval '14 days')
 WHERE media_purge_after IS NOT NULL;
