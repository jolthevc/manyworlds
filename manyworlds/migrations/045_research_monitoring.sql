-- Cost-aware monitoring of the reference creator universe (manyworlds.research.monitor; docs/research/
-- 2026-11-25-research-providers.md). Each creator has a tier (hot, warm, cold, dormant) from relevance, growth,
-- performance and intelligence value, and is checked on its tier's cadence with cheap metadata only. A check whose
-- newest post ids hash to the same content_hash found nothing new. Posts carry a content hash (caption, media type,
-- media file names) and `selected` (why deterministic filters picked them for vision: outperformer, new format, new
-- tags, commercial); only selected posts get their video downloaded and a vision read. An annotation remembers the
-- content hash it read, so unchanged content is never analysed twice.
ALTER TABLE reference_creators ADD COLUMN tier text NOT NULL DEFAULT 'cold'
    CHECK (tier IN ('hot', 'warm', 'cold', 'dormant'));
ALTER TABLE reference_creators ADD COLUMN tier_scores jsonb NOT NULL DEFAULT '{}';
ALTER TABLE reference_creators ADD COLUMN next_check_at timestamptz;
ALTER TABLE reference_creators ADD COLUMN last_checked_at timestamptz;
ALTER TABLE reference_creators ADD COLUMN last_post_id text;
ALTER TABLE reference_creators ADD COLUMN content_hash text;
ALTER TABLE reference_creators ADD COLUMN check_failures int NOT NULL DEFAULT 0;
CREATE INDEX reference_creators_due_idx ON reference_creators (next_check_at) WHERE next_check_at IS NOT NULL;

ALTER TABLE reference_posts ADD COLUMN content_hash text;
ALTER TABLE reference_posts ADD COLUMN selected jsonb;          -- {at, reasons: [...]}; NULL: not selected
CREATE INDEX reference_posts_selected_idx ON reference_posts (creator_id) WHERE selected IS NOT NULL;

ALTER TABLE reference_annotations ADD COLUMN content_hash text;
