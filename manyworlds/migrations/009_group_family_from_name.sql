-- SaladCloud groups from the manual launcher recorded no job kinds, so 008 could not give
-- them a family. Their names are mw-<family>-<gpu>[-...]; without a family, pool scoring
-- saw no history for those pools and gave them fresh trials they had already failed.
UPDATE gpu_groups SET family = split_part(name, '-', 2)
WHERE family IS NULL AND name LIKE 'mw-%-%'
  AND split_part(name, '-', 2) IN ('image', 'video', 'tts', 'train');
