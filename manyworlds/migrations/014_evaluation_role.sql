-- Which evaluator produced a row: the volume evaluator or the strong-model check. Model names can
-- change or coincide, so the role is recorded explicitly; time series read role = 'evaluator'.
ALTER TABLE evaluations ADD COLUMN role text NOT NULL DEFAULT 'evaluator'
    CHECK (role IN ('evaluator', 'evaluator_check'));
CREATE INDEX evaluations_asset_idx ON evaluations (asset_id);
