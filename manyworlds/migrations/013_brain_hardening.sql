-- Simulated findings get their own table, like simulated metrics: the real tables can never hold
-- simulated rows, so no query on them (any JOIN, any module) can mistake simulation for demand.
-- Only manyworlds/brain/evidence.py touches simulated_* tables (enforced by a test).
CREATE TABLE simulated_findings (LIKE findings INCLUDING DEFAULTS INCLUDING CONSTRAINTS);
ALTER TABLE simulated_findings ADD PRIMARY KEY (id);
ALTER TABLE simulated_findings ADD CONSTRAINT simulated_findings_creator_fk FOREIGN KEY (creator_id) REFERENCES creators(id);
ALTER TABLE simulated_findings ADD CONSTRAINT simulated_findings_only CHECK (evidence = 'simulated');
INSERT INTO simulated_findings SELECT * FROM findings WHERE evidence = 'simulated';
DELETE FROM findings WHERE evidence = 'simulated';
ALTER TABLE findings ADD CONSTRAINT findings_real_only CHECK (evidence <> 'simulated');

-- One override version per change, so "+o<n>" names exactly one effective policy.
ALTER TABLE policy_overrides ADD CONSTRAINT policy_overrides_version_uq UNIQUE (version);

-- A publication is simulated exactly when it went to the simulated platform.
ALTER TABLE publications ADD CONSTRAINT publications_simulated_platform CHECK (simulated = (platform = 'simulated'));
