-- Machine faults (GPU out of memory, broken CUDA) don't use up a job's attempts, but a job
-- that faults on every machine it touches must still stop eventually.
ALTER TABLE jobs ADD COLUMN worker_faults int NOT NULL DEFAULT 0;
