# GPU factory audit (2026-11-25)

The owner's direction (2026-11-25): run Manyworlds as a high-throughput production factory, not individually
provisioned experiments. This note checks each principle against the code on `main` (b5f18b2), measures the last 72 h
of production data (read-only SELECTs on Neon, 2026-11-25), and ranks the fixes. Nothing was rented or run for it.
**E** marks an estimate.

## 1. Principles against the code

| Principle | Exists (file:function) | Partial | Missing |
|---|---|---|---|
| Persistent weights | RunPod network volume, 150 GB in US-NC-1 (`gpu_pools.runpod_secure_l40s_vol`; `autoscale.launch_one` points `MW_MODEL_DIR`/`HF_HOME` at it); boot prune and free-space floor (`worker/volume.check`, `autoscale.volume_low`, fallback to the plain pool); R2 mirror (`worker/models.ensure`, `image.mirror`) | the volume fix is in talkcomfy v24 but **no machine has recorded a reading yet** (`gpu_groups.volume_*` empty); tts is pinned off the volume; image weights come from R2/HF on every boot | a volume per data centre (one DC, LOW L40S stock); pre-seeding; small models baked into images |
| Warm workers | enforcer counts a machine busy while any job of its kinds is open (`enforce.tick`); idle stop 5 min (RunPod/Vast), 2 min with no demand (`pull_forward.demand`); `pull_forward` fills a warm image machine in live mode | the worker's own idle self-stop (`runner.Worker.self_stop_reason`) is demand-blind: 5 idle minutes and it destroys its machine | a keep-warm hold for soon-due work of the loaded model (needs a worker rebuild) |
| Model-affinity scheduling | claim prefers the loaded family (`queue.claim(prefer=...)`, `runner.Worker.run`); family switch frees the engine (`runner._switch_to`); `min_vram_gb` (`runner.kinds_that_fit`); batch wait for production (`autoscale.batch_wait`); merged machines for tests (`autoscale.merge_groups`) | merged machines exist only in the test overlay; production unpinned work (`pools IS NULL`) already shares any warm machine across creators and posts | pinned test work could not use a warm machine in another pool (**fixed here**: `route_to_warm`) |
| Hardware allocation | pools ranked by measured $/accepted output (`scoring.launch_order`); per-kind pools and VRAM floors; Vast bandwidth filter | one machine per merged test plan serialises voice, clips and lip sync (story clips waited p50 87 min) | per-family placement inside a run is by pin, not by measured throughput |
| Concurrency | brain, producer, reviewer, hosted, finisher are separate Fly processes; GPU workers on rented hosts talk to Neon directly | the daily GPU cap stops machines mid-job (3 in 72 h, each losing its attempt); `breadth.refuse_beside_open_tests` serialises test runs (relaxed for CPU kinds 2026-11-25) | a cap that drains in-flight work instead of killing it |
| Independent deployments | GPU workers do not depend on Fly: deploys never touch them (confirmed: they hold only `WORKER_DATABASE_URL` and storage); durable queue with leases and reaping; finisher deploy grace | post leases (30 min) are released by `finally` on SIGINT (Fly's default stop signal); a brain job killed by a deploy waited out its 10 min lease | the breadth driver was a shell loop in /tmp (**fixed here**: a durable brain job); brain jobs handed back on shutdown (**fixed here**) |
| Pipelining | lead_hours planning, `pull_forward`, wakeups, finish on the finisher, LLM batches | test runs drive one stage at a time per post | — |
| Measurement | `report.throughput`: boot/busy/idle $, $/accepted, accepted/GPU-h, first-pass rate, retries; `report.machines`; `scoring` | download, model load and provisioning were not split; queue latency and idle tail not reported (**added here**) | autoscaler decisions (batching, cooldown, no capacity) were stdout only; `no_capacity` now an owner event |

## 2. Measured, last 72 h (machines started 2026-11-22..25, almost all test and benchmark work)

47 machines, **$17.87 billed** (ledger `gpu_billed` $18.23; Claude $5.26; Apify $4.04). Split by time:
boot **$2.18 (12%)**, idle **$2.29 (13%)**, busy $13.40 (75%). Busy hides the first job's weights download and model
load: first jobs ran 2-6 min longer than warm ones (**E** $0.95, 5%). **$2.23 (12%)** was billed by 11 machines that
finished nothing. Busy share of working machines: 65%. Three machines (**$3.72**) were stopped by the $10 daily GPU cap
with a job in flight (11-22 and 11-24 both hit the cap).

| family / pool | machines | billed $ | boot p50 min | idle tail p50 min | busy share | done / attempts | accepted | $/accepted |
|---|---|---|---|---|---|---|---|---|
| talkcomfy (merged) / l40s | 9 | 11.40 | 4.3 (max 18) | 4.9 | 82% | 113 / 171 | 947 | 0.012 |
| talk / l40s_vol | 3 | 1.62 | 2.3 | 4.3 | 84% | 13 / 23 | 73 | 0.022 |
| comfy / l40s_vol | 3 | 1.53 | 2.3 | — | 83% | 0 / 14 | 0 | — |
| music / l40s | 3 | 1.03 | 3.7 | 4.6 | 40% | 12 / 26 | 44 | 0.023 |
| image / vast_3090 | 5 | 0.58 | 1.6 | 5.1 | 71% | 56 / 56 | 194 | 0.003 |
| edit / l40s | 1 | 0.30 | 0.8 | 5.5 | 62% | 2 / 2 | 8 | 0.038 |
| voice / l40s | 1 | 0.26 | 2.7 | 5.0 | 23% | 2 / 2 | 5 | 0.052 |
| talkcomfy / l40s_vol | 3 | 0.26 | 3.8 | 0.2 | 2% | 0 / 9 | 0 | — |
| other (image 4090/3090, talk 4090, tts, music 4090) | 10 | 0.84 | 0.6-14 | 1.6-6.7 | 28-59% | | | |
| finish / fly_finisher | 9 | 0.06 | 0.1 | 3.0 | 24% | 12 / 12 | — | — |

Cold first job vs warm (median job seconds, done attempts): Z-Image 385 vs 66 (≈5.3 min of weights and load per image
machine), InfiniteTalk 402 vs 174, voice.speak 129 vs 11, Wan 2.2 I2V 244 vs 120, comfy.graph 332 vs 72 (its own
`bench.download_seconds` max 280 s), music.song 101 vs 21, Qwen edit 467 vs 151.

Queue latency (job created → first attempt; p50 / p90 min): talk 14 / 50, Z-Image 28 / 58, voice.speak 20 / 74,
Wan I2V **87 / 145**, Wan-Animate 21 / 40, comfy.graph 14 / 43, music.song 7 / 8, finish 0.9 / 1.5. 39% of story
clips waited over an hour and 22% of talk jobs over 30 min; in part the daily cap (no launches until 00:00 UTC), in part one
merged machine running voice, clips and lip sync one job at a time.

Sharing: 10 of 47 boots started while a machine on the same worker image was up (6 in the same pool); 6 boots came
within 15 min after a same-pool, same-image machine stopped (their boots alone $0.57). Jobs per machine: talkcomfy 8.2,
image 7, talk 3.3, music 3.3, voice 2, finish 1.3. Machines ran 11 different talkcomfy tags in 72 h, so warm reuse
across image rebuilds is impossible by construction; weights on a volume survive rebuilds.

**Not recorded before this change**: provisioning (provider start + image pull) apart from weights download and model
load (now `gpu_groups.worker_ready_at`, written by workers built from this branch; the cold-start premium is estimated
from first vs warm jobs); bytes downloaded per boot; autoscaler decisions (batching, cooldown, no capacity: stdout
only; `no_capacity` is now an owner event); the volume's free space (column exists, no reading yet).

## 3. Fixes ranked by $ and wall-clock saved per week (at today's test volume, ×7/3 of the window)

| # | Fix | Reuses | $/wk (**E**) | Wall-clock | Code | Status |
|---|---|---|---|---|---|---|
| 1 | (a) breadth driver as a durable brain job: one step per job run, deferred between steps, handed back on shutdown | `breadth.run(once)`, brain runtime, `queue.defer`, leases | 0.5-1 (a killed driver strands a warm machine to idle out and reboot) | hours per deploy-killed run | S | **done** |
| 2 | (f) no capacity: owner event after 15 min; pinned test work widened to the same card's other pools after 30 min | `scale_up`, `notify.owner_event` | 0 | up to an hour per silent wait | S | **done** |
| 3 | (g) factory clock in `report.throughput`: provisioning, boot, cold start, idle tail, queue latency, jobs/machine | `report.throughput`, `mw machines` | enabling | — | S | **done** (provisioning after a worker rebuild) |
| 4 | (c) warm reuse: pinned test work goes to a running machine with its kind on the same card before a new rental; benchmarks keep pins | `scale_up`, `queue.claim` pools rule | ~4.5 (≈10 boots/72 h × boot + first-job load + tail) | 5-10 min per avoided boot | S | **done** |
| 5 | (d) verify the volume fix live (first talkcomfy v24 machine on the volume must record `volume_free_gb`); then point tts at the volume and add a second volume in another DC | `worker/volume`, `volume_low` | ~4 (volume machines with no output: $1.79/72 h) | boots that die at first download | policy + one GPU run | code **done** on `feat/gpu-factory-2` (tts on the volume first, headroom check, volume order, second-volume entry not created); live check needs a run |
| 6 | daily cap drains instead of killing: stop launches at the cap, let in-flight attempts finish up to an overrun allowance, stop machines between jobs | `enforce.tick`, `over_budget` | ~1.5 (3 lost attempts + next-day reboots per 72 h) | a lost render per cap hit | M (needs workers to stop claiming: rebuild) | **done** on `feat/gpu-factory-2` (enforcer side live at deploy; workers stop claiming after a rebuild) |
| 7 | (e) demand-aware hold: the worker's self-stop keeps its machine while queued or soon-due work for its loaded kinds exists (an autoscaler-set `keep_warm_until`) | `pull_forward.demand`, enforcer `busy` | ~2.4 (6 re-boots within 15 min per 72 h) minus extra idle | 5-10 min per avoided boot | M (migration + worker rebuild) | **done** on `feat/gpu-factory-2` (enforcer side live at deploy; worker self-stop after a rebuild) |
| 8 | bake small models into images or seed them on a volume (Z-Image ≈5 min per image boot, voice ≈2 min) | `models.ensure`, image build | ~0.4 | ~5 min per image boot | M | open |
| 9 | relax `breadth.refuse_beside_open_tests` once warm reuse is proven; split `run_spend` so shared machines are not double counted | `breadth.open_test_work` | — | runs no longer queue behind each other | S | open |
| 10 | RunPod Serverless for talkcomfy (prior note, rec. 1) | `deploy/serverless.py` | largest at volume | no boots | L (owner: endpoint) | open |

**E**: weekly $ = 72 h figures × 7/3. At production volume the boot and idle shares fall as machines run longer, and
the queue-latency and warm-reuse fixes matter more than the dollars here suggest.

## 4. What changed on `feat/gpu-factory`

- `mw test breadth --queue` (`breadth.enqueue_driver`, agent `brain.breadth`, `brain/agents/breadth_driver.py`): the
  run is a test job the paused brain worker drives one step at a time (`agent.Defer` → `queue.defer`, no attempt
  charged, progress on `jobs.result`). A step past the deadline or cap ends the run (`--once` now checks them too).
- `brain/runtime`: a shutdown (SIGINT/SystemExit) mid-job hands the job back at once.
- `deploy/autoscale.route_to_warm`, `capacity_fallback`; policy `autoscale.hard_pin_batches`, `warm_reuse`,
  `capacity`; `notify.priority.no_capacity`.
- `deploy/report.throughput`: `provision_minutes_p50`, `boot_minutes_p50`, `cold_start_seconds_p50`,
  `tail_minutes_p50`, `queue_minutes_p50/p90`, `jobs_per_machine`; migration 042 `gpu_groups.worker_ready_at`, written
  by `worker/runner.mark_ready`.

## 5. What changed on `feat/gpu-factory-2`

- Warm hold (e): `deploy/autoscale.warm_holds` sets `gpu_groups.keep_warm_until` (migration 043) every tick from work due
  for a machine's kinds (`warm_demand`: a queued job it may claim, a producing post whose format needs its families,
  a planned post about to start or within pull-forward's horizon), from its last job's end plus
  `autoscale.warm_hold.max_hold_minutes`, capped by deadline and spend cap, off while draining or at the daily cap.
  The enforcer's idle stop and the worker's idle self-stop (`runner.Worker.self_stop_reason`) wait for it.
- Daily cap drains (6): `deploy/enforce.drain` marks a machine with a job in flight `draining_since` (migration 043);
  its worker claims nothing new (`runner.Worker.draining`) and self-stops between jobs; the enforcer stops it when the
  job ends or at `gpu_runs.daily_drain_grace_minutes` / `daily_drain_max_overshoot_usd`. Idle machines stop at once.
- Volumes (d): tts pinned to the volume first in the test overlay (default `job_kinds` already were);
  `autoscale.volume_skip` keeps a family off a volume whose last reading leaves less than `min_free_gb` plus
  `network_volume.headroom_gb_by_family` unless that family has finished a job there; `volume_order` tries volumes
  holding the family's weights first, then by measured free space; pool `runpod_secure_l40s_vol2` is a policy entry
  with `id: null` (skipped, `volume_not_created`) until the owner creates a second volume.
- Image churn: not solved in code (a machine on an older tag serving new work risks running stale handlers);
  `mw machines` now shows each machine's worker image so the churn is visible. Fewer rebuilds of the talkcomfy image,
  or weights on the volume (which survive rebuilds), are the remedy.
