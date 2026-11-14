# Brain M1: Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The ground every Brain agent stands on: its tables, an agent contract and runtime on the existing queue, decision and cost tracing per agent, a code-enforced wall between real and simulated evidence, and runtime policy overrides bounded by authority.

**Architecture:** Agents are Python modules registered by name and run as `brain.<name>` jobs on pool `local` by a brain runtime that reuses `queue.claim/complete/fail`. Each run gets an `AgentContext` (conn, effective policy, an LLM tagged with the agent and job, the job). Findings and metrics are read only through `brain/evidence.py`, which takes a mode and never returns simulated rows in real mode. Runtime settings the Brain changes live in `policy_overrides`; `brain/overrides.py` merges them over the policy file within `brain.authority`, and anything outside becomes an `approvals` row.

**Tech Stack:** Python 3.11, psycopg 3, Pydantic 2, Postgres (Neon / local compose), pytest with stubbed Claude.

**Spec:** `docs/superpowers/specs/2026-11-14-brain-design.md`; milestone context in `docs/superpowers/plans/2026-11-14-brain-roadmap.md` (M1).

## Global Constraints

- No magic numbers in code: every budget, threshold, window and limit comes from `policy/default.yaml` (CLAUDE.md).
- Everything that costs money writes to the ledger, including failures; LLM entries carry the agent name.
- Decisions record the policy version they ran under, plus the override version (spec 6a).
- Simulated results are never treated, displayed or used as evidence of real demand (spec anti-goals, principle 2, 6a).
- Migrations are forward-only: add `012_brain.sql`, never edit an applied file.
- Tests stub Claude; no network, no spend. Run with `docker compose build mw -q && docker compose run --rm -T --entrypoint pytest mw -q`.
- Secrets only in `.env`; `accounts` stores the name of the env key, never a token.

## Review Focus

1. A brain worker dies mid-agent and the job reruns: the agent must not record a second decision for the same work (decisions keyed by job id + step).
2. An agent has spent its daily LLM budget: the job fails as non-retryable with the reason recorded, and is not retried in a loop.
3. A creator is in `instagram` mode but simulated findings exist for it (from its sandbox days): a real-mode read returns none of them.
4. An override whose value equals the authority bound exactly is allowed; one beyond it becomes a pending approval and the effective policy is unchanged.
5. A job kind `brain.<unknown>` reaches the runtime: it fails non-retryable with "no agent named …", and other jobs keep running.

Each has its test in the owning task below.

---

### Task 1: Brain schema

**Files:**
- Create: `manyworlds/migrations/012_brain.sql`
- Test: `tests/test_brain_schema.py`
- Create: `tests/brain_helpers.py` — shared test helpers: `creator(conn, *, publish_mode="simulated") -> uuid`, `make_post(conn, creator_id=None) -> uuid`. Later tasks add to it.

**Interfaces:**
- Produces: tables `accounts, creator_states, creator_memory, ideas, evaluations, rubric_versions, findings, allocations, publications, simulated_metrics, markets, clusters, convictions, budgets, approvals, events, policy_overrides`; columns `creators.publish_mode text NOT NULL DEFAULT 'simulated' CHECK (publish_mode IN ('simulated','instagram'))`, `creators.market_id uuid`, `creators.cluster_id uuid`, `cost_entries.agent text`; `experiments` gains `arms jsonb DEFAULT '[]'`, `metric text`, `min_sample int`, `result jsonb`, `updated_at timestamptz`.
- Key columns later tasks rely on:
  - `findings(id uuid pk, scope text CHECK IN ('post','creator','cluster','market','portfolio'), creator_id uuid NULL, cluster_id uuid NULL, market_id uuid NULL, statement text, evidence text NOT NULL CHECK IN ('observational','experimental','simulated','research'), grade text NOT NULL CHECK IN ('proven','tentative','speculative'), effect numeric NULL, interval_low numeric NULL, interval_high numeric NULL, n int NULL, status text DEFAULT 'open', sources jsonb DEFAULT '[]', created_at, updated_at)`
  - `decisions` gains `agent text`, `job_id uuid`, `step text`, `rationale text`, `override_version int`, and a unique index on `(job_id, step)` where job_id is not null.
  - `policy_overrides(id bigserial pk, path text NOT NULL, value jsonb NOT NULL, version int NOT NULL, author text NOT NULL, reason text NOT NULL, decision_id uuid NULL, created_at)`; effective value per path = highest version.
  - `approvals(id uuid pk, kind text, subject jsonb, reason text, amount_usd numeric NULL, status text DEFAULT 'pending' CHECK IN ('pending','approved','rejected'), requested_by text, decided_by text NULL, created_at, decided_at NULL)`
  - `simulated_metrics(id bigserial pk, post_id uuid REFERENCES posts, captured_at, post_age_s int, model text, panel_version text, metrics jsonb)`; `metric_snapshots` gains `post_age_s int` and a CHECK that `simulated = false` (real only from now on; existing rows are all false).
  - `evaluations(id uuid pk, item_kind text CHECK IN ('idea','asset','feed'), idea_id uuid NULL, asset_id uuid NULL, creator_id uuid, rubric text, rubric_version int, evaluator_model text, prompt_version text, scores jsonb, spreads jsonb, rationale jsonb, samples int, profile_version int NULL, format text NULL, created_at)`
  - Remaining tables carry the columns listed in spec 4.10; the implementer names extra columns.

- [ ] **Step 1: Write the failing tests**

```python
def test_brain_tables_exist(conn):
    names = {r["table_name"] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'").fetchall()}
    assert {"accounts", "creator_states", "creator_memory", "ideas", "evaluations", "rubric_versions",
            "findings", "allocations", "publications", "simulated_metrics", "markets", "clusters",
            "convictions", "budgets", "approvals", "events", "policy_overrides"} <= names

def test_real_metric_snapshots_cannot_hold_simulated_rows(conn):
    post = make_post(conn)                     # helper in the test file: creator + post
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("INSERT INTO metric_snapshots (post_id, simulated, metrics) VALUES (%s, true, '{}')", (post,))

def test_new_creators_start_in_simulated_mode(conn):
    cid = conn.execute("INSERT INTO creators (handle) VALUES ('x') RETURNING publish_mode").fetchone()
    assert cid["publish_mode"] == "simulated"

def test_one_decision_per_job_step(conn):
    job = queue.enqueue(conn, "brain.echo", {})
    conn.execute("INSERT INTO decisions (kind, job_id, step) VALUES ('k', %s, 's')", (job,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute("INSERT INTO decisions (kind, job_id, step) VALUES ('k', %s, 's')", (job,))
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose build mw -q && docker compose run --rm -T --entrypoint pytest mw -q tests/test_brain_schema.py`
Expected: FAIL (tables missing).

- [ ] **Step 3: Write `012_brain.sql`** with the tables and columns above. Existing `creators` rows (Maren) get `publish_mode = 'simulated'` by default; that is correct until M4 connects her account.

- [ ] **Step 4: Run the whole suite**

Run: `docker compose build mw -q && docker compose run --rm -T --entrypoint pytest mw -q`
Expected: all pass (124 existing + 4 new).

- [ ] **Step 5: Commit** `git commit -m "Brain M1: schema"`

---

### Task 2: LLM cost per agent and agent budgets

**Files:**
- Modify: `manyworlds/ledger.py` (`record` gains `agent: str | None = None`)
- Modify: `manyworlds/llm.py` (`Claude` gains `tagged(agent: str, job_id) -> contextlib.AbstractContextManager[None]`; `_charge` passes `agent` and `job_id` from the active tag)
- Create: `manyworlds/brain/budget.py`
- Modify: `policy/default.yaml` (new `brain:` section)
- Test: `tests/test_brain_budget.py`

**Interfaces:**
- Consumes: Task 1 `cost_entries.agent`.
- Produces: `budget.spent_today(conn, agent: str) -> float`; `budget.remaining(conn, policy, agent: str) -> float`; `class OverBudget(NonRetryable)`; `budget.check(conn, policy, agent: str) -> None` raises `OverBudget`.
- Policy `brain:` keys used later: `brain.llm_usd_per_day: {default: <n>, <agent>: <n>}`, `brain.runtime: {loop_seconds, max_jobs_per_tick, lease_seconds}`, `brain.authority` (Task 5). Starting values: `llm_usd_per_day.default: 2.0`, `showrunner: 3.0`, `evaluator: 3.0`, `strategist: 2.0`, `scouts: 3.0`; `runtime: {loop_seconds: 30, max_jobs_per_tick: 10, lease_seconds: 600}` (builder's call; inside the $750/month master budget).

- [ ] **Step 1: Failing tests**

```python
def test_llm_cost_is_recorded_under_the_agent_and_job(conn, policy):
    llm = Claude(conn, policy, client=FakeAnthropic(input_tokens=1000, output_tokens=100))
    job = queue.enqueue(conn, "brain.echo", {})
    with llm.tagged("echo", job):
        llm.structured("brain", system="s", content="c", schema=Echo)
    row = conn.execute("SELECT agent, job_id, category FROM cost_entries").fetchone()
    assert row["agent"] == "echo" and row["job_id"] == job and row["category"] == "llm"

def test_an_agent_over_its_daily_budget_is_stopped(conn, policy):
    cap = budget.remaining(conn, policy, "echo")
    ledger.record(conn, category="llm", provider="m", quantity=1, unit="token", usd=cap + 0.01, agent="echo")
    with pytest.raises(budget.OverBudget, match="echo"):
        budget.check(conn, policy, "echo")

def test_spend_by_other_agents_does_not_count(conn, policy):
    ledger.record(conn, category="llm", provider="m", quantity=1, unit="token", usd=999, agent="scouts")
    budget.check(conn, policy, "echo")      # no raise
```

`Echo` is a one-field Pydantic model defined in the test file. `FakeAnthropic` returns a parsed object and a usage with the given token counts (reuse the stub style in `tests/test_day2.py` if one exists there; otherwise define it in this test file).

- [ ] **Step 2: Run, expect FAIL** (`tagged` and `budget` missing).
- [ ] **Step 3: Implement** the signatures above. "Today" is the UTC day, matching `enforce.billed_today`.
- [ ] **Step 4: Run the suite, expect PASS.**
- [ ] **Step 5: Commit** `git commit -m "Brain M1: LLM cost per agent and agent budgets"`

---

### Task 3: Agent contract, decisions and the brain runtime

**Files:**
- Create: `manyworlds/brain/agent.py`
- Create: `manyworlds/brain/runtime.py`
- Create: `manyworlds/brain/agents/__init__.py`, `manyworlds/brain/agents/echo.py`
- Modify: `tests/brain_helpers.py` — add `StubLLM(cost: float = 0.0)` (implements `structured` returning `schema.model_construct()` and `tagged(agent, job_id)`; each `structured` call writes one `llm` ledger entry of `cost` under the active tag) and `make_ctx(conn, policy, job_id) -> AgentContext` (claims the job as worker "test" and builds the context with a `StubLLM`).
- Modify: `policy/default.yaml` (`job_kinds: brain.echo: {pools: [local], outputs_per_job: 0}`; the autoscaler already skips `local`-only kinds)
- Test: `tests/test_brain_runtime.py`

**Interfaces:**
- Consumes: Task 2 `Claude.tagged`, `budget.check`; existing `queue.claim/complete/fail/enqueue`.
- Produces:
  - `@dataclass AgentContext: conn; policy: Policy; llm: StructuredLLM; job: queue.Job; now: datetime`
  - `class Agent(Protocol): name: str; def run(self, ctx: AgentContext) -> dict`
  - `agent.register(a: Agent) -> Agent`; `agent.get(name: str) -> Agent` (raises `KeyError`); `agent.REGISTRY: dict[str, Agent]`
  - `agent.record_decision(ctx: AgentContext, kind: str, *, step: str, inputs: dict, output: dict, rationale: str, creator_id=None, model: str | None = None) -> uuid.UUID` — inserts with `agent`, `job_id`, `step`, policy version and current override version; on the `(job_id, step)` conflict returns the existing id (idempotent rerun).
  - `agent.commission(ctx: AgentContext, target: str, payload: dict, *, reason: str, key: str) -> uuid.UUID` — enqueues `brain.<target>` with `parent_job_id=ctx.job.id`, `idempotency_key=f"{ctx.job.id}:{key}"`, payload including `reason` and `requested_by`.
  - `runtime.tick(conn, policy, llm, *, worker_id: str, now=None) -> list[dict]` — claims up to `brain.runtime.max_jobs_per_tick` `brain.*` jobs on pool `local`, runs each: `budget.check`, build context with `effective_policy` (Task 5; until then the file policy), `with llm.tagged(name, job.id)`, `agent.run`, `queue.complete`; on `NonRetryable` (including `OverBudget`, unknown agent) `queue.fail(retryable=False)`; on other exceptions `queue.fail(retryable=True)`. Returns one dict per job: `{job, agent, status, error?}`.
  - `echo` agent: records one decision `kind='echo'`, `step='echo'` with the payload and returns it.

- [ ] **Step 1: Failing tests**

```python
def test_an_agent_job_runs_and_records_its_decision(conn, policy):
    job = queue.enqueue(conn, "brain.echo", {"say": "hi"})
    out = runtime.tick(conn, policy, StubLLM(), worker_id="w1")
    assert out == [{"job": job, "agent": "echo", "status": "done"}]
    d = conn.execute("SELECT agent, job_id, step, output FROM decisions WHERE job_id = %s", (job,)).fetchone()
    assert d["agent"] == "echo" and d["output"]["say"] == "hi"

def test_a_rerun_after_a_crash_does_not_duplicate_the_decision(conn, policy):
    job = queue.enqueue(conn, "brain.echo", {"say": "hi"})
    ctx = make_ctx(conn, policy, job)              # helper building AgentContext for a claimed job
    first = agent.record_decision(ctx, "echo", step="echo", inputs={}, output={}, rationale="r")
    again = agent.record_decision(ctx, "echo", step="echo", inputs={}, output={}, rationale="r")
    assert first == again
    assert conn.execute("SELECT count(*) n FROM decisions WHERE job_id = %s", (job,)).fetchone()["n"] == 1

def test_an_unknown_agent_fails_once_and_others_still_run(conn, policy):
    bad = queue.enqueue(conn, "brain.nobody", {})
    good = queue.enqueue(conn, "brain.echo", {})
    out = {r["job"]: r for r in runtime.tick(conn, policy, StubLLM(), worker_id="w1")}
    assert out[bad]["status"] == "failed" and "no agent named 'nobody'" in out[bad]["error"]
    assert out[good]["status"] == "done"
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (bad,)).fetchone()["status"] == "failed"

def test_an_agent_over_budget_fails_without_retry(conn, policy):
    ledger.record(conn, category="llm", provider="m", quantity=1, unit="token", usd=10_000, agent="echo")
    job = queue.enqueue(conn, "brain.echo", {})
    runtime.tick(conn, policy, StubLLM(), worker_id="w1")
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job,)).fetchone()["status"] == "failed"

def test_commissioned_work_is_linked_and_not_duplicated(conn, policy):
    job = queue.enqueue(conn, "brain.echo", {})
    ctx = make_ctx(conn, policy, job)
    a = agent.commission(ctx, "echo", {"q": 1}, reason="why", key="k")
    b = agent.commission(ctx, "echo", {"q": 1}, reason="why", key="k")
    row = conn.execute("SELECT parent_job_id, payload FROM jobs WHERE id = %s", (a,)).fetchone()
    assert a == b and row["parent_job_id"] == job and row["payload"]["requested_by"] == "echo"
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `agent.py`, `runtime.py`, `agents/echo.py`. Agents register on import of `manyworlds.brain.agents`.
- [ ] **Step 4: Run the suite, expect PASS.**
- [ ] **Step 5: Commit** `git commit -m "Brain M1: agent contract, decisions and runtime"`

---

### Task 4: Evidence guard (real vs simulated)

**Files:**
- Create: `manyworlds/brain/evidence.py`
- Test: `tests/test_brain_evidence.py`

**Interfaces:**
- Consumes: Task 1 `findings`, `metric_snapshots`, `simulated_metrics`, `creators.publish_mode`.
- Produces:
  - `Mode = Literal["real", "sandbox"]`
  - `evidence.mode_for(conn, creator_id) -> Mode` (`instagram` → `real`, `simulated` → `sandbox`)
  - `evidence.findings(conn, *, mode: Mode, scope: str | None = None, creator_id=None, cluster_id=None, market_id=None, grades: tuple[str, ...] = ("proven", "tentative", "speculative")) -> list[dict]` — `real` excludes `evidence='simulated'`; `sandbox` includes all, and every simulated row carries `"label": "SIMULATED"`.
  - `evidence.metrics(conn, post_id, *, mode: Mode) -> list[dict]` — `real` reads `metric_snapshots` only; `sandbox` reads `simulated_metrics` only, each row labelled `"SIMULATED"`.
  - `evidence.for_allocation(conn, **filters) -> list[dict]` — always `mode="real"`; the only reader the strategist and allocator may use (M8, M9).
  - `evidence.record_finding(conn, *, scope, statement, evidence, grade, creator_id=None, cluster_id=None, market_id=None, effect=None, interval=None, n=None, sources=()) -> uuid.UUID`

- [ ] **Step 1: Failing tests**

```python
def test_real_reads_never_see_simulated_findings(conn):
    cid = creator(conn, publish_mode="instagram")
    evidence.record_finding(conn, scope="creator", creator_id=cid, statement="sim", evidence="simulated", grade="tentative")
    evidence.record_finding(conn, scope="creator", creator_id=cid, statement="real", evidence="observational", grade="tentative")
    assert [f["statement"] for f in evidence.findings(conn, mode="real", creator_id=cid)] == ["real"]
    assert [f["statement"] for f in evidence.for_allocation(conn, creator_id=cid)] == ["real"]

def test_a_creator_that_left_the_sandbox_keeps_its_simulated_history_out_of_real_reads(conn):
    cid = creator(conn, publish_mode="simulated")
    evidence.record_finding(conn, scope="post", creator_id=cid, statement="old sim", evidence="simulated", grade="speculative")
    conn.execute("UPDATE creators SET publish_mode = 'instagram' WHERE id = %s", (cid,))
    assert evidence.findings(conn, mode=evidence.mode_for(conn, cid), creator_id=cid) == []

def test_sandbox_reads_label_simulated_rows(conn):
    cid = creator(conn)
    evidence.record_finding(conn, scope="post", creator_id=cid, statement="s", evidence="simulated", grade="speculative")
    assert evidence.findings(conn, mode="sandbox", creator_id=cid)[0]["label"] == "SIMULATED"

def test_metrics_come_from_the_table_for_the_mode(conn):
    post = make_post(conn)
    conn.execute("INSERT INTO metric_snapshots (post_id, metrics, post_age_s) VALUES (%s, '{\"reach\": 10}', 3600)", (post,))
    conn.execute("INSERT INTO simulated_metrics (post_id, metrics, post_age_s, model, panel_version) "
                 "VALUES (%s, '{\"reach\": 999}', 3600, 'm', 'v1')", (post,))
    assert [m["metrics"]["reach"] for m in evidence.metrics(conn, post, mode="real")] == [10]
    assert [m["metrics"]["reach"] for m in evidence.metrics(conn, post, mode="sandbox")] == [999]
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `evidence.py`. Add a module docstring stating it is the only sanctioned reader of findings and metrics for agents.
- [ ] **Step 4: Add a guard test** that no module under `manyworlds/brain/` other than `evidence.py` contains `FROM findings`, `FROM metric_snapshots` or `FROM simulated_metrics` (read the files, assert by regex). Run the suite, expect PASS.
- [ ] **Step 5: Commit** `git commit -m "Brain M1: evidence guard between real and simulated data"`

---

### Task 5: Runtime overrides, authority and approvals

**Files:**
- Create: `manyworlds/brain/overrides.py`
- Modify: `manyworlds/brain/runtime.py` (contexts get `effective_policy`)
- Modify: `policy/default.yaml` (`brain.authority`)
- Test: `tests/test_brain_overrides.py`

**Interfaces:**
- Consumes: Task 1 `policy_overrides`, `approvals`; Task 3 `AgentContext`.
- Produces:
  - `overrides.effective_policy(conn, policy: Policy) -> Policy` — the file policy with every path's latest override applied; `version` becomes `f"{policy.version}+o{n}"` where n is the max override version (0 when none).
  - `overrides.propose(conn, policy: Policy, path: str, value, *, author: str, reason: str) -> dict` — returns `{"applied": True, "version": n}` when `path` matches an entry in `brain.authority.autonomous` and the value is within that entry's `min`/`max` (inclusive) for numbers or its `allowed` list for strings; otherwise writes a pending `approvals` row and returns `{"applied": False, "approval_id": id}`. Paths under `limits.` and `budget.master` are never autonomous.
  - `overrides.decide(conn, approval_id, *, approved: bool, by: str) -> None` — approving applies the override.
- Policy shape (builder's starting values):

```yaml
brain:
  authority:
    autonomous:                       # paths the Brain may change itself, and the bounds
      - {path: "planner.formats.*", min: 0, max: 10}
      - {path: "production.candidates_per_scene", min: 2, max: 8}
      - {path: "brain.exploration_share", min: 0.1, max: 0.6}
```

  Glob `*` matches one path segment. More entries are added by later milestones as they introduce tunables.

- [ ] **Step 1: Failing tests**

```python
def test_an_override_within_authority_applies(conn, policy):
    r = overrides.propose(conn, policy, "production.candidates_per_scene", 6, author="strategist", reason="r")
    eff = overrides.effective_policy(conn, policy)
    assert r["applied"] and eff.get("production", "candidates_per_scene") == 6 and eff.version.endswith("+o1")

def test_the_bound_itself_is_allowed(conn, policy):
    assert overrides.propose(conn, policy, "production.candidates_per_scene", 8, author="a", reason="r")["applied"]

def test_beyond_authority_waits_for_approval_and_changes_nothing(conn, policy):
    r = overrides.propose(conn, policy, "production.candidates_per_scene", 9, author="a", reason="r")
    assert not r["applied"]
    assert overrides.effective_policy(conn, policy).get("production", "candidates_per_scene") == \
        policy.get("production", "candidates_per_scene")
    overrides.decide(conn, r["approval_id"], approved=True, by="owner")
    assert overrides.effective_policy(conn, policy).get("production", "candidates_per_scene") == 9

def test_limits_are_never_autonomous(conn, policy):
    assert not overrides.propose(conn, policy, "limits.daily_usd_max", 1, author="a", reason="r")["applied"]

def test_agents_run_under_the_effective_policy_and_decisions_record_it(conn, policy):
    overrides.propose(conn, policy, "brain.exploration_share", 0.5, author="a", reason="r")
    job = queue.enqueue(conn, "brain.echo", {})
    runtime.tick(conn, policy, StubLLM(), worker_id="w1")
    d = conn.execute("SELECT policy_version, override_version FROM decisions WHERE job_id = %s", (job,)).fetchone()
    assert d["override_version"] == 1 and d["policy_version"].endswith("+o1")
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `overrides.py`; switch `runtime.tick` to build contexts with `effective_policy`.
- [ ] **Step 4: Run the suite, expect PASS.**
- [ ] **Step 5: Commit** `git commit -m "Brain M1: runtime overrides within authority, approvals"`

---

### Task 6: Operating it: CLI, service, docs

**Files:**
- Modify: `manyworlds/cli.py` (subcommands below)
- Modify: `docker-compose.yml` (service `brain-worker`, profile `live`, `command: ["brain", "worker", "--loop"]`, `env_file: .env`, `restart: unless-stopped`)
- Modify: `CLAUDE.md` (layout lines for `manyworlds/brain/agent.py`, `runtime.py`, `evidence.py`, `overrides.py`; rule: "agents read findings and metrics only through `brain/evidence.py`")
- Test: `tests/test_brain_cli.py`

**Interfaces:**
- Consumes: Tasks 3–5.
- Produces commands (the existing `mw brain …` group, if one exists in `cli.py`, is extended; otherwise create it):
  - `mw brain worker [--loop]` — runs `runtime.tick` every `brain.runtime.loop_seconds`; prints only non-empty results.
  - `mw brain run <agent> [--payload JSON]` — enqueues `brain.<agent>` and runs one tick (for operators and tests).
  - `mw brain trace <job_id>` — the job, its decisions, its commissioned children (recursive), and their LLM cost by agent.
  - `mw brain approvals [--decide ID --approve|--reject]` — lists pending approvals or decides one.

- [ ] **Step 1: Failing test**

```python
def test_trace_shows_decisions_children_and_cost(conn, policy, capsys):
    parent = queue.enqueue(conn, "brain.echo", {})
    runtime.tick(conn, policy, StubLLM(cost=0.01), worker_id="w1")
    cli.main(["brain", "trace", str(parent)])
    out = json.loads(capsys.readouterr().out)
    assert out["job"]["id"] == str(parent) and out["decisions"][0]["agent"] == "echo"
    assert out["llm_usd_by_agent"]["echo"] == 0.01
```

(`StubLLM(cost=…)` records a ledger entry through the active tag, so the test exercises the same path as `Claude`.)

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the commands and the compose service; update CLAUDE.md.
- [ ] **Step 4: Run the suite, expect PASS.** Then against the real database: `docker compose --profile live up -d --build brain-worker` and `docker compose run --rm mw brain run echo --payload '{"say":"hello"}'`; expected: a `done` result and a decision visible in `mw brain trace`.
- [ ] **Step 5: Commit** `git commit -m "Brain M1: brain worker, trace and approvals commands"`

---

## M1 acceptance (from the roadmap)

- `brain.echo` runs through the live brain worker, records a decision and its LLM cost under `echo`.
- A simulated finding is invisible to every real-mode read and to `for_allocation`.
- An override outside authority becomes a pending approval and leaves the effective policy unchanged.
- Full test suite green; brain worker running in the live profile.
