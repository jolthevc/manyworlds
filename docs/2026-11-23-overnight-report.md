# Overnight build report (2026-11-22 to 23)

Nothing ran on a GPU. No paid service was added. Operations stayed paused, launches 0, publishing paused. Each change
went to `main` with CI passing.

## What changed

**1. Brain: the agency layer (the main piece).**
- **Archetype library** (`policy/archetypes/archetypes-v1.yaml`). Nine playbooks: fashion and style, beauty/GRWM,
  slow lifestyle, food, storyteller, educator, movement, city explorer, maker. Each one sets out:
  - what the audience comes for, and how this kind of creator communicates
  - the content mix and how often they talk to camera
  - the capabilities that carry it, and series patterns
  - hooks, edit grammar, success signals and pitfalls

  The director, showrunner and talent manager all read this one file. It replaces the old word table.
- **Creator dossier** (`creator/dossier.py`). It adds five sections to the profile: archetypes, positioning (why
  follow, the gap, point of view, signature angle, how she differs from siblings), audience insight (the one person,
  desires, frustrations, what they save and share, their words, turn-offs), story (backstory, values in action,
  quirks, routines, small goals) and content system (mix, talk share, 3-5 series built only from capabilities that
  render now, hooks, what she never makes).
  - Each section is one small structured call.
  - Code checks what a model can't be trusted with.
  - A dossier that comes out too close to a sibling creator gets one rewrite.
- **Talent manager agent** (`brain/agents/talent_manager.py`), one per creator.
  - It builds the dossier at launch (a new launch step).
  - Every Monday it reviews the career, using posts, idea scores, what production dropped and its cost, and findings
    labelled real or simulated.
  - It writes the brief the showrunner plans from.
  - It changes the profile only within an allowed list. Identity, the never-list and disclosure are never its call.
    Anything outside the list goes to you as `talent_needs_owner`.
  - The strategist (the CEO) sees every creator's latest brief and their requests.

**2. Review: only media defects kill once media exists.**
- **Identity is judged by the face only.** Clothes, hats, hair, props, pose and setting no longer count. Most identity
  rejects on real posts were complaints about the brief: "beanie instead of claw clip", "the cat is missing".
- **Direction and scene are scored, never rejecting.** The reason to drop is bad media, not a missed brief.
- **POV and cutaway shots** prompt her hands, not her face, and get no identity gate.
- **Before any GPU:** props are capped at 2 per shot, and the director has a new rule that every shot must be drawable
  in one frame.
- **Cheaper judging:**
  - The second look is Sonnet instead of Opus.
  - It only checks close calls: a clear pass on the cheap screen skips it, for photos as already for clips.
  - At most one per job.
  - The agreement check also moved off Opus.
  - Long system prompts are now cached.

**3. GPU waste.**
- **Code faults** (TypeError, a missing package, a missing compiler, ComfyUI not starting) now fail a job at once.
  Before, they were retried on fresh machines at about 5 GPU-minutes each.
- **Disk-full** now counts as a machine fault.
- **Memory floors per job kind.** A card below a kind's floor never takes it, and a card that can take nothing
  destroys its own machine at start.
- **Model affinity.** A machine runs one model's work back to back, and frees the last engine when it switches
  (ComfyUI `/free`, VoxCPM dropped). This matters for the merged test machine.

**4. Fixes from an independent review of the last two waves** (3 reviewers, 23 findings, the confirmed ones fixed):
- **Transitions shifted every later caption, label and sound.** They also faded a take's last syllable under the
  next take. Fixed: the outgoing frame now holds.
- **The final-file review never wrote its rows.** Fixed.
- **`edit.hook` stage directions were drawn on screen.** Fixed.
- **Labels sat on top of captions.** Fixed.
- **Breadth posts could later be produced by the normal producer.** Fixed.
- **After the test deadline, a retry could still boot a new machine.** Fixed.
- **The merged test machine** now has its own caps (6 h / $7, 64 GB RAM) and a small disk on the volume pool. The
  Qwen edit weights stay off the shared volume.
- **An empty overlay file would have wiped the whole policy.** Fixed.

**5. Retired the old diffusers video path.**
- Removed: `video.wan_i2v` (14 min per 5 s clip), FastWan (0 of 10 clips passed) and the DMD sampler.
- Story clips are ComfyUI only, with hosted Wan 2.7 as the fallback.

**6. Docs.** CLAUDE.md layout, rules and status are rewritten, the tech debt is updated, and the roadmap notes what
was built.

## What is not verified
- Nothing new has run on a GPU or against a real model. Specifically unproven:
  - the dossier and briefs (stub tests only)
  - the cheaper confirm's agreement with Opus
  - model affinity and freeing memory on a real merged machine
  - ComfyUI story clips, which still have never finished a job
- **The director's larger schema might hit the grammar-size limit again.** Checking needs one paid director call.
- **Memory floors are set only where the floor is known** (Wan-Animate-2).

## Still open (in `docs/tech-debt.md`)
- **Filling paid idle minutes with extra takes:** designed, not built.
- **Narrated-edit label timing.**
- **Set plates while paused.**
- **Security items:**
  - bucket-scoped R2 token for workers
  - `MW_HASH_SALT`
  - the worker database role can update any job

## Needs you
- **The breadth test.** Your go, plus RunPod or Vast credit. About $5-9.
- **Look at the first real dossier** before it steers anything.
- **When you can:** `NTFY_TOPIC`, `FLY_API_TOKEN` in GitHub, `MW_HASH_SALT`, an R2 token scoped to the bucket.
