"""What image and video review share: the stages, the references, how rows land, the loop."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Callable

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from .. import direction, logs
from ..storage import ObjectStore

log = logging.getLogger("manyworlds.review")

# The QA's three judgments (owner 2026-11-25), all in the one existing judge call:
#   content safety       hard: the `sexualized` flag, defined by SAFETY_RULE (policy strategy.desirability.restrictions)
#   brand alignment      scored against the creator's presentation register (BRAND_RULE); rejects only a clear misfit
#                        (review.<image|video>.min_brand_alignment)
#   distribution         a risk score and label per platform (DISTRIBUTION_RULE): report and routing only, never a reject
# Safety is judged on VISIBLE content, context, composition and intent, never on speculation about what is under
# clothing: an oversized shirt, a hotel bathrobe, a backless dress or a cardigan worn as a top pass when covered.
SAFETY_RULE = ("true ONLY for what is VISIBLE: explicit sexual content or sexual activity; visible nudity (genitals, "
               "nipples) or a deliberately staged nudity simulation (posed to look naked, e.g. strategically hidden "
               "with hands or objects); fetish-oriented content; sexual solicitation; content plainly made for "
               "sexual gratification. Never speculate about what is under clothing: an oversized shirt, a hotel "
               "bathrobe, off-the-shoulder or backless clothes, a cardigan worn as a top are fine when covered. Judge "
               "by context, never by clothing alone: swimwear at a beach or pool, a shirtless man swimming or "
               "training, cleavage, fitted or revealing fashion, covered underwear in a fashion campaign, fitness "
               "and swimwear-campaign photography and tasteful sensuality are NOT sexualized; attractive and "
               "confident is the goal")
SAFETY_SYSTEM = ("Safety is judged on visible content in context, never by clothing alone and never by guessing what "
                 "is underneath: mainstream fashion, swimwear and fitness photography (Vogue, GQ, swimwear campaigns) "
                 "is fine; only explicit, visibly nude or staged-nude, fetish, soliciting or gratification-first "
                 "content, or someone who could read as under age, is unsafe. Judge it like a world-class fashion "
                 "editor and stylist: appearance, styling, accessories, grooming and surroundings.")
BRAND_RULE = ("only when HER PRESENTATION is given: 0-10, how well the image fits her presentation register, "
              "wardrobe range and photographic style (10 = exactly her; 0 = clearly someone else's register, e.g. a "
              "sultry editorial for an understated creator or a sterile catalogue shot for a glamorous one); "
              "null when not given")
DISTRIBUTION_RULE = ("0-10 risk that the platforms would limit recommendation of this post to non-followers "
                     "(0 = mainstream; 10 = would surely be limited); a report, never a reason to reject")
DETAIL_RULE = ("styling and detail defects an ordinary viewer would notice, worst first, each 'severity: what', "
               "severity minor or major: malformed jewelry, unnatural or melted nails, accessories that change or "
               "merge, grooming artifacts (patchy beard, smeared makeup), distorted hands, implausible clothing "
               "(seams, straps or buttons that make no sense); empty if none")
CONTEXT_FIELDS = ("brand_alignment", "distribution_risk", "distribution_label", "detail_defects")
CONTEXT_SCORES = {"brand": "brand_alignment", "distribution": "distribution_risk"}


# --- performance (manyworlds.performance; owner 2026-11-26): observed in the call we already make, rank only ---------
# Among candidates that pass the defect gates, selection prefers appeal, charisma, personality, expression and
# variety; nothing here ever rejects. Naturalness is judged on every asset: more expressive is not better.
class ObservedLevel(BaseModel):
    control: str = Field(default="", description="a control named in PERFORMANCE")
    level: int | None = Field(default=None, description="the level you see, 0-4")


PERFORMANCE_FIELDS = ("performance_read",)
SCORE_NAMES = ("appeal", "charisma", "expression", "naturalness")
READ_RULE = ("only when PERFORMANCE is asked, in exactly this form, else empty: "
             "'appeal=N; charisma=N; expression=N; naturalness=N; levels: control=L, control=L; pose: a few words' "
             "(N 1-10, or 0 when there is no face to judge; levels only for the controls PERFORMANCE lists, L 0-4)")
APPEAL_RULE = ("only when PERFORMANCE is asked: 1-10, how much a viewer would stop for and want to look at this: "
               "attractiveness, presence, style, the moment; 0 otherwise")
CHARISMA_RULE = ("only when PERFORMANCE is asked: 1-10, the personality and magnetism that come through; 0 without "
                 "a face")
EXPRESSION_RULE = ("only when PERFORMANCE is asked: 1-10, does the face carry the intended feeling, alive and specific "
                   "rather than blank; 0 without a face")
NATURAL_RULE = ("only when PERFORMANCE is asked: 1-10, believable human behaviour: 10 = a real person caught in a real "
                "moment; exaggerated, theatrical, stiff, frozen or uncanny behaviour scores low EVEN when it matches "
                "the request (more expressive is not better); 0 otherwise")
POSE_RULE = ("only when PERFORMANCE is asked: a few words for the pose and expression, e.g. 'leaning on the counter, "
             "half-smile, eyes down' (for variety between frames); empty otherwise")
LEVELS_RULE = ("only when PERFORMANCE lists controls: the level 0-4 you see for each control listed, as "
               "'control=level' pairs separated by commas, e.g. 'sass=2, warmth=3'; empty otherwise")


def parse_read(text: str) -> dict:
    """The judge's performance_read ('appeal=7; charisma=6; expression=8; naturalness=7; levels: sass=2, warmth=3;
    pose: leaning on the counter, half-smile') as {scores: {name: int}, levels: [{control, level}], pose: str}.
    Parts may come in any order; anything unreadable is skipped."""
    import re
    out = {"scores": {}, "levels": [], "pose": ""}
    for part in str(text or "").split(";"):
        key, sep, val = part.partition(":")
        k = key.strip().lower()
        if sep and k == "levels":
            out["levels"] = parse_levels(val)
            continue
        if sep and k == "pose":
            out["pose"] = val.strip()
            continue
        m = re.fullmatch(r"\s*([a-z]+)\s*=\s*(\d+)\s*", part.lower())
        if m and m.group(1) in SCORE_NAMES:
            out["scores"][m.group(1)] = int(m.group(2))
    return out


def parse_levels(value) -> list[dict]:
    """The judge's observed levels as [{control, level}]: from 'sass=2, warmth=3', or from an older list of
    {control, level} (objects or dicts). Anything unreadable is skipped."""
    import re
    if isinstance(value, str):
        return [{"control": m.group(1).lower(), "level": int(m.group(2))}
                for m in re.finditer(r"([a-z_]+)\s*[=:]\s*([0-4])", value.lower())]
    out = []
    for o in value or []:
        c = o.get("control") if isinstance(o, dict) else getattr(o, "control", None)
        l = o.get("level") if isinstance(o, dict) else getattr(o, "level", None)
        if c and l is not None:
            out.append({"control": str(c), "level": l})
    return out


def observe_enabled(policy) -> bool:
    return bool((policy.get("performance", "observe", default={}) or {}).get("enabled")) if policy is not None else False


def performance_controls(shot: dict | None, policy) -> list[str]:
    """The shot's directed controls a judge observes (manyworlds.performance.observe_controls), most salient first."""
    from .. import performance
    if not shot:
        return []
    try:
        controls = direction.shot_controls(shot)
        base = (((shot.get("performance") or {}).get("profile") or {}).get("baseline")
                if isinstance(shot, dict) else None)
    except Exception:  # noqa: BLE001 - a malformed sheet is judged without performance
        return []
    return performance.observe_controls(controls, policy, baseline=base)


def performance_prompt(shot: dict | None, policy, *, still: bool) -> str:
    """The judge's PERFORMANCE block: the scores (appeal, charisma, expression, naturalness, a pose descriptor) and,
    for a directed shot, the controls to observe with what each means. Blind to the requested levels. '' when
    observation is off."""
    from .. import performance
    if not observe_enabled(policy):
        return ""
    lib = performance.library(policy)
    names = performance_controls(shot, policy)
    intent = ""
    if shot:
        try:
            intent = direction.ShotSheet.model_validate(shot).performance.intent.strip()
        except Exception:  # noqa: BLE001
            intent = ""
    what = "each candidate" if still else "the clip"
    text = (f"PERFORMANCE (rank only, never a reason to reject): for {what} also give appeal, charisma, expression and "
            f"naturalness (1-10) and a pose descriptor, all in performance_read. Naturalness: exaggerated, theatrical or uncanny behaviour scores "
            f"low even when it is what was asked; a natural, specific, alive moment scores high.")
    if intent:
        text += f" The intended performance (for expression): {intent}."
    if names:
        levels = ", ".join(f"{k} {v}" for k, v in (lib.get("levels") or {}).items())
        means = "; ".join(f"{n}: {((lib['controls'].get(n) or {}).get('means') or '')}" for n in names)
        text += f" levels: the level you see ({levels}) for each of these controls: {means}."
    return text


def performance_result(r, shot: dict | None, policy, *, target: str) -> dict | None:
    """What a judged candidate says of its performance: {points, requested, compiled, observed, scores, descriptor,
    closeness, intent, phrases, target} (manyworlds.performance: closeness, selection points), or None when the judge
    gave no performance score. Observed levels only for the controls asked."""
    from .. import performance
    if not observe_enabled(policy) or r is None:
        return None
    read = parse_read(getattr(r, "performance_read", "") or "")
    scores = {}
    for k in SCORE_NAMES:
        v = read["scores"].get(k)
        if v:   # 0 or missing: not asked, or no face to judge
            try:
                scores[k] = max(1, min(10, int(v)))
            except (TypeError, ValueError):
                continue
    asked = performance_controls(shot, policy)
    observed = {k: v for k, v in performance.levels_of(read["levels"]).items()
                if k in asked}
    if not scores and not observed:
        return None
    requested = direction.shot_controls(shot) if shot else {}
    close = performance.closeness({k: requested[k] for k in asked if k in requested}, observed)
    used, intent, phrases = {}, "", {}
    if shot:
        try:
            s = direction.ShotSheet.model_validate(shot)
            intent = s.performance.intent.strip()
            out = performance.compose(intent, requested, target, policy, context=s.performance.context or None,
                                      profile=s.performance.profile or {}, extreme_reason=s.performance.extreme_reason)
            used, phrases = {u["control"]: u["compiled"] for u in out["used"]}, out["phrases"]
        except Exception:  # noqa: BLE001 - the record is still useful without the compiled view
            pass
    return {"points": performance.selection_points(scores, close, policy), "requested": requested,
            "compiled": used, "observed": observed, "scores": scores,
            "descriptor": " ".join(read["pose"].split())[:200],
            "closeness": close, "intent": intent, "phrases": phrases, "target": target,
            "library_version": performance.library(policy).get("version")}


def performance_row(asset_id, perf: dict, details: dict) -> tuple:
    """The `performance` review row: always passes (rank only), its score the selection points."""
    return (asset_id, "performance", True, perf["points"],
            Jsonb({**details, "report_only": True,
                   "performance": {k: perf[k] for k in ("observed", "scores", "closeness", "descriptor")}}))


def major_details(r, cfg: dict | None = None) -> list[str]:
    """The styling/detail defects an ordinary viewer would notice (severity major): they fail realism. Minor ones are
    kept in the row's details as a report."""
    return [d for d in (getattr(r, "detail_defects", None) or []) if str(d).strip().lower().startswith("major")]


def context_verdicts(cfg: dict, r) -> dict[str, tuple[bool, dict]]:
    """Brand alignment (rejects only a clear misfit: under review.<kind>.min_brand_alignment) and distribution
    suitability (a risk score and label: report and routing only, always passes), when the judge scored them."""
    out = {}
    b = getattr(r, "brand_alignment", None)
    if b is not None:
        b = max(0, min(10, int(b)))
        bar = float(cfg.get("min_brand_alignment", 3))
        out["brand"] = (b >= bar, {"brand_alignment": b, "bar": bar, "enforced": True})
    d = getattr(r, "distribution_risk", None)
    if d is not None:
        out["distribution"] = (True, {"distribution_risk": max(0, min(10, int(d))),
                                      "label": getattr(r, "distribution_label", None), "report_only": True})
    return out


def presentation_of(conn, creator_id) -> dict | None:
    """The creator's presentation register (brand_core.presentation in her latest profile), or None."""
    row = conn.execute("SELECT body FROM creator_profiles WHERE creator_id = %s ORDER BY version DESC LIMIT 1",
                       (creator_id,)).fetchone()
    return (((row or {}).get("body") or {}).get("brand_core") or {}).get("presentation") or None


def presentation_prompt(policy, presentation: dict) -> str:
    import json
    guidelines = (policy.get("strategy", "platform_guidelines", default={}) or {}) if policy is not None else {}
    return (f"HER PRESENTATION (score brand_alignment against it, per candidate; and distribution_risk with a "
            f"label against these platform guidelines: {json.dumps(guidelines)}): {json.dumps(presentation)}")


def policy_for(policy, batch: str | None):
    """The policy a job is judged under: the test overlay for a batch it names (operations.test_overlay.batches,
    manyworlds.policy.for_batch), else `policy`. Applying it to a policy that already carries the overlay (the
    breadth driver's) changes nothing."""
    from .. import policy as policy_mod
    return policy_mod.for_batch(policy, batch) if policy is not None else policy


def judges_identity(shot: dict | None = None, *, plate: bool = False) -> bool:
    """Whether a candidate is judged on identity at all: a set plate or outfit reference (payload `plate`,
    manyworlds.creator.sets) has nobody in it, and a POV, cutaway or faceless aesthetic shot (direction.faceless)
    shows no face. The one test the gates, the verdicts and both clear passes share."""
    return not (plate or direction.faceless(shot))


def stages(cfg: dict, role: str | None = None) -> list[tuple[str, str, str]]:
    """(role, effort, name) per stage: a cheap screen on everything, then a stronger confirm on
    what the screen passed. Passing `role` runs that one model alone (calibration, dry runs)."""
    out = [(role or cfg.get("role", "judge"), cfg.get("effort", "low"), "screen")]
    if cfg.get("confirm_role") and not role:
        out.append((cfg["confirm_role"], cfg.get("confirm_effort", "low"), "confirm"))
    return out


def direction_cfg(policy) -> dict:
    return (policy.get("review", "direction", default={}) or {}) if policy is not None else {}


def direction_passes(score: int | None, policy) -> bool:
    """The `direction` gate: a score below review.direction.min_score rejects. A judge that leaves it out passes,
    as for `scene` (the output is still judged on identity, realism and safety)."""
    return score is None or int(score) >= int(direction_cfg(policy).get("min_score", 6))


def direction_brief(shot: dict, policy, *, still: bool) -> str:
    """What the judge scores `direction` against: the shot's one-line summary, its beats (a photo is one moment:
    the first beat only) and, for video, the camera move and the performance; then the rubric's anchors
    (policy/rubrics/direction-v<n>.yaml), so scores mean the same thing on every judge."""
    from .. import direction
    from ..brain import rubrics
    s = direction.ShotSheet.model_validate(shot)
    beats = s.blocking[:1] if still else sorted(s.blocking, key=lambda b: b.at_s)
    looks = {"lens": "into the lens", "off": "off-camera", "prop": "at the object in hand", "screen": "at a screen"}

    def beat(b) -> str:
        text = ", ".join(x for x in (b.action.strip(), f"looking {looks[b.eyeline]}" if b.eyeline else "",
                                     f"hands {b.hands.strip()}" if b.hands.strip() else "") if x)
        return text if still or not text else f"at {b.at_s:g} s: {text}"

    lines = [f"Shot: {direction.summary(s) or 'no summary'}."]
    beat_text = "; then ".join(t for t in (beat(b) for b in beats) if t)
    if beat_text:
        lines.append(f"Beats: {beat_text}.")
    if not still:
        move = direction.camera_move(s, policy)
        perf = ", ".join(x for x in (f"camera {move.replace('_', ' ')}" if move else "",
                                     f"{s.performance.energy} energy" if s.performance.energy else "",
                                     f"mood {s.performance.emotion_arc}" if s.performance.emotion_arc.strip() else "",
                                     s.performance.body_language.strip()) if x)
        if perf:
            lines.append(f"Camera and performance: {perf}.")
    rubric = rubrics.load(str(direction_cfg(policy).get("rubric", "direction")))
    return " ".join(lines) + "\n" + rubric.as_text()


def reference_images(creator: dict, store: ObjectStore, workdir: Path, n: int) -> list[bytes]:
    refs = (creator["anchors"] or {}).get("reference_images", [])[:n]
    if not refs:
        raise ValueError(f"creator {creator['id']} has no reference images to review against")
    return [store.get_file(r["uri"], Path(workdir) / f"ref_{i}.png").read_bytes() for i, r in enumerate(refs)]


@contextlib.contextmanager
def charged_to(llm, job_id):
    """Charge the judge's calls to the job under review, so review counts in that output's cost
    (ledger.cost_per_output). No agent: review is production cost, not the Brain's."""
    tag = getattr(llm, "tagged", None)
    with (tag(None, job_id) if tag else contextlib.nullcontext()):
        yield


def write_rows(conn, rows: list[tuple]) -> None:
    """All of a job's review rows at once, after every stage: anyone reading reviews (production
    choosing what to use) must never see a screen without its confirm."""
    if not rows:
        return
    with conn.transaction():
        for row in rows:
            conn.execute("INSERT INTO reviews (asset_id, gate, passed, score, details) VALUES (%s, %s, %s, %s, %s)", row)


def give_up(conn, policy, job_id, error: Exception | str, *, asset_kind: str, section: str) -> bool:
    """Count a review of this job that raised (jobs.review_errors). Once review.<section>.max_review_errors is used
    up, every unreviewed `asset_kind` asset of the job gets one failing row (gate `review`, details.stage `error`,
    details.error), so production sees the output rejected and repairs it with a new job, and the reviewer stops
    paying a screen for it every loop. Repair diagnosis ignores rows with details.error. True when it gave up.
    Errors count at most once per review.<section>.review_error_spacing_seconds, so a judge outage of a few
    minutes (every review raising, loop after loop) uses up no job's tries."""
    cfg = (policy.get("review", section, default={}) or {}) if policy is not None else {}
    row = conn.execute(
        """UPDATE jobs SET review_errors = review_errors + 1, review_error_at = now()
           WHERE id = %s AND (review_error_at IS NULL OR review_error_at <= now() - make_interval(secs => %s))
           RETURNING review_errors""",
        (job_id, float(cfg.get("review_error_spacing_seconds", 600)))).fetchone()
    if row is None:   # counted too recently (or no such job)
        return False
    if row["review_errors"] < int(cfg.get("max_review_errors", 3)):
        return False
    text = (f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error))[:300]
    assets = conn.execute(
        "SELECT a.id FROM assets a WHERE a.job_id = %s AND a.kind = %s "
        "AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id)", (job_id, asset_kind)).fetchall()
    write_rows(conn, [(a["id"], "review", False, None,
                       Jsonb({"stage": "error", "error": text, "review_errors": row["review_errors"]}))
                      for a in assets])
    log.warning("review of job %s failed %d times; its %d %s output(s) count as rejected: %s",
                job_id, row["review_errors"], len(assets), asset_kind, text)
    return True


def run_jobs(job_ids: list, review: Callable[[object, Path], list[dict]], workdir: Path, prefix: str = "", *,
             on_error: Callable[[object, Exception], bool] | None = None) -> dict:
    """Review each job in its own folder; one bad job never stops the others. `on_error(job_id, error)` is told of
    each job whose review raised (give_up); True means it was given up and its outputs count as rejected."""
    accepted = rejected = 0
    errors = []
    for job_id in job_ids:
        jd = Path(workdir) / f"{prefix}{job_id}"
        jd.mkdir(parents=True, exist_ok=True)
        try:
            with logs.context(job_id=job_id):   # every line of this review names its job
                rs = list(review(job_id, jd))
            for r in rs:
                accepted += r["accepted"]
                rejected += not r["accepted"]
        except Exception as e:  # noqa: BLE001
            err = {"job": str(job_id), "error": str(e)[:300]}
            if on_error is not None:
                try:
                    if on_error(job_id, e):
                        err["gave_up"] = True
                except Exception as e2:  # noqa: BLE001 - counting must not stop the other jobs' reviews
                    log.warning("could not count the review error of job %s: %s", job_id, e2)
            errors.append(err)
    return {"accepted": accepted, "rejected": rejected, "errors": errors, "jobs": len(job_ids)}
