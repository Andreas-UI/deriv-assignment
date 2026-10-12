"""Validate the artifacts produced by run_pipeline.py.

Usage:
    python validate.py          # exits 0 if every check passes, 1 otherwise
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List

import run_pipeline as rp

REQUIRED = [
    rp.TICKETS_FILE, rp.CONFIG_FILE, rp.NORMALIZED_FILE, rp.PREDICTIONS_FILE,
    rp.OVERRIDES_FILE, rp.FINAL_QUEUE_FILE, rp.SUMMARY_FILE,
]
OPTIONAL = [rp.ESCALATIONS_FILE, rp.LLM_CALLS_FILE, rp.MANIFEST_FILE, rp.REASONING_FILE]
JSON_FILES = [f for f in REQUIRED + OPTIONAL if f.endswith(".json")]


def sha256_path(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_checks(base_dir: str) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    data: Dict[str, Any] = {}

    def p(name: str) -> str:
        return os.path.join(base_dir, name)

    def check(name: str, fn: Callable[[], List[str]]) -> None:
        try:
            problems = fn()
        except Exception as e:  # a crashing check is a failing check
            problems = ["check raised {}: {}".format(type(e).__name__, e)]
        checks.append({"name": name, "passed": not problems, "detail": "; ".join(problems) or "ok"})

    # 1. artifacts exist
    def artifacts_exist() -> List[str]:
        missing = [f for f in REQUIRED + OPTIONAL if not os.path.exists(p(f))]
        return ["missing {}".format(f) for f in missing]

    # 2. JSON parses
    def json_valid() -> List[str]:
        problems = []
        for f in JSON_FILES:
            if not os.path.exists(p(f)):
                continue
            try:
                with open(p(f), encoding="utf-8") as fh:
                    data[f] = json.load(fh)
            except ValueError as e:
                problems.append("{} is not valid JSON: {}".format(f, e))
        if os.path.exists(p(rp.LLM_CALLS_FILE)):
            data[rp.LLM_CALLS_FILE] = []
            with open(p(rp.LLM_CALLS_FILE), encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if line.strip():
                        try:
                            data[rp.LLM_CALLS_FILE].append(json.loads(line))
                        except ValueError as e:
                            problems.append("{} line {} invalid: {}".format(rp.LLM_CALLS_FILE, i, e))
        return problems

    check("required_artifacts_exist", artifacts_exist)
    check("json_valid", json_valid)

    tickets = data.get(rp.TICKETS_FILE, [])
    config = data.get(rp.CONFIG_FILE, {})
    normalized = data.get(rp.NORMALIZED_FILE, [])
    preds = data.get(rp.PREDICTIONS_FILE, [])
    overrides = data.get(rp.OVERRIDES_FILE, [])
    final = data.get(rp.FINAL_QUEUE_FILE, [])
    manifest = data.get(rp.MANIFEST_FILE, {})
    llm_calls = data.get(rp.LLM_CALLS_FILE, [])
    cats = config.get("allowed_categories", [])
    prios = config.get("allowed_priorities", [])
    routes = config.get("routing_rules", {})
    max_words = config.get("reply_style", {}).get("max_words", 0)
    ticket_ids = [t.get("ticket_id") for t in tickets]

    # 3. inputs read from disk
    def inputs_from_disk() -> List[str]:
        problems = []
        hashes = manifest.get("input_hashes", {})
        for f in (rp.TICKETS_FILE, rp.CONFIG_FILE):
            if f not in hashes:
                problems.append("manifest has no hash for {}".format(f))
            elif hashes[f] != sha256_path(p(f)):
                problems.append("{} changed since the run (hash mismatch)".format(f))
        if [n.get("ticket_id") for n in normalized] != [rp.clean_text(i) for i in ticket_ids]:
            problems.append("normalized tickets do not match ticket IDs in {}".format(rp.TICKETS_FILE))
        return problems

    # 4. normalization deterministic + before LLM
    def normalization_before_llm() -> List[str]:
        problems = []
        expected = [dict(rp.normalize_ticket(t)) for t in tickets]
        if expected != normalized:
            problems.append("normalized_tickets.json differs from deterministic re-normalization of tickets.json")
        stages = [s["stage"] for s in manifest.get("stages", [])]
        if stages != rp.STAGES[: len(stages)]:
            problems.append("stage order in manifest is wrong: {}".format(stages))
        if "FINAL_QUEUE_GENERATED" not in stages:
            problems.append("pipeline never reached FINAL_QUEUE_GENERATED")
        norm_at = manifest.get("normalized_written_at")
        llm_at = manifest.get("llm_call_started_at")
        if not norm_at or not llm_at:
            problems.append("manifest missing normalization/LLM timestamps")
        elif ts(norm_at) > ts(llm_at):
            problems.append("LLM call started before normalization was written")
        if not llm_calls:
            problems.append("no LLM calls logged")
        for call in llm_calls:
            if norm_at and ts(call["timestamp"]) < ts(norm_at):
                problems.append("LLM call at {} precedes normalization".format(call["timestamp"]))
            if rp.NORMALIZED_FILE not in call.get("input_artifacts", []):
                problems.append("LLM call did not consume {}".format(rp.NORMALIZED_FILE))
            missing = {"stage", "timestamp", "provider", "model", "prompt_hash", "input_artifacts",
                       "output_artifact"} - set(call)
            if missing:
                problems.append("LLM call log missing fields: {}".format(sorted(missing)))
        return problems

    # 5. one prediction per ticket
    def one_prediction_each() -> List[str]:
        got = [x.get("ticket_id") for x in preds]
        problems = []
        if sorted(got) != sorted(ticket_ids):
            problems.append("prediction IDs {} != ticket IDs {}".format(got, ticket_ids))
        if [x.get("ticket_id") for x in final] != got:
            problems.append("final queue IDs do not match predictions")
        return problems

    # 6. values constrained by config
    def values_in_config() -> List[str]:
        problems = []
        for x in preds:
            if x.get("category") not in cats:
                problems.append("{} prediction category {!r} not allowed".format(x.get("ticket_id"), x.get("category")))
            if x.get("priority") not in prios:
                problems.append("{} prediction priority {!r} not allowed".format(x.get("ticket_id"), x.get("priority")))
            c = x.get("confidence")
            if not isinstance(c, (int, float)) or not 0.0 <= c <= 1.0:
                problems.append("{} confidence {!r} outside 0..1".format(x.get("ticket_id"), c))
        for x in final:
            if x.get("final_category") not in cats:
                problems.append("{} final category {!r} not allowed".format(x.get("ticket_id"), x.get("final_category")))
            if x.get("final_priority") not in prios:
                problems.append("{} final priority {!r} not allowed".format(x.get("ticket_id"), x.get("final_priority")))
        return problems

    # 7. routing matches config
    def routes_match() -> List[str]:
        problems = []
        for x in preds:
            if x.get("route_to") != routes.get(x.get("category")):
                problems.append("{} prediction route {!r} != config".format(x.get("ticket_id"), x.get("route_to")))
        for x in final:
            if x.get("final_route_to") != routes.get(x.get("final_category")):
                problems.append("{} final route {!r} != config".format(x.get("ticket_id"), x.get("final_route_to")))
        return problems

    # 8. overrides valid and applied
    def overrides_applied() -> List[str]:
        problems = []
        if not isinstance(overrides, list):
            return ["review_overrides.json must be a list"]
        pred_by_id = {x["ticket_id"]: x for x in preds}
        final_by_id = {x["ticket_id"]: x for x in final}
        seen = set()
        for o in overrides:
            tid = o.get("ticket_id")
            if tid not in pred_by_id:
                problems.append("override for unknown ticket {!r}".format(tid))
                continue
            if tid in seen:
                problems.append("duplicate override for {}".format(tid))
            seen.add(tid)
            if o.get("new_category") not in cats or o.get("new_priority") not in prios:
                problems.append("{} override uses values outside config".format(tid))
            if (o.get("old_category"), o.get("old_priority")) != (pred_by_id[tid]["category"], pred_by_id[tid]["priority"]):
                problems.append("{} override old values do not match prediction".format(tid))
            f = final_by_id.get(tid, {})
            if (f.get("final_category"), f.get("final_priority")) != (o.get("new_category"), o.get("new_priority")):
                problems.append("{} override not applied in final queue".format(tid))
            if not f.get("was_overridden"):
                problems.append("{} was_overridden should be true".format(tid))
        for tid, f in final_by_id.items():
            if tid not in seen:
                if f.get("was_overridden"):
                    problems.append("{} marked overridden but has no override".format(tid))
                pr = pred_by_id.get(tid, {})
                if (f.get("final_category"), f.get("final_priority")) != (pr.get("category"), pr.get("priority")):
                    problems.append("{} final values differ from prediction without an override".format(tid))
        return problems

    # 9. reply length
    def reply_length() -> List[str]:
        problems = []
        for x in preds + final:
            n = rp.word_count(x.get("suggested_reply", ""))
            if n > max_words:
                problems.append("{} reply has {} words (max {})".format(x.get("ticket_id"), n, max_words))
        return problems

    # 10. escalations deterministic
    def escalations_correct() -> List[str]:
        conf = {x["ticket_id"]: x.get("confidence", 0.0) for x in preds}
        expected = sorted(
            x["ticket_id"] for x in final
            if x["final_category"] == rp.FALLBACK_CATEGORY or conf[x["ticket_id"]] < rp.ESCALATION_CONFIDENCE_THRESHOLD
        )
        got = sorted(e.get("ticket_id") for e in data.get(rp.ESCALATIONS_FILE, []))
        return [] if got == expected else ["escalations {} != expected {}".format(got, expected)]

    # 11. structured chain of thought: one record per ticket, grounding flags recomputed in code
    def reasoning_recorded() -> List[str]:
        reasoning = data.get(rp.REASONING_FILE, {}).get("tickets", [])
        problems = []
        if [r.get("ticket_id") for r in reasoning] != [n["ticket_id"] for n in normalized]:
            problems.append("reasoning records do not match tickets one-to-one")
        texts = {n["ticket_id"]: n["text_for_model"].lower() for n in normalized}
        for r in reasoning:
            ungrounded = [e for e in r.get("evidence", []) if e.strip("\"'").lower() not in texts.get(r.get("ticket_id"), "")]
            if ungrounded != r.get("ungrounded_evidence"):
                problems.append("{} ungrounded_evidence flag is wrong".format(r.get("ticket_id")))
        return problems

    def summary_contents() -> List[str]:
        with open(p(rp.SUMMARY_FILE), encoding="utf-8") as fh:
            text = fh.read()
        problems = []
        if "**Total tickets:** {}".format(len(final)) not in text:
            problems.append("summary total count missing or wrong")
        for heading in ("By final category", "By final priority", "Queue breakdown by destination", "Overridden tickets"):
            if heading not in text:
                problems.append("summary missing section '{}'".format(heading))
        return problems

    check("inputs_read_from_disk", inputs_from_disk)
    check("normalization_before_llm", normalization_before_llm)
    check("one_prediction_per_ticket", one_prediction_each)
    check("values_restricted_to_config", values_in_config)
    check("routes_match_config", routes_match)
    check("overrides_valid_and_applied", overrides_applied)
    check("reply_length_within_max_words", reply_length)
    check("escalations_deterministic", escalations_correct)
    check("reasoning_recorded", reasoning_recorded)
    check("summary_contents", summary_contents)

    return {
        "validated_at": rp.now_iso(),
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }


def main() -> int:
    report = run_checks(rp.BASE_DIR)
    rp.write_json(rp.VALIDATION_REPORT_FILE, report)
    for c in report["checks"]:
        print("[{}] {:<32} {}".format("PASS" if c["passed"] else "FAIL", c["name"], c["detail"]))
    print("\nOverall: {}".format("PASSED" if report["passed"] else "FAILED"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
