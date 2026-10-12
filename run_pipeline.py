"""Replayable support-triage pipeline.

Stages (enforced in code, see Pipeline.advance):
    INIT -> INPUTS_LOADED -> TICKETS_NORMALIZED -> TRIAGE_PREDICTED
         -> HUMAN_REVIEW_COMPLETE -> FINAL_QUEUE_GENERATED
         -> VALIDATION_COMPLETE -> RESULTS_FINALISED

Usage:
    python run_pipeline.py                     # Gemini (needs GEMINI_API_KEY in .env)
    python run_pipeline.py --provider mock     # offline deterministic stand-in, for testing without a key
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TICKETS_FILE = "tickets.json"
CONFIG_FILE = "triage_config.json"
NORMALIZED_FILE = "normalized_tickets.json"
PREDICTIONS_FILE = "triage_predictions.json"
REASONING_FILE = "triage_reasoning.json"
OVERRIDES_FILE = "review_overrides.json"
FINAL_QUEUE_FILE = "final_queue.json"
SUMMARY_FILE = "queue_summary.md"
ESCALATIONS_FILE = "escalations.json"
LLM_CALLS_FILE = "llm_calls.jsonl"
LLM_RAW_FILE = "llm_raw_response.txt"
MANIFEST_FILE = "run_manifest.json"
VALIDATION_REPORT_FILE = "validation_report.json"
PIPELINE_LOG_FILE = "pipeline.log"

GENERATED_FILES = [
    NORMALIZED_FILE, PREDICTIONS_FILE, REASONING_FILE, OVERRIDES_FILE, FINAL_QUEUE_FILE, SUMMARY_FILE,
    ESCALATIONS_FILE, LLM_CALLS_FILE, LLM_RAW_FILE, MANIFEST_FILE, VALIDATION_REPORT_FILE, PIPELINE_LOG_FILE,
]

STAGES = [
    "INIT",
    "INPUTS_LOADED",
    "TICKETS_NORMALIZED",
    "TRIAGE_PREDICTED",
    "HUMAN_REVIEW_COMPLETE",
    "FINAL_QUEUE_GENERATED",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]

ESCALATION_CONFIDENCE_THRESHOLD = 0.60
FALLBACK_CATEGORY = "other"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
DEFAULT_MODEL = "gemini-3.8-flash"
GEMINI_THINKING_LEVEL = "medium"
COT_FIELDS = ["customer_issue", "evidence", "category_reasoning", "urgency_signals", "priority_reasoning"]
LLM_PARSE_ATTEMPTS = 2
HTTP_ATTEMPTS = 5

REVIEW_PROMPT = (
    "Enter any overrides as: ticket_id,category,priority\n"
    "Press Enter on an empty line when done."
)


# --------------------------------------------------------------------------- helpers

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def path(name: str) -> str:
    return os.path.join(BASE_DIR, name)


def sha256_file(name: str) -> str:
    with open(path(name), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_json(name: str) -> Any:
    with open(path(name), "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(name: str, data: Any) -> None:
    with open(path(name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_dotenv(name: str = ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ; real environment variables win."""
    if not os.path.exists(path(name)):
        return
    with open(path(name), "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.replace("export ", "", 1).strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)


def word_count(text: str) -> int:
    return len(text.split())


class PipelineError(Exception):
    pass


# --------------------------------------------------------------------------- stage machine

class Pipeline:
    """Holds run state and refuses to move between stages out of order."""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.stage = "INIT"
        self.manifest: Dict[str, Any] = {
            "run_started_at": now_iso(),
            "provider": provider,
            "model": model,
            "stages": [{"stage": "INIT", "timestamp": now_iso()}],
            "input_hashes": {},
        }
        self.tickets: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {}
        self.normalized: List[Dict[str, Any]] = []
        self.predictions: List[Dict[str, Any]] = []
        self.reasoning: List[Dict[str, Any]] = []
        self.overrides: List[Dict[str, Any]] = []
        self.final_queue: List[Dict[str, Any]] = []
        self.log_lines: List[str] = []

    def advance(self, to_stage: str) -> None:
        expected = STAGES[STAGES.index(self.stage) + 1]
        if to_stage != expected:
            raise PipelineError(
                "Illegal stage transition {} -> {} (expected {})".format(self.stage, to_stage, expected)
            )
        self.stage = to_stage
        self.manifest["stages"].append({"stage": to_stage, "timestamp": now_iso()})
        self.write_manifest()
        self.log("Stage -> {}".format(to_stage))

    def require(self, stage: str) -> None:
        if self.stage != stage:
            raise PipelineError("Step requires stage {} but pipeline is at {}".format(stage, self.stage))

    def log(self, message: str) -> None:
        line = "{} {}".format(now_iso(), message)
        self.log_lines.append(line)
        with open(path(PIPELINE_LOG_FILE), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_manifest(self) -> None:
        write_json(MANIFEST_FILE, self.manifest)


# --------------------------------------------------------------------------- 1. load + normalize

def load_inputs(p: Pipeline) -> None:
    p.require("INIT")
    for name in (TICKETS_FILE, CONFIG_FILE):
        if not os.path.exists(path(name)):
            raise PipelineError("Missing input file: {}".format(name))

    tickets = read_json(TICKETS_FILE)
    config = read_json(CONFIG_FILE)
    if not isinstance(tickets, list):
        raise PipelineError("{} must contain a JSON array".format(TICKETS_FILE))
    validate_config(config)

    ids = [t.get("ticket_id") for t in tickets]
    if any(not isinstance(i, str) or not i.strip() for i in ids):
        raise PipelineError("Every ticket needs a non-empty string ticket_id")
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise PipelineError("Duplicate ticket_id(s): {}".format(", ".join(dupes)))

    p.tickets, p.config = tickets, config
    p.manifest["input_hashes"] = {
        TICKETS_FILE: sha256_file(TICKETS_FILE),
        CONFIG_FILE: sha256_file(CONFIG_FILE),
    }
    p.manifest["inputs_read_at"] = now_iso()
    p.log("Loaded {} tickets and config from disk".format(len(tickets)))
    p.advance("INPUTS_LOADED")


def validate_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise PipelineError("{} must be a JSON object".format(CONFIG_FILE))
    for key in ("allowed_categories", "allowed_priorities", "reply_style", "routing_rules"):
        if key not in config:
            raise PipelineError("{} missing key: {}".format(CONFIG_FILE, key))
    cats, prios = config["allowed_categories"], config["allowed_priorities"]
    if not cats or not prios:
        raise PipelineError("allowed_categories and allowed_priorities must be non-empty")
    missing_routes = [c for c in cats if c not in config["routing_rules"]]
    if missing_routes:
        raise PipelineError("routing_rules has no route for: {}".format(", ".join(missing_routes)))
    if not isinstance(config["reply_style"].get("max_words"), int):
        raise PipelineError("reply_style.max_words must be an integer")


def clean_text(value: Any) -> str:
    """Deterministic text cleanup: NFKC, strip control chars, collapse whitespace."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    subject = clean_text(ticket.get("subject"))
    message = clean_text(ticket.get("message"))
    text_for_model = "Subject: {}\nMessage: {}".format(subject, message)
    return OrderedDict([
        ("ticket_id", clean_text(ticket.get("ticket_id"))),
        ("subject", subject),
        ("message", message),
        ("channel", clean_text(ticket.get("channel")).lower()),
        ("created_at", clean_text(ticket.get("created_at"))),
        ("text_for_model", text_for_model),
        ("char_count", len(text_for_model)),
    ])


def normalize_tickets(p: Pipeline) -> None:
    p.require("INPUTS_LOADED")
    p.normalized = [normalize_ticket(t) for t in p.tickets]
    write_json(NORMALIZED_FILE, p.normalized)
    p.manifest["normalized_written_at"] = now_iso()
    p.log("Wrote {} ({} tickets)".format(NORMALIZED_FILE, len(p.normalized)))
    p.advance("TICKETS_NORMALIZED")


# --------------------------------------------------------------------------- 2. triage prediction

SYSTEM_PROMPT = """You are a customer support triage assistant.
For every ticket you receive, choose exactly one category and one priority from the allowed values in the
triage configuration, explain the choice briefly, give a confidence between 0.0 and 1.0, and draft a suggested
reply to the customer.

Work through each ticket step by step, filling the analysis fields before the decision fields:
1. customer_issue: restate in one sentence what the customer needs.
2. evidence: copy 1-3 short phrases VERBATIM from the ticket text that drive your decision.
3. category_reasoning: name the one or two most plausible categories and say why the evidence fits one better.
4. urgency_signals: list concrete signals of urgency or impact found in the ticket (money at risk, locked out,
   time pressure, security, many users affected). Use an empty list if there are none.
5. priority_reasoning: weigh those signals against the allowed priorities.
6. Only then fill category, priority, confidence, reason and suggested_reply, consistent with the analysis.

Rules:
- category must be one of allowed_categories; use "other" when nothing fits well.
- priority must be one of allowed_priorities. Reserve the most urgent level for money at risk, locked-out users
  with time pressure, or security concerns.
- confidence reflects how sure you are of the category; use a value below 0.6 when the ticket is ambiguous.
- suggested_reply must match reply_style.tone and contain at most reply_style.max_words words. Do not promise
  refunds, timelines or actions you cannot verify, and do not ask for passwords or full card numbers.
- Return exactly one prediction per ticket_id, in the same order as the input.
- Routing is computed by the system from routing_rules; you do not need to choose a queue."""


def build_user_prompt(config: Dict[str, Any], normalized: List[Dict[str, Any]]) -> str:
    tickets_for_model = [
        {"ticket_id": t["ticket_id"], "channel": t["channel"], "text": t["text_for_model"]}
        for t in normalized
    ]
    return (
        "<triage_config>\n{}\n</triage_config>\n\n<tickets>\n{}\n</tickets>".format(
            json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False),
            json.dumps(tickets_for_model, indent=2, ensure_ascii=False),
        )
    )


def response_schema(config: Dict[str, Any], ticket_ids: List[str]) -> Dict[str, Any]:
    """Gemini responseSchema (OpenAPI subset); enums come from the config so only allowed values decode.

    propertyOrdering puts the reasoning fields first, so the model writes its analysis (structured
    chain of thought) before it commits to a category and priority.
    """
    fields = ["ticket_id"] + COT_FIELDS + ["category", "priority", "confidence", "reason", "suggested_reply"]
    return {
        "type": "OBJECT",
        "properties": {
            "predictions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "ticket_id": {"type": "STRING", "enum": ticket_ids},
                        "customer_issue": {"type": "STRING"},
                        "evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "category_reasoning": {"type": "STRING"},
                        "urgency_signals": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "priority_reasoning": {"type": "STRING"},
                        "category": {"type": "STRING", "enum": list(config["allowed_categories"])},
                        "priority": {"type": "STRING", "enum": list(config["allowed_priorities"])},
                        "confidence": {"type": "NUMBER"},
                        "reason": {"type": "STRING"},
                        "suggested_reply": {"type": "STRING"},
                    },
                    "required": fields,
                    "propertyOrdering": fields,
                },
            }
        },
        "required": ["predictions"],
    }


def call_gemini(model: str, system: str, user: str, schema: Dict[str, Any]) -> Tuple[str, str]:
    """Return (JSON answer text, native thought summary)."""
    import time
    import requests

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise PipelineError(
            "GEMINI_API_KEY is not set. Add it to .env, or run with --provider mock for an offline run."
        )

    # responseSchema makes Gemini decode against the schema; every field is still
    # re-validated in code by coerce_prediction().
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
            # Native thinking: the model reasons internally before answering; includeThoughts returns
            # a summary of that reasoning as separate parts flagged "thought": true.
            "thinkingConfig": {"thinkingLevel": GEMINI_THINKING_LEVEL, "includeThoughts": True},
        },
    }
    headers = {"x-goog-api-key": api_key}

    last_error = ""
    for attempt in range(HTTP_ATTEMPTS):
        if attempt:
            wait = 2 ** attempt  # 2, 4, 8, 16s: rides out short "high demand" 503 spikes
            print("  Gemini unavailable ({}), retrying in {}s ...".format(last_error[:60], wait))
            time.sleep(wait)
        try:
            response = requests.post(GEMINI_URL.format(model), headers=headers, json=body, timeout=300)
        except requests.RequestException as e:
            last_error = "could not reach Gemini API: {}".format(e)
        else:
            if response.status_code in (401, 403):
                raise PipelineError("Gemini API rejected the key (HTTP {}). Check GEMINI_API_KEY.".format(response.status_code))
            if response.status_code == 200:
                candidates = response.json().get("candidates") or [{}]
                parts = candidates[0].get("content", {}).get("parts") or []
                answer = "".join(part.get("text", "") for part in parts if not part.get("thought"))
                thoughts = "\n".join(part.get("text", "") for part in parts if part.get("thought"))
                return answer, thoughts
            last_error = "Gemini API error {}: {}".format(response.status_code, response.text[:500])
            if response.status_code == 429 and gemini_retry_delay(response) > 60:
                # Daily free-tier quota: retrying within seconds cannot succeed.
                raise PipelineError(
                    "Gemini quota exhausted for {} (retry in ~{}s). Set GEMINI_MODEL in .env to another "
                    "model, enable billing, or wait for the reset.".format(model, gemini_retry_delay(response)))
            if response.status_code < 500 and response.status_code != 429:
                break  # other 4xx errors will not succeed on retry
    raise PipelineError(last_error)


def gemini_retry_delay(response: Any) -> int:
    """Seconds Gemini asks us to wait (RetryInfo.retryDelay, e.g. "64529s"); 0 if not given."""
    try:
        for detail in response.json()["error"].get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                return int(float(detail.get("retryDelay", "0s").rstrip("s")))
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    return 0


MOCK_KEYWORDS = [
    ("billing_issue", ["charge", "charged", "refund", "deposit", "payment", "invoice", "card", "withdraw"]),
    ("account_access", ["log in", "login", "password", "locked", "2fa", "credentials", "verify", "access"]),
    ("bug_report", ["crash", "error", "bug", "broken", "freeze", "closes", "not working"]),
    ("product_how_to", ["how do i", "how to", "where can i", "export", "download", "change my"]),
]
MOCK_URGENT = ["urgent", "urgently", "asap", "immediately", "charged twice", "fraud", "stolen"]


def call_mock(config: Dict[str, Any], normalized: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Deterministic keyword stand-in for the LLM, so the pipeline can be exercised offline."""
    cats, prios = config["allowed_categories"], config["allowed_priorities"]
    max_words = config["reply_style"]["max_words"]
    preds = []
    for t in normalized:
        text = t["text_for_model"].lower()
        scores = [(sum(k in text for k in kws), cat) for cat, kws in MOCK_KEYWORDS if cat in cats]
        best_score, best_cat = max(scores) if scores else (0, FALLBACK_CATEGORY)
        category = best_cat if best_score > 0 else (FALLBACK_CATEGORY if FALLBACK_CATEGORY in cats else cats[-1])
        if any(k in text for k in MOCK_URGENT):
            priority = prios[0]
        elif category in ("billing_issue", "account_access", "bug_report"):
            priority = prios[min(1, len(prios) - 1)]
        else:
            priority = prios[min(2, len(prios) - 1)]
        reply = (
            "Hi, thank you for contacting us about \"{}\". We have passed your ticket to the right team, "
            "who will review it and reply shortly. Please do not share passwords or full card numbers."
        ).format(t["subject"])
        matched = [k for _, kws in MOCK_KEYWORDS for k in kws if k in text]
        urgent = [k for k in MOCK_URGENT if k in text]
        preds.append({
            "ticket_id": t["ticket_id"],
            "customer_issue": t["subject"],
            "evidence": matched[:3],
            "category_reasoning": "Keyword matches: {}".format(", ".join(matched) or "none"),
            "urgency_signals": urgent,
            "priority_reasoning": "Urgent keywords found" if urgent else "No urgent keywords",
            "category": category,
            "priority": priority,
            "confidence": round(min(0.5 + 0.15 * best_score, 0.95), 2),
            "reason": "Mock keyword match (score {}) for {}".format(best_score, category),
            "suggested_reply": " ".join(reply.split()[:max_words]),
        })
    return json.dumps({"predictions": preds}), ""


def coerce_prediction(
    raw: Any, ticket_id: str, config: Dict[str, Any], issues: List[str]
) -> Dict[str, Any]:
    """Validate one model prediction; anything malformed falls back to manual review."""
    cats, prios = config["allowed_categories"], config["allowed_priorities"]
    max_words = config["reply_style"]["max_words"]
    fallback_priority = "normal" if "normal" in prios else prios[len(prios) // 2]
    fallback_category = FALLBACK_CATEGORY if FALLBACK_CATEGORY in cats else cats[-1]

    def fallback(problem: str) -> Dict[str, Any]:
        issues.append("{}: {} -> routed to {}".format(ticket_id, problem, fallback_category))
        return {
            "ticket_id": ticket_id,
            "category": fallback_category,
            "priority": fallback_priority,
            "reason": "Automatic fallback: {}".format(problem),
            "suggested_reply": "",
            "route_to": config["routing_rules"][fallback_category],
            "confidence": 0.0,
            "llm_output_valid": False,
        }

    if raw is None:
        return fallback("missing from LLM response")
    if not isinstance(raw, dict):
        return fallback("prediction is not an object")
    category, priority = raw.get("category"), raw.get("priority")
    if category not in cats:
        return fallback("invalid category {!r}".format(category))
    if priority not in prios:
        return fallback("invalid priority {!r}".format(priority))

    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        issues.append("{}: missing/invalid confidence, set to 0.0".format(ticket_id))
        confidence = 0.0
    confidence = round(min(max(confidence, 0.0), 1.0), 3)

    reply = clean_text(raw.get("suggested_reply"))
    if word_count(reply) > max_words:
        issues.append("{}: reply had {} words, truncated to {}".format(ticket_id, word_count(reply), max_words))
        reply = " ".join(reply.split()[:max_words])
    if not reply:
        issues.append("{}: empty suggested_reply".format(ticket_id))

    return {
        "ticket_id": ticket_id,
        "category": category,
        "priority": priority,
        "reason": clean_text(raw.get("reason")),
        "suggested_reply": reply,
        "route_to": config["routing_rules"][category],  # derived from config, never from the model
        "confidence": confidence,
        "llm_output_valid": True,
    }


def build_reasoning(raw: Any, ticket: Dict[str, Any], issues: List[str]) -> Dict[str, Any]:
    """Keep the model's structured chain of thought, and check in code that its evidence is grounded."""
    raw = raw if isinstance(raw, dict) else {}

    def text_list(value: Any) -> List[str]:
        return [clean_text(v) for v in value if clean_text(v)] if isinstance(value, list) else []

    evidence = text_list(raw.get("evidence"))
    haystack = ticket["text_for_model"].lower()
    ungrounded = [e for e in evidence if e.strip("\"'").lower() not in haystack]
    if ungrounded:
        issues.append("{}: evidence not found verbatim in ticket: {}".format(ticket["ticket_id"], ungrounded))
    return OrderedDict([
        ("ticket_id", ticket["ticket_id"]),
        ("customer_issue", clean_text(raw.get("customer_issue"))),
        ("evidence", evidence),
        ("evidence_grounded", bool(evidence) and not ungrounded),
        ("ungrounded_evidence", ungrounded),
        ("category_reasoning", clean_text(raw.get("category_reasoning"))),
        ("urgency_signals", text_list(raw.get("urgency_signals"))),
        ("priority_reasoning", clean_text(raw.get("priority_reasoning"))),
    ])


def parse_predictions(raw_text: str, ticket_ids: List[str], issues: List[str]) -> Dict[str, Any]:
    parsed = json.loads(raw_text)
    items = parsed.get("predictions") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        raise ValueError("no 'predictions' array")
    by_id: Dict[str, Any] = {}
    for item in items:
        tid = item.get("ticket_id") if isinstance(item, dict) else None
        if tid not in ticket_ids:
            issues.append("Ignored prediction for unknown ticket_id {!r}".format(tid))
        elif tid in by_id:
            issues.append("{}: duplicate prediction ignored".format(tid))
        else:
            by_id[tid] = item
    return by_id


def log_llm_call(p: Pipeline, started: str, prompt_hash: str) -> None:
    with open(path(LLM_CALLS_FILE), "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "stage": "TRIAGE_PREDICTED",
            "timestamp": started,
            "provider": "google" if p.provider == "gemini" else "mock",
            "model": p.model if p.provider == "gemini" else "keyword-mock",
            "prompt_hash": prompt_hash,
            "input_artifacts": [NORMALIZED_FILE, CONFIG_FILE],
            "output_artifact": PREDICTIONS_FILE,
        }) + "\n")


def predict_triage(p: Pipeline) -> None:
    p.require("TICKETS_NORMALIZED")
    # Normalization must be on disk before the LLM sees anything.
    if not os.path.exists(path(NORMALIZED_FILE)):
        raise PipelineError("normalized_tickets.json must exist before the LLM call")

    ticket_ids = [t["ticket_id"] for t in p.normalized]
    user_prompt = build_user_prompt(p.config, p.normalized)
    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n\n" + user_prompt).encode("utf-8")).hexdigest()

    issues: List[str] = []
    by_id: Dict[str, Any] = {}
    call_started = ""
    raw_responses = []
    thoughts = ""
    # One batch call; a second attempt is made only if the whole response is unparseable.
    for attempt in range(1, LLM_PARSE_ATTEMPTS + 1):
        started = now_iso()
        call_started = call_started or started
        if p.provider == "mock":
            raw_text, thoughts = call_mock(p.config, p.normalized)
        else:
            raw_text, thoughts = call_gemini(p.model, SYSTEM_PROMPT, user_prompt, response_schema(p.config, ticket_ids))
        raw_responses.append("=== attempt {} ({}) ===\n--- thought summary ---\n{}\n--- answer ---\n{}\n".format(
            attempt, started, thoughts, raw_text))
        log_llm_call(p, started, prompt_hash)
        try:
            by_id = parse_predictions(raw_text, ticket_ids, issues)
            break
        except (ValueError, AttributeError) as e:
            issues.append("Attempt {}: could not parse LLM response ({})".format(attempt, e))
    else:
        issues.append("No parseable LLM response; all tickets fall back")

    with open(path(LLM_RAW_FILE), "w", encoding="utf-8") as f:
        f.write("\n".join(raw_responses))

    p.predictions = [coerce_prediction(by_id.get(tid), tid, p.config, issues) for tid in ticket_ids]
    write_json(PREDICTIONS_FILE, p.predictions)

    p.reasoning = [build_reasoning(by_id.get(t["ticket_id"]), t, issues) for t in p.normalized]
    write_json(REASONING_FILE, OrderedDict([
        ("thinking_level", GEMINI_THINKING_LEVEL if p.provider == "gemini" else None),
        ("native_thought_summary", thoughts),
        ("tickets", p.reasoning),
    ]))

    p.manifest["llm_call_started_at"] = call_started
    p.manifest["prediction_issues"] = issues
    for issue in issues:
        p.log("PREDICTION ISSUE " + issue)
    p.log("Wrote {}".format(PREDICTIONS_FILE))
    p.advance("TRIAGE_PREDICTED")


# --------------------------------------------------------------------------- 3. human review

def parse_override(line: str, p: Pipeline, current: Dict[str, Dict[str, str]]) -> Tuple[Optional[Tuple[str, str, str]], str]:
    parts = [x.strip() for x in line.split(",")]
    if len(parts) != 3:
        return None, "expected 3 comma-separated values: ticket_id,category,priority"
    tid, cat, prio = parts
    if tid not in current:
        return None, "unknown ticket_id {!r}".format(tid)
    cat = cat or current[tid]["category"]   # blank field = keep current value
    prio = prio or current[tid]["priority"]
    if cat not in p.config["allowed_categories"]:
        return None, "category must be one of: {}".format(", ".join(p.config["allowed_categories"]))
    if prio not in p.config["allowed_priorities"]:
        return None, "priority must be one of: {}".format(", ".join(p.config["allowed_priorities"]))
    return (tid, cat, prio), ""


def human_review(p: Pipeline) -> None:
    p.require("TRIAGE_PREDICTED")
    print("\n=== Triage predictions for review ===")
    print("{:<12} {:<18} {:<8} {:<6} {}".format("ticket_id", "category", "priority", "conf", "subject"))
    subjects = {t["ticket_id"]: t["subject"] for t in p.normalized}
    for pred in p.predictions:
        print("{:<12} {:<18} {:<8} {:<6} {}".format(
            pred["ticket_id"], pred["category"], pred["priority"],
            "{:.2f}".format(pred["confidence"]), subjects[pred["ticket_id"]][:50],
        ))

    print("\n--- Model reasoning ---")
    for r in p.reasoning:
        print("{}: {}".format(r["ticket_id"], r["category_reasoning"] or "(none)"))
        print("    urgency: {}{}".format(
            "; ".join(r["urgency_signals"]) or "none",
            "" if r["evidence_grounded"] else "   [evidence not grounded in ticket]"))
    print("\nAllowed categories: {}".format(", ".join(p.config["allowed_categories"])))
    print("Allowed priorities: {}".format(", ".join(p.config["allowed_priorities"])))
    print(REVIEW_PROMPT)

    original = {x["ticket_id"]: {"category": x["category"], "priority": x["priority"]} for x in p.predictions}
    current = {tid: dict(v) for tid, v in original.items()}
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        result, error = parse_override(line, p, current)
        if error:
            print("  Rejected: {}".format(error))
            continue
        tid, cat, prio = result
        current[tid] = {"category": cat, "priority": prio}
        same = current[tid] == original[tid]
        print("  Accepted: {} -> {} / {}{}".format(tid, cat, prio, " (same as prediction, not an override)" if same else ""))

    p.overrides = [
        OrderedDict([
            ("ticket_id", tid),
            ("old_category", original[tid]["category"]),
            ("new_category", current[tid]["category"]),
            ("old_priority", original[tid]["priority"]),
            ("new_priority", current[tid]["priority"]),
        ])
        for tid in original
        if current[tid] != original[tid]
    ]
    write_json(OVERRIDES_FILE, p.overrides)
    p.log("Wrote {} ({} overrides)".format(OVERRIDES_FILE, len(p.overrides)))
    p.advance("HUMAN_REVIEW_COMPLETE")


# --------------------------------------------------------------------------- 4/5. final queue, summary, escalations

def generate_final_queue(p: Pipeline) -> None:
    p.require("HUMAN_REVIEW_COMPLETE")
    overrides = {o["ticket_id"]: o for o in p.overrides}
    routes = p.config["routing_rules"]

    p.final_queue = []
    for pred in p.predictions:
        o = overrides.get(pred["ticket_id"])
        category = o["new_category"] if o else pred["category"]
        priority = o["new_priority"] if o else pred["priority"]
        p.final_queue.append(OrderedDict([
            ("ticket_id", pred["ticket_id"]),
            ("final_category", category),
            ("final_priority", priority),
            ("final_route_to", routes[category]),  # recomputed so category overrides change routing
            ("suggested_reply", pred["suggested_reply"]),
            ("was_overridden", o is not None),
        ]))
    write_json(FINAL_QUEUE_FILE, p.final_queue)

    confidence = {x["ticket_id"]: x["confidence"] for x in p.predictions}
    escalations = []
    for item in p.final_queue:
        reasons = []
        if item["final_category"] == FALLBACK_CATEGORY:
            reasons.append("category is '{}'".format(FALLBACK_CATEGORY))
        if confidence[item["ticket_id"]] < ESCALATION_CONFIDENCE_THRESHOLD:
            reasons.append("confidence {:.2f} < {:.2f}".format(confidence[item["ticket_id"]], ESCALATION_CONFIDENCE_THRESHOLD))
        if reasons:
            escalations.append(OrderedDict([
                ("ticket_id", item["ticket_id"]),
                ("final_category", item["final_category"]),
                ("final_priority", item["final_priority"]),
                ("confidence", confidence[item["ticket_id"]]),
                ("reasons", reasons),
            ]))
    write_json(ESCALATIONS_FILE, escalations)

    write_summary(p, escalations)
    p.log("Wrote {}, {}, {}".format(FINAL_QUEUE_FILE, ESCALATIONS_FILE, SUMMARY_FILE))
    p.advance("FINAL_QUEUE_GENERATED")


def count_table(header: str, counts: Counter, order: List[str]) -> List[str]:
    lines = ["| {} | Count |".format(header), "|---|---|"]
    for key in order:
        lines.append("| {} | {} |".format(key, counts.get(key, 0)))
    return lines


def write_summary(p: Pipeline, escalations: List[Dict[str, Any]]) -> None:
    q = p.final_queue
    cats = p.config["allowed_categories"]
    prios = p.config["allowed_priorities"]
    destinations = list(OrderedDict.fromkeys(p.config["routing_rules"][c] for c in cats))

    lines = ["# Support Queue Summary", "", "Generated: {}".format(now_iso()), "",
             "**Total tickets:** {}".format(len(q)), "", "## By final category", ""]
    lines += count_table("Category", Counter(x["final_category"] for x in q), cats)
    lines += ["", "## By final priority", ""]
    lines += count_table("Priority", Counter(x["final_priority"] for x in q), prios)

    lines += ["", "## Queue breakdown by destination", ""]
    for dest in destinations:
        items = [x for x in q if x["final_route_to"] == dest]
        lines.append("### {} ({})".format(dest, len(items)))
        for x in sorted(items, key=lambda x: prios.index(x["final_priority"])):
            lines.append("- {} [{}] {}".format(x["ticket_id"], x["final_priority"], x["final_category"]))
        if not items:
            lines.append("- _empty_")
        lines.append("")

    lines += ["## Overridden tickets", ""]
    if p.overrides:
        lines += ["| Ticket | Category | Priority |", "|---|---|---|"]
        for o in p.overrides:
            lines.append("| {} | {} -> {} | {} -> {} |".format(
                o["ticket_id"], o["old_category"], o["new_category"], o["old_priority"], o["new_priority"]))
        lines += ["", "_Suggested replies were drafted for the original category; check them before sending._"]
    else:
        lines.append("No tickets were overridden.")

    lines += ["", "## Escalations ({})".format(len(escalations)), ""]
    if escalations:
        for e in escalations:
            lines.append("- {}: {}".format(e["ticket_id"], "; ".join(e["reasons"])))
    else:
        lines.append("No tickets flagged for manual escalation.")

    issues = p.manifest.get("prediction_issues") or []
    if issues:
        lines += ["", "## LLM output issues", ""] + ["- {}".format(i) for i in issues]

    with open(path(SUMMARY_FILE), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- 6. validation + finalise

def run_validation(p: Pipeline) -> bool:
    p.require("FINAL_QUEUE_GENERATED")
    import validate  # local module

    report = validate.run_checks(BASE_DIR)
    write_json(VALIDATION_REPORT_FILE, report)
    p.manifest["validation_passed"] = report["passed"]
    for check in report["checks"]:
        if not check["passed"]:
            p.log("VALIDATION FAILED {}: {}".format(check["name"], check["detail"]))
    p.log("Validation {}".format("passed" if report["passed"] else "FAILED"))
    p.advance("VALIDATION_COMPLETE")
    return report["passed"]


def finalise(p: Pipeline) -> None:
    p.require("VALIDATION_COMPLETE")
    p.manifest["run_finished_at"] = now_iso()
    p.manifest["output_hashes"] = {
        name: sha256_file(name)
        for name in (NORMALIZED_FILE, PREDICTIONS_FILE, OVERRIDES_FILE, FINAL_QUEUE_FILE, SUMMARY_FILE, ESCALATIONS_FILE)
    }
    p.advance("RESULTS_FINALISED")


def clean_previous_outputs() -> None:
    for name in GENERATED_FILES:
        if os.path.exists(path(name)):
            os.remove(path(name))


def main() -> int:
    parser = argparse.ArgumentParser(description="Support triage pipeline")
    parser.add_argument("--provider", choices=["gemini", "mock"], default="gemini",
                        help="LLM provider (default: gemini; 'mock' is an offline keyword stand-in)")
    parser.add_argument("--model", default=None,
                        help="Gemini model (default: $GEMINI_MODEL or {})".format(DEFAULT_MODEL))
    args = parser.parse_args()
    load_dotenv()
    args.model = args.model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    clean_previous_outputs()  # every run regenerates all artifacts from the inputs
    p = Pipeline(args.provider, args.model)
    try:
        load_inputs(p)
        normalize_tickets(p)
        predict_triage(p)
        human_review(p)
        generate_final_queue(p)
        passed = run_validation(p)
        finalise(p)
    except PipelineError as e:
        p.log("ERROR " + str(e))
        print("Pipeline error: {}".format(e), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        p.log("ABORTED by user at stage {}".format(p.stage))
        print("\nAborted at stage {}; final queue was not generated.".format(p.stage), file=sys.stderr)
        return 130

    print("\nDone. {} tickets, {} overrides. Validation: {}.".format(
        len(p.final_queue), len(p.overrides), "PASSED" if passed else "FAILED (see validation_report.json)"))
    print("Summary: {}".format(SUMMARY_FILE))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
