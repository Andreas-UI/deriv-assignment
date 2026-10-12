# Support Triage Pipeline

A replayable pipeline that reads `tickets.json` and `triage_config.json` from disk. It normalizes the tickets deterministically, then classifies and drafts replies for all of them in one LLM call (Google Gemini, `gemini-3.8-flash` by default). A person reviews the predictions in the terminal before the final agent queue is produced.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env                     # then put your key in .env
```

The pipeline reads `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`) from `.env`. A variable already set in the environment takes precedence. `.env` is listed in `.gitignore`.

The Gemini free tier allows only a small number of requests per model per day (20 for `gemini-3.8-flash`). Each run uses one request. When the daily quota is used up, the pipeline stops straight away with a clear message. To keep going, set `GEMINI_MODEL` to another Flash model (for example `gemini-3.5-flash`) or enable billing.

## Run

```bash
python run_pipeline.py                   # real run with Gemini
python run_pipeline.py --provider mock   # offline run with a deterministic keyword stand-in (no API key)
python validate.py                       # re-check the artifacts from the last run
```

Each run deletes the previously generated artifacts and rebuilds them all from the two input files.

During review, enter overrides as `ticket_id,category,priority`, one per line. Leave a field blank to keep its value (for example `T-1002,,normal`). Press Enter on an empty line to finish. Invalid values are rejected and you are asked again. The review step also accepts piped input:

```bash
printf 'T-1003,bug_report,high\n\n' | python run_pipeline.py --provider mock
```

## Stages

`INIT → INPUTS_LOADED → TICKETS_NORMALIZED → TRIAGE_PREDICTED → HUMAN_REVIEW_COMPLETE → FINAL_QUEUE_GENERATED → VALIDATION_COMPLETE → RESULTS_FINALISED`

`Pipeline.advance()` refuses any transition except the next stage. Each step calls `require()` to check that the pipeline is at the expected stage before it runs. Every transition is timestamped in `run_manifest.json`.

## Design notes

- **Normalization is plain code:** NFKC, removal of control characters, and whitespace collapsing. `text_for_model = "Subject: …\nMessage: …"`. The file is written to disk before the LLM is called, and `validate.py` re-runs normalization to confirm the output is identical.
- **Allowed labels are enforced in two places:**
  - The request sends a JSON schema whose `enum`s for `category`, `priority` and `ticket_id` are built from the config. Gemini enforces it during generation (`responseSchema`, temperature 0).
  - Every prediction is checked again in code.
- **Chain of thought is used in two ways, and both are recorded:**
  - *Structured CoT.* The response schema has analysis fields: `customer_issue`, verbatim `evidence`, `category_reasoning`, `urgency_signals` and `priority_reasoning`. `propertyOrdering` makes Gemini fill these in before `category` and `priority`, so the labels are produced after the reasoning.
  - *Native thinking.* `thinkingConfig.thinkingLevel = "medium"` lets Gemini reason internally before answering. `includeThoughts` returns a summary of that reasoning, which is kept separate from the JSON answer.
  - Code checks that every evidence quote appears word for word in the ticket text, and flags any that don't. The reviewer sees the category reasoning and urgency signals at the review step. Everything is saved to `triage_reasoning.json`, so `triage_predictions.json` keeps the shape the spec asks for.
- **The model never chooses the route.** `route_to` and `final_route_to` are looked up in `routing_rules`, so changing a ticket's category in review also changes its route.
- **Bad model output is handled per ticket.** A missing, malformed or invalid prediction sends that ticket to `other` / `manual_review_queue` with confidence 0.0. The problem is logged in `pipeline.log` and `queue_summary.md`. Replies over `max_words` are cut at a word boundary. Only an API or authentication failure stops the whole run.
- **Escalations are decided in code:** a ticket is escalated when its final category is `other` or its confidence is below 0.60.
- **Retries.** Rate limits and server errors (such as Gemini's "high demand" 503) are retried up to 5 times, waiting 2, 4, 8 and then 16 seconds. If the whole response can't be parsed, the call is retried once. Each attempt is logged in `llm_calls.jsonl`, and the raw responses are saved to `llm_raw_response.txt`.

## Requirement coverage

| `problem.md` item | Where it is implemented |
| --- | --- |
| Stages enforced in code | `Pipeline.advance()` / `require()` in `run_pipeline.py` |
| 1. Deterministic normalization before the LLM | `normalize_ticket()`; `validate.py` checks the output and timestamps |
| 2. One LLM call, values limited to the config, routing from the config | `predict_triage()`, `response_schema()`, `coerce_prediction()` |
| 3. Interactive human review that changes the outputs | `human_review()`; the review prompt uses the exact text from the spec |
| 4. Final queue and summary | `generate_final_queue()`, `write_summary()` |
| 5. Confidence and escalations | `confidence` field; escalation rule decided in code |
| 6. Validation | `python validate.py` (also run automatically at `VALIDATION_COMPLETE`) |
| 7. Batch resilience | Per-ticket fallback to `other` / `manual_review_queue`, logged |
| 8. LLM call logging | `llm_calls.jsonl` |

The pipeline does not depend on the sample ticket wording, IDs, categories or priorities. It was tested with replacement inputs: other ticket IDs, a different category set, priorities `P1`/`P2`/`P3`, a smaller `max_words`, a config with no `other` category, and messy text.

## Artifacts

| File | Contents |
| --- | --- |
| `normalized_tickets.json` | Deterministic normalized tickets |
| `triage_predictions.json` | Category, priority, reason, reply, route and confidence for each ticket |
| `triage_reasoning.json` | Structured CoT for each ticket, evidence grounding flags, Gemini's thought summary |
| `review_overrides.json` | Human corrections (`[]` if there were none) |
| `final_queue.json` | Values after review, used by agents |
| `queue_summary.md` | Counts, breakdown by queue, overrides, escalations |
| `escalations.json` | Tickets flagged for manual escalation |
| `llm_calls.jsonl` | One line per LLM call (provider, model, prompt hash, input and output artifacts) |
| `llm_raw_response.txt` | Raw thought summary and answer from each attempt, for debugging |
| `run_manifest.json` | Stage timestamps and hashes of the input and output files |
| `validation_report.json` | Results of the `validate.py` checks |
| `pipeline.log` | Stage transitions and problems found in model output |

Tested on Python 3.8+.
