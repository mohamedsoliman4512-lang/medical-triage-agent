# AegisHealth Triage System - Architecture Handover

**Author:** Mohamed Soliman | **To:** Dr. Elena Rostova, Chief AI Architect | **Version:** V1

## 1. Overview

The triage system takes a patient ID and a free-text symptom description, assesses the case, and routes it to the
right specialist. Its core safety rule is **human in the loop**: no case below the confidence threshold (0.85)
reaches a specialist without an explicit, named doctor approval. This closes the failure mode of the earlier
prototype, which routed autonomously.

Three decoupled parts:

| Part | File | Responsibility |
|---|---|---|
| Orchestration | `src/triage_workflow.py` | LangGraph pipeline, confidence routing, pause/resume |
| Tool layer | `src/mcp_server.py` | Only way agents read patient records (MCP, JSON-RPC errors) |
| Agent protocol | `schemas/a2a_schema.json` | Strict message envelope between agents |

Users: triage nurses (submit cases), on-call doctors (review paused cases), specialist teams (receive plans).

## 2. Graph architecture

Nodes: `triage_agent`, `doctor_review`, `specialist_agent`.

```
START -> triage_agent --[confidence >= 0.85]--------------------------> specialist_agent -> END
                      \--[confidence <  0.85]--> (PAUSE) doctor_review --[approved]--> specialist_agent
                                                                       \--[rejected / no decision]--> END
```

1. `triage_agent` fetches records via MCP, assesses the case, writes `confidence_score`, and builds a validated A2A message.
2. `route_after_triage` (conditional edge) compares `confidence_score` with `CONFIDENCE_THRESHOLD = 0.85`.
3. Below the threshold, the graph is suspended **before** `doctor_review` (`interrupt_before`).
4. `doctor_review` records the decision. `route_after_review` only continues when `status == "doctor_approved"`.
5. `specialist_agent` re-validates the A2A message and produces the plan.

Fail-safe defaults: an MCP failure gives confidence `0.0` (so escalation), and a resume with no decision counts as **rejected**.

## 3. State and state passing

| Key | Type | Written by |
|---|---|---|
| `messages` | `list[AnyMessage]` (`add_messages`) | every node |
| `patient_id`, `symptoms` | `str` | caller (input) |
| `records` | `dict` | `triage_agent` (from MCP) |
| `confidence_score` | `float` | `triage_agent` |
| `urgency`, `suggested_specialty`, `rationale` | `str` | `triage_agent` (specialty overridable by doctor) |
| `red_flags` | `list[str]` | `triage_agent` |
| `a2a_message` | `dict` (schema-validated) | `triage_agent`, `doctor_review` |
| `human_review` | `dict` | doctor, via `update_state` |
| `final_plan` | `dict` | `specialist_agent` |
| `status` | `str` | every node |

`messages` is annotated with the `add_messages` reducer, so each node *returns only its own new message* and the
reducer **appends** it instead of overwriting the list. The audit trail therefore keeps every agent's contribution
(patient, triage, doctor, specialist). All other keys use last-write-wins.

Agents never share raw state. The Triage Agent hands over a **schema-validated A2A message**
(`state["a2a_message"]`), and the Specialist re-validates it before acting.

## 4. Confidence routing and human review

- **Threshold:** `confidence_score < 0.85` means human review. Rule-engine scores are reduced when two specialties
  are ambiguous, and **capped at 0.70 when red flags are present** (e.g. chest tightness, dizziness), so
  dangerous presentations are always escalated.
- **Pause point:** the graph is compiled with `checkpointer=MemorySaver()` and `interrupt_before=["doctor_review"]`.
  State is saved to the checkpointer, then execution stops.
- **Resume:** the run is identified by `thread_id` (`case-<patient_id>`). The doctor's decision is written with
  `graph.update_state(config, {"human_review": {...}})`, then `graph.invoke(None, config)` continues the *same*
  run from its checkpoint (triage is not re-run).
- **Decision payload:** `{"approved": bool, "reviewer": str, "notes": str, "override_specialty": str?}`.
  Approval requires a non-empty `reviewer` name.
- **Production note:** `MemorySaver` is in-process. Swap in `PostgresSaver` / `SqliteSaver` so paused cases survive restarts.

## 5. MCP server

`src/mcp_server.py` (FastMCP, async). Transports: stdio (default, used by the workflow) or
`--transport streamable-http --port 8000` for a network endpoint.

**Tool:** `get_medical_records(patient_id: str)`

- **Input:** `patient_id` matching `^P-[0-9]{4}$`.
- **Success output:** `{"jsonrpc": "2.0", "id": null, "result": {<patient record>}}`
- **Error output (JSON-RPC style):** `{"jsonrpc": "2.0", "id": null, "error": {"code", "message", "data"}}`

| Case | Code | `data.reason` |
|---|---|---|
| Missing / empty `patient_id` | `-32602` | `missing` |
| Malformed (e.g. `bad`) | `-32602` | `malformed` |
| Unknown ID (e.g. `P-9999`) | `-32004` | `not_found` |
| Record store unreadable | `-32603` | `store_error` |

Example: `{"error": {"code": -32004, "message": "Patient P-9999 not found", "data": {"param": "patient_id", "reason": "not_found", "patient_id": "P-9999"}}}`

The workflow also converts an unreachable MCP server (timeout 20 s) into a `-32603` error, so it escalates instead of crashing.

## 6. Agent-to-agent (A2A) messages

`schemas/a2a_schema.json` (JSON Schema 2020-12, `additionalProperties: false` everywhere).

- **Required:** `protocol_version` (`"1.0"`), `message_id` (uuid), `thread_id`, `timestamp`, `sender`, `receiver`
  (`agent_id` + `role`), `intent`, `patient_id`, `triage`, `human_review`.
- **`triage`:** required `symptoms`, `confidence_score` (0-1), `urgency`, `suggested_specialty`; optional `rationale`, `red_flags`.
- **Optional:** `context` (minimum-necessary data only: age, chronic conditions, visit count, last visit date).
- **`intent`:** `handoff_to_specialist` | `request_human_review` | `report_result`.
- **Built-in safety rules:** a `handoff_to_specialist` is only valid when `human_review.status` is `not_required` or
  `approved`; an approved/rejected review must carry `reviewer` and `reviewed_at`.

Example (auto-routed case):

```json
{
  "protocol_version": "1.0",
  "message_id": "0b6c6f0e-4c1d-4b53-9f0a-6d1c2f3a9e11",
  "thread_id": "case-P-1007",
  "timestamp": "2026-11-14T10:00:00Z",
  "sender": {"agent_id": "triage-agent-01", "role": "triage_agent"},
  "receiver": {"agent_id": "specialist-agent-01", "role": "specialist_agent"},
  "intent": "handoff_to_specialist",
  "patient_id": "P-1007",
  "triage": {"symptoms": "High fever, body aches, severe dry cough", "confidence_score": 0.9,
             "urgency": "medium", "suggested_specialty": "Infectious Disease", "red_flags": []},
  "context": {"age": 19, "chronic_conditions": [], "recent_visit_count": 1, "last_visit_date": "2024-02-28"},
  "human_review": {"required": false, "status": "not_required"}
}
```

## 7. Communication traces

Real output of `python src/triage_workflow.py` (case P-1003, symptoms: palpitations, dizziness, chest tightness):

```
INFO node=triage_agent start patient_id=P-1003
INFO route confidence=0.70 threshold=0.85 -> doctor_review
INFO PAUSED before ('doctor_review',) (thread=case-P-1003) - awaiting doctor
INFO route confidence=0.70 threshold=0.85 -> doctor_review        <- resume: edge re-evaluated from checkpoint
INFO node=doctor_review decision=approved reviewer=Dr. Amina Patel
INFO route review approved=True
INFO node=specialist_agent start specialty=Cardiology
```

And the auto-routed case (P-1007, flu symptoms):

```
INFO node=triage_agent start patient_id=P-1007
INFO route confidence=0.97 threshold=0.85 -> specialist_agent
INFO node=specialist_agent start specialty=Infectious Disease
```

What it shows: confidence below 0.85 stops the run **before** `doctor_review`; after the doctor's decision the same
thread resumes without re-running triage (`triage_agent start` appears once). Other verified paths: rejection ends
the run with no specialist plan (`doctor_rejected`); an unknown patient (`P-9999`) gives confidence `0.0` and escalates.

## 8. Risks and next steps

- **Rule engine is a demo.** The default assessor is keyword-based. Set `TRIAGE_USE_LLM=1` + `OPENAI_API_KEY` for the
  LLM assessor (falls back to rules on error). LLM confidence is self-reported and must be calibrated against
  labelled cases before it gates anything clinical.
- **Persistence:** replace `MemorySaver` with a durable checkpointer; add review SLAs and timeouts for paused cases.
- **Auth and audit:** the MCP server has no authentication. Add mTLS/OAuth, per-agent scopes and an immutable access
  log before exposing it over HTTP. Records are real PHI in production (encryption at rest, data minimisation).
- **Reviewer identity:** `reviewer` is currently a free-text name; bind it to an authenticated clinician identity.
- **Schema versioning:** bump `protocol_version` on breaking changes; receivers reject unknown versions.
- **Testing:** add unit tests for routing thresholds, red-flag caps and the schema's safety rules to CI.
