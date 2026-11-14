"""AegisHealth triage workflow (LangGraph).

Flow::

    START -> triage_agent --(confidence >= 0.85)--------------> specialist_agent -> END
                          \\-(confidence <  0.85)-> doctor_review --(approved)--> specialist_agent
                                                       \\-(rejected / no decision)-> END

Execution is suspended *before* ``doctor_review`` by the checkpointer; a doctor
supplies a decision with ``update_state`` and the same ``thread_id`` resumes the run.

Set ``TRIAGE_USE_LLM=1`` (and ``OPENAI_API_KEY``) to let an LLM assess the case;
otherwise (or on any LLM failure) a deterministic rule engine is used.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict

from jsonschema import Draft202012Validator
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("triage")

SRC_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = SRC_DIR.parent / "schemas" / "a2a_schema.json"
MCP_SERVER_PATH = SRC_DIR / "mcp_server.py"

CONFIDENCE_THRESHOLD = 0.85
MCP_TIMEOUT_SECONDS = 20.0

# (specialty, urgency, keywords, base confidence)
RULES: list[tuple[str, str, tuple[str, ...], float]] = [
    ("Cardiology", "high", ("palpitation", "chest", "dizziness", "irregular heart"), 0.80),
    ("Pulmonology", "medium", ("shortness of breath", "wheez", "asthma", "breathing"), 0.80),
    ("Infectious Disease", "medium", ("fever", "cough", "body aches", "flu", "sore throat"), 0.82),
    ("Orthopedics", "low", ("ankle", "knee", "joint", "sprain", "fracture", "fall"), 0.82),
    ("Neurology", "medium", ("headache", "migraine", "visual disturbance", "numbness"), 0.78),
    ("Endocrinology", "medium", ("thirst", "fatigue", "weight gain", "blood sugar", "hba1c"), 0.75),
]
RED_FLAGS: tuple[str, ...] = (
    "chest pain",
    "chest tightness",
    "dizziness",
    "shortness of breath",
    "fainting",
    "visual disturbance",
    "numbness",
)
URGENCY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class State(TypedDict, total=False):
    """Shared graph state. ``messages`` uses ``add_messages`` so updates accumulate."""

    messages: Annotated[list[AnyMessage], add_messages]  # every node appends
    patient_id: str  # input
    symptoms: str  # input
    records: dict[str, Any]  # triage_agent (via MCP)
    confidence_score: float  # triage_agent
    urgency: str  # triage_agent
    suggested_specialty: str  # triage_agent
    rationale: str  # triage_agent
    red_flags: list[str]  # triage_agent
    a2a_message: dict[str, Any]  # triage_agent / doctor_review (validated)
    human_review: dict[str, Any]  # doctor (update_state) -> doctor_review
    final_plan: dict[str, Any]  # specialist_agent
    status: str  # every node


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# MCP client (secure record access)
# --------------------------------------------------------------------------- #
async def _call_mcp_tool(patient_id: str) -> dict[str, Any]:
    """Call ``get_medical_records`` on the MCP server over stdio."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import TextContent

    params = StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER_PATH)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_medical_records", {"patient_id": patient_id})
            text = next((c.text for c in result.content if isinstance(c, TextContent)), "")
            return json.loads(text)


def fetch_medical_records(patient_id: str) -> dict[str, Any]:
    """Fetch records through MCP; never raises. Failures become JSON-RPC errors."""
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                lambda: asyncio.run(asyncio.wait_for(_call_mcp_tool(patient_id), MCP_TIMEOUT_SECONDS))
            )
            return future.result()
    except Exception as exc:  # noqa: BLE001 - fault tolerance boundary
        log.error("MCP call failed: %s", exc)
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32603, "message": "MCP server unreachable", "data": {"detail": str(exc)}},
        }


# --------------------------------------------------------------------------- #
# Assessment engines
# --------------------------------------------------------------------------- #
def _rule_assess(symptoms: str, record: dict[str, Any]) -> dict[str, Any]:
    """Deterministic triage assessment (offline fallback)."""
    text = symptoms.lower()
    scored: list[tuple[float, str, str, list[str]]] = []
    history_depts = {v.get("department") for v in record.get("recent_visits", [])}
    history_text = " ".join(
        [*record.get("chronic_conditions", [])]
        + [f"{v.get('diagnosis', '')} {v.get('reason', '')}" for v in record.get("recent_visits", [])]
    ).lower()

    for specialty, urgency, keywords, base in RULES:
        hits = [k for k in keywords if k in text]
        if not hits:
            continue
        score = base + min(0.05 * (len(hits) - 1), 0.10)
        if specialty in history_depts or any(k in history_text for k in hits):
            score += 0.05  # consistent with documented history
        scored.append((min(score, 0.99), specialty, urgency, hits))

    flags = [f for f in RED_FLAGS if f in text]
    if not scored:
        return {
            "confidence_score": 0.30,
            "suggested_specialty": "General Medicine",
            "urgency": "medium",
            "rationale": "No known symptom pattern matched; manual review needed.",
            "red_flags": flags,
        }

    scored.sort(reverse=True)
    score, specialty, urgency, hits = scored[0]
    notes = [f"Matched {specialty} on: {', '.join(hits)}."]
    if len(scored) > 1 and score - scored[1][0] < 0.10:
        score -= 0.20
        notes.append(f"Ambiguous with {scored[1][1]}; confidence reduced.")
    if flags:
        score = min(score, 0.70)
        urgency = max(urgency, "high", key=lambda u: URGENCY_RANK[u])
        notes.append(f"Red flags present ({', '.join(flags)}); confidence capped for safety.")
    return {
        "confidence_score": round(max(score, 0.0), 2),
        "suggested_specialty": specialty,
        "urgency": urgency,
        "rationale": " ".join(notes),
        "red_flags": flags,
    }


def _llm_assess(symptoms: str, record: dict[str, Any]) -> dict[str, Any]:
    """LLM-based assessment with structured output; caller handles failures."""
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    class Assessment(BaseModel):
        confidence_score: float = Field(ge=0, le=1)
        suggested_specialty: str
        urgency: Literal["low", "medium", "high", "critical"]
        rationale: str
        red_flags: list[str] = Field(default_factory=list)

    llm = ChatOpenAI(model=os.getenv("TRIAGE_MODEL", "gpt-4o-mini"), temperature=0)
    structured = llm.with_structured_output(Assessment)
    prompt = (
        "You are a clinical triage assistant. Be conservative: report LOW confidence when the "
        "case is ambiguous or has red flags. Never invent facts.\n"
        f"Symptoms: {symptoms}\nPatient record: {json.dumps(record)}"
    )
    return structured.invoke(prompt).model_dump()


def _assess(symptoms: str, record: dict[str, Any]) -> dict[str, Any]:
    """Pick the LLM engine when enabled, falling back to rules on any error."""
    if os.getenv("TRIAGE_USE_LLM") == "1" and os.getenv("OPENAI_API_KEY"):
        try:
            return _llm_assess(symptoms, record)
        except Exception as exc:  # noqa: BLE001 - fault tolerance boundary
            log.warning("LLM assessment failed (%s); using rule engine", exc)
    return _rule_assess(symptoms, record)


# --------------------------------------------------------------------------- #
# A2A message helpers
# --------------------------------------------------------------------------- #
_VALIDATOR = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def build_a2a_message(
    state: State,
    thread_id: str,
    intent: str,
    receiver: dict[str, str],
    human_review: dict[str, Any],
) -> dict[str, Any]:
    """Build an A2A message from state and validate it against the JSON schema."""
    record = state.get("records", {})
    visits = record.get("recent_visits", [])
    message: dict[str, Any] = {
        "protocol_version": "1.0",
        "message_id": str(uuid.uuid4()),
        "thread_id": thread_id,
        "timestamp": _utc_now(),
        "sender": {"agent_id": "triage-agent-01", "role": "triage_agent"},
        "receiver": receiver,
        "intent": intent,
        "patient_id": state["patient_id"],
        "triage": {
            "symptoms": state["symptoms"],
            "confidence_score": state["confidence_score"],
            "urgency": state["urgency"],
            "suggested_specialty": state["suggested_specialty"],
            "rationale": state.get("rationale", ""),
            "red_flags": state.get("red_flags", []),
        },
        "context": {
            "age": record.get("age", 0),
            "chronic_conditions": record.get("chronic_conditions", []),
            "recent_visit_count": len(visits),
        },
        "human_review": human_review,
    }
    if visits:
        message["context"]["last_visit_date"] = max(v["date"] for v in visits)
    _VALIDATOR.validate(message)
    return message


def _thread_id(config: Optional[RunnableConfig]) -> str:
    """Extract the checkpointer thread id from a runnable config."""
    return str(((config or {}).get("configurable") or {}).get("thread_id", "unknown"))


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def triage_agent(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Fetch records via MCP, assess the case and score confidence."""
    pid = state["patient_id"]
    log.info("node=triage_agent start patient_id=%s", pid)
    response = fetch_medical_records(pid)

    assessment: dict[str, Any]
    record: dict[str, Any]
    if "error" in response:
        err = response["error"]
        log.warning("node=triage_agent record fetch failed code=%s msg=%s", err.get("code"), err.get("message"))
        assessment = {
            "confidence_score": 0.0,
            "suggested_specialty": "General Medicine",
            "urgency": "high",
            "rationale": f"Records unavailable ({err.get('message')}); escalating to a doctor.",
            "red_flags": [],
        }
        record = {}
    else:
        record = response["result"]
        assessment = _assess(state["symptoms"], record)

    update: dict[str, Any] = {
        "records": record,
        "status": "triaged",
        **assessment,
    }
    merged: State = {**state, **update}  # type: ignore[typeddict-item]
    needs_review = assessment["confidence_score"] < CONFIDENCE_THRESHOLD
    update["a2a_message"] = build_a2a_message(
        merged,
        _thread_id(config),
        intent="request_human_review" if needs_review else "handoff_to_specialist",
        receiver=(
            {"agent_id": "doctor-on-call", "role": "doctor"}
            if needs_review
            else {"agent_id": "specialist-agent-01", "role": "specialist_agent"}
        ),
        human_review=(
            {"required": True, "status": "pending"}
            if needs_review
            else {"required": False, "status": "not_required"}
        ),
    )
    update["messages"] = [
        AIMessage(
            content=(
                f"Triage: {assessment['suggested_specialty']} "
                f"(confidence={assessment['confidence_score']}, urgency={assessment['urgency']}). "
                f"{assessment['rationale']}"
            ),
            name="triage_agent",
        )
    ]
    log.info(
        "node=triage_agent done confidence=%s specialty=%s",
        assessment["confidence_score"],
        assessment["suggested_specialty"],
    )
    return update


def doctor_review(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Record the doctor's decision. No decision on file means NOT approved (fail-safe)."""
    review = dict(state.get("human_review") or {})
    approved = review.get("approved") is True and bool(review.get("reviewer"))
    status = "approved" if approved else "rejected"
    log.info("node=doctor_review decision=%s reviewer=%s", status, review.get("reviewer"))

    update: dict[str, Any] = {"status": f"doctor_{status}"}
    if approved and review.get("override_specialty"):
        update["suggested_specialty"] = review["override_specialty"]

    merged: State = {**state, **update}  # type: ignore[typeddict-item]
    update["a2a_message"] = build_a2a_message(
        merged,
        _thread_id(config),
        intent="handoff_to_specialist" if approved else "report_result",
        receiver={"agent_id": "specialist-agent-01", "role": "specialist_agent"},
        human_review={
            "required": True,
            "status": status,
            "reviewer": review.get("reviewer", "unknown"),
            "notes": review.get("notes", "No decision recorded."),
            "reviewed_at": _utc_now(),
        },
    )
    update["messages"] = [
        AIMessage(
            content=f"Doctor review {status} by {review.get('reviewer', 'nobody')}: {review.get('notes', '')}",
            name="doctor_review",
        )
    ]
    return update


def specialist_agent(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Consume the validated A2A message and produce the specialist plan."""
    message = state["a2a_message"]
    _VALIDATOR.validate(message)  # reject malformed or unsafe handoffs
    triage = message["triage"]
    log.info("node=specialist_agent start specialty=%s", triage["suggested_specialty"])
    plan = {
        "patient_id": message["patient_id"],
        "specialty": triage["suggested_specialty"],
        "urgency": triage["urgency"],
        "next_steps": [
            f"Schedule {triage['suggested_specialty']} consult ({triage['urgency']} urgency).",
            "Review chronic conditions: " + (", ".join(message["context"]["chronic_conditions"]) or "none"),
        ],
        "human_review_status": message["human_review"]["status"],
    }
    return {
        "final_plan": plan,
        "status": "completed",
        "messages": [AIMessage(content=f"Specialist plan ready: {plan['next_steps'][0]}", name="specialist_agent")],
    }


# --------------------------------------------------------------------------- #
# Routing + graph
# --------------------------------------------------------------------------- #
def route_after_triage(state: State) -> Literal["doctor_review", "specialist_agent"]:
    """Escalate to a human when confidence is below the safety threshold."""
    score = state.get("confidence_score", 0.0)
    target: Literal["doctor_review", "specialist_agent"] = (
        "specialist_agent" if score >= CONFIDENCE_THRESHOLD else "doctor_review"
    )
    log.info("route confidence=%.2f threshold=%.2f -> %s", score, CONFIDENCE_THRESHOLD, target)
    return target


def route_after_review(state: State) -> Literal["specialist_agent", "__end__"]:
    """Only an approved review may continue to the specialist."""
    approved = state.get("status") == "doctor_approved"
    log.info("route review approved=%s", approved)
    if approved:
        return "specialist_agent"
    return "__end__"


def build_graph(checkpointer: Optional[Any] = None) -> Any:
    """Compile the workflow, pausing before ``doctor_review``."""
    builder = StateGraph(State)
    builder.add_node("triage_agent", triage_agent)
    builder.add_node("doctor_review", doctor_review)
    builder.add_node("specialist_agent", specialist_agent)
    builder.add_edge(START, "triage_agent")
    builder.add_conditional_edges(
        "triage_agent",
        route_after_triage,
        {"doctor_review": "doctor_review", "specialist_agent": "specialist_agent"},
    )
    builder.add_conditional_edges(
        "doctor_review",
        route_after_review,
        {"specialist_agent": "specialist_agent", END: END},
    )
    builder.add_edge("specialist_agent", END)
    return builder.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=["doctor_review"],
    )


def start_case(graph: Any, patient_id: str, symptoms: str) -> dict[str, Any]:
    """Start a run on its own thread; returns the snapshot (paused when review is needed)."""
    thread = {"configurable": {"thread_id": f"case-{patient_id}"}}
    graph.invoke(
        {
            "patient_id": patient_id,
            "symptoms": symptoms,
            "messages": [HumanMessage(content=symptoms, name="patient")],
        },
        thread,
    )
    snapshot = graph.get_state(thread)
    if snapshot.next:
        log.info("PAUSED before %s (thread=%s) - awaiting doctor", snapshot.next, thread["configurable"]["thread_id"])
    return {"paused": bool(snapshot.next), "next": list(snapshot.next), "state": snapshot.values}


def resume_case(graph: Any, patient_id: str, doctor: dict[str, Any]) -> dict[str, Any]:
    """Inject the doctor's decision and resume the SAME thread from its checkpoint."""
    thread = {"configurable": {"thread_id": f"case-{patient_id}"}}
    graph.update_state(thread, {"human_review": doctor})
    graph.invoke(None, thread)
    snapshot = graph.get_state(thread)
    return {"paused": bool(snapshot.next), "next": list(snapshot.next), "state": snapshot.values}


if __name__ == "__main__":
    app = build_graph()  # one graph + one checkpointer shared by all cases

    print("\n=== Case 1: high confidence -> straight to specialist ===")
    out = start_case(app, "P-1007", "High fever, body aches, severe dry cough")
    print(json.dumps(out["state"]["final_plan"], indent=2))

    print("\n=== Case 2: low confidence -> pause for human review ===")
    paused = start_case(app, "P-1003", "Palpitations, dizziness and chest tightness since last night")
    print("Paused:", paused["paused"], "| next:", paused["next"])
    print("Confidence:", paused["state"]["confidence_score"], "|", paused["state"]["rationale"])

    print("\n=== Case 2 resumed after doctor approval (same thread) ===")
    done = resume_case(
        app,
        "P-1003",
        {"approved": True, "reviewer": "Dr. Amina Patel", "notes": "Confirmed cardiology referral."},
    )
    print(json.dumps(done["state"]["final_plan"], indent=2))

    print("\n=== Case 3: doctor rejects -> pipeline stops ===")
    start_case(app, "P-1004", "Severe headache with visual disturbance")
    rej = resume_case(
        app,
        "P-1004",
        {"approved": False, "reviewer": "Dr. Emily Tran", "notes": "Needs in-person exam first."},
    )
    print("Final status:", rej["state"]["status"], "| plan:", rej["state"].get("final_plan"))

    print("\n=== Case 4: unknown patient -> MCP error -> escalate ===")
    err = start_case(app, "P-9999", "Cough")
    print("Paused:", err["paused"], "| confidence:", err["state"]["confidence_score"])
