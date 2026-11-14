"""MCP server exposing the ``get_medical_records`` tool.

Reads patient data from ``mock_patients.json`` (same folder) so that agents never
touch the record store directly: all access goes through this single, auditable
endpoint.

Run (stdio, used by the workflow)::

    python src/mcp_server.py

Run as an HTTP endpoint::

    python src/mcp_server.py --transport streamable-http --port 8000
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

PATIENTS_FILE = Path(__file__).with_name("mock_patients.json")
PATIENT_ID_PATTERN = re.compile(r"^P-\d{4}$")

# Logs go to stderr: on the stdio transport stdout carries the protocol itself.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s [mcp] %(message)s",
)
log = logging.getLogger("mcp_server")

# JSON-RPC 2.0 error codes (-32602 is the spec's "Invalid params").
ERR_INVALID_PARAMS = -32602
ERR_NOT_FOUND = -32004  # implementation-defined server error range
ERR_INTERNAL = -32603

# JSON schemas advertised for the tool (also useful for the implementation team).
TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patient_id": {"type": "string", "pattern": r"^P-\d{4}$", "examples": ["P-1001"]}
    },
    "required": ["patient_id"],
    "additionalProperties": False,
}

TOOL_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["jsonrpc", "id", "error"],
    "properties": {
        "jsonrpc": {"const": "2.0"},
        "id": {"type": ["string", "integer", "null"]},
        "error": {
            "type": "object",
            "required": ["code", "message"],
            "properties": {
                "code": {"type": "integer"},
                "message": {"type": "string"},
                "data": {"type": "object"},
            },
        },
    },
}

mcp = FastMCP("aegishealth-medical-records", host="127.0.0.1", port=8000)


def _error(code: int, message: str, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Build a structured JSON-RPC 2.0 error payload."""
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": code, "message": message},
    }
    if data:
        payload["error"]["data"] = data
    return payload


def _load_patients() -> list[dict[str, Any]]:
    """Load the patient store from disk."""
    with PATIENTS_FILE.open(encoding="utf-8") as fh:
        return json.load(fh)


def lookup_records(patient_id: Any) -> dict[str, Any]:
    """Validate ``patient_id`` and return a result or a JSON-RPC error payload.

    Success shape: ``{"jsonrpc": "2.0", "id": null, "result": {...patient...}}``.
    """
    if patient_id is None or (isinstance(patient_id, str) and not patient_id.strip()):
        return _error(
            ERR_INVALID_PARAMS,
            "Missing required parameter: patient_id",
            {"param": "patient_id", "reason": "missing"},
        )
    if not isinstance(patient_id, str) or not PATIENT_ID_PATTERN.match(patient_id.strip()):
        return _error(
            ERR_INVALID_PARAMS,
            "Malformed patient_id; expected format 'P-NNNN'",
            {"param": "patient_id", "reason": "malformed", "received": str(patient_id)[:32]},
        )

    pid = patient_id.strip()
    try:
        patients = _load_patients()
    except (OSError, json.JSONDecodeError) as exc:
        log.error("record store unavailable: %s", exc)
        return _error(ERR_INTERNAL, "Record store unavailable", {"reason": "store_error"})

    for patient in patients:
        if patient.get("patient_id") == pid:
            log.info("records served patient_id=%s", pid)
            return {"jsonrpc": "2.0", "id": None, "result": patient}

    log.warning("patient not found patient_id=%s", pid)
    return _error(
        ERR_NOT_FOUND,
        f"Patient {pid} not found",
        {"param": "patient_id", "reason": "not_found", "patient_id": pid},
    )


@mcp.tool()
async def get_medical_records(patient_id: Optional[str] = None) -> dict[str, Any]:
    """Fetch a patient's historical medical records.

    Args:
        patient_id: Patient identifier in the form ``P-NNNN`` (e.g. ``P-1001``).
            Optional at the transport level so a missing value yields our
            JSON-RPC error payload instead of a framework validation error.

    Returns:
        A JSON-RPC style payload: ``result`` holds the patient record on success,
        ``error`` (code, message, data) on a missing, malformed or unknown id.
    """
    return lookup_records(patient_id)


def main() -> None:
    """Parse CLI arguments and start the server."""
    parser = argparse.ArgumentParser(description="AegisHealth MCP medical-records server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    log.info("starting server transport=%s", args.transport)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
