"""
api/schemas.py

All Pydantic models for the EquityMind API.

This is the single source of truth for the data contract between:
  - The FastAPI backend
  - The WebSocket streaming protocol
  - The sync REST endpoint
  - All clients: web, iOS, Android, third-party microservices

Structure
---------
  1. WebSocket inbound   — what the client sends to open a stream
  2. WebSocket outbound  — the events the server streams back
  3. Sync REST inbound   — what the client sends to /query/sync
  4. Sync REST outbound  — the full JSON response from /query/sync
  5. Health              — /health endpoint response

WebSocket event types
---------------------
  connected   — sent immediately on WebSocket open, carries job_id
  progress    — one per LangGraph node (Layer 1)
  token       — one per GPT-4o output token (Layer 2)
  done        — stream complete, carries summary metadata
  error       — something went wrong at any point

Naming convention
-----------------
  *Request   — inbound from client
  *Response  — outbound from server (sync REST)
  *Event     — outbound from server (WebSocket stream)
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# 1. WebSocket inbound
# ─────────────────────────────────────────────────────────────────────────────

class StreamRequest(BaseModel):
    """
    Sent by the client as the first message after opening a WebSocket
    connection to ws://.../api/v1/query/stream.

    Example JSON:
        {"question": "Analyse Apple", "session_id": "web-abc123"}
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language question about a stock, comparison, or portfolio.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional client-generated session identifier. "
            "Used for request tracing and conversation history. "
            "If omitted, the server generates a UUID."
        ),
    )
    messages: Optional[list] = Field(
        default=None,
        description="Conversation history from the previous turn, sent back by the client.",
    )
    session_memory: Optional[dict] = Field(
        default=None,
        description="Structured session memory from the previous turn, sent back by the client.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. WebSocket outbound events
# ─────────────────────────────────────────────────────────────────────────────

class ConnectedEvent(BaseModel):
    """
    Sent immediately after the WebSocket connection is accepted.
    Carries the job_id the client can use for logging or correlation.

    Example:
        {"type": "connected", "job_id": "550e8400-e29b-41d4-a716-446655440000"}
    """
    type: Literal["connected"] = "connected"
    job_id: str


class ProgressEvent(BaseModel):
    """
    Layer 1 streaming — one event per LangGraph node execution.
    Allows the UI to show what the agent is doing in real time.

    node values match the node names registered in graph.py:
        classify_top_intent | classify_sub_intent | explain_concept |
        extract | ensure_sec | market_data | news | report |
        discovery_suggest | greeting | out_of_scope | clarification |
        follow_up | no_ticker

    Example:
        {"type": "progress", "node": "fetch", "message": "Fetching SEC 10-K from EDGAR..."}
    """
    type: Literal["progress"] = "progress"
    node: str = Field(description="LangGraph node name that just started executing.")
    message: str = Field(description="Human-readable status message for the UI.")

class SubProgressEvent(BaseModel):
    type: Literal["sub_progress"] = "sub_progress"
    node: str = Field(description="Parent node that emitted this sub_progress event.")
    message: str

class TokenEvent(BaseModel):
    """
    Layer 2 streaming — one event per GPT-4o output token.
    The client appends each token to the report panel to create the
    ChatGPT-style typewriter effect.

    Example:
        {"type": "token", "text": "## Apple Inc (AAPL)"}
        {"type": "token", "text": " — Investment Analysis\n\n"}
    """
    type: Literal["token"] = "token"
    text: str = Field(description="A single token or small chunk of the generated report.")


class DoneEvent(BaseModel):
    """
    Sent after all tokens have been streamed. Signals the client that
    the WebSocket stream is complete and the full report is available.

    ticker and intent fields are included so the client can update its UI
    state (e.g. store the report under the correct ticker key) without
    having to parse the markdown report text.

    Example:
        {"type": "done", "job_id": "...", "tickers": ["AAPL"], "top_intent": "TASK", "sub_intent": "SPECIFIC_STOCK", "messages": [...]}
    """
    type: Literal["done"] = "done"
    job_id: str
    tickers: Optional[list[str]] = Field(
        default=None,
        description="All tickers involved in the query.",
    )
    top_intent: Optional[str] = Field(
        default=None,
        description="Layer 1 coarse category: TASK | GREETING | OUT_OF_SCOPE | GENERAL_KNOWLEDGE",
    )
    sub_intent: Optional[str] = Field(
        default=None,
        description="Layer 2 fine-grained task type, only present when top_intent is TASK: "
                    "SPECIFIC_STOCK | COMPARISON | DISCOVERY | ANALYZE_POSITION | "
                    "ANALYZE_PORTFOLIO | FOLLOW_UP | CLARIFICATION",
    )
    messages: Optional[list] = Field(
        default=None,
        description="Conversation history to be sent back by the client on the next request.",
    )
    session_memory: Optional[dict] = Field(
        default=None,
        description="Updated session memory to be sent back by the client on the next request.",
    )


class ErrorEvent(BaseModel):
    """
    Sent when an unrecoverable error occurs at any point in the pipeline.
    The WebSocket connection is closed after this event.

    Example:
        {"type": "error", "message": "SEC EDGAR is currently unavailable. Please try again."}
    """
    type: Literal["error"] = "error"
    message: str = Field(description="Human-readable error description.")


# Union type for all outbound WebSocket events.
# Used in query.py to ensure every send() call is typed correctly.
WebSocketEvent = ConnectedEvent | ProgressEvent | SubProgressEvent | TokenEvent | DoneEvent | ErrorEvent


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sync REST inbound
# ─────────────────────────────────────────────────────────────────────────────

class SyncRequest(BaseModel):
    """
    Request body for POST /api/v1/query/sync.

    Intended for programmatic consumers (microservices, scripts) that
    do not need streaming and prefer a single blocking HTTP response.

    Example JSON:
        {"question": "Analyse Apple", "session_id": "svc-xyz"}
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language question about a stock, comparison, or portfolio.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for tracing.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sync REST outbound
# ─────────────────────────────────────────────────────────────────────────────

class SyncResponse(BaseModel):
    """
    Full response from POST /api/v1/query/sync.

    The agent runs to completion before this is returned.
    Designed for microservice consumers that parse structured data
    rather than render markdown in a UI.

    Example JSON:
        {
            "job_id": "550e8400-...",
            "tickers": ["AAPL"],
            "top_intent": "TASK",
            "sub_intent": "SPECIFIC_STOCK",
            "answer": "## Apple Inc (AAPL)...",
            "status": "success"
        }
    """
    job_id: str = Field(description="Server-generated unique identifier for this request.")
    tickers: Optional[list[str]] = Field(
        default=None,
        description="All tickers involved (relevant for COMPARISON intent).",
    )
    top_intent: Optional[str] = Field(
        default=None,
        description="Layer 1 coarse category: TASK | GREETING | OUT_OF_SCOPE | GENERAL_KNOWLEDGE",
    )
    sub_intent: Optional[str] = Field(
        default=None,
        description="Layer 2 fine-grained task type, only present when top_intent is TASK: "
                    "SPECIFIC_STOCK | COMPARISON | DISCOVERY | ANALYZE_POSITION | ANALYZE_PORTFOLIO",
    )
    answer: str = Field(description="Full markdown report generated by the agent.")
    status: Literal["success", "error"] = Field(
        default="success",
        description="success if the agent completed normally, error otherwise.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is error, otherwise null.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """
    Response from GET /api/v1/health.
    Used by Railway uptime checks and monitoring tools.

    Example:
        {"status": "ok", "app": "EquityMind", "version": "0.4.0"}
    """
    status: Literal["ok"] = "ok"
    app: str
    version: str