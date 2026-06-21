"""
tests/test_api.py

API layer tests for EquityMind Phase 4.

Tests cover:
    1. Health endpoint
    2. Sync REST endpoint
    3. WebSocket streaming endpoint — Layer 1 + Layer 2
    4. Edge cases — empty question, invalid JSON, out_of_scope

Run with server stopped (uses TestClient — no server needed):
    pytest tests/test_api.py -v

Note: Some tests make real API calls to OpenAI, PostgreSQL, and yfinance.
These take 30-90 seconds each.
"""

import json
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ─────────────────────────────────────────────
# 1. Health endpoint
# ─────────────────────────────────────────────

def test_health_returns_200():
    """Health endpoint must return 200 OK."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_response_shape():
    """Health response must have status, app, version fields."""
    response = client.get("/api/v1/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "EquityMind"
    assert data["version"] == "0.4.0"


# ─────────────────────────────────────────────
# 2. Sync REST endpoint — validation
# ─────────────────────────────────────────────

def test_sync_empty_question_returns_422():
    """Empty question must be rejected with 422 validation error."""
    response = client.post(
        "/api/v1/query/sync",
        json={"question": ""}
    )
    assert response.status_code == 422


def test_sync_missing_question_returns_422():
    """Missing question field must be rejected with 422."""
    response = client.post(
        "/api/v1/query/sync",
        json={"session_id": "test"}
    )
    assert response.status_code == 422


def test_sync_question_too_long_returns_422():
    """Question over 1000 chars must be rejected with 422."""
    response = client.post(
        "/api/v1/query/sync",
        json={"question": "x" * 1001}
    )
    assert response.status_code == 422


def test_sync_response_shape():
    """Sync response must have all required fields."""
    response = client.post(
        "/api/v1/query/sync",
        json={"question": "Hello"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "tickers" in data
    assert "top_intent" in data
    assert "sub_intent" in data
    assert "answer" in data
    assert "status" in data
    assert "error" in data


def test_sync_greeting_intent():
    """Greeting question must return GREETING top_intent and non-empty answer."""
    response = client.post(
        "/api/v1/query/sync",
        json={"question": "Hello"}
    )
    data = response.json()
    assert data["status"] == "success"
    assert data["top_intent"] == "GREETING"
    assert len(data["answer"]) > 0


def test_sync_out_of_scope_intent():
    """Out of scope question must return OUT_OF_SCOPE top_intent."""
    response = client.post(
        "/api/v1/query/sync",
        json={"question": "What is the weather today?"}
    )
    data = response.json()
    assert data["status"] == "success"
    assert data["top_intent"] == "OUT_OF_SCOPE"
    assert len(data["answer"]) > 0



def test_sync_specific_stock_intent():
    """Stock question must return SPECIFIC_STOCK sub_intent with ticker and answer."""
    response = client.post(
        "/api/v1/query/sync",
        json={"question": "Analyse Apple"}
    )
    data = response.json()
    assert data["status"] == "success"
    assert data["top_intent"] == "TASK"
    assert data["sub_intent"] == "SPECIFIC_STOCK"
    assert "AAPL" in data["tickers"]
    assert len(data["answer"]) > 100



def test_sync_comparison_intent():
    """Comparison question must return COMPARISON sub_intent with multiple tickers."""
    response = client.post(
        "/api/v1/query/sync",
        json={"question": "Compare Apple and Microsoft"}
    )
    data = response.json()
    assert data["status"] == "success"
    assert data["top_intent"] == "TASK"
    assert data["sub_intent"] == "COMPARISON"
    assert len(data["tickers"]) >= 2
    assert len(data["answer"]) > 100


# ─────────────────────────────────────────────
# 3. WebSocket endpoint — connection and events
# ─────────────────────────────────────────────

def test_websocket_connects():
    """WebSocket must accept connection and send connected event."""
    with client.websocket_connect(
        "/api/v1/query/stream",
        headers={"Authorization": "Bearer test_key"}
    ) as ws:
        ws.send_text(json.dumps({"question": "Hello"}))
        message = ws.receive_text()
        data = json.loads(message)
        assert data["type"] == "connected"
        assert "job_id" in data


def test_websocket_connected_event_has_job_id():
    """Connected event must contain a non-empty job_id."""
    with client.websocket_connect(
        "/api/v1/query/stream",
        headers={"Authorization": "Bearer test_key"}
    ) as ws:
        ws.send_text(json.dumps({"question": "Hello"}))
        message = ws.receive_text()
        data = json.loads(message)
        assert len(data["job_id"]) > 0


def test_websocket_greeting_receives_answer():
    """Greeting question must receive an answer via WebSocket."""
    events = []
    with client.websocket_connect(
        "/api/v1/query/stream",
        headers={"Authorization": "Bearer test_key"}
    ) as ws:
        ws.send_text(json.dumps({"question": "Hello"}))
        while True:
            message = ws.receive_text()
            data = json.loads(message)
            events.append(data)
            if data["type"] in ("done", "error"):
                break

    event_types = [e["type"] for e in events]
    tokens = [e["text"] for e in events if e["type"] == "token"]
    full_answer = "".join(tokens)

    assert "connected" in event_types
    assert "done" in event_types
    assert len(full_answer) > 0


def test_websocket_progress_events_received():
    """At least one progress event must be received for any question."""
    events = []
    with client.websocket_connect(
        "/api/v1/query/stream",
        headers={"Authorization": "Bearer test_key"}
    ) as ws:
        ws.send_text(json.dumps({"question": "Hello"}))
        while True:
            message = ws.receive_text()
            data = json.loads(message)
            events.append(data)
            if data["type"] in ("done", "error"):
                break

    progress_events = [e for e in events if e["type"] == "progress"]
    assert len(progress_events) >= 1


def test_websocket_done_event_received():
    """Done event must always be the last event received."""
    events = []
    with client.websocket_connect(
        "/api/v1/query/stream",
        headers={"Authorization": "Bearer test_key"}
    ) as ws:
        ws.send_text(json.dumps({"question": "Hello"}))
        while True:
            message = ws.receive_text()
            data = json.loads(message)
            events.append(data)
            if data["type"] in ("done", "error"):
                break

    assert events[-1]["type"] == "done"


def test_websocket_invalid_json_handled():
    """Invalid JSON must not crash the server."""
    with client.websocket_connect(
        "/api/v1/query/stream",
        headers={"Authorization": "Bearer test_key"}
    ) as ws:
        ws.send_text("this is not json")
        message = ws.receive_text()
        data = json.loads(message)
        assert data["type"] == "error"



def test_websocket_specific_stock_streams_tokens():
    """Stock question must stream tokens via WebSocket."""
    events = []
    with client.websocket_connect(
        "/api/v1/query/stream",
        headers={"Authorization": "Bearer test_key"}
    ) as ws:
        ws.send_text(json.dumps({"question": "Analyse Apple"}))
        while True:
            message = ws.receive_text()
            data = json.loads(message)
            events.append(data)
            if data["type"] in ("done", "error"):
                break

    tokens = [e for e in events if e["type"] == "token"]
    done_event = next(e for e in events if e["type"] == "done")

    assert len(tokens) > 100
    assert "AAPL" in done_event["tickers"]
    assert done_event["top_intent"] == "TASK"
    assert done_event["sub_intent"] == "SPECIFIC_STOCK"