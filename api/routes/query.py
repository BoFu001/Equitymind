"""
api/routes/query.py

Endpoints:

1. WS  /api/v1/query/stream — WebSocket, two-layer streaming
   Layer 1: LangGraph node progress events
   Layer 2: GPT-4o token-by-token streaming


"""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.auth import verify_api_key
from api.schemas import (
    ConnectedEvent,
    ProgressEvent,
    SubProgressEvent,
    TokenEvent,
    DoneEvent,
    ErrorEvent,
    StreamRequest,
)

from core.context import token_queue_var
from src.agent.state import build_initial_state

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────
# Endpoint 1: WebSocket streaming
# ─────────────────────────────────────────────

@router.websocket("/query/stream")
async def query_stream(websocket: WebSocket):
    """
    WebSocket endpoint for two-layer streaming.

    Client connects, sends one JSON message:
        {"question": "Analyse Apple", "session_id": "abc123"}

    Server streams back events until done:
        {"type": "connected",  "job_id": "..."}
        {"type": "progress",   "node": "classify", "message": "..."}
        {"type": "token",      "text": "## Apple Inc"}
        {"type": "done",       "job_id": "...", "tickers": ["AAPL"], "top_intent": "TASK", "sub_intent": "SPECIFIC_STOCK"}
    """
    await websocket.accept()

    # ── API Key Authentication ────────────────────────────────
    auth_header = websocket.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        await websocket.close(code=4001)
        return
    api_key = auth_header.split(" ")[1]

    if not await verify_api_key(api_key):
        await websocket.close(code=4001)
        return
    # ─────────────────────────────────────────────────────────


    job_id = str(uuid.uuid4())
    logger.info("WebSocket connected (job_id=%s)", job_id)

    try:
        # ── Receive question from client ──
        raw = await websocket.receive_text()
        data = json.loads(raw)
        request = StreamRequest(**data)
        question = request.question
        messages = request.messages or []
        session_memory = request.session_memory or None

        logger.info("─" * 50)
        logger.info("Question : %s", question)
        logger.info("History  : %s messages", len(messages))
        for msg in messages:
            role = msg.get("role", "")
            preview = msg.get("content", "")[:50]
            logger.info("  [%s] %s...", role, preview)

        # ── Send connected event ──
        await websocket.send_text(
            ConnectedEvent(job_id=job_id).model_dump_json()
        )

        # ── Create token queue and set in context ──
        queue: asyncio.Queue = asyncio.Queue()
        token_queue_var.set(queue)

        # ── Import graph ──
        from src.agent.graph import build_graph
        graph = build_graph()

        initial_state = build_initial_state(question, messages, session_memory)

        # ── Sentinel: signals the token queue is done ──
        DONE = object()

        final_state = initial_state

        async def run_graph():
            """Runs the LangGraph graph, emits progress events, puts DONE when finished."""
            nonlocal final_state
            try:
                async for event in graph.astream(
                    initial_state,
                    stream_mode=["updates", "custom"]
                ):
                    kind, data = event

                    if kind == "updates":
                        for node_name, state_update in data.items():
                            final_state.update(state_update)

                    elif kind == "custom":
                        event_type = data.get("type")
                        if event_type == "progress":
                            await websocket.send_text(
                                ProgressEvent(
                                    node=data["node"],
                                    message=data["message"],
                                ).model_dump_json()
                            )
                        elif event_type == "sub_progress":
                            await websocket.send_text(
                                SubProgressEvent(node=data["node"], message=data["message"]).model_dump_json()
                            )

            except Exception as e:
                logger.exception("Graph error (job_id=%s)", job_id)
                await queue.put(e)
            finally:
                await queue.put(DONE)




        async def stream_tokens():
            """Reads tokens from queue and sends to WebSocket until DONE."""
            while True:
                item = await queue.get()
                if item is DONE:
                    break
                if isinstance(item, Exception):
                    await websocket.send_text(
                        ErrorEvent(message="An error occurred. Please try again.").model_dump_json()
                    )
                    return
                # item is a token string
                await websocket.send_text(
                    TokenEvent(text=item).model_dump_json()
                )

        # ── Run graph, sub_progress and stream tokens concurrently ──
        await asyncio.gather(run_graph(), stream_tokens())

        # ── Send done event ──
        await websocket.send_text(
            DoneEvent(
                job_id=job_id,
                tickers=final_state.get("tickers"),
                top_intent=final_state.get("top_intent"),
                sub_intent=final_state.get("sub_intent"),
                messages=final_state.get("messages"),
                session_memory=final_state.get("session_memory"),
            ).model_dump_json()
        )

        logger.info("Answer   : %s...", (final_state.get("answer") or "")[:100])
        logger.info("─" * 50)
        logger.info("WebSocket completed (job_id=%s)", job_id)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client (job_id=%s)", job_id)
    except Exception as e:
        logger.exception("WebSocket error (job_id=%s)", job_id)
        try:
            await websocket.send_text(
                ErrorEvent(message="An unexpected error occurred.").model_dump_json()
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


