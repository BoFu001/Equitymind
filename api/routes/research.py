"""
api/routes/research.py

Sync batch endpoint for research and testing only.
Not part of the commercial product.

Used for:
- Dissertation test dataset evaluation
- Batch processing of multiple questions
- Automated testing without WebSocket

Not publicly documented — internal research use only.
"""

import logging
import uuid

from fastapi import APIRouter

from api.schemas import (
    SyncRequest,
    SyncResponse,
)
from src.agent.state import build_initial_state

logger = logging.getLogger(__name__)

router = APIRouter()





@router.post("/query/sync")
async def query_sync(request: SyncRequest):
    """
    Sync endpoint — runs the full agent and returns complete JSON.
    No streaming. For research and batch testing only.
    """
    job_id = str(uuid.uuid4())

    logger.info("Sync query received (job_id=%s): %s", job_id, request.question[:80])

    try:
        from src.agent.graph import build_graph
        graph = build_graph()

        initial_state = build_initial_state(request.question)

        # Run graph to completion — no streaming
        final_state = await graph.ainvoke(initial_state)

        return SyncResponse(
            job_id=job_id,
            tickers=final_state.get("tickers"),
            top_intent=final_state.get("top_intent"),
            sub_intent=final_state.get("sub_intent"),
            answer=final_state.get("answer", ""),
            status="success",
        )

    except Exception as e:
        logger.exception("Sync query error (job_id=%s)", job_id)
        return SyncResponse(
            job_id=job_id,
            tickers=None,
            top_intent=None,
            sub_intent=None,
            answer="",
            status="error",
            error=str(e),
        )
