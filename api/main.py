"""
api/main.py

FastAPI application entry point.

Startup order
-------------
1. Logging configured.
2. Lifespan: pre-loads FinBERT on startup so the first request is not slow.
3. CORS middleware — allows all origins during development.
4. API routes mounted at /api/v1.

Running locally
---------------
    conda activate equitymind-core
    uvicorn api.main:app --reload --port 8000

API docs available at:
    http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.query import router as query_router
from api.routes.research import router as research_router
from api.routes.health import router as health_router
from api.routes.pipeline import router as pipeline_router
from config import APP_NAME

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Lifespan — startup and shutdown
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("%s Core starting up...", APP_NAME)

    # Pre-load FinBERT so the first user request is not slow.
    try:
        from src.quant.news_sentiment_signal import finbert  # noqa: F401
        logger.info("FinBERT sentiment model loaded.")
    except Exception as exc:
        logger.warning("FinBERT pre-warm skipped: %s", exc)

    yield

    logger.info("%s Core shutting down.", APP_NAME)


# ─────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────
app = FastAPI(
    title=f"{APP_NAME} Core",
    description=(
        "Financial Agentic RAG System. "
        "Query any stock ticker and receive a structured equity research report "
        "generated from SEC filings, live market data, and news sentiment."
    ),
    version="0.4.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# CORS middleware
# Allow all origins during development.
# Tighten before production — replace "*" with your frontend URL.
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
app.include_router(query_router, prefix="/api/v1")
app.include_router(research_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")
app.include_router(pipeline_router, prefix="/api/v1")