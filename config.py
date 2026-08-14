import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
APP_NAME = "EquityMind"

# ─────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FINLIGHT_API_KEY = os.getenv("FINLIGHT_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
LLM_MODEL = "gpt-4o"
LLM_MODEL_LIGHT = "gpt-4o-mini"  # for simple, low-stakes tasks (e.g. company name normalization) — cheaper/faster, no complex reasoning needed

# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────
# 6 messages = 3 exchanges (1 exchange = 1 user + 1 assistant)
CONVERSATION_HISTORY_LIMIT = 6

# ─────────────────────────────────────────────
# RAG
# ─────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_BATCH_SIZE = 100
PGVECTOR_BATCH_SIZE = 100
SEC_TOP_K = 5

# ─────────────────────────────────────────────
# Data pipeline (Kestra internal trigger)
# ─────────────────────────────────────────────
# Used by Kestra, not exposed to external developers — separate from
# api/auth.py's future portal-based API key system.
PIPELINE_TRIGGER_SECRET = os.getenv("PIPELINE_TRIGGER_SECRET")

# Per-ticker delay in update_quant_signals.py — 0 locally, nonzero on
# Railway only (avoids yfinance rate limits Railway hits but local doesn't).
PIPELINE_THROTTLE_SECONDS = float(os.getenv("PIPELINE_THROTTLE_SECONDS", "0"))