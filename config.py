import os
from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_MODEL = "gpt-4o"

# Conversation history — number of messages to include in LLM context
# 6 messages = 3 exchanges (1 exchange = 1 user + 1 assistant)
CONVERSATION_HISTORY_LIMIT = 6

# APP
APP_NAME = "EquityMind"

# Paths
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
SEC_FILINGS_DIR = os.path.join(DATA_DIR, "sec-edgar-filings")

# Chunking
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# API Keys
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
FINLIGHT_API_KEY = os.getenv("FINLIGHT_API_KEY")

# Database
DATABASE_URL = os.getenv("DATABASE_URL")

# Batch sizes
EMBEDDING_BATCH_SIZE = 100
PGVECTOR_BATCH_SIZE  = 100