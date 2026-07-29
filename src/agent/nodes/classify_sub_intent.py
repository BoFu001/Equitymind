"""
src/agent/nodes/classify_sub_intent.py

Node: Task Classification (Layer 2)
"""

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL, CONVERSATION_HISTORY_LIMIT
from langgraph.config import get_stream_writer
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.formatters.conversation_formatter import format_conversation_context
from colors import gprint

client = OpenAI(api_key=OPENAI_API_KEY)


def classify_sub_intent(state: AgentState) -> dict:
    """
    Layer 2 — fine classification.
    Only runs when top_intent == TASK.
    Classifies into one of seven task-types.
    """

    writer = get_stream_writer()
    writer({"type": "progress", "node": "classify_sub_intent", "message": NODE_PROGRESS["classify_sub_intent"]})

    question = state.get("contextualized_question") or state["question"]
    messages = state.get("messages") or []
    session_memory  = state.get("session_memory") or {}
    in_clarification = (session_memory.get("structured") or {}).get("in_clarification", False)

    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT, max_chars=200)

    prompt = f"""You are {APP_NAME}'s intent classifier.
The user's question has already been confirmed as a TASK — something {APP_NAME} should actually do.

CRITICAL RULE: If the user mentions ANY specific company name or stock ticker
(e.g. Microsoft, Apple, Tesla, TSLA, GOOGL), ALWAYS classify as SPECIFIC_STOCK
or COMPARISON — regardless of how the question is phrased.
NEVER classify as CLARIFICATION or DISCOVERY if a specific company is named.

Classify it into exactly one of these categories:

- SPECIFIC_STOCK: user asks about one NAMED specific company (e.g. "What are Apple's risks?", "Analyse NVIDIA", "Tell me about Tesla", "Who are Google's peers?", "What sector is Apple in?"). The company must be explicitly named OR clearly implied by conversation history (e.g. if the previous discussion was about KO and user asks "what's the percentage?", classify as SPECIFIC_STOCK).
  NEVER classify as CLARIFICATION if:
  - the question is a follow-up calculation or metric question about a company already discussed in conversation history
  - the question contains specific numbers or financial figures clearly referencing a previous answer
- COMPARISON: user wants to compare two or more EXPLICITLY NAMED companies with real identifiable stock tickers (e.g. "Compare Apple and Microsoft", "AAPL vs GOOGL", "Tesla versus BMW"). Also classify as COMPARISON if the user refers to previously suggested companies (e.g. "Compare the last 5 suggested", "Compare those stocks", "Which of those is better?"). IMPORTANT: if no specific company names are mentioned AND no reference to previous suggestions, classify as DISCOVERY instead.
- DISCOVERY: user wants general investment recommendations, asks about a sector, or asks general financial market questions without naming a specific company (e.g. "Find me a low risk stock", "Analyse a tech company", "Tell me about semiconductor stocks", "Tell me about the stock market", "What is a good investment?")
- ANALYZE_POSITION: user asks about their own holding in one stock (e.g. "I bought AAPL at $165, should I sell?", "I have 200 Apple shares, what should I do?")
- ANALYZE_PORTFOLIO: user wants to analyse their full portfolio of multiple stocks (e.g. "Review my portfolio: AAPL 200 shares, NVDA 50 shares")
- CLARIFICATION: user wants investment recommendations but hasn't provided enough 
  criteria to make good suggestions. Classify as CLARIFICATION when the request is 
  too vague (e.g. "Find me a good stock", "What should I buy?", "I have $1000 to invest",
  "suggest some stocks", "recommend me something").
  NEVER classify as CLARIFICATION if:
  - user names a specific company → SPECIFIC_STOCK or COMPARISON
  - user provides enough criteria (sector + risk or sector + time horizon) → DISCOVERY
  - user is answering a clarification question with a new intent → classify by new intent
  - user is asking a follow-up question about a specific number or calculation from the previous answer (e.g. "what's the percentage?", "how much is that in total?") → SPECIFIC_STOCK

CONVERSATION HISTORY (for context):

{conversation_context if conversation_context else "NONE — this is the first message in this session."}
{"⚠️ IMPORTANT: User is currently in an active clarification flow — they are answering your questions to help find a good stock. Classify as CLARIFICATION unless they explicitly name a specific company or ask something completely unrelated." if in_clarification else ""}

User question: {question}

Reply with ONLY the category name. Nothing else."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    sub_intent = response.choices[0].message.content.strip()
    gprint(f"  [classify_sub_intent] sub_intent: {sub_intent}")
    return {"sub_intent": sub_intent}
