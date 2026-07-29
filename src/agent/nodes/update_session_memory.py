"""
src/agent/nodes/update_session_memory.py

Node: Update Session Memory
"""

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL, CONVERSATION_HISTORY_LIMIT
from src.agent.state import AgentState
from src.agent.formatters.conversation_formatter import format_conversation_context
from colors import gprint, rprint

client = OpenAI(api_key=OPENAI_API_KEY)


def update_session_memory(state: AgentState) -> dict:
    """
    Runs after every terminal node.
    Updates structured facts and regenerates narrative summary.
    """

    # ── Get current state ──
    question       = state["question"]
    answer         = state.get("answer") or ""
    top_intent     = state.get("top_intent") or ""
    sub_intent     = state.get("sub_intent") or ""
    tickers        = state.get("tickers") or []
    messages       = state.get("messages") or []
    session_memory = state.get("session_memory") or {}

    # ── Append this turn to conversation history (single point of truth) ──
    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    # ── Get previoursly saved memory ──
    structured = session_memory.get("structured", {
        "last_tickers": [],                             
    })

    structured["last_tickers"] = tickers

    # ── Update clarification state ──
    if sub_intent == "CLARIFICATION":
        structured["in_clarification"] = not state.get("clarification_complete", False)
    else:
        structured["in_clarification"] = False

    gprint(f"  [update_session_memory] in_clarification: {structured.get('in_clarification')}")

    # ── Build conversation context for narrative ──
    conversation_context = format_conversation_context(updated_messages, CONVERSATION_HISTORY_LIMIT, max_chars=300)

    existing_narrative = session_memory.get("narrative", "")

    # ── Generate updated narrative ──
    narrative_prompt = f"""You are a memory summariser for {APP_NAME}, an AI investment research assistant.

EXISTING SUMMARY:
{existing_narrative if existing_narrative else "No previous summary."}

CONVERSATION HISTORY:
{conversation_context}

LATEST TURN:
User asked: {question}
Top-level category: {top_intent}
Task type: {sub_intent if sub_intent else "N/A"}
Tickers involved: {tickers}
Answer summary: {answer[:300]}

Update the summary to include the latest turn. Keep it concise — maximum 5 sentences.
Focus on: what stocks were discussed, user preferences revealed, recommendations made, and any user feedback.
Write in third person. Do not include disclaimers or formatting."""

    narrative_response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": narrative_prompt}],
        temperature=0,
    )

    narrative = narrative_response.choices[0].message.content.strip()

    updated_session_memory = {
        "structured": structured,
        "narrative":  narrative,
    }

    gprint(f"  [update_session_memory] Session memory updated — tickers: {tickers}")
    gprint(f"  [update_session_memory] Narrative: {narrative[:100]}...")

    rprint("  [update_session_memory] Final messages:")
    for i, msg in enumerate(updated_messages):
        rprint(f"    {i}: [{msg['role']}] {msg['content'][:100]}")

    return {
        "messages":       updated_messages,
        "session_memory": updated_session_memory,
    }
