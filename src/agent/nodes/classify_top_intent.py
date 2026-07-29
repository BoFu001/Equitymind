"""
src/agent/nodes/classify_top_intent.py

Node: Top Intent Classification (Layer 1)
"""

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL_LIGHT, CONVERSATION_HISTORY_LIMIT
from langgraph.config import get_stream_writer
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.formatters.conversation_formatter import format_conversation_context
from colors import gprint

client = OpenAI(api_key=OPENAI_API_KEY)


def classify_top_intent(state: AgentState) -> dict:
    """
    Layer 1 — coarse classification.
    Decides whether this question needs the task pipeline at all,
    before any tool-related complexity enters.
    """

    writer = get_stream_writer()
    writer({"type": "progress", "node": "classify_top_intent", "message": NODE_PROGRESS["classify_top_intent"]})

    question = state.get("contextualized_question") or state["question"]
    messages = state.get("messages") or []
    session_memory  = state.get("session_memory") or {}
    in_clarification = (session_memory.get("structured") or {}).get("in_clarification", False)

    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT, max_chars=200)

    prompt = f"""You are {APP_NAME}'s coarse classifier.
Classify the user question into exactly one of these categories:

- GREETING: user is saying hello or asking what {APP_NAME} can do (e.g. "Hi", "What can you do?")
- OUT_OF_SCOPE: question has no relation to investing, stocks, or financial markets (e.g. "What's the weather?", "Am I handsome?"). Also OUT_OF_SCOPE: cryptocurrency questions (EquityMind only covers US-listed equities filing 10-Ks), scam/fraud recovery questions (e.g. "I lost money to a fake investment site, how do I get it back"), and "make money with no investment" / side-hustle questions. Do NOT classify as OUT_OF_SCOPE based on assumptions about whether a company is publicly traded — many companies have recently gone public. If the user mentions a company name and asks for analysis, classify as TASK.
- GENERAL_KNOWLEDGE: user is asking a general conceptual question about investing — they want to be taught, not helped with a specific task. Look for the ABSENCE of concrete parameters: no sector, no risk tolerance, no dollar amount, no named company. Signals: "what is...", "how do I start...", "I'm new to this", "I don't understand...". Examples: "What is a stock?", "How do I start investing as a beginner?", "What's the difference between stocks and ETFs?", "I'm new to investing, where do I start?"
- TASK: user wants {APP_NAME} to actually do something — analyse a company, compare companies, find recommendations, or answer a question using real data. This includes vague-but-parameterized requests (e.g. "I have $1000 to invest" — has a dollar amount, even though other details are still missing).

The key distinction between GENERAL_KNOWLEDGE and TASK: if the user has given ANY concrete parameter (a number, a sector, a risk preference, a named company) and seems to want a personalised answer, classify TASK even if details are still missing. If the user is asking what something means or how investing works in general, with no parameters at all, classify GENERAL_KNOWLEDGE.

CRITICAL OVERRIDE: phrases like "what is...", "what does...mean", "how do I..."
are only GENERAL_KNOWLEDGE signals when they stand alone with NO named company
or ticker anywhere in the question. If the question names a specific company
(e.g. "what does that F-Score mean for Microsoft?", "what is Apple's PEG
ratio?"), this is ALWAYS a TASK asking about that company's specific data —
never GENERAL_KNOWLEDGE — regardless of how the question is phrased. The
presence of a named company always takes priority over surface-level
phrasing patterns.

CONVERSATION HISTORY (for context):

{f"⚠️ IMPORTANT: The user is currently in the middle of answering {APP_NAME}'s clarification questions to find a stock recommendation. Their message is almost certainly continuing that conversation — even if phrased as a question, or if it gives indirect/contextual information (e.g. their age, life stage, a general statement about their goals) rather than a direct keyword answer. Classify as TASK unless the message is unmistakably a new greeting, a completely unrelated topic, or genuinely off-scope (e.g. asking about the weather or world news)." if in_clarification else ""}

{conversation_context if conversation_context else "NONE — this is the first message in this session."}

User question: {question}

Reply with ONLY the category name. Nothing else."""

    response = client.chat.completions.create(
        model=LLM_MODEL_LIGHT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    top_intent = response.choices[0].message.content.strip()

    gprint(f"  [classify_top_intent] top_intent: {top_intent}")
    return {"top_intent": top_intent}
