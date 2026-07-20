"""
src/agent/nodes/contextualize_question.py

Node: Contextualize Question
"""

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, CONVERSATION_HISTORY_LIMIT
from langgraph.config import get_stream_writer
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.formatters import format_conversation_context
from colors import gprint, rprint

client = OpenAI(api_key=OPENAI_API_KEY)


def contextualize_question(state: AgentState) -> dict:
    """
    Runs before classify_top_intent. Rewrites the user's question into a
    standalone, self-contained form if it depends on conversation history
    (pronouns, implicit references, short follow-ups about a specific
    detail from the previous answer). Leaves the question unchanged if
    it's already self-contained.

    This exists because classifiers and downstream nodes should never have
    to guess what a pronoun or an implicit reference points to — that
    guessing is exactly what caused follow-up questions to be misrouted
    to explain_concept instead of the data-backed fetch_all_data path.

    The original state["question"] is preserved for session memory and
    conversation history — only a new field, contextualized_question, is
    added for downstream nodes to use instead.
    """

    writer = get_stream_writer()
    writer({"type": "progress", "node": "contextualize_question", "message": NODE_PROGRESS["contextualize_question"]})

    question = state["question"]
    messages = state.get("messages") or []

    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT, max_chars=200)

    if not conversation_context:
        # First message in the session — nothing to contextualize against.
        gprint(f"  [contextualize_question] No history — using question as-is")
        return {"contextualized_question": question}

    prompt = f"""You are a query contextualization assistant.

Your ONLY job: determine whether the user's latest message can be understood
on its own, or whether it depends on the conversation history to make sense.

If the message is already self-contained (names its own subject, doesn't rely
on pronouns or implicit references to prior turns), return it UNCHANGED.

If the message depends on context — uses pronouns ("this", "it", "that"),
refers to something mentioned earlier without naming it, or is a short
follow-up question about a specific detail from the previous answer —
rewrite it into a standalone question that includes the specific company
being asked about, drawing only from the conversation history below.

CRITICAL — never invent a specific number, date, or fact that does not
literally appear in the conversation history. If the reference is
unambiguous (only one matching number/fact exists in the history), use
that exact value. If the reference could plausibly point to more than
one number or fact in the history (e.g. the previous answer mentioned
several percentages and the user just says "that percentage"), do NOT
guess which one and do NOT invent a new value — instead, keep the
reference in its original, unresolved form (e.g. "the drawdown
percentage you mentioned earlier") and let the company name be the
only thing you add. It is always better to preserve an ambiguous
reference than to resolve it with a fabricated number.

Do not add any information that isn't in the history or the question.

CONVERSATION HISTORY:
{conversation_context}

LATEST MESSAGE: {question}

Reply with ONLY the resulting question — either unchanged or rewritten.
No explanation, no markdown."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    contextualized_question = response.choices[0].message.content.strip()

    rprint(f"  [contextualize_question] original question: {question}")
    gprint(f"  [contextualize_question] contextualized question: {contextualized_question}")

    return {"contextualized_question": contextualized_question}
