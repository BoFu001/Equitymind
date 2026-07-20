"""
src/agent/nodes/handle_clarification.py

Node: Handle Clarification
"""

import json
import re
import time

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL, CONVERSATION_HISTORY_LIMIT
from langgraph.config import get_stream_writer
from core.context import token_queue_var
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.formatters import format_conversation_context
from colors import gprint

client = OpenAI(api_key=OPENAI_API_KEY)


def handle_clarification(state: AgentState) -> dict:
    """
    Multi-turn clarification for vague discovery requests.
    Asks one question at a time until enough criteria collected.
    When ready, builds enriched question and routes to DISCOVERY.
    """

    writer = get_stream_writer()
    writer({"type": "progress", "node": "clarification", "message": NODE_PROGRESS["clarification"]})

    question = state["question"]
    messages = state.get("messages") or []

    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT)
    full_context = conversation_context + f"USER: {question}\n"

    prompt = f"""You are {APP_NAME}, a professional AI investment research assistant.

The user wants investment recommendations but hasn't provided enough criteria.
Your job is to collect the necessary information through friendly conversation.

IMPORTANT: Users often answer indirectly rather than with exact keywords. Treat these as valid signals:
- Age or life stage ("I'm 47", "I'm retired", "I just graduated") → infer a reasonable time horizon (e.g. closer to retirement age suggests shorter horizon; young age suggests longer horizon). Do NOT ask the same question again if the user has already given you something you can reasonably infer from.
- General statements about goals ("I want to earn money now", "I'm saving for a house") → can imply risk tolerance or time horizon even without using those exact words.
- If the user asks YOU to make the inference ("what do you think my time horizon is?"), make a reasonable inference yourself rather than deflecting the question back to them.

Look at the conversation history below and decide:

1. Do you have ENOUGH information to make good recommendations?
   Minimum needed: at least ONE of these pairs:
   - sector + risk tolerance
   - sector + time horizon  
   - budget + sector
   - risk tolerance + time horizon
   Use reasonable inference from indirect signals (as described above) to fill in any of these — do not require the user to use exact keywords.

2. Reply with ONLY valid JSON. No markdown, no code fences, no explanation.


If NOT enough info:
{{"complete": false, "clarifying_question": "Your friendly question here"}}

If ENOUGH info:
{{"complete": true, "enriched_question": "Find me a good stock in tech sector, medium risk, long term investment"}}

CONVERSATION HISTORY (including the user's latest message):
{full_context}"""

    queue = token_queue_var.get()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=False,
    )

    content = response.choices[0].message.content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        gprint(f"  [handle_clarification] Invalid JSON: {content}")
        data = {"complete": False, "clarifying_question": "Could you tell me more about what you're looking for?"}

    complete = data.get("complete", False)

    if complete:
        enriched_question = data.get("enriched_question", "")
        gprint(f"  [handle_clarification] Complete — enriched question: {enriched_question}")
        return {
            "clarification_complete": True,
            "enriched_query":         enriched_question,
        }

    else:
        # only fire this sub_progress when we know another question is needed
        writer({"type": "sub_progress", "node": "clarification", "message": NODE_PROGRESS["clarification_sub"]})
        # Stream a clarifying question to user
        clarifying_question = data.get("clarifying_question", "")
        if queue:
            for word in re.findall(r'\S+|\s+', clarifying_question):
                queue.put_nowait(word)
                time.sleep(0.03)

        gprint(f"  [handle_clarification] Asking clarification ({len(clarifying_question)} chars)")
        return {
            "clarification_complete":  False,
            "answer":                  clarifying_question,
        }
