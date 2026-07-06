import json
import re
import random
import time
from datetime import datetime

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL, CONVERSATION_HISTORY_LIMIT
from langgraph.config import get_stream_writer
from core.context import token_queue_var
from src.agent.state import AgentState
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.formatters import format_market_data, format_news, format_sec_chunks, format_conversation_context
from colors import gprint, rprint

client = OpenAI(api_key=OPENAI_API_KEY)




# ─────────────────────────────────────────────
# Node: Contextualize Question
# ─────────────────────────────────────────────
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
    to explain_concept instead of the data-backed research_loop path.

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




# ─────────────────────────────────────────────
# Node: Top Intent Classification (Layer 1)
# ─────────────────────────────────────────────

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

CONVERSATION HISTORY (for context):

{f"⚠️ IMPORTANT: The user is currently in the middle of answering {APP_NAME}'s clarification questions to find a stock recommendation. Their message is almost certainly continuing that conversation — even if phrased as a question, or if it gives indirect/contextual information (e.g. their age, life stage, a general statement about their goals) rather than a direct keyword answer. Classify as TASK unless the message is unmistakably a new greeting, a completely unrelated topic, or genuinely off-scope (e.g. asking about the weather or world news)." if in_clarification else ""}

{conversation_context if conversation_context else "NONE — this is the first message in this session."}

User question: {question}

Reply with ONLY the category name. Nothing else."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    top_intent = response.choices[0].message.content.strip()

    gprint(f"  [classify_top_intent] top_intent: {top_intent}")
    return {"top_intent": top_intent}


# ─────────────────────────────────────────────
# Node: Task Classification (Layer 2)
# ─────────────────────────────────────────────

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



# ─────────────────────────────────────────────
# Node: Explain Concept (GENERAL_KNOWLEDGE handler)
# ─────────────────────────────────────────────

def explain_concept(state: AgentState) -> dict:
    """
    Answers general conceptual questions about investing.
    No tools, no SEC/market/news data — pure LLM explanation.
    Ends with a soft invitation back into TASK territory.
    """

    writer = get_stream_writer()
    writer({"type": "progress", "node": "explain_concept", "message": NODE_PROGRESS["explain_concept"]})

    question = state["question"]
    messages = state.get("messages") or []

    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT, max_chars=200)

    prompt = f"""You are {APP_NAME}, a professional AI investment research assistant.

The user is asking a general conceptual question about investing — they want to learn, not get a specific recommendation.
Give a clear, friendly, beginner-appropriate explanation. Keep it concise — a few short paragraphs, not an exhaustive essay.
Do NOT ask about sector preference, risk tolerance, or any other specific criteria — that would be premature for someone at this stage.
End with a brief, natural invitation: if they want help finding or analysing a specific stock once they're ready, they can just ask.

CONVERSATION HISTORY (for context):
{conversation_context if conversation_context else "NONE — this is the first message in this session."}

USER QUESTION: {question}

Use markdown and emojis sparingly where it aids clarity."""

    queue = token_queue_var.get()
    answer = ""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        stream=True,
    )

    for stream_chunk in response:
        token = stream_chunk.choices[0].delta.content or ""
        if token:
            answer += token
            if queue:
                queue.put_nowait(token)

    gprint(f"  [explain_concept] Explanation generated ({len(answer)} chars)")
    return {"answer": answer}

# ─────────────────────────────────────────────
# Node: Extract Parameters
# ─────────────────────────────────────────────
def extract_parameters(state: AgentState) -> dict:
    """
    Extracts ticker(s) and year from the user's question.
    Returns primary ticker, list of all tickers, and year.
    """

    writer = get_stream_writer()
    writer({"type": "progress", "node": "extract", "message": NODE_PROGRESS["extract"]})

    question = state.get("contextualized_question") or state["question"]
    messages = state.get("messages") or []
    session_memory = state.get("session_memory") or {}
    last_tickers = (session_memory.get("structured") or {}).get("last_tickers", [])

    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT, max_chars=200)

    prompt = f"""You are a financial data extractor.
Extract the stock ticker(s) and year from the user question.

Rules:
- tickers: list of ALL stock ticker symbols. Convert ANY company name to its ticker symbol. If no company or ticker mentioned, return [].
- year: the year mentioned. If not mentioned, return null.
- If the user refers to a previously discussed company using a pronoun ("this", "it", "that") instead of naming it, include the ticker from LAST TICKERS FROM PREVIOUS TURN below, IN ADDITION TO any newly named company in the current question. Example: if LAST TICKERS = ["GOOGL"] and the user asks "is this better than Tesla?", return ["GOOGL", "TSLA"] — both the referenced company and the newly named one.
- If the user refers to the whole previous list ("last", "those", "them", "the suggested ones"), return all of LAST TICKERS FROM PREVIOUS TURN.
- Examples of conversions: Apple → AAPL, Microsoft → MSFT, Tesla → TSLA, NVIDIA → NVDA, Google → GOOGL, Amazon → AMZN, Alibaba → BABA, Meta → META, Samsung → 005930.KS, Tencent → 0700.HK

CONVERSATION HISTORY (for context):
{conversation_context}
User question: {question}

LAST TICKERS FROM PREVIOUS TURN:
{last_tickers if last_tickers else "None"}

Reply with ONLY valid JSON. No markdown, no code fences, no explanation. Example:
{{"tickers": ["AAPL"], "year": null}}
{{"tickers": ["AAPL", "MSFT"], "year": null}}
{{"tickers": ["BABA"], "year": null}}
{{"tickers": ["AMZN"], "year": null}}
{{"tickers": [], "year": null}}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    if not content:
        gprint(f"  [extract_parameters] Empty response from {LLM_MODEL}, using defaults")
        return {"tickers": [], "year": None}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
            gprint(f"  [extract_parameters] Invalid JSON: {content}")
            return {"tickers": [], "year": None}

    tickers = data.get("tickers", [])
    year    = str(data.get("year")) if data.get("year") else None

    gprint(f"  [extract_parameters] Tickers: {tickers}, Year: {year}")
    return {"tickers": tickers, "year": year}



# ─────────────────────────────────────────────
# Node: Handle Out of Scope
# ─────────────────────────────────────────────
def handle_out_of_scope(state: AgentState) -> dict:
    """
    Returns a polite refusal for out-of-scope questions.
    """


    writer = get_stream_writer()
    writer({"type": "progress", "node": "out_of_scope", "message": NODE_PROGRESS["out_of_scope"]})


    answer = f"""I'm {APP_NAME}, an AI investment research assistant. I specialise in stock analysis, company research, and investment insights.

I can help you with:
- 📊 Analysing a specific stock (e.g. "Analyse Apple")
- ⚖️ Comparing companies (e.g. "Compare NVIDIA and Microsoft")
- 🔍 Finding investment opportunities (e.g. "Find low risk stocks")
- 📰 News and sentiment analysis
- ⚠️ Risk analysis from SEC filings

What stock would you like me to research?"""

    queue = token_queue_var.get()
    if queue:
        for word in re.findall(r'\S+|\s+', answer):
            queue.put_nowait(word)
            time.sleep(0.03)

    gprint(f"  [handle_out_of_scope] Response generated ({len(answer)} chars)")
    return {"answer": answer}


# ─────────────────────────────────────────────
# Node: Handle Greeting
# ─────────────────────────────────────────────
def handle_greeting(state: AgentState) -> dict:
    """
    Returns a friendly greeting and explains what this app can do.
    """

    writer = get_stream_writer()
    writer({"type": "progress", "node": "greeting", "message": NODE_PROGRESS["greeting"]})

    messages = state.get("messages") or []
    question = state["question"]


    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT, max_chars=200)

    prompt = f"""You are {APP_NAME}, a professional AI investment research assistant.

CONVERSATION HISTORY:
{conversation_context}

USER MESSAGE: {question}

If this is the first message (no history) — introduce yourself warmly and explain what you can do.
If the user is saying thank you, well done, or giving positive feedback — respond naturally and briefly, then invite them to ask another question.
If the user is saying goodbye — respond warmly and briefly.

Keep the response concise and contextual. Use markdown and emojis where appropriate."""


    queue = token_queue_var.get()
    answer = ""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=True,
    )    
    
    for stream_chunk in response:
        token = stream_chunk.choices[0].delta.content or ""
        if token:
            answer += token
            if queue:
                queue.put_nowait(token)
    
    gprint(f"  [handle_greeting] Greeting generated ({len(answer)} chars)")

    return {"answer": answer}


# ─────────────────────────────────────────────
# Node: Discovery Suggest
# ─────────────────────────────────────────────
def discovery_suggest(state: AgentState) -> dict:
    """
    Single responsibility: LLM suggests 5 candidate tickers based on user criteria.
    Writes candidate tickers to state["tickers"] so retrieve_sec, market_data, news can process them.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "discovery", "message": NODE_PROGRESS["discovery_suggest"]})

    # use synthesized question if clarification ran, else the raw user question
    question = state.get("enriched_query") or state["question"]
    

    rprint(f"  [discovery_suggest] question: {question}")



    ticker_prompt = f"""You are a financial analyst.
The user wants investment recommendations based on their criteria.

USER QUESTION: {question}

Return exactly 5 stock tickers that could match the user's criteria.
IMPORTANT: Only suggest US-listed companies that file 10-K annual reports with the SEC.
Do NOT suggest foreign companies or ADRs (e.g. Alibaba, ASML, Toyota, TSM).
Avoid always suggesting the same popular mega-cap companies (AAPL, MSFT, GOOGL, AMZN, NVDA) unless the user specifically asks for them.
Be creative and consider less obvious but relevant companies that genuinely match the user's criteria.
Exploration seed: {random.randint(1000, 9999)}

Reply with ONLY valid JSON. No markdown, no code fences, no explanation. Example:
{{"tickers": ["JNJ", "WMT", "BRK-B", "PFE", "JPM"]}}"""

    # TODO: Once 20-F pipeline is built, remove the 10-K constraint above.

    ticker_response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": ticker_prompt}],
        temperature=0.3,
    )

    try:
        ticker_data = json.loads(ticker_response.choices[0].message.content.strip())
        candidate_tickers = ticker_data.get("tickers", [])
    except Exception:
        gprint(f"  [discovery_suggest] Could not parse candidate tickers")
        candidate_tickers = []

    gprint(f"  [discovery_suggest] Candidates: {candidate_tickers}")
    return {"tickers": candidate_tickers}




# ─────────────────────────────────────────────
# Node: Handle No Ticker
# ─────────────────────────────────────────────
def handle_no_ticker(state: AgentState) -> dict:
    """
    User asked a valid financial question but no ticker could be extracted.
    Different from out_of_scope — the intent was valid, just no company identified.
    """


    writer = get_stream_writer()
    writer({"type": "progress", "node": "no_ticker", "message": NODE_PROGRESS["no_ticker"]})

    sub_intent = state.get("sub_intent", "")

    if sub_intent == "COMPARISON":
        answer = f"""I couldn't identify which companies you want to compare.

Please name the companies specifically, for example:
- "Compare Apple and Microsoft"
- "Compare AAPL vs GOOGL"
- "Tesla versus BMW" 

Note: Foreign companies like Airbus, Toyota, ASML, Alibaba are not yet supported (coming soon)."""
        
    else:
        answer = f"""I couldn't identify which company or stock you are asking about.

Please name the company specifically, for example:
- "Analyse Apple"
- "What are NVIDIA's risks?"
- "Tell me about Tesla" 

Note: Foreign companies like Airbus, Toyota, ASML, Alibaba are not yet supported (coming soon)."""

    queue = token_queue_var.get()
    if queue:
        for word in re.findall(r'\S+|\s+', answer):
            queue.put_nowait(word)
            time.sleep(0.03)

    gprint(f"  [handle_no_ticker] Response generated ({len(answer)} chars)")
    return {"answer": answer}




# ─────────────────────────────────────────────
# Node: Update Session Memory
# ─────────────────────────────────────────────
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





# ─────────────────────────────────────────────
# Node: Generate Report
# ─────────────────────────────────────────────
def generate_report(state: AgentState) -> dict:
    """
    Answers the user's question directly using all available data.
    No fixed template — LLM decides format based on what was asked.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "report", "message": NODE_PROGRESS["generate_report"]})

    question    = state.get("contextualized_question") or state["question"]
    tickers     = state.get("tickers") or []
    all_chunks  = state.get("chunks") or {}
    all_market  = state.get("market_data") or {}
    all_news    = state.get("news") or {}
    quant_signals = state.get("quant_signals") or {}
    messages    = state.get("messages") or []

    # ── Format SEC filing chunks ──
    sec_context = "".join(
        format_sec_chunks(all_chunks.get(t, []), t) if all_chunks.get(t) else f"\n{t}: No SEC 10-K filing available.\n"
        for t in tickers
    )

    # ── Format quant signals ──
    quant_context = ""
    for t in tickers:
        signals = quant_signals.get(t, {})
        quant_context += f"\n{t} Quantitative Signals:\n"

        # Valuation
        val = signals.get("valuation")
        if val:
            quant_context += f"  Valuation: {val['valuation_label']} (score={val['valuation_score']}, method={val['method']})\n"
            quant_context += f"  {val['detail']}\n"
            if val.get("reference_only"):
                quant_context += f"  ⚠️ Valuation reference only — company is loss-making, P/S used instead of P/E.\n"
            if val.get("stale_benchmark"):
                quant_context += f"  ⚠️ Sector benchmarks may be outdated (>90 days old).\n"
        else:
            quant_context += f"  Valuation: Insufficient data.\n"

        # Momentum
        mom = signals.get("momentum")
        if mom:
            quant_context += f"  Momentum: {mom['momentum_label']} (score={mom['momentum_score']})\n"
            quant_context += f"  {mom['detail']}\n"
        else:
            quant_context += f"  Momentum: Insufficient data.\n"

        # Risk
        risk = signals.get("risk")
        if risk:
            quant_context += f"  Risk: {risk['risk_level']} (score={risk['risk_score']})\n"
            quant_context += f"  {risk['detail']}\n"
            if risk.get("low_confidence"):
                quant_context += f"  ⚠️ Risk signal based on less than 1 year of price history — lower confidence.\n"
            if risk.get("beta") is None:
                quant_context += f"  ⚠️ Beta could not be computed — market benchmark data unavailable.\n"
            if risk.get("max_drawdown", {}).get("stress_tested") is False:
                quant_context += f"  ⚠️ This stock has not experienced a significant decline in the observed window — risk may be understated.\n"
        else:
            quant_context += f"  Risk: Insufficient data.\n"

    # ── Format market data ──
    market_context = "".join(format_market_data(all_market.get(t, {}), t) for t in tickers)

    # ── Format news and sentiment ──
    news_context   = "".join(format_news(all_news.get(t, []), t) for t in tickers if all_news.get(t))

    # ── Format conversation history ──
    conversation_context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT)

    prompt = f"""You are {APP_NAME}, a professional AI investment research assistant.

Answer the user's question directly and naturally using the data provided below.
Let the question determine the length and format of your response:
- Simple factual question (e.g. "What is the P/E ratio?", "What about the risk?") → answer in 1-3 sentences with the relevant numbers. No headers, no tables.
- Request for full analysis (e.g. "Analyse NVDA", "Give me a full report on AAPL") → generate a comprehensive structured report with headers, tables, and sections.
- Comparison request (e.g. "Compare NVDA and AMD") → generate a structured side-by-side comparison.
- Discovery request (e.g. "Find me a low risk stock") → rank and recommend from the candidates with real data.

Always use specific numbers from the data. Never be vague.
Always include ALL tickers in the response — never drop any company from the analysis.
Format large numbers cleanly: $24.5B not $24,452,999,168. Round to 2 decimal places.
Use markdown and emojis where appropriate for the format chosen.

When presenting analyst sentiment, use recommendation_mean for accuracy:
- 1.0–1.5 = Strong Buy, 1.5–2.5 = Buy, 2.5–3.5 = Hold, 3.5–4.5 = Sell, 4.5–5.0 = Strong Sell
- If recommendation_mean > 2.5, do not describe analysts as bullish or positive.
- Always mention analyst_count and recommendation_mean for transparency.
- If recommendation_mean is between 2.0–3.0, describe as "mixed" or "cautiously positive".
- When analyst target price range is very wide (high target > 3x low target),
  explicitly mention the divergence to highlight analyst uncertainty.
  
If QUANTITATIVE SIGNALS are provided, integrate them naturally into your analysis:
- If reference_only is flagged, explicitly state the valuation is indicative only.
- If stale_benchmark is flagged, note that sector benchmarks may be outdated.
- If quant signals show "Insufficient data", do not attempt to estimate valuation.
- When comparing companies that use different valuation methods 
  (e.g. one uses P/E and another uses P/S due to losses), explicitly 
  state that their valuation scores are NOT directly comparable, and 
  explain why each method was used for each company separately.
- When valuation and momentum signals point in opposite directions
  (e.g. overvalued but bullish, or undervalued but bearish),
  explicitly highlight this conflict and explain what it means for investors.
- When presenting the Risk signal, treat Beta, Sharpe Ratio, VaR, and Max
  Drawdown as historical statistics, not predictions of future risk.
  Never phrase them as guarantees (e.g. do not say "this stock cannot
  fall more than X%" — say "historically, losses have not exceeded X%
  under normal conditions").
- If Beta could not be computed (market benchmark unavailable), state
  this plainly and do not substitute your own estimate of the stock's
  market sensitivity.
- If the Risk signal is flagged as low confidence (less than 1 year of
  price history), mention this caveat explicitly rather than presenting
  the risk score with full confidence.
- If the stock's observed price window did not include a significant
  decline, mention that its historical Max Drawdown may understate
  risk in a genuine market downturn — do not present a low Max Drawdown
  alone as evidence the stock is safe in a crisis.
- Always state the observation window (e.g. "based on the past 2 years
  of price history") when presenting Risk signal metrics, so the user
  understands the time horizon these statistics are drawn from.
- When discussing Max Drawdown, always state BOTH the specific peak/trough
  dates AND the specific peak/trough prices provided in the data (e.g.
  "from $257.38 on Dec 26, 2024 to $171.51 on Apr 8, 2025"). Provide both
  pieces even if the user's question only asks about timing (e.g. "when
  did that happen?") — the price movement is part of a complete answer
  about a drawdown, not an optional extra.

USER QUESTION: {question}
TICKERS: {', '.join(tickers)}
DATE: {datetime.now().strftime('%B %d, %Y')}

CONVERSATION HISTORY:
{conversation_context}

MARKET DATA:
{market_context}

SEC FILING DATA:
{sec_context}

NEWS & SENTIMENT:
{news_context}

QUANTITATIVE SIGNALS:
{quant_context if quant_context else "No quantitative signals available."}"""
    
    queue = token_queue_var.get()
    answer = ""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        stream=True,
    )

    for stream_chunk in response:
        token = stream_chunk.choices[0].delta.content or ""
        if token:
            answer += token
            if queue:
                queue.put_nowait(token)

    gprint(f"  [generate_report] Response generated for {tickers} ({len(answer)} chars)")
    return {"answer": answer}



# ─────────────────────────────────────────────
# Node: Handle Clarification
# ─────────────────────────────────────────────
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