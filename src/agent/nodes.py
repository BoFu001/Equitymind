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
from src.tools.market_data import get_stock_data
from src.tools.news_sentiment import get_news_and_sentiment
from src.tools.sec_retrieval import retrieve, fetch_embed_store_retrieve

client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# Node: Intent Classification
# ─────────────────────────────────────────────

def classify_intent(state: AgentState) -> dict:
    """
    Classifies the user's question into one of seven categories.
    Uses LLM with a strict prompt — returns only the category name.
    """


    writer = get_stream_writer()
    writer({"type": "progress", "node": "classify", "message": NODE_PROGRESS["classify"]})

    question = state["question"]
    messages = state.get("messages") or []
    session_memory  = state.get("session_memory") or {}
    in_clarification = (session_memory.get("structured") or {}).get("in_clarification", False)

    has_history = len(messages) > 0

    # Hard rule: no history = cannot be FOLLOW_UP
    # Skip LLM ambiguity by injecting explicit signal into prompt
    history_context = ""
    for msg in messages[-CONVERSATION_HISTORY_LIMIT:]:
        role = msg.get("role", "")
        content = msg.get("content", "")[:200]
        history_context += f"{role.upper()}: {content}\n"

    prompt = f"""You are {APP_NAME}'s intent classifier.
Classify the user question into exactly one of these categories:

- GREETING: user is saying hello or asking what {APP_NAME} can do (e.g. "Hi", "What can you do?")
- OUT_OF_SCOPE: question has NO relation to investing, stocks, or financial markets (e.g. "I want to be rich", "What's the weather?", "Am I handsome?")
- SPECIFIC_STOCK: user asks about one NAMED specific company (e.g. "What are Apple's risks?", "Analyse NVIDIA", "Tell me about Tesla"). The company must be explicitly named — NOT vague like "a tech company" or "a healthcare stock".
- COMPARISON: user wants to compare two or more EXPLICITLY NAMED companies with real identifiable stock tickers (e.g. "Compare Apple and Microsoft", "AAPL vs GOOGL", "Tesla versus BMW"). Also classify as COMPARISON if the user refers to previously suggested companies (e.g. "Compare the last 5 suggested", "Compare those stocks", "Which of those is better?"). IMPORTANT: if no specific company names are mentioned AND no reference to previous suggestions, classify as DISCOVERY instead.
- DISCOVERY: user wants general investment recommendations, asks about a sector, or asks general financial market questions without naming a specific company (e.g. "Find me a low risk stock", "Analyse a tech company", "Tell me about semiconductor stocks", "Tell me about the stock market", "What is a good investment?")
- ANALYZE_POSITION: user asks about their own holding in one stock (e.g. "I bought AAPL at $165, should I sell?", "I have 200 Apple shares, what should I do?")
- ANALYZE_PORTFOLIO: user wants to analyse their full portfolio of multiple stocks (e.g. "Review my portfolio: AAPL 200 shares, NVDA 50 shares")
- FOLLOW_UP: user asks a follow-up question about a company already discussed in conversation history, without requesting a full new analysis (e.g. "What is their P/E ratio?", "What about the risk?", "Is it oversold?", "What's the dividend?", "Tell me more about their revenue"). Only classify as FOLLOW_UP if there is conversation history — if no history exists, classify normally.
  NEVER classify as FOLLOW_UP if:
  - user asks for a NEW full analysis ("Analyse", "Give me a report", "Tell me about", "Research")
  - user mentions a DIFFERENT or NEW company not yet discussed in conversation history
  - user says "again" or "refresh" or "update"
  - there is NO conversation history
- CLARIFICATION: user wants investment recommendations but hasn't provided enough 
  criteria to make good suggestions. Classify as CLARIFICATION when the request is 
  too vague (e.g. "Find me a good stock", "What should I buy?", "I have $1000 to invest",
  "suggest some stocks", "recommend me something").
  NEVER classify as CLARIFICATION if:
  - user names a specific company → SPECIFIC_STOCK or COMPARISON
  - user provides enough criteria (sector + risk or sector + time horizon) → DISCOVERY
  - user is answering a clarification question with a new intent → classify by new intent

CONVERSATION HISTORY (for context):

{history_context if history_context else "NONE — this is the first message in this session."}

{"⚠️ IMPORTANT: There is NO conversation history. FOLLOW_UP is impossible. Do NOT classify as FOLLOW_UP." if not has_history else "Conversation history exists above."}
{"⚠️ IMPORTANT: User is currently in an active clarification flow — they are answering your questions to help find a good stock. Classify as CLARIFICATION unless they explicitly name a specific company or ask something completely unrelated." if in_clarification else ""}

User question: {question}

Reply with ONLY the category name. Nothing else."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    intent = response.choices[0].message.content.strip()
    print(f"  [classify_intent] Intent: {intent}")
    return {"intent": intent}





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

    question = state["question"]
    messages = state.get("messages") or []
    session_memory = state.get("session_memory") or {}
    last_tickers = (session_memory.get("structured") or {}).get("last_tickers", [])

    history_context = ""
    for msg in messages[-CONVERSATION_HISTORY_LIMIT:]:
        role = msg.get("role", "")
        content = msg.get("content", "")[:200]
        history_context += f"{role.upper()}: {content}\n"

    prompt = f"""You are a financial data extractor.
Extract the stock ticker(s) and year from the user question.

Rules:
- tickers: list of ALL stock ticker symbols. Convert ANY company name to its ticker symbol. If no company or ticker mentioned, return [].
- year: the year mentioned. If not mentioned, return null.
- Examples of conversions: Apple → AAPL, Microsoft → MSFT, Tesla → TSLA, NVIDIA → NVDA, Google → GOOGL, Amazon → AMZN, Alibaba → BABA, Meta → META, Samsung → 005930.KS, Tencent → 0700.HK

CONVERSATION HISTORY (for context):
{history_context}
User question: {question}

LAST TICKERS FROM PREVIOUS TURN (use these if user refers to "last", "those", "them", "the suggested ones"):
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
        print(f"  [extract_parameters] Empty response from {LLM_MODEL}, using defaults")
        return {"tickers": [], "year": None}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
            print(f"  [extract_parameters] Invalid JSON: {content}")
            return {"tickers": [], "year": None}

    tickers = data.get("tickers", [])
    year    = str(data.get("year")) if data.get("year") else None

    print(f"  [extract_parameters] Tickers: {tickers}, Year: {year}")
    return {"tickers": tickers, "year": year}


# ─────────────────────────────────────────────
# Node: Retrieve SEC Data
# ─────────────────────────────────────────────
def ensure_sec_data(state: AgentState) -> dict:
    """
    Single responsibility: get SEC chunks for all tickers.
    Works for 1 ticker (SPECIFIC_STOCK) or N tickers (COMPARISON, DISCOVERY).
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "ensure_sec", "message": NODE_PROGRESS["ensure_sec_data"]})

    tickers  = state.get("tickers") or []
    question = state["question"]

    all_chunks = {}
    for t in tickers:
        try:
            chunks = retrieve(question, t)
            if chunks:
                writer({"type": "sub_progress", "node": "ensure_sec", "message": NODE_PROGRESS["retrieve"].format(ticker=t)})
                all_chunks[t] = chunks
            else:
                writer({"type": "sub_progress", "node": "ensure_sec", "message": NODE_PROGRESS["fetch"].format(ticker=t)})
                all_chunks[t] = fetch_embed_store_retrieve(question, t)
        except Exception as e:
            print(f"  [ensure_sec_data] Could not fetch SEC data for {t}: {e}")
            all_chunks[t] = []

    print(f"  [ensure_sec_data] SEC data fetched for {list(all_chunks.keys())}")
    return {"chunks": all_chunks}

# ─────────────────────────────────────────────
# Node: Get Market Data
# ─────────────────────────────────────────────
def get_market_data(state: AgentState) -> dict:
    """
    Single responsibility: fetch market data for all tickers.
    Works for 1 ticker (SPECIFIC_STOCK) or N tickers (COMPARISON, DISCOVERY).
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "market_data", "message": NODE_PROGRESS["market_data"]})

    tickers = state.get("tickers") or []

    if not tickers:
        return {"market_data": {}}

    all_market_data = {}
    for t in tickers:
        writer({"type": "sub_progress", "node": "market_data", "message": NODE_PROGRESS["market_data_sub"].format(ticker=t)})
        data = get_stock_data(t)
        if data:
            all_market_data[t] = data
        print(f"  [get_market_data] Market data fetched for {t}")

    return {"market_data": all_market_data}






# ─────────────────────────────────────────────
# Node: Get News and Sentiment
# ─────────────────────────────────────────────
def get_news(state: AgentState) -> dict:
    writer = get_stream_writer()
    writer({"type": "progress", "node": "news", "message": NODE_PROGRESS["news"]})

    tickers = state.get("tickers") or []

    if not tickers:
        return {"news": {}}

    all_news = {}
    for t in tickers:
        writer({"type": "sub_progress", "node": "news", "message": NODE_PROGRESS["news_sub"].format(ticker=t)})
        articles = get_news_and_sentiment(t)
        all_news[t] = articles
        print(f"  [get_news] {len(articles)} articles fetched for {t}")

    return {"news": all_news}




# # ─────────────────────────────────────────────
# # Node: Specific Report
# # ─────────────────────────────────────────────
# def specific_report(state: AgentState) -> dict:
#     """
#     Combines all data and generates a structured investment report.
#     Uses LLM with full context: SEC chunks, market data, news.
#     Implements XAI by including evidence and sources.
#     """


#     writer = get_stream_writer()
#     writer({"type": "progress", "node": "report", "message": NODE_PROGRESS["specific_report"]})


#     question    = state["question"]
#     ticker      = (state.get("tickers") or [None])[0]
#     chunks      = (state.get("chunks") or {}).get(ticker, [])
#     market_data = (state.get("market_data") or {}).get(ticker, {})
#     news        = (state.get("news") or {}).get(ticker, [])
#     messages    = state.get("messages") or []

#     # ── Format SEC chunks for prompt ──
#     if not chunks:
#         sec_context = f"No SEC 10-K filing available for {ticker}. This company may be a foreign filer (20-F) or the filing could not be retrieved."
#     else:
#         sec_context = ""
#         for i, chunk in enumerate(chunks):
#             source = f"{ticker}_{chunk['chunk']['filing_type']}_{chunk['chunk']['section']}_{chunk['chunk']['filing_date']}"
#             sec_context += f"\n[SEC Source {i+1}: {source} | Score: {chunk['score']:.2f}]\n"
#             sec_context += chunk["chunk"]["text"] + "\n"

#     # ── Format market data for prompt ──
#     md = market_data
#     market_context = f"""
# Company: {md.get('company_name', ticker)}
# Price: ${md.get('current_price')} | Market Cap: {md.get('market_cap')}
# P/E: {md.get('pe_ratio')} | Forward P/E: {md.get('forward_pe')}
# Revenue: {md.get('revenue')} | Profit Margin: {md.get('profit_margin')}
# EPS (trailing): {md.get('eps_trailing')} | EPS (forward): {md.get('eps_forward')}
# 52w High: {md.get('52w_high')} | 52w Low: {md.get('52w_low')}
# Dividend Yield: {md.get('dividend_yield')} | Dividend Rate: {md.get('dividend_rate')}
# Analyst Target High: {md.get('target_high')} | Low: {md.get('target_low')} | Mean: {md.get('target_mean')}
# Analyst Recommendation: {md.get('recommendation')}
# RSI: {md.get('rsi')} | MACD: {md.get('macd')} | Signal: {md.get('macd_signal')}
# SMA50: {md.get('sma_50')} | SMA200: {md.get('sma_200')}
# Sector: {md.get('sector')} | Industry: {md.get('industry')}
# """

#     # ── Format news for prompt ──
#     news_context = ""
#     for i, article in enumerate(news):
#         news_context += f"\n[News {i+1}] {article.get('sentiment','').upper()} ({article.get('score',0):.2f})\n"
#         news_context += f"Title: {article.get('title','')}\n"
#         news_context += f"Summary: {article.get('summary','')}\n"
#         news_context += f"URL: {article.get('url','')}\n"
#         news_context += f"Published: {article.get('published','')}\n"

#     # ── Format chat history ──
#     history_context = ""
#     for msg in messages[-CONVERSATION_HISTORY_LIMIT:]: 
#         role = msg.get("role", "")
#         content = msg.get("content", "")
#         history_context += f"{role.upper()}: {content}\n"

#     # ── Build prompt ──
#     prompt = f"""You are {APP_NAME}, a professional AI investment research analyst.
# Generate a comprehensive, well-structured investment report in markdown format.

# IMPORTANT RULES:
# 1. Put the recommendation FIRST — users want the answer before the details.
# 2. Include ALL evidence and sources for XAI (explainable AI) transparency.
# 3. Include clickable news URLs so users can verify information.
# 4. Be specific with numbers — never vague.
# 5. Always include the disclaimer at the end.
# 6. Use markdown formatting with emojis for visual clarity.
# 7. Format large numbers cleanly — use $24.5B not $24,452,999,168. Use $7.5B not $7,506,999,808.
# 8. Round decimal numbers to 2 decimal places — use 15.10 not 15.105477.

# USER QUESTION: {question}
# TICKER: {ticker}
# DATE: {datetime.now().strftime('%B %d, %Y')}

# CONVERSATION HISTORY:
# {history_context}

# MARKET DATA:
# {market_context}

# SEC FILING DATA (from {ticker} 10-K annual report):
# {sec_context}

# NEWS & SENTIMENT (last 30 days):
# {news_context}

# Generate the report in this EXACT structure:

# # 📊 [Company Name] ([TICKER]) — Investment Analysis
# *Generated by {APP_NAME} · [date]*

# ---

# ## 💡 AI Recommendation: [BUY/HOLD/SELL]
# **Target Price:** $[mean analyst target] | **Current:** $[price] | **Upside:** [%]
# **Confidence:** [High/Medium/Low] | **Time Horizon:** [Short/Medium/Long term]

# > [2-3 sentence summary of why this recommendation]

# ---

# ## 📊 Quick Summary
# - [Key fact 1]
# - [Key fact 2]
# - [Key fact 3]
# - [Key fact 4]

# ---

# ## 💰 Valuation & Fundamentals
# | Metric | Value | Signal |
# |--------|-------|--------|
# | P/E Ratio | [value] | [Cheap/Fair/Expensive] |
# | Forward P/E | [value] | [signal] |
# | Revenue | [value] | [signal] |
# | Profit Margin | [value] | [signal] |
# | EPS (TTM) | [value] | |
# | Dividend Yield | [value] | |

# ---

# ## 📈 Technical Analysis
# | Indicator | Value | Signal |
# |-----------|-------|--------|
# | RSI (14) | [value] | [Oversold/Neutral/Overbought] |
# | MACD | [value] | [Bullish/Bearish] |
# | SMA 50 | [value] | [Above/Below price] |
# | SMA 200 | [value] | [Above/Below price] |
# | 52w Range | [low] - [high] | [position] |

# ---

# ## 📰 News & Sentiment (Last 30 Days)
# **Overall Sentiment:** [BULLISH/NEUTRAL/BEARISH] | **Avg Score:** [x.xx]

# [For each article:]
# [emoji] [[title]]([url]) — [sentiment] ([score]) — [date]

# ---

# ## ⚠️ Key Risks (from SEC 10-K Filing)
# [Extract top 3-5 key risks from the SEC chunks provided]

# ---

# ## 📎 Evidence & Sources (XAI)
# **This recommendation is based on:**
# - **Fundamentals:** [key metrics used]
# - **Technicals:** [indicators used]
# - **News sentiment:** [x positive, y negative, z neutral articles]
# - **SEC Filing:** [sources cited with scores]

# ---

# ## ⚠️ Disclaimer
# *This report is generated by AI for educational and research purposes only. It does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.*
# """

#     queue = token_queue_var.get()
#     answer = ""

#     response = client.chat.completions.create(
#         model=LLM_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#         stream=True,
#     )

#     for stream_chunk in response:
#         token = stream_chunk.choices[0].delta.content or ""
#         if token:
#             answer += token
#             if queue:
#                 queue.put_nowait(token)

#     # Update conversation history
#     updated_messages = messages + [
#         {"role": "user",      "content": question},
#         {"role": "assistant", "content": answer},
#     ]

#     print(f"  [specific_report] Report generated for {ticker} ({len(answer)} chars)")
#     return {"answer": answer, "messages": updated_messages}





# ─────────────────────────────────────────────
# Node: Handle Out of Scope
# ─────────────────────────────────────────────
def handle_out_of_scope(state: AgentState) -> dict:
    """
    Returns a polite refusal for out-of-scope questions.
    """


    writer = get_stream_writer()
    writer({"type": "progress", "node": "out_of_scope", "message": NODE_PROGRESS["out_of_scope"]})



    messages = state.get("messages") or []
    question = state["question"]

    answer = f"""I'm {APP_NAME}, an AI investment research assistant. I specialise in stock analysis, company research, and investment insights.

I can help you with:
- 📊 Analysing a specific stock (e.g. "Analyse Apple")
- ⚖️ Comparing companies (e.g. "Compare NVIDIA and Microsoft")
- 🔍 Finding investment opportunities (e.g. "Find low risk stocks")
- 📰 News and sentiment analysis
- ⚠️ Risk analysis from SEC filings

What stock would you like me to research?"""

    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    queue = token_queue_var.get()
    if queue:
        for word in re.findall(r'\S+|\s+', answer):
            queue.put_nowait(word)
            time.sleep(0.03)

    print(f"  [handle_out_of_scope] Response generated ({len(answer)} chars)")
    return {"answer": answer, "messages": updated_messages}


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


    history_context = ""
    for msg in messages[-CONVERSATION_HISTORY_LIMIT:]:
        role = msg.get("role", "")
        content = msg.get("content", "")[:200]
        history_context += f"{role.upper()}: {content}\n"

    prompt = f"""You are {APP_NAME}, a professional AI investment research assistant.

CONVERSATION HISTORY:
{history_context}

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

    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    
    print(f"  [handle_greeting] Greeting generated ({len(answer)} chars)")

    return {"answer": answer, "messages": updated_messages}


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

    question = state["question"]
    messages = state.get("messages") or []

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
        print(f"  [discovery_suggest] Could not parse candidate tickers")
        candidate_tickers = []

    print(f"  [discovery_suggest] Candidates: {candidate_tickers}")
    return {"tickers": candidate_tickers}


# # ─────────────────────────────────────────────
# # Node: Discovery Report
# # ─────────────────────────────────────────────
# def discovery_report(state: AgentState) -> dict:
#     """
#     Single responsibility: format real data and generate discovery report.
#     Reads chunks, market_data, news from state — all populated by upstream nodes.
#     """
#     writer = get_stream_writer()
#     writer({"type": "progress", "node": "discovery_report", "message": NODE_PROGRESS["discovery_report"]})

#     question   = state["question"]
#     tickers    = state.get("tickers") or []
#     messages   = state.get("messages") or []
#     all_chunks = state.get("chunks") or {}
#     all_market = state.get("market_data") or {}
#     all_news   = state.get("news") or {}

#     # ── Format SEC chunks ──
#     sec_context = ""
#     for t in tickers:
#         chunks = all_chunks.get(t, [])
#         sec_context += f"\n{t} SEC Filing:\n"
#         if not chunks:
#             sec_context += "  No SEC 10-K data available.\n"
#         else:
#             for chunk in chunks:
#                 sec_context += chunk["chunk"]["text"][:300] + "\n"


#     # ── Format market data ──
#     market_context = ""
#     for t in tickers:
#         md = all_market.get(t, {})
#         market_context += f"\n{t} ({md.get('company_name')}):\n"
#         market_context += f"  Price: ${md.get('current_price')} | P/E: {md.get('pe_ratio')} | Forward P/E: {md.get('forward_pe')}\n"
#         market_context += f"  Revenue: {md.get('revenue')} | Profit Margin: {md.get('profit_margin')}\n"
#         market_context += f"  RSI: {md.get('rsi')} | Dividend Yield: {md.get('dividend_yield')}\n"
#         market_context += f"  Analyst Target: ${md.get('target_mean')} | Recommendation: {md.get('recommendation')}\n"


#     # ── Format news ──
#     news_context = ""
#     for t in tickers:
#         articles = all_news.get(t, [])
#         if articles:
#             news_context += f"\n{t} News:\n"
#             for article in articles[:3]:
#                 news_context += f"  [{article.get('sentiment','').upper()}] {article.get('title','')} ({article.get('score',0):.2f})\n"



#     prompt = f"""You are {APP_NAME}, a professional AI investment research analyst.
# The user wants investment recommendations. You have real market data and SEC filing data
# for 5 candidate companies. Use this real data to rank and recommend the top 3.

# IMPORTANT RULES:
# - Format large numbers cleanly — use $24.5B not $24,452,999,168.
# - Round decimal numbers to 2 decimal places — use 15.10 not 15.105477.

# USER QUESTION: {question}
# DATE: {datetime.now().strftime('%B %d, %Y')}

# SEC FILING EXCERPTS:
# {sec_context}

# REAL MARKET DATA FOR 5 CANDIDATES:
# {market_context}

# NEWS & SENTIMENT (last 30 days):
# {news_context}

# Based on the REAL DATA above:
# 1. Rank all 5 companies against the user's criteria
# 2. Select the TOP 3 that best match
# 3. Explain why each was selected using specific numbers from the real data
# 4. Briefly explain why the other 2 were not selected

# Generate in markdown format:

# # Investment Recommendations
# *Generated by {APP_NAME} · {datetime.now().strftime('%B %d, %Y')}*

# ## Top 3 Recommendations
# [For each of top 3 — explain with real numbers why it fits the user's criteria]

# ## Why we excluded the others
# [Brief explanation for the 2 not selected — based on real data]

# ## Summary Comparison (Top 3)
# | Metric | [ticker1] | [ticker2] | [ticker3] |
# |--------|-----------|-----------|-----------|
# | Price | | | |
# | P/E Ratio | | | |
# | RSI | | | |
# | Dividend Yield | | | |
# | Analyst View | | | |

# ## 📰 News Sentiment (Top 3)
# [For each of the top 3 companies — list 2-3 key news headlines with sentiment and score]
# Format: [emoji] [title] — [POSITIVE/NEGATIVE/NEUTRAL] ([score]) — [date]

# ## 👉 Next Steps
# Ask me for a detailed analysis of any of these companies.

# *This is AI-generated for educational purposes only. Not financial advice.*"""

#     queue  = token_queue_var.get()
#     answer = ""

#     response = client.chat.completions.create(
#         model=LLM_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#         stream=True,
#     )

#     for stream_chunk in response:
#         token = stream_chunk.choices[0].delta.content or ""
#         if token:
#             answer += token
#             if queue:
#                 queue.put_nowait(token)

#     updated_messages = messages + [
#         {"role": "user",      "content": question},
#         {"role": "assistant", "content": answer},
#     ]

#     print(f"  [discovery_report] Report generated for {tickers} ({len(answer)} chars)")
#     return {"answer": answer, "messages": updated_messages}


# # ─────────────────────────────────────────────
# # Node: Comparison Report
# # ─────────────────────────────────────────────
# def comparison_report(state: AgentState) -> dict:
#     """
#     Single responsibility: format real data and generate comparison report.
#     Reads chunks, market_data, news from state — all populated by upstream nodes.
#     """
#     writer = get_stream_writer()
#     writer({"type": "progress", "node": "comparison", "message": NODE_PROGRESS["comparison_report"]})

#     question   = state["question"]
#     tickers    = state.get("tickers") or []
#     messages   = state.get("messages") or []
#     all_chunks = state.get("chunks") or {}
#     all_market = state.get("market_data") or {}
#     all_news   = state.get("news") or {}

#     # ── Format market data ──
#     market_context = ""
#     for t in tickers:
#         md = all_market.get(t, {})
#         market_context += f"\n{t}:\n"
#         market_context += f"  Company: {md.get('company_name')}\n"
#         market_context += f"  Price: ${md.get('current_price')} | P/E: {md.get('pe_ratio')} | Forward P/E: {md.get('forward_pe')}\n"
#         market_context += f"  Revenue: {md.get('revenue')} | Profit Margin: {md.get('profit_margin')}\n"
#         market_context += f"  RSI: {md.get('rsi')} | MACD: {md.get('macd')}\n"
#         market_context += f"  Analyst Target Mean: ${md.get('target_mean')} | Recommendation: {md.get('recommendation')}\n"
#         market_context += f"  EPS: {md.get('eps_trailing')} | Dividend Yield: {md.get('dividend_yield')}\n"

#     # ── Format SEC chunks ──
#     sec_context = ""
#     for t in tickers:
#         chunks = all_chunks.get(t, [])
#         sec_context += f"\n{t} SEC Filing:\n"
#         if not chunks:
#             sec_context += "  No SEC 10-K data available.\n"
#         else:
#             for chunk in chunks:
#                 sec_context += chunk["chunk"]["text"][:300] + "\n"

#     # ── Format news ──
#     news_context = ""
#     for t in tickers:
#         articles = all_news.get(t, [])
#         if articles:
#             news_context += f"\n{t} News:\n"
#             for article in articles[:3]:
#                 news_context += f"  [{article.get('sentiment','').upper()}] {article.get('title','')} ({article.get('score',0):.2f})\n"

#     prompt = f"""You are {APP_NAME}, a professional AI investment research analyst.
# Generate a detailed comparison report between these companies.

# IMPORTANT RULES:
# - Format large numbers cleanly — use $24.5B not $24,452,999,168.
# - Round decimal numbers to 2 decimal places — use 15.10 not 15.105477.

# USER QUESTION: {question}
# COMPANIES: {', '.join(tickers)}
# DATE: {datetime.now().strftime('%B %d, %Y')}

# MARKET DATA:
# {market_context}

# SEC FILING EXCERPTS:
# {sec_context}

# NEWS & SENTIMENT (last 30 days):
# {news_context}

# Generate a structured comparison report in markdown format:

# # ⚖️ {' vs '.join(tickers)} — Comparison Analysis
# *Generated by {APP_NAME} · {datetime.now().strftime('%B %d, %Y')}*

# ---

# ## 💡 Verdict
# [Which company wins for the user's specific criteria and why — be direct]

# ---

# ## 📊 Side-by-Side Comparison
# | Metric | {' | '.join(tickers)} |
# |--------|{'|'.join(['--------' for _ in tickers])}|
# | Price | [values] |
# | P/E Ratio | [values] |
# | Forward P/E | [values] |
# | Revenue | [values] |
# | Profit Margin | [values] |
# | RSI | [values] |
# | Analyst Target | [values] |
# | Recommendation | [values] |

# ---

# ## 📈 Technical Comparison
# [Compare RSI, MACD, price performance for each company]

# ---

# ## 📰 News Sentiment
# [For each company — 1-2 key headlines with sentiment and score]

# ---

# ## ⚠️ Key Risks
# [Main risks for each company from SEC filings]

# ---

# ## 📎 Evidence & Sources (XAI)
# **Data sources used:**
# - Market data: yfinance
# - SEC filings: {', '.join([f'{t}_10-K' for t in tickers])}
# - News sentiment: Finlight + FinBERT

# ---

# ## ⚠️ Disclaimer
# *This report is AI-generated for educational purposes only. Not financial advice.*"""

#     queue  = token_queue_var.get()
#     answer = ""

#     response = client.chat.completions.create(
#         model=LLM_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#         stream=True,
#     )

#     for stream_chunk in response:
#         token = stream_chunk.choices[0].delta.content or ""
#         if token:
#             answer += token
#             if queue:
#                 queue.put_nowait(token)

#     updated_messages = messages + [
#         {"role": "user",      "content": question},
#         {"role": "assistant", "content": answer},
#     ]

#     print(f"  [comparison_report] Report generated for {tickers} ({len(answer)} chars)")
#     return {"answer": answer, "messages": updated_messages}

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


    
    messages = state.get("messages") or []
    question = state["question"]
    intent   = state.get("intent", "")

    if intent == "COMPARISON":
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

    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    queue = token_queue_var.get()
    if queue:
        for word in re.findall(r'\S+|\s+', answer):
            queue.put_nowait(word)
            time.sleep(0.03)

    print(f"  [handle_no_ticker] Response generated ({len(answer)} chars)")
    return {"answer": answer, "messages": updated_messages}






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
    intent         = state.get("intent") or ""
    tickers        = state.get("tickers") or []
    messages       = state.get("messages") or []
    session_memory = state.get("session_memory") or {}

    # ── Get previoursly saved memory ──
    structured = session_memory.get("structured", {
        "tickers_discussed":   [],       # active
        "last_tickers":        [],       # active
        "last_intent":         "",       # active
        "top_recommendations": [],       # future
        "user_preferences": {            # future
            "sectors": [],
            "risk":    "",
            "style":   "",
        }
    })

    # ── Update structured facts ──
    existing_tickers = structured.get("tickers_discussed", [])
    for t in tickers:
        if t not in existing_tickers:
            existing_tickers.append(t)

    structured["tickers_discussed"] = existing_tickers        # active
    structured["last_tickers"]      = tickers                 # active
    structured["last_intent"]       = intent                  # active
    new_market_data = state.get("market_data")
    if new_market_data:
        structured["last_market_data"] = new_market_data      # active
    # structured["top_recommendations"]                         future
    # structured["user_preferences"]                            future (separate LLM call)

    # ── Update clarification state ──
    if intent == "CLARIFICATION":
        structured["in_clarification"] = not state.get("clarification_ready", False)
    else:
        structured["in_clarification"] = False

    print(f"  [update_session_memory] in_clarification: {structured.get('in_clarification')}")

    # ── Build conversation context for narrative ──
    conversation_context = ""
    for msg in messages[-CONVERSATION_HISTORY_LIMIT:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")[:300]
        conversation_context += f"{role.upper()}: {content}\n"

    existing_narrative = session_memory.get("narrative", "")

    # ── Generate updated narrative ──
    narrative_prompt = f"""You are a memory summariser for {APP_NAME}, an AI investment research assistant.

EXISTING SUMMARY:
{existing_narrative if existing_narrative else "No previous summary."}

LATEST TURN:
User asked: {question}
Intent: {intent}
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

    print(f"  [update_session_memory] Session memory updated — tickers: {structured['tickers_discussed']}")
    print(f"  [update_session_memory] Narrative: {narrative[:100]}...")

    return {"session_memory": updated_session_memory}





# ─────────────────────────────────────────────
# Node: Simple Report
# ─────────────────────────────────────────────
def simple_report(state: AgentState) -> dict:
    """
    Answers the user's question directly using all available data.
    No fixed template — LLM decides format based on what was asked.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "report", "message": NODE_PROGRESS["specific_report"]})

    question    = state["question"]
    tickers     = state.get("tickers") or []
    all_chunks  = state.get("chunks") or {}
    all_market  = state.get("market_data") or {}
    all_news    = state.get("news") or {}
    messages    = state.get("messages") or []

    # ── Format SEC chunks ──
    sec_context = ""
    for t in tickers:
        chunks = all_chunks.get(t, [])
        if not chunks:
            sec_context += f"\n{t}: No SEC 10-K filing available.\n"
        else:
            sec_context += f"\n{t} SEC Filing:\n"
            for i, chunk in enumerate(chunks):
                source = f"{t}_{chunk['chunk']['filing_type']}_{chunk['chunk']['section']}_{chunk['chunk']['filing_date']}"
                sec_context += f"[Source {i+1}: {source} | Score: {chunk['score']:.2f}]\n"
                sec_context += chunk["chunk"]["text"] + "\n"

    # ── Format market data ──
    market_context = ""
    for t in tickers:
        md = all_market.get(t, {})
        market_context += f"""
{t} — {md.get('company_name')}
  Price: ${md.get('current_price')} | Market Cap: {md.get('market_cap')}
  P/E: {md.get('pe_ratio')} | Forward P/E: {md.get('forward_pe')}
  Revenue: {md.get('revenue')} | Profit Margin: {md.get('profit_margin')}
  EPS (TTM): {md.get('eps_trailing')} | EPS (Fwd): {md.get('eps_forward')}
  52w High: {md.get('52w_high')} | 52w Low: {md.get('52w_low')}
  Dividend Yield: {md.get('dividend_yield')}
  Analyst Target: ${md.get('target_mean')} | Recommendation: {md.get('recommendation')}
  RSI: {md.get('rsi')} | MACD: {md.get('macd')}
  SMA50: {md.get('sma_50')} | SMA200: {md.get('sma_200')}
  Sector: {md.get('sector')} | Industry: {md.get('industry')}
"""

    # ── Format news ──
    news_context = ""
    for t in tickers:
        articles = all_news.get(t, [])
        if articles:
            news_context += f"\n{t} News:\n"
            for i, article in enumerate(articles):
                news_context += f"[{article.get('sentiment','').upper()}] ({article.get('score',0):.2f}) {article.get('title','')}\n"
                news_context += f"  Summary: {article.get('summary','')}\n"
                news_context += f"  URL: {article.get('url','')}\n"
                news_context += f"  Published: {article.get('published','')}\n"

    # ── Format conversation history ──
    history_context = ""
    for msg in messages[-CONVERSATION_HISTORY_LIMIT:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        history_context += f"{role.upper()}: {content}\n"

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
Always end with a one-line disclaimer for investment-related responses.

USER QUESTION: {question}
TICKERS: {', '.join(tickers)}
DATE: {datetime.now().strftime('%B %d, %Y')}

CONVERSATION HISTORY:
{history_context}

MARKET DATA:
{market_context}

SEC FILING DATA:
{sec_context}

NEWS & SENTIMENT:
{news_context}"""

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

    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    print(f"  [simple_report] Response generated for {tickers} ({len(answer)} chars)")
    return {"answer": answer, "messages": updated_messages}




def handle_follow_up(state: AgentState) -> dict:
    """
    Answers follow-up questions directly from conversation history.
    No pipeline — reads from messages only.
    If data not found, tells user honestly what data is available.
    """
    writer = get_stream_writer()
    writer({"type": "progress", "node": "report", "message": NODE_PROGRESS["follow_up"]})

    question = state["question"]
    messages = state.get("messages") or []
    session_memory  = state.get("session_memory") or {}
    last_market_data = (session_memory.get("structured") or {}).get("last_market_data", {})

    # ── Format last known market data ──
    market_context = ""
    for t, md in last_market_data.items():
        if md:
            market_context += f"""
{t} — {md.get('company_name')}
  Price: ${md.get('current_price')} | Market Cap: {md.get('market_cap')}
  P/E: {md.get('pe_ratio')} | Forward P/E: {md.get('forward_pe')}
  Revenue: {md.get('revenue')} | Profit Margin: {md.get('profit_margin')}
  RSI: {md.get('rsi')} | MACD: {md.get('macd')}
  Analyst Target: ${md.get('target_mean')} | Recommendation: {md.get('recommendation')}
  52w High: {md.get('52w_high')} | 52w Low: {md.get('52w_low')}
  Dividend Yield: {md.get('dividend_yield')}
"""
    history_context = ""
    for msg in messages[-CONVERSATION_HISTORY_LIMIT:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        history_context += f"{role.upper()}: {content}\n"

    prompt = f"""You are {APP_NAME}, a professional AI investment research assistant.

The user is asking a follow-up question about a company already discussed.
Answer the question using the data available. Priority order:
1. LAST KNOWN MARKET DATA — use this first for any metric questions
2. CONVERSATION HISTORY — use this for context and previous analysis
3. If genuinely not answerable from either → say so briefly, then suggest what the user could ask instead.

Never say "I don't have data" if the data is clearly present above.
Keep the response short and direct. No headers. No tables unless necessary.
Format large numbers cleanly: use $5.07T for trillions, $892.4B for billions. Round to 2 decimal places.
End with a one-line disclaimer.

USER QUESTION: {question}

LAST KNOWN MARKET DATA (use this to answer metric questions):
{market_context if market_context else "No market data available."}

CONVERSATION HISTORY:
{history_context}"""

    # ── Debug: verify market data is available ──
    print(f"  [handle_follow_up] last_market_data keys: {list(last_market_data.keys())}")
    print(f"  [handle_follow_up] NVDA market_cap: {last_market_data.get('NVDA', {}).get('market_cap')}")
    print(f"  [handle_follow_up] TSLA market_cap: {last_market_data.get('TSLA', {}).get('market_cap')}")

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

    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    print(f"  [handle_follow_up] Response generated ({len(answer)} chars)")
    return {"answer": answer, "messages": updated_messages}



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

    history_context = ""
    for msg in messages[-CONVERSATION_HISTORY_LIMIT:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        history_context += f"{role.upper()}: {content}\n"

    prompt = f"""You are {APP_NAME}, a professional AI investment research assistant.

The user wants investment recommendations but hasn't provided enough criteria.
Your job is to collect the necessary information through friendly conversation.

Look at the conversation history below and decide:

1. Do you have ENOUGH information to make good recommendations?
   Minimum needed: at least ONE of these pairs:
   - sector + risk tolerance
   - sector + time horizon  
   - budget + sector
   - risk tolerance + time horizon

2. Reply with ONLY valid JSON. No markdown, no code fences, no explanation.


If NOT enough info:
{{"ready": false, "message": "Your friendly question here"}}

If ENOUGH info:
{{"ready": true, "message": "Find me a good stock in tech sector, medium risk, long term investment"}}

Where "message" is either:
- The clarifying question to ask the user (if not ready)
- The enriched question for DISCOVERY pipeline (if ready)

USER QUESTION: {question}

CONVERSATION HISTORY:
{history_context}"""

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
        print(f"  [handle_clarification] Invalid JSON: {content}")
        data = {"ready": False, "message": "Could you tell me more about what you're looking for?"}

    is_ready = data.get("ready", False)
    message  = data.get("message", "")

    # Stream message word by word to user
    if queue and not is_ready:
        for word in re.findall(r'\S+|\s+', message):
            queue.put_nowait(word)
            time.sleep(0.03)

    updated_messages = messages + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": message},
    ]

    # Check if ready to route to DISCOVERY
    if is_ready:
        print(f"  [handle_clarification] Ready — enriched question: {message}")
        return {
            "answer":              message,
            "question":            message,
            "clarification_ready": True,
            "messages":            updated_messages,
        }
    else:
        print(f"  [handle_clarification] Asking clarification ({len(message)} chars)")
        return {
            "answer":              message,
            "clarification_ready": False,
            "messages":            updated_messages,
        }