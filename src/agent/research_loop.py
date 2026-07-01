"""
src/agent/research_loop.py

Smart Tool Selection via a real agent loop.
The model decides which tools to call, how many times, and in what order.
"""

from openai import OpenAI
import json

from config import OPENAI_API_KEY, LLM_MODEL
from src.tools.market_data import get_stock_data
from src.tools.news_sentiment import get_news_and_sentiment
from src.tools.sec_retrieval import retrieve, fetch_embed_store_retrieve
from src.agent.state import AgentState

from langgraph.config import get_stream_writer
from src.agent.nodes_notifications import NODE_PROGRESS
from colors import gprint, bprint

from src.agent.formatters import format_market_data, format_news, format_sec_chunks


client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# Tool: Market Data
# ─────────────────────────────────────────────

def get_market_data_tool(ticker: str) -> dict | None:

    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "research_loop", "message": NODE_PROGRESS["market_data_sub"].format(ticker=ticker)})


    data = get_stock_data(ticker)
    if not data:
        return None

    bprint(f"  [get_market_data_tool] Called for {ticker}")
    return data


get_market_data_json = {
    "name": "get_market_data",
    "description": "Fetch current market data for a single stock ticker — price, valuation ratios, technical indicators, analyst targets.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol, e.g. AAPL",
            },
        },
        "required": ["ticker"],
        "additionalProperties": False,
    },
}


# ─────────────────────────────────────────────
# Tool: News and Sentiment
# ─────────────────────────────────────────────

def get_news_tool(ticker: str) -> list:
    writer = get_stream_writer()
    writer({"type": "sub_progress", "node": "research_loop", "message": NODE_PROGRESS["news_sub"].format(ticker=ticker)})

    articles = get_news_and_sentiment(ticker)
    bprint(f"  [get_news_tool] Called for {ticker}")
    return articles


get_news_json = {
    "name": "get_news",
    "description": "Fetch recent news headlines and sentiment analysis for a single stock ticker.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol, e.g. AAPL",
            },
        },
        "required": ["ticker"],
        "additionalProperties": False,
    },
}


# ─────────────────────────────────────────────
# Tool: SEC Filing Retrieval
# ─────────────────────────────────────────────

def ensure_sec_data_tool(ticker: str, question: str) -> list:

    writer = get_stream_writer()

    try:
        chunks = retrieve(question, ticker)
        if chunks:
            writer({"type": "sub_progress", "node": "research_loop", "message": NODE_PROGRESS["retrieve"].format(ticker=ticker)})
        else:
            writer({"type": "sub_progress", "node": "research_loop", "message": NODE_PROGRESS["fetch"].format(ticker=ticker)})
            chunks = fetch_embed_store_retrieve(question, ticker)
    except Exception as e:
        bprint(f"  [ensure_sec_data_tool] Could not fetch SEC data for {ticker}: {e}")
        return []

    bprint(f"  [ensure_sec_data_tool] Called for {ticker}")
    return chunks


ensure_sec_data_json = {
    "name": "ensure_sec_data",
    "description": "Retrieve relevant excerpts from a company's SEC 10-K annual report filing, based on the user's question.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol, e.g. AAPL",
            },
            "question": {
                "type": "string",
                "description": "The user's question, used to retrieve the most relevant filing excerpts",
            },
        },
        "required": ["ticker", "question"],
        "additionalProperties": False,
    },
}


# ─────────────────────────────────────────────
# Loop: Tool registry and dispatch
# ─────────────────────────────────────────────

TOOLS = [
    {"type": "function", "function": get_market_data_json},
    {"type": "function", "function": get_news_json},
    {"type": "function", "function": ensure_sec_data_json},
]

TOOL_REGISTRY = {
    "get_market_data": get_market_data_tool,
    "get_news": get_news_tool,
    "ensure_sec_data": ensure_sec_data_tool,
}



def handle_tool_calls(tool_calls) -> tuple[list[dict], dict, dict, dict]:

    results          = []
    collected_chunks = {}
    collected_market = {}
    collected_news   = {}


    
    # notification
    writer = get_stream_writer()
    sent_progress = set()

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool = TOOL_REGISTRY.get(tool_name)

        if tool:            
            try:
                arguments = json.loads(tool_call.function.arguments)


                
                # notification
                if tool_name == "get_market_data" and "market_data" not in sent_progress:
                    writer({"type": "progress", "node": "market_data", "message": NODE_PROGRESS["market_data"]})
                    sent_progress.add("market_data")
                elif tool_name == "get_news" and "news" not in sent_progress:
                    writer({"type": "progress", "node": "news", "message": NODE_PROGRESS["news"]})
                    sent_progress.add("news")
                elif tool_name == "ensure_sec_data" and "ensure_sec" not in sent_progress:
                    writer({"type": "progress", "node": "ensure_sec", "message": NODE_PROGRESS["ensure_sec_data"]})
                    sent_progress.add("ensure_sec")
                # notification




                result = tool(**arguments)

                # collect structured data
                ticker    = arguments.get("ticker", "")

                if tool_name == "get_market_data":
                    if result:
                        collected_market[ticker] = result
                    content = format_market_data(result, ticker) if result else f"No market data available for {ticker}."

                elif tool_name == "get_news":
                    if result:
                        collected_news[ticker] = result
                    content = format_news(result, ticker) if result else f"No recent news found for {ticker}."

                elif tool_name == "ensure_sec_data":
                    if result:
                        collected_chunks[ticker] = result
                    content = format_sec_chunks(result, ticker) if result else f"No SEC 10-K filing available for {ticker}."

                else:
                    content = str(result)
                # collect structured data



            except json.JSONDecodeError:
                content = f"Malformed arguments for {tool_name}"
            except TypeError as e:
                content = f"Invalid arguments for {tool_name}: {e}"
        else:
            content = f"Unknown tool: {tool_name}"


        results.append({
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id,
        })

    return results, collected_chunks, collected_market, collected_news


# ─────────────────────────────────────────────
# Loop: Main agent loop node
# ─────────────────────────────────────────────

def research_loop(state: AgentState) -> dict:

    writer = get_stream_writer()
    writer({"type": "progress", "node": "research_loop", "message": NODE_PROGRESS["research_loop"]})


    # Use enriched_query if clarification ran, else fall back to the raw question
    question = state.get("enriched_query") or state["question"]
    tickers = state.get("tickers") or []

    system_prompt = f"""You are a financial research assistant.
The user is asking about: {', '.join(tickers) if tickers else 'a company not yet identified'}.

Use only the tools that are actually needed to answer the question. For example:
- A question only about recent news needs only get_news.
- A question only about price or financial metrics needs only get_market_data.
- A request for a full analysis needs all three tools.
- A comparison question needs each tool called once per company being compared.

Once you have enough information, answer the question directly and naturally."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    max_turns = 6
    turns = 0
    done = False

    all_chunks = {}
    all_market = {}
    all_news   = {}

    while not done and turns < max_turns:
        turns += 1
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
        )
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls":

            gprint(f"  [research_loop] Turn {turns}")
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results, new_chunks, new_market, new_news = handle_tool_calls(tool_calls)
            all_chunks.update(new_chunks)
            all_market.update(new_market)
            all_news.update(new_news)

            messages.append(message)
            messages.extend(results)
        else:
            done = True


    gprint(f"  [research_loop] Completed in {turns} turn(s)")

    return {
        "chunks":      all_chunks,
        "market_data": all_market,
        "news":        all_news,
    }