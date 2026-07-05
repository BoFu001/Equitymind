"""
src/agent/formatters.py

Shared formatters for converting structured data into LLM-readable strings.
Used by research_loop.py and nodes.py to ensure consistent data presentation.
"""


def format_market_data(data: dict, ticker: str) -> str:
    return f"""
{ticker} — {data.get('company_name')}
  Price: ${data.get('current_price')} | Market Cap: {data.get('market_cap')}
  P/E: {data.get('pe_ratio')} | Forward P/E: {data.get('forward_pe')}
  Revenue: {data.get('revenue')} | Profit Margin: {data.get('profit_margin')}
  EPS (TTM): {data.get('eps_trailing')} | EPS (Fwd): {data.get('eps_forward')}
  52w High: {data.get('52w_high')} | 52w Low: {data.get('52w_low')}
  Dividend Yield: {data.get('dividend_yield')}
  Analyst Target: ${data.get('target_mean')} (Low: ${data.get('target_low')} / High: ${data.get('target_high')}) | Recommendation: {data.get('recommendation')}
  Analyst Count: {data.get('analyst_count')} | Recommendation Mean: {data.get('recommendation_mean')} (1=Strong Buy, 3=Hold, 5=Strong Sell)
  RSI: {data.get('rsi')} | MACD: {data.get('macd')}
  SMA50: {data.get('sma_50')} | SMA200: {data.get('sma_200')}
  Sector: {data.get('sector')} | Industry: {data.get('industry')}
"""


def format_news(articles: list, ticker: str) -> str:
    news_context = f"\n{ticker} News:\n"
    for article in articles:
        news_context += f"[{article.get('sentiment','').upper()}] ({article.get('score',0):.2f}) {article.get('title','')}\n"
        news_context += f"  Summary: {article.get('summary','')}\n"
        news_context += f"  URL: {article.get('url','')}\n"
        news_context += f"  Published: {article.get('published','')}\n"
    return news_context


def format_sec_chunks(chunks: list, ticker: str) -> str:
    sec_context = f"\n{ticker} SEC Filing:\n"
    for i, chunk in enumerate(chunks):
        source = f"{ticker}_{chunk['chunk']['filing_type']}_{chunk['chunk']['section']}_{chunk['chunk']['filing_date']}"
        sec_context += f"[Source {i+1}: {source} | Score: {chunk['score']:.2f}]\n"
        sec_context += chunk["chunk"]["text"] + "\n"
    return sec_context


def format_conversation_context(messages: list, limit: int, max_chars: int = None) -> str:
    conversation_context = ""
    for msg in messages[-limit:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if max_chars:
            content = content[:max_chars]
        conversation_context += f"{role.upper()}: {content}\n"
    return conversation_context