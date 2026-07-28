"""
src/agent/formatters/snapshot_formatter.py

Formats snapshot_reader.get_stock_snapshot() results for the LLM —
pure display information, not tied to any signal's computation (see
snapshot_reader.py's module docstring: this data is shared by the
report's display layer, not owned by valuation_signal() or
consensus_signal(), both of which now fetch their own inputs
independently via valuation_reader.py / consensus_reader.py).

Split out from the former single-file formatters.py on 2026-07-27,
alongside consensus_formatter.py and valuation_formatter.py.
"""


def format_stock_snapshot(data: dict, ticker: str) -> str:
    return f"""
{ticker} — {data.get('company_name')}
  Price: ${data.get('current_price')} | Market Cap: {data.get('market_cap')}
  Forward P/E: {data.get('forward_pe')}
  Revenue: {data.get('revenue')} | Profit Margin: {data.get('profit_margin')}
  EPS (TTM): {data.get('eps_trailing')} | EPS (Fwd): {data.get('eps_forward')}
  52w High: {data.get('52w_high')} | 52w Low: {data.get('52w_low')}
  Dividend Yield: {data.get('dividend_yield')}
  Sector: {data.get('sector')} | Industry: {data.get('industry')}
"""
