"""
src/agent/capabilities.py

What EquityMind can be asked, in the user's words.

Three nodes tell the user what this system does — the greeting, the
out-of-scope refusal, and the no-ticker fallback — and each had drifted
into naming features rather than showing questions. "News and sentiment
analysis" says a capability exists without saying how to reach it, and
a user who reads it still has to guess the wording.

Every line below is a question that can be copied straight into the
box. The screening examples carry the most weight: they are the part
of the system that a feature list cannot convey, since "find low risk
stocks" gives no hint that an industry, several criteria and a count
can all go in one sentence.

Kept here rather than in each node so the three stay in step.
"""

CAPABILITIES = """**Analyse one company**
"Analyse Apple" · "What are NVIDIA's risks?" · "Is Tesla expensive?"

**Compare two**
"Apple or Microsoft?" · "Compare AMD and Intel on quality"

**Screen the universe**
"Cheapest large-cap tech stocks with high quality"
"The five lowest-risk stocks"
"Semiconductor companies that are undervalued"

**Financial statement figures**
"Apple's FY2024 revenue" · "How has Microsoft's net income grown?"

**News**
"Any negative news on Apple recently?"

Ask in whatever words come naturally — I'll ask back if I need to know
what you mean by "best" or "good"."""
