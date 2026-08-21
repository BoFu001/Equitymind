"""
src/agent/nodes/prepare_discovery.py

Node: Prepare Discovery

Stands in front of Discovery and decides whether the question, as asked,
is one Discovery can execute. If it is, the parsed query is passed
through untouched. If it is not, the user is asked a question that
points at whatever could not be resolved, and the turn ends.

Why a gate rather than letting Discovery try
--------------------------------------------
Discovery narrows a candidate pool by ranking it on fields it has data
for. With no field to rank on, every stage of that chain is a no-op:
group_fields_by_priority returns no stages, the loop body never runs,
and the pool is returned exactly as it arrived — the whole 250-ticker
universe when no industry was named either. Measured on 2026-08-20,
"Which US companies have a promising future?" returned all 250, and
"Find me tech stocks" returned all 111 companies tagged technology.
Neither raises; both look like answers.

The judgement is delegated to the parser below, not made separately.
extract_discovery_query only records fields it can name (see RankField's
docstring — an unmappable word is left out rather than guessed at), so an
empty fields list already means "nothing the user asked for could be
expressed as a ranking". Asking an LLM here whether the question is
answerable would be a second opinion that can disagree with the one
that actually runs.

An industry with no fields is still blocked. It executes, but returning
a whole sector unranked is not an answer to a question that asked for
one, and the pool is then large enough to be a problem for every node
downstream.
"""

import re
import time
from typing import get_args

from openai import OpenAI

from config import OPENAI_API_KEY, APP_NAME, LLM_MODEL, CONVERSATION_HISTORY_LIMIT
from langgraph.config import get_stream_writer
from core.context import token_queue_var
from src.agent.state import AgentState
from src.agent.discovery_types import VALID_FIELD_NAMES, DiscoveryQuery
from src.agent.nodes_notifications import NODE_PROGRESS
from src.agent.formatters.conversation_formatter import format_conversation_context
from colors import gprint

client = OpenAI(api_key=OPENAI_API_KEY)

FIELD_NAMES = list(get_args(VALID_FIELD_NAMES))


def _build_enriched_question(question: str, messages: list) -> str:
    """Folds a multi-turn exchange into one standalone question.

    Needed because the parser sees one string. After a clarifying
    question the user's reply is often a fragment — "estimates are what
    I meant", "yes, cheapest" — which carries no industry and no
    ranking on its own.
    """
    ENRICH_PROMPT = """You are {app_name}, an AI investment research assistant.

Below is a conversation. Rewrite the user's request as ONE self-contained
question, folding in anything they established in earlier turns.

Change nothing about what they are asking for. Do not add a criterion
they did not give, do not resolve a vague word into a specific one, and
do not drop a requirement because it seems hard to answer. If they said
"the most awesome company" and never said what awesome means, the
rewritten question still says "the most awesome company" — deciding
that it means momentum is not your job here.

CONVERSATION (the last line is their newest message):
{context}

Reply with ONLY the rewritten question."""

    context = format_conversation_context(messages, CONVERSATION_HISTORY_LIMIT)
    full_context = context + f"USER: {question}\n"

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": ENRICH_PROMPT.format(app_name=APP_NAME, context=full_context),
        }],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def extract_discovery_query(question: str) -> DiscoveryQuery:
    """
    Turn a user's free-text question into a structured DiscoveryQuery.
    LLM only transcribes what the sentence says — no ranking logic, no
    judgment. If parsing fails, returns an empty query rather than raising.
    """
    
    extraction_prompt = f'''You are parsing a user\'s stock question into structured
data. Do NOT answer the question or judge anything — only faithfully transcribe
what the sentence says. All logic (ranking, filtering) happens in code afterward.

USER QUESTION: {question}

Extract two things:

1. industry: the user\'s own wording for any industry/theme mentioned (e.g.
   "semiconductor stocks", "robotics companies", "healthcare"). Copy their
   phrasing verbatim, do not translate or normalize it. If no industry is
   mentioned, use null.

2. fields: a list of ranking fields the user mentioned. Each field has:
   - name: the field being ranked. MUST be exactly one of these values —
     do not invent a name, do not use spaces (use underscores), and if the
     user's question does not clearly map to any of these, leave it out of
     the fields list entirely rather than guessing:
     valuation_score, momentum_12_1_score, position_52w_score,
     risk_beta_score, risk_sharpe_score, risk_var_score, risk_drawdown_score,
     quality_score, consensus_recommendation_score, consensus_upside_score,
     consensus_trend_score, short_interest_pct, days_to_cover, market_cap,
     total_revenue, cost_of_revenue, gross_profit, research_and_development,
     selling_general_and_administration, operating_expense, operating_income,
     ebit, ebitda, pretax_income, net_income, diluted_eps, basic_eps,
     interest_expense, operating_cash_flow, capital_expenditure,
     free_cash_flow, repurchase_of_capital_stock, cash_dividends_paid,
     depreciation_amortization_depletion, total_assets, total_liabilities,
     stockholders_equity, cash_and_equivalents, long_term_debt,
     current_assets, current_liabilities, shares_outstanding,
     retained_earnings, net_ppe, accounts_receivable, inventory, total_debt

     When the user says something generic, map it to the closest specific
     field: "risk" alone -> risk_beta_score; "momentum" alone ->
     momentum_12_1_score; "consensus"/"analyst rating" alone ->
     consensus_recommendation_score.
   - order: "ascending" (smallest value first) or "descending" (largest
     value first). Decide this from the VALUE STORED IN THE FIELD, not
     from the adjective the user used — for most of these fields the two
     point the same way, but for the scores below they are opposites.

     The fourteen fields ending in _score are normalised judgements, not
     raw quantities, and every one of them is signed the same way:
     POSITIVE ALWAYS MEANS THE FAVOURABLE END, whatever the field
     measures. valuation_score is +1 when a company is cheap against its
     peers and -1 when expensive; risk_*_score is +1 when risk is LOW;
     quality_score is +1 when financials are strong; the consensus_*
     scores are +1 when analysts are positive; the momentum and
     position scores are +1 when the stock ranks at the top of the
     universe.

     So for a _score field, work out which END the user is asking for:
       - the favourable end ("cheapest", "best valuation", "safest",
         "lowest risk", "highest quality", "strongest momentum", "most
         favoured by analysts") -> descending
       - the unfavourable end ("most overvalued", "riskiest", "worst
         quality", "weakest momentum", "least favoured by analysts")
         -> ascending
     Note that "cheapest" and "safest" take DESCENDING here, even though
     they sound like they should be ascending: the field holds a score,
     and a cheap or safe company scores high.

     short_interest_pct and days_to_cover are the exception inside this
     group — despite living alongside the scores they are raw figures (a
     percentage of float, and a number of days), with no scoring applied,
     so read them literally: "most heavily shorted" -> descending, "least
     shorted" -> ascending.

     Every other field — market_cap and the financial-statement figures —
     is a raw quantity, read literally: "highest revenue" -> descending,
     "least debt" -> ascending.
     IMPORTANT — subjective overall judgements are NOT rankable fields.

     Words like "worth buying", "good", "should I buy", "worth investing",
     "promising", "attractive" express an overall judgement about a company,
     not a single measurable dimension. They MUST NOT be mapped to any field.

     consensus_recommendation_score is NOT a proxy for "worth buying" —
     it reflects analyst opinion, not the user's own criteria.

     If the question contains ONLY such a judgement and no measurable
     criterion, return an empty fields list.

     However, when the same words are paired with an explicit dimension,
     they ARE rankable and should be mapped normally:
       - "best valuation" -> valuation_score
       - "best analyst rating" -> consensus_recommendation_score
       - "attractive valuation" -> valuation_score

   - count: a number if THIS SPECIFIC field has one attached in the sentence
     (e.g. "the 10 largest" -> count=10 on the market-cap field). Otherwise null.
   - priority: an integer. Assign this by sentence structure, not by which
     fields are involved:
     - If two or more fields are joined by "and"/"but", or simply listed
       together with no containment wording ("highest net income and lowest
       valuation", "low-risk and high-momentum stocks") -> give them the
       SAME priority number. They will be averaged together with equal
       weight, no field is ever treated as more important than another by
       default.
     - If the sentence has a containment structure ("of the 10 largest
       companies, which has the lowest X" / "among X stocks, Y" / "X 里 Y")
       -> give the outer (containing) field a SMALLER priority number than
       the inner (contained) field. Lower number = executed first.
     - A sentence can have more than two priority levels if it nests more
       than once (e.g. "of the 20 largest, the 10 cheapest, the most
       analyst-favored" -> three distinct priority numbers, 1/2/3).

3. final_count: a number if the user asked for a specific count of final
   results that does NOT belong to any single field (e.g. "give me 10
   stocks with low valuation and high quality" -> the 10 applies to the
   combined result, not to valuation or quality individually -> final_count=10,
   and neither field gets its own count). Null if unspecified. Do not
   confuse this with a field-specific count in a nested sentence (e.g. "of
   the 10 largest" -> that 10 belongs to the market-cap field, not here).

EXAMPLES:

"Give me 10 stocks with low valuation and high quality"
(both wanted at their favourable end, so both descending — a cheap
company scores HIGH on valuation_score)
{{"industry": null, "fields": [
  {{"name": "valuation_score", "order": "descending", "count": null, "priority": 1}},
  {{"name": "quality_score", "order": "descending", "count": null, "priority": 1}}
], "final_count": 10}}

"Of the 10 largest companies, which has the lowest valuation?"
(market_cap is a raw figure read literally; valuation_score is not —
"lowest valuation" means cheapest, the favourable end, so descending)
{{"industry": null, "fields": [
  {{"name": "market_cap", "order": "descending", "count": 10, "priority": 1}},
  {{"name": "valuation_score", "order": "descending", "count": null, "priority": 2}}
], "final_count": null}}

"Of the 20 largest companies, the 10 with the lowest valuation, which 5 do
analysts favor most?"
{{"industry": null, "fields": [
  {{"name": "market_cap", "order": "descending", "count": 20, "priority": 1}},
  {{"name": "valuation_score", "order": "descending", "count": 10, "priority": 2}},
  {{"name": "consensus_recommendation_score", "order": "descending", "count": 5, "priority": 3}}
], "final_count": null}}

"Which healthcare companies are the most overvalued?"
(the unfavourable end of valuation_score — an expensive company scores
LOW — so ascending. A _score field is not always descending; which end
the user asked for is what decides it.)
{{"industry": "healthcare", "fields": [
  {{"name": "valuation_score", "order": "ascending", "count": null, "priority": 1}}
], "final_count": null}}

"Which healthcare companies are worth buying?"
(a judgement about the company overall, with no dimension named — the
industry is still recorded, but there is nothing to rank on, so fields
stays empty rather than being filled with the nearest plausible score)
{{"industry": "healthcare", "fields": [], "final_count": null}}

"Recommend some low-risk, high-momentum semiconductor stocks"
(low risk is the favourable end of risk_beta_score, so descending —
the same direction as high momentum, despite the opposite adjective)
{{"industry": "semiconductor", "fields": [
  {{"name": "risk_beta_score", "order": "descending", "count": null, "priority": 1}},
  {{"name": "momentum_12_1_score", "order": "descending", "count": null, "priority": 1}}
], "final_count": null}}

Reply with ONLY valid JSON matching this shape, no markdown, no commentary:
{{"industry": "...", "fields": [{{"name": "...", "order": "...", "count": null, "priority": 1}}], "final_count": null}}'''

    extraction_response = client.responses.parse(
        model=LLM_MODEL,
        input=[{"role": "user", "content": extraction_prompt}],
        temperature=0,
        text_format=DiscoveryQuery,
    )
    try:
        query = extraction_response.output_parsed
    except Exception as e:
        gprint(f"  [extract_discovery_query] Could not parse query: {e}")
        query = DiscoveryQuery(industry=None, fields=[], final_count=None)

    # The model occasionally writes the string "null" where the schema
    # asks for JSON null. The two are indistinguishable to a reader and
    # opposite to the code: None means "no industry named, search the
    # whole universe", while "null" is looked up as a tag, matches
    # nothing, and hands Discovery an empty candidate pool. Nothing
    # raises — the user is simply told no companies were found. Fixing
    # this in the prompt would be one more instruction to be followed
    # most of the time; the shape is fixed enough to settle in code.
    if isinstance(query.industry, str) and query.industry.strip().lower() in {"null", "none", ""}:
        query.industry = None

    print(f"  [extract_discovery_query] parsed:\n\n{query.model_dump_json(indent=2)}")
    return query


def _ask_for_a_field(question: str) -> str:
    """Writes the clarifying question. Only called once the parse has
    already come back empty, so the cost falls on the questions that
    could not be answered rather than on every question."""
    CLARIFY_PROMPT = """You are {app_name}, an AI investment research assistant.

A user asked this, and nothing in it could be turned into a ranking we
can actually run:

    {question}

You can rank companies on any of these stored fields:

{fields}

Ask the user ONE short question that turns their wording into something
from that list. Quote the word of theirs that was the problem, and offer
two or three concrete alternatives drawn from the fields above — a user
who says "the most awesome company" should be asked whether they mean
the strongest recent price momentum, the largest company, or the
cheapest on valuation.

Write for someone who does not know these field names: say "strongest
recent price momentum", never "momentum_12_1_score". Ask about what
they want to rank by, not which direction to sort — sorting has a
sensible default and asking about it wastes the user's turn.

Reply with ONLY the question, in a natural conversational voice."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": CLARIFY_PROMPT.format(
                app_name=APP_NAME,
                question=question,
                fields="\n".join(f"- {f}" for f in FIELD_NAMES),
            ),
        }],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def _stream(text: str) -> None:
    """Sends the clarifying question to the user a word at a time, the
    same way handle_clarification does — this node's output is the whole
    reply for this turn, not a progress line."""
    queue = token_queue_var.get()
    if queue:
        for word in re.findall(r"\S+|\s+", text):
            queue.put_nowait(word)
            time.sleep(0.03)


def prepare_discovery(state: AgentState) -> dict:
    writer = get_stream_writer()
    writer({"type": "progress", "node": "clarification", "message": NODE_PROGRESS["clarification"]})

    question = state["question"]
    messages = state.get("messages") or []

    enriched_question = _build_enriched_question(question, messages)
    gprint(f"  [prepare_discovery] enriched question: {enriched_question}")

    query = extract_discovery_query(enriched_question)

    if query.fields:
        gprint(f"  [prepare_discovery] executable — {len(query.fields)} field(s), industry={query.industry!r}")
        return {
            "clarification_complete": True,
            "enriched_query":         enriched_question,
            "discovery_query":        query.model_dump(),
        }

    else:
        writer({"type": "sub_progress", "node": "clarification", "message": NODE_PROGRESS["clarification_sub"]})

        clarifying_question = _ask_for_a_field(enriched_question)
        _stream(clarifying_question)

        gprint(f"  [prepare_discovery] no rankable field — asking: {clarifying_question}")
        return {
            "clarification_complete": False,
            "answer":                 clarifying_question,
        }


if __name__ == "__main__":
    # prepare_discovery calls get_stream_writer, and _stream reads
    # token_queue_var; both exist only inside a LangGraph run. Rebinding
    # them here is enough, since each is resolved by global lookup at
    # call time — which means the node itself gets exercised, not just
    # the helpers beneath it.
    def _fake_get_stream_writer():
        def writer(event):
            pass
        return writer

    class _FakeQueue:
        def put_nowait(self, token):
            print(token, end="", flush=True)

    class _FakeVar:
        def get(self):
            return _FakeQueue()

    globals()["get_stream_writer"] = _fake_get_stream_writer
    globals()["token_queue_var"] = _FakeVar()

    # Carrying messages forward is what makes a multi-turn exchange
    # testable: a blocked turn is appended, so the next answer reaches
    # _build_enriched_question with something to fold in.
    # Carrying messages forward is what a multi-turn test needs: a blocked
    # turn leaves its clarifying question behind for the next answer to
    # fold into. It is exactly wrong for a repeated single-turn test,
    # though — the same question asked twice is no longer the same input
    # once a clarification sits in front of it. --isolate clears the
    # history each turn, so one turn can be measured on its own.
    import sys as _sys
    isolate = "--isolate" in _sys.argv

    messages = []

    print("=== prepare_discovery ===")
    print("blank line to quit")

    while True:
        q = input("\nEnter question: ").strip()
        if not q:
            break

        if isolate:
            messages = []

        result = prepare_discovery({"question": q, "messages": messages})

        print("\n--- RESULT ---")
        if result["clarification_complete"]:
            print("EXECUTABLE")
            print(f"  enriched_query:  {result['enriched_query']}")
            print(f"  discovery_query: {result['discovery_query']}")
            messages = []
        else:
            print("BLOCKED")
            messages = messages + [
                {"role": "user", "content": q},
                {"role": "assistant", "content": result["answer"]},
            ]
