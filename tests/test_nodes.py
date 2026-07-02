import pytest
from unittest.mock import patch, MagicMock
from src.agent.state import AgentState
from src.agent.nodes import (
    classify_top_intent,
    classify_sub_intent,
    explain_concept,
    extract_parameters,
    handle_out_of_scope,
    handle_greeting,
    discovery_suggest,
    generate_report,
    handle_no_ticker,
    handle_clarification,
)
from src.agent.research_loop import research_loop
import json

@pytest.fixture(autouse=True)
def mock_stream_writer():
    with patch('src.agent.nodes.get_stream_writer') as mock:
        mock.return_value = MagicMock()
        yield


@pytest.fixture(autouse=True)
def mock_research_loop_writer():
    with patch('src.agent.research_loop.get_stream_writer') as mock:
        mock.return_value = MagicMock()
        yield

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_state(**kwargs) -> AgentState:
    """Create a minimal AgentState for testing."""
    defaults = {
        "question": "What are Apple's biggest risks?",
        "messages": [],
        "session_memory": None,
        "top_intent": None,
        "sub_intent": None,
        "tickers": [], 
        "year": None,
        "chunks": None,
        "market_data": None,
        "news": None,
        "answer": None,
    }
    defaults.update(kwargs)
    return defaults


# ─────────────────────────────────────────────
# Node: Top Intent Classification (Layer 1)
# ─────────────────────────────────────────────

def test_classify_top_intent_out_of_scope():
    state = make_state(question="I want to become rich")
    result = classify_top_intent(state)
    assert result["top_intent"] == "OUT_OF_SCOPE"

def test_classify_top_intent_greeting():
    state = make_state(question="Hello, what can you do?")
    result = classify_top_intent(state)
    assert result["top_intent"] == "GREETING"

def test_classify_top_intent_general_knowledge():
    state = make_state(question="What is a stock?")
    result = classify_top_intent(state)
    assert result["top_intent"] == "GENERAL_KNOWLEDGE"

def test_classify_top_intent_task_specific_stock():
    state = make_state(question="What are Apple's biggest risks?")
    result = classify_top_intent(state)
    assert result["top_intent"] == "TASK"

def test_classify_top_intent_task_discovery():
    state = make_state(question="Find me a low risk stock")
    result = classify_top_intent(state)
    assert result["top_intent"] == "TASK"


# ─────────────────────────────────────────────
# Node: Sub Intent Classification (Layer 2)
# ─────────────────────────────────────────────

def test_classify_sub_intent_specific_stock():
    state = make_state(question="What are Apple's biggest risks?")
    result = classify_sub_intent(state)
    assert result["sub_intent"] == "SPECIFIC_STOCK"

def test_classify_sub_intent_comparison():
    state = make_state(question="Compare Apple and Microsoft")
    result = classify_sub_intent(state)
    assert result["sub_intent"] == "COMPARISON"

def test_classify_sub_intent_discovery():
    state = make_state(question="Find me a low risk stock")
    result = classify_sub_intent(state)
    assert result["sub_intent"] == "DISCOVERY"

def test_classify_sub_intent_stock_market():
    state = make_state(question="Tell me about the stock market")
    result = classify_sub_intent(state)
    assert result["sub_intent"] == "DISCOVERY"

def test_classify_sub_intent_vague_sector():
    state = make_state(question="Analyse a tech company")
    result = classify_sub_intent(state)
    assert result["sub_intent"] == "DISCOVERY"

def test_classify_sub_intent_clarification():
    state = make_state(question="Find me a good stock")
    result = classify_sub_intent(state)
    assert result["sub_intent"] == "CLARIFICATION"

# ─────────────────────────────────────────────
# Node: Explain Concept
# ─────────────────────────────────────────────

def test_explain_concept():
    state = make_state(question="What is a stock?")
    result = explain_concept(state)
    assert "answer" in result
    assert len(result["answer"]) > 0

# ─────────────────────────────────────────────
# Node: Extract Parameters
# ─────────────────────────────────────────────

def test_extract_parameters_aapl():
    state = make_state(question="What are Apple's biggest risks?")
    result = extract_parameters(state)
    assert result["tickers"] == ["AAPL"]
    assert result["year"] is None

def test_extract_parameters_with_year():
    state = make_state(question="What were Microsoft's risks in 2024?")
    result = extract_parameters(state)
    assert result["tickers"] == ["MSFT"]
    assert result["year"] == "2024"

def test_extract_parameters_no_ticker():
    state = make_state(question="Find me a low risk stock")
    result = extract_parameters(state)
    assert result["tickers"] == []



def test_extract_parameters_multiple_tickers():
    state = make_state(question="Compare Apple and Microsoft")
    result = extract_parameters(state)
    assert result["tickers"] == ["AAPL", "MSFT"]
    



def test_extract_parameters_amazon():
    state = make_state(question="Analyse Amazon")
    result = extract_parameters(state)
    assert result["tickers"] == ["AMZN"]

def test_extract_parameters_alibaba():
    state = make_state(question="Analyse Alibaba")
    result = extract_parameters(state)
    assert result["tickers"] == ["BABA"]

def test_extract_parameters_tencent():
    state = make_state(question="Analyse Tencent")
    result = extract_parameters(state)
    assert result["tickers"][0] in ["0700.HK", "TCEHY"]

# ─────────────────────────────────────────────
# Node: Intent Classification + Extract Parameters for Edge case tests: valid intent but no ticker
# ─────────────────────────────────────────────

def test_no_ticker_edge_cases():
    """
    Tests questions that may be classified as SPECIFIC_STOCK or COMPARISON
    but have no extractable ticker. All should route to DISCOVERY or have no ticker.
    """
    edge_cases = [
        "Analyse a tech company",
        "Tell me about a good stock",
        "What about that AI company?",
        "Compare two tech companies",
        "Compare them",
        "Which is better, A or B?",
    ]

    for question in edge_cases:
        # Step 1 — classify sub intent
        classify_state = make_state(question=question)
        sub_result = classify_sub_intent(classify_state)
        sub_intent = sub_result["sub_intent"]

        # Step 2 — extract
        extract_state = make_state(question=question, sub_intent=sub_intent)
        extract_result = extract_parameters(extract_state)
        tickers = extract_result.get("tickers", [])

        print(f"\nQ: '{question}'")
        print(f"  Sub-intent: {sub_intent}")

        # Assert: if COMPARISON or SPECIFIC_STOCK — no ticker should be found
        # These vague questions should either route to DISCOVERY or have no ticker
        if sub_intent == "COMPARISON":
            assert not tickers, f"Expected no tickers for vague COMPARISON: '{question}' but got {tickers}"
        if sub_intent == "SPECIFIC_STOCK":
            assert not tickers, f"Expected no tickers for vague SPECIFIC_STOCK: '{question}' but got {tickers}"



# ─────────────────────────────────────────────
# Node: Out of Scope
# ─────────────────────────────────────────────

def test_handle_out_of_scope():
    state = make_state(question="I want to be rich")
    result = handle_out_of_scope(state)
    assert "answer" in result
    assert len(result["answer"]) > 0


# ─────────────────────────────────────────────
# Node: Greeting
# ─────────────────────────────────────────────

def test_handle_greeting():
    state = make_state(question="Hello")
    result = handle_greeting(state)
    assert "answer" in result
    assert len(result["answer"]) > 0






# ─────────────────────────────────────────────
# Node: Discovery 
# ─────────────────────────────────────────────


def test_discovery_suggest():
    state = make_state(question="Find me a low risk stock")
    result = discovery_suggest(state)
    assert "tickers" in result
    assert len(result["tickers"]) == 5

# ─────────────────────────────────────────────
# Node: Report
# ─────────────────────────────────────────────

def test_generate_report():
    state = make_state(
        question="Analyse Apple",
        tickers=["AAPL"],
        chunks={"AAPL": []},
        market_data={"AAPL": {}},
        news={"AAPL": []},
    )
    result = generate_report(state)
    assert "answer" in result
    assert len(result["answer"]) > 0

# ─────────────────────────────────────────────
# Node: No Ticker
# ─────────────────────────────────────────────

def test_handle_no_ticker_specific_stock():
    state = make_state(question="Analyse XYZ Corporation", sub_intent="SPECIFIC_STOCK")
    result = handle_no_ticker(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "company" in result["answer"].lower()

def test_handle_no_ticker_comparison():
    state = make_state(question="Compare them", sub_intent="COMPARISON")
    result = handle_no_ticker(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "compare" in result["answer"].lower()


# ─────────────────────────────────────────────
# Node: Clarification
# ─────────────────────────────────────────────

def test_handle_clarification_asks_question():
    """With no criteria — should ask a question, complete should be False."""
    state = make_state(
        question="Find me a good stock",
        messages=[]
    )
    result = handle_clarification(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert result.get("clarification_complete") == False

def test_handle_clarification_complete_with_enough_criteria():
    """With enough criteria in history — should return clarification_complete=True."""
    state = make_state(
        question="long term",
        messages=[
            {"role": "user",      "content": "Find me a good stock"},
            {"role": "assistant", "content": "Which sector interests you?"},
            {"role": "user",      "content": "Technology"},
            {"role": "assistant", "content": "What's your risk tolerance?"},
            {"role": "user",      "content": "Medium risk"},
            {"role": "assistant", "content": "Time horizon — short or long term?"},
        ]
    )
    result = handle_clarification(state)
    assert result.get("clarification_complete") == True
    assert "enriched_query" in result
    assert len(result["enriched_query"]) > 0


# ─────────────────────────────────────────────
# Node: Research Loop
# ─────────────────────────────────────────────


def make_tool_call(name, arguments: dict, call_id="call_001"):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    tc.id = call_id
    return tc


def make_llm_response(finish_reason, tool_calls=None, content=""):
    msg = MagicMock()
    msg.tool_calls = tool_calls
    msg.content = content
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@patch('src.agent.research_loop.get_stock_data', return_value={"current_price": 200.0, "company_name": "Apple"})
@patch('src.agent.research_loop.get_news_and_sentiment', return_value=[])
@patch('src.agent.research_loop.retrieve', return_value=[])
@patch('src.agent.research_loop.fetch_embed_store_retrieve', return_value=[])
@patch('src.agent.research_loop.client')
def test_research_loop_market_only(mock_client, *_):
    """When user asks a market metric question, only get_market_data should be called."""
    tool_call = make_tool_call("get_market_data", {"ticker": "AAPL"})
    mock_client.chat.completions.create.side_effect = [
        make_llm_response("tool_calls", tool_calls=[tool_call]),
        make_llm_response("stop", content="Apple's P/E is 35."),
    ]
    state = make_state(question="What is Apple's P/E ratio?", tickers=["AAPL"])
    result = research_loop(state)
    assert "market_data" in result
    assert "AAPL" in result["market_data"]
    assert result["news"] == {}
    assert result["chunks"] == {}


@patch('src.agent.research_loop.get_stock_data', return_value={"current_price": 200.0, "company_name": "Apple"})
@patch('src.agent.research_loop.get_news_and_sentiment', return_value=[{"title": "Apple news", "sentiment": "positive", "score": 0.9, "summary": "", "url": "", "published": ""}])
@patch('src.agent.research_loop.retrieve', return_value=[{"chunk": {"text": "Risk factors...", "filing_type": "10-K", "section": "1A", "filing_date": "2024"}, "score": 0.9}])
@patch('src.agent.research_loop.fetch_embed_store_retrieve', return_value=[])
@patch('src.agent.research_loop.client')
def test_research_loop_all_tools(mock_client, *_):
    """When user asks for a full analysis, all three tools should be called."""
    tool_calls = [
        make_tool_call("get_market_data", {"ticker": "AAPL"}, "call_001"),
        make_tool_call("get_news",        {"ticker": "AAPL"}, "call_002"),
        make_tool_call("ensure_sec_data", {"ticker": "AAPL", "question": "Analyse Apple"}, "call_003"),
    ]
    mock_client.chat.completions.create.side_effect = [
        make_llm_response("tool_calls", tool_calls=tool_calls),
        make_llm_response("stop", content="Apple analysis complete."),
    ]
    state = make_state(question="Analyse Apple", tickers=["AAPL"])
    result = research_loop(state)
    assert "AAPL" in result["market_data"]
    assert "AAPL" in result["news"]
    assert "AAPL" in result["chunks"]


@patch('src.agent.research_loop.get_stock_data', return_value=None)
@patch('src.agent.research_loop.get_news_and_sentiment', return_value=[])
@patch('src.agent.research_loop.retrieve', return_value=[])
@patch('src.agent.research_loop.fetch_embed_store_retrieve', return_value=[])
@patch('src.agent.research_loop.client')
def test_research_loop_returns_state_fields(mock_client, *_):
    """research_loop must always return chunks, market_data, news regardless of tools called."""
    mock_client.chat.completions.create.return_value = make_llm_response("stop", content="Done.")
    state = make_state(question="Hello", tickers=["AAPL"])
    result = research_loop(state)
    assert "chunks" in result
    assert "market_data" in result
    assert "news" in result