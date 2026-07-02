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

@pytest.fixture(autouse=True)
def mock_stream_writer():
    with patch('src.agent.nodes.get_stream_writer') as mock:
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