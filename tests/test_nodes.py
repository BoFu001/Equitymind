import pytest
from unittest.mock import patch, MagicMock
from src.agent.state import AgentState
from src.agent.nodes import (
    classify_intent,
    extract_parameters,
    ensure_sec_data,
    get_market_data,
    get_news,
    handle_out_of_scope,
    handle_greeting,
    discovery_suggest,
    # discovery_report,
    # comparison_report,
    # specific_report,
    simple_report,
    handle_no_ticker,
    handle_follow_up,
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
        "intent": None,
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
# Node: Intent Classification
# ─────────────────────────────────────────────

def test_classify_intent_specific_stock():
    state = make_state(question="What are Apple's biggest risks?")
    result = classify_intent(state)
    assert result["intent"] == "SPECIFIC_STOCK"

def test_classify_intent_out_of_scope():
    state = make_state(question="I want to become rich")
    result = classify_intent(state)
    assert result["intent"] == "OUT_OF_SCOPE"

def test_classify_intent_greeting():
    state = make_state(question="Hello, what can you do?")
    result = classify_intent(state)
    assert result["intent"] == "GREETING"

def test_classify_intent_comparison():
    state = make_state(question="Compare Apple and Microsoft")
    result = classify_intent(state)
    assert result["intent"] == "COMPARISON"

def test_classify_intent_discovery():
    state = make_state(question="Find me a low risk stock")
    result = classify_intent(state)
    assert result["intent"] == "DISCOVERY"

def test_classify_intent_stock_market():
    state = make_state(question="Tell me about the stock market")
    result = classify_intent(state)
    assert result["intent"] == "DISCOVERY"

def test_classify_intent_vague_sector():
    state = make_state(question="Analyse a tech company")
    result = classify_intent(state)

    assert result["intent"] == "DISCOVERY"
def test_classify_intent_clarification():
    state = make_state(question="Find me a good stock")
    result = classify_intent(state)
    assert result["intent"] == "CLARIFICATION"

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
        # Step 1 — classify
        classify_state = make_state(question=question)
        classify_result = classify_intent(classify_state)
        intent = classify_result["intent"]

        # Step 2 — extract
        extract_state = make_state(question=question, intent=intent)
        extract_result = extract_parameters(extract_state)
        tickers = extract_result.get("tickers", [])

        print(f"\nQ: '{question}'")
        print(f"  Intent: {intent}")

        # Assert: if COMPARISON or SPECIFIC_STOCK — no ticker should be found
        # These vague questions should either route to DISCOVERY or have no ticker
        if intent == "COMPARISON":
            assert not tickers, f"Expected no tickers for vague COMPARISON: '{question}' but got {tickers}"
        if intent == "SPECIFIC_STOCK":
            assert not tickers, f"Expected no tickers for vague SPECIFIC_STOCK: '{question}' but got {tickers}"

# ─────────────────────────────────────────────
# Node: Retrieve SEC Data
# ─────────────────────────────────────────────

def test_retrieve_sec_data_existing_ticker():
    state = make_state(
        question="What are Apple's biggest risks?",
        tickers=["AAPL"]
    )
    result = ensure_sec_data(state)
    assert "chunks" in result
    assert "AAPL" in result["chunks"]
    assert len(result["chunks"]["AAPL"]) > 0
    assert "chunk" in result["chunks"]["AAPL"][0]
    assert "score" in result["chunks"]["AAPL"][0]
    assert "text" in result["chunks"]["AAPL"][0]["chunk"]

def test_retrieve_sec_data_unknown_ticker():
    state = make_state(
        question="What are the risks?",
        tickers=["FAKE123"]
    )
    result = ensure_sec_data(state)
    assert "chunks" in result
    assert isinstance(result["chunks"], dict)
    assert result["chunks"].get("FAKE123") == []


# ─────────────────────────────────────────────
# Node: Market Data
# ─────────────────────────────────────────────

def test_get_market_data_aapl():
    state = make_state(tickers=["AAPL"])
    result = get_market_data(state)
    assert result["market_data"] is not None
    assert "AAPL" in result["market_data"]
    assert result["market_data"]["AAPL"]["current_price"] is not None
    assert result["market_data"]["AAPL"]["pe_ratio"] is not None

def test_get_market_data_no_ticker():
    state = make_state(tickers=[])
    result = get_market_data(state)
    assert result["market_data"] == {}


# ─────────────────────────────────────────────
# Node: News and Sentiment
# ─────────────────────────────────────────────

def test_get_news_aapl():
    state = make_state(tickers=["AAPL"])
    result = get_news(state)
    assert "news" in result
    assert "AAPL" in result["news"]
    assert isinstance(result["news"]["AAPL"], list)
    if len(result["news"]["AAPL"]) > 0:
        article = result["news"]["AAPL"][0]
        assert "title" in article
        assert "sentiment" in article
        assert "score" in article
        assert "url" in article

def test_get_news_no_ticker():
    state = make_state(tickers=[])
    result = get_news(state)
    assert result["news"] == {}


# ─────────────────────────────────────────────
# Node: Out of Scope
# ─────────────────────────────────────────────

def test_handle_out_of_scope():
    state = make_state(question="I want to be rich")
    result = handle_out_of_scope(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "messages" in result
    assert len(result["messages"]) == 2


# ─────────────────────────────────────────────
# Node: Greeting
# ─────────────────────────────────────────────

def test_handle_greeting():
    state = make_state(question="Hello")
    result = handle_greeting(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "messages" in result
    assert len(result["messages"]) == 2






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


# def test_specific_report():
#     state = make_state(
#         question="Analyse Apple",
#         tickers=["AAPL"],
#         chunks={"AAPL": []},
#         market_data={"AAPL": {}},
#         news={"AAPL": []},
#     )
#     result = specific_report(state)
#     assert "answer" in result
#     assert len(result["answer"]) > 0
#     assert "messages" in result

# def test_comparison_report():
#     state = make_state(
#         question="Compare Apple and Microsoft",
#         tickers=["AAPL", "MSFT"],
#         chunks={"AAPL": [], "MSFT": []},
#         market_data={"AAPL": {}, "MSFT": {}},
#         news={"AAPL": [], "MSFT": []},
#     )
#     result = comparison_report(state)
#     assert "answer" in result
#     assert len(result["answer"]) > 0
#     assert "messages" in result

# def test_discovery_report():
#     state = make_state(
#         question="Find me a low risk stock",
#         tickers=["JNJ", "PG", "KO"],
#         chunks={"JNJ": [], "PG": [], "KO": []},
#         market_data={"JNJ": {}, "PG": {}, "KO": {}},
#         news={"JNJ": [], "PG": [], "KO": []},
#     )
#     result = discovery_report(state)
#     assert "answer" in result
#     assert len(result["answer"]) > 0
#     assert "messages" in result

def test_simple_report():
    state = make_state(
        question="Analyse Apple",
        tickers=["AAPL"],
        chunks={"AAPL": []},
        market_data={"AAPL": {}},
        news={"AAPL": []},
    )
    result = simple_report(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "messages" in result

# ─────────────────────────────────────────────
# Node: No Ticker
# ─────────────────────────────────────────────

def test_handle_no_ticker_specific_stock():
    state = make_state(question="Analyse XYZ Corporation", intent="SPECIFIC_STOCK")
    result = handle_no_ticker(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "company" in result["answer"].lower()
    assert "messages" in result
    assert len(result["messages"]) == 2

def test_handle_no_ticker_comparison():
    state = make_state(question="Compare them", intent="COMPARISON")
    result = handle_no_ticker(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "compare" in result["answer"].lower()
    assert "messages" in result
    assert len(result["messages"]) == 2


# ─────────────────────────────────────────────
# Node: Handle Follow Up
# ─────────────────────────────────────────────

def test_handle_follow_up():
    state = make_state(
        question="What is their P/E ratio?",
        messages=[
            {"role": "user",      "content": "Analyse NVDA"},
            {"role": "assistant", "content": "NVIDIA P/E ratio is 32.58, revenue $253.49B..."},
        ]
    )
    result = handle_follow_up(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "messages" in result
    assert len(result["messages"]) == 4  # 2 history + 2 new



# ─────────────────────────────────────────────
# Node: Clarification
# ─────────────────────────────────────────────

def test_handle_clarification_asks_question():
    """With no criteria — should ask a question, not return READY."""
    state = make_state(
        question="Find me a good stock",
        messages=[]
    )
    result = handle_clarification(state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert not result["answer"].strip().startswith("READY:")
    assert "messages" in result
    assert len(result["messages"]) == 2

def test_handle_clarification_ready_with_enough_criteria():
    """With enough criteria in history — should return clarification_ready=True."""
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
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert result.get("clarification_ready") == True
    assert "question" in result
    assert len(result["question"]) > 0