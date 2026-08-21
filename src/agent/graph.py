from colors import yprint

from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes.contextualize_question import contextualize_question
from src.agent.nodes.classify_top_intent import classify_top_intent
from src.agent.nodes.classify_sub_intent import classify_sub_intent
from src.agent.nodes.explain_concept import explain_concept
from src.agent.nodes.extract_parameters import extract_parameters
from src.agent.nodes.determine_data_scope import determine_data_scope
from src.agent.nodes.handle_out_of_scope import handle_out_of_scope
from src.agent.nodes.handle_greeting import handle_greeting
from src.agent.nodes.handle_no_ticker import handle_no_ticker
from src.agent.nodes.discovery_execution import discovery_execution
from src.agent.nodes.update_session_memory import update_session_memory
from src.agent.nodes.discovery_preparation import discovery_preparation
from src.agent.nodes.generate_report import generate_report
from src.agent.nodes.fetch_data import fetch_data
from src.agent.nodes.quant_engine import quant_engine

# ─────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────

def route_after_top_intent(state: AgentState) -> str:
    """Routes after Layer 1 — coarse classification."""
    top_intent = state.get("top_intent", "")
    yprint(f"  [route_after_top_intent] top_intent={top_intent}")

    if top_intent == "OUT_OF_SCOPE":
        return "out_of_scope"
    elif top_intent == "GREETING":
        return "greeting"
    elif top_intent == "GENERAL_KNOWLEDGE":
        return "explain_concept"
    else:
        return "classify_sub_intent"
    

def route_after_sub_intent(state: AgentState) -> str:
    """Routes after Layer 2 — fine classification. Only reached when top_intent == TASK."""
    sub_intent = state.get("sub_intent", "")
    yprint(f"  [route_after_sub_intent] sub_intent={sub_intent}")

    # Every Discovery request goes through discovery_preparation first —
    # there is no longer a direct edge to discovery. Whether a request
    # is specific enough to run is decided there, by parsing it, rather
    # than guessed at by this classifier.
    if sub_intent == "DISCOVERY":
        return "discovery_preparation"
    else:
        return "extract"

def route_after_extract(state: AgentState) -> str:
    """Routes after Node extract based on intent and ticker availability."""
    tickers = state.get("tickers") or []
    yprint(f"  [route_after_extract] tickers={tickers}")

    if not tickers:
        return "no_ticker"
    else:
        return "determine_data_scope"

def route_after_discovery_preparation(state: AgentState) -> str:
    """Runs discovery when the request parsed into something rankable,
    otherwise ends the turn on the follow-up question discovery_preparation
    already wrote into answer."""
    complete = state.get("clarification_complete")
    yprint(f"  [route_after_discovery_preparation] complete={complete}")
    if complete:
        return "discovery"
    else:
        return "update_session_memory"
    
# ─────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("contextualize_question", contextualize_question)
    graph.add_node("classify_top_intent",    classify_top_intent)
    graph.add_node("classify_sub_intent",    classify_sub_intent)
    graph.add_node("explain_concept",        explain_concept)
    graph.add_node("extract",                extract_parameters)
    graph.add_node("determine_data_scope",   determine_data_scope)
    graph.add_node("fetch_data",             fetch_data)
    graph.add_node("generate_report",        generate_report) 
    graph.add_node("out_of_scope",           handle_out_of_scope)
    graph.add_node("greeting",               handle_greeting)
    graph.add_node("discovery",              discovery_execution)
    graph.add_node("no_ticker",              handle_no_ticker) 
    graph.add_node("update_session_memory",  update_session_memory)
    graph.add_node("discovery_preparation",      discovery_preparation)
    graph.add_node("quant_engine",           quant_engine)

    # Conditional edge after Layer 1
    graph.add_conditional_edges(
        "classify_top_intent",
        route_after_top_intent,
        {
            "out_of_scope":        "out_of_scope",
            "greeting":            "greeting",
            "explain_concept":     "explain_concept",
            "classify_sub_intent": "classify_sub_intent",
        }
    )

    # Conditional edge after Layer 2
    graph.add_conditional_edges(
        "classify_sub_intent",
        route_after_sub_intent,
        {
            "discovery_preparation": "discovery_preparation",
            "extract":           "extract",
        }
    )

    # Conditional edge after Node extract_parameters
    graph.add_conditional_edges(
        "extract",
        route_after_extract,
        {
            "no_ticker":            "no_ticker",
            "determine_data_scope": "determine_data_scope",
        }
    )

    graph.add_conditional_edges(
        "discovery_preparation",
        route_after_discovery_preparation,
        {
            "discovery":             "discovery",
            "update_session_memory": "update_session_memory",
        }
    )

    # Linear edges
    graph.add_edge(START,                    "contextualize_question")
    graph.add_edge("contextualize_question", "classify_top_intent")
    # Discovery reaches fetch_data through determine_data_scope rather
    # than around it. Left to itself, fetch_data reads an empty scope as
    # "fetch everything" — nine sources per ticker, mostly live yfinance
    # calls, across a candidate pool that can run to a dozen companies.
    # Scope is decided from the question rather than from the parsed
    # ranking fields, because the two answer different questions: asked
    # for "companies with major recent risks", the fields carry
    # risk_beta_score, which measures price volatility, while what the
    # question needs is sec_filing — and no numeric field maps to that.
    graph.add_edge("discovery",              "determine_data_scope")
    graph.add_edge("determine_data_scope",   "fetch_data")
    graph.add_edge("fetch_data",             "quant_engine")
    graph.add_edge("quant_engine",           "generate_report")
    graph.add_edge("generate_report",        "update_session_memory")
    graph.add_edge("explain_concept",        "update_session_memory")
    graph.add_edge("out_of_scope",           "update_session_memory")
    graph.add_edge("greeting",               "update_session_memory")
    graph.add_edge("no_ticker",              "update_session_memory") 
    graph.add_edge("update_session_memory",  END)

    return graph.compile()


# Compile once at module level
equitymind_graph = build_graph()