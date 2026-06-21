from colors import yprint

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    classify_top_intent,
    classify_sub_intent,
    explain_concept,
    extract_parameters,
    ensure_sec_data,
    get_market_data,
    get_news,
    simple_report,
    handle_out_of_scope,
    handle_greeting,
    handle_no_ticker,
    discovery_suggest,
    update_session_memory,
    handle_follow_up,
    handle_clarification,
)


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

    if sub_intent == "FOLLOW_UP":
        return "follow_up"
    elif sub_intent == "CLARIFICATION":
        return "clarification"
    elif sub_intent == "DISCOVERY":
        return "discovery_suggest"
    else:
        return "extract"

def route_after_extract(state: AgentState) -> str:
    """Routes after Node extract based on intent and ticker availability."""
    tickers = state.get("tickers") or []
    yprint(f"  [route_after_extract] tickers={tickers}")

    if not tickers:
        return "no_ticker"
    else:
        return "ensure_sec"

def route_after_clarification(state: AgentState) -> str:
    complete = state.get("clarification_complete")
    yprint(f"  [route_after_clarification] complete={complete}")
    if complete:
        return "discovery_suggest"
    else:
        return "update_session_memory"
    
# ─────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("classify_top_intent",    classify_top_intent)
    graph.add_node("classify_sub_intent",    classify_sub_intent)
    graph.add_node("explain_concept",        explain_concept)
    graph.add_node("extract",                extract_parameters)
    graph.add_node("ensure_sec",             ensure_sec_data)
    graph.add_node("market_data",            get_market_data)
    graph.add_node("news",                   get_news)
    graph.add_node("simple_report",          simple_report)
    graph.add_node("out_of_scope",           handle_out_of_scope)
    graph.add_node("greeting",               handle_greeting)
    graph.add_node("discovery_suggest",      discovery_suggest)
    graph.add_node("no_ticker",              handle_no_ticker) 
    graph.add_node("update_session_memory",  update_session_memory)
    graph.add_node("follow_up",              handle_follow_up)
    graph.add_node("clarification",          handle_clarification)

    # Entry point — Layer 1
    graph.set_entry_point("classify_top_intent")

    # Conditional edge after Layer 1
    graph.add_conditional_edges(
        "classify_top_intent",
        route_after_top_intent,
        {
            "out_of_scope":       "out_of_scope",
            "greeting":           "greeting",
            "explain_concept":    "explain_concept",
            "classify_sub_intent": "classify_sub_intent",
        }
    )

    # Conditional edge after Layer 2
    graph.add_conditional_edges(
        "classify_sub_intent",
        route_after_sub_intent,
        {
            "follow_up":          "follow_up",
            "clarification":      "clarification",
            "discovery_suggest":  "discovery_suggest",
            "extract":            "extract",
        }
    )

    # Conditional edge after Node extract_parameters
    graph.add_conditional_edges(
        "extract",
        route_after_extract,
        {
            "no_ticker":      "no_ticker",
            "ensure_sec":     "ensure_sec",
        }
    )

    graph.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "discovery_suggest":  "discovery_suggest",
            "update_session_memory": "update_session_memory",
        }
    )

    # Linear flow after market_data
    graph.add_edge("discovery_suggest",      "ensure_sec")
    graph.add_edge("ensure_sec",             "market_data")
    graph.add_edge("market_data",            "news")
    graph.add_edge("news",                   "simple_report")
    graph.add_edge("explain_concept",        "update_session_memory")
    graph.add_edge("follow_up",              "update_session_memory")
    graph.add_edge("simple_report",          "update_session_memory")
    graph.add_edge("out_of_scope",           "update_session_memory")
    graph.add_edge("greeting",               "update_session_memory")
    graph.add_edge("no_ticker",              "update_session_memory") 
    graph.add_edge("update_session_memory",  END)

    return graph.compile()


# Compile once at module level
equitymind_graph = build_graph()