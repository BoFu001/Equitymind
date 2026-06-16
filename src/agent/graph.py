from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    classify_intent,
    extract_parameters,
    ensure_sec_data,
    get_market_data,
    get_news,
    simple_report,
    #specific_report,
    handle_out_of_scope,
    handle_greeting,
    #comparison_report,
    handle_no_ticker,
    discovery_suggest,
    #discovery_report,
    update_session_memory,
    handle_follow_up,
    handle_clarification,
)


# ─────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────

def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "")
    if intent == "OUT_OF_SCOPE":
        return "out_of_scope"
    elif intent == "GREETING":
        return "greeting"
    elif intent == "FOLLOW_UP":
        return "follow_up"
    elif intent == "CLARIFICATION":
        return "clarification"
    elif intent == "DISCOVERY":
        return "discovery_suggest"
    else:
        return "extract"  # simple_report

def route_after_extract(state: AgentState) -> str:
    """Routes after Node extract based on intent and ticker availability."""
    tickers = state.get("tickers") or []

    if not tickers:
        return "no_ticker"
    else:
        return "ensure_sec"

# def route_after_news(state: AgentState) -> str:
#     intent = state.get("intent", "")
#     if intent == "DISCOVERY":
#         return "discovery_report"
#     elif intent == "COMPARISON":
#         return "comparison_report"
#     else:
#         return "specific_report"

def route_after_clarification(state: AgentState) -> str:
    answer = state.get("answer", "")
    if answer.strip().startswith("READY:"):
        return "discovery_suggest"
    else:
        return "update_session_memory"
    
# ─────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("classify",               classify_intent)
    graph.add_node("extract",                extract_parameters)
    graph.add_node("ensure_sec",             ensure_sec_data)
    graph.add_node("market_data",            get_market_data)
    graph.add_node("news",                   get_news)
    graph.add_node("simple_report",          simple_report)
    #graph.add_node("specific_report",        specific_report)
    graph.add_node("out_of_scope",           handle_out_of_scope)
    graph.add_node("greeting",               handle_greeting)
    graph.add_node("discovery_suggest",      discovery_suggest)
    #graph.add_node("discovery_report",       discovery_report)
    #graph.add_node("comparison_report",      comparison_report)
    graph.add_node("no_ticker",              handle_no_ticker) 
    graph.add_node("update_session_memory",  update_session_memory)
    graph.add_node("follow_up",              handle_follow_up)
    graph.add_node("clarification",          handle_clarification)

    # Entry point
    graph.set_entry_point("classify")

    # Conditional edge after Node classify_intent
    graph.add_conditional_edges(
        "classify",
        route_intent,
        {
            "out_of_scope":      "out_of_scope",
            "greeting":          "greeting",
            "follow_up":         "follow_up",
            "clarification":     "clarification",
            "discovery_suggest": "discovery_suggest",
            "extract":           "extract",
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

    # graph.add_conditional_edges(
    #     "news",
    #     route_after_news,
    #     {
    #         "specific_report":     "specific_report",
    #         "discovery_report":    "discovery_report",
    #         "comparison_report":   "comparison_report",
    #     }
    # )

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
    graph.add_edge("follow_up",              "update_session_memory")
    # graph.add_edge("specific_report",        "update_session_memory")
    # graph.add_edge("discovery_report",       "update_session_memory")
    # graph.add_edge("comparison_report",      "update_session_memory")
    graph.add_edge("simple_report",          "update_session_memory")
    graph.add_edge("out_of_scope",           "update_session_memory")
    graph.add_edge("greeting",               "update_session_memory")
    graph.add_edge("no_ticker",              "update_session_memory") 
    graph.add_edge("update_session_memory",  END)

    return graph.compile()


# Compile once at module level
equitymind_graph = build_graph()