from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from backend.graph.nodes import (
    classify_question,
    generate_answer,
    merge_results,
    retrieve_bm25,
    retrieve_chroma,
    rewrite_query,
    save_message_node,
    tool_node,
    validate_answer_node,
)
from backend.graph.state import GraphState


def route_after_classify(state: GraphState) -> Literal["rag", "chat", "tool"]:
    route = state.get("route", "chat")

    if route in ["rag", "chat", "tool"]:
        return route

    return "chat"


def build_graph():
    """
    LangGraph 主工作流。

    chat 分支：classify -> generate -> validate -> save
    rag 分支：classify -> rewrite -> chroma -> bm25 -> merge -> generate -> validate -> save
    tool 分支：classify -> tool_node -> generate -> validate -> save
    """
    graph = StateGraph(GraphState)

    graph.add_node("classify_question", classify_question)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_chroma", retrieve_chroma)
    graph.add_node("retrieve_bm25", retrieve_bm25)
    graph.add_node("merge_results", merge_results)
    graph.add_node("tool_node", tool_node)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("validate_answer", validate_answer_node)
    graph.add_node("save_message", save_message_node)

    graph.add_edge(START, "classify_question")

    graph.add_conditional_edges(
        "classify_question",
        route_after_classify,
        {
            "rag": "rewrite_query",
            "chat": "generate_answer",
            "tool": "tool_node",
        },
    )

    graph.add_edge("rewrite_query", "retrieve_chroma")
    graph.add_edge("retrieve_chroma", "retrieve_bm25")
    graph.add_edge("retrieve_bm25", "merge_results")
    graph.add_edge("merge_results", "generate_answer")
    graph.add_edge("tool_node", "generate_answer")
    graph.add_edge("generate_answer", "validate_answer")
    graph.add_edge("validate_answer", "save_message")
    graph.add_edge("save_message", END)

    checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)


rag_graph = build_graph()
