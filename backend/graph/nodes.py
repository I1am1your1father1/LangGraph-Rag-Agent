import json
from typing import Any
from backend.utils.logger import log_node_time

from langchain_ollama import ChatOllama

from backend.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from backend.db.sqlite import ensure_session, save_message, save_eval_record
from backend.graph.state import GraphState, RetrievedDoc
from backend.retrieval.chroma_store import chroma_service
from backend.retrieval.bm25_store import search_bm25
from backend.retrieval.hybrid import reciprocal_rank_fusion
from backend.retrieval.chroma_store import chroma_service
from backend.evaluation.citation_validator import validate_citations
from backend.tools.basic_tools import CalculatorInput, calculator_tool


llm = ChatOllama(
    model="qwen2.5:0.5b",
    temperature=0.2,
    base_url="http://127.0.0.1:11434",
)


@log_node_time("classify_question")
def classify_question(state: GraphState) -> dict[str, Any]:
    """
    Version 1.0 
    Using simple rules.
    """
    question = state["question"].strip()

    greeting_words = {"你好", "hello", "hi", "在吗"}
    if question.lower() in greeting_words:
        return {"route": "chat"}
    
    if any(op in question for op in ["+", "-", "*", "/", "计算"]):
        expression = (
            question
            .replace("计算", "")
            .replace("帮我", "")
            .replace("请", "")
            .strip()
        )

        return {
            "route": "tool",
            "tool_name": "calculator",
            "tool_args": {
                "expression": expression,
            },
            "tool_call_count": 0,
        }

    return {"route": "rag"}


@log_node_time("rewrite_query")
def rewrite_query(state: GraphState) -> dict[str, Any]:
    question = state["question"]

    prompt = f"""
你是一个 RAG 检索查询改写器。
请把用户问题改写成更适合知识库检索的查询。
要求：
1. 保留原始问题的核心含义
2. 补充可能的关键词
3. 不要回答问题
4. 只输出改写后的检索查询

用户问题：
{question}
"""

    response = llm.invoke([("human", prompt)])
    rewritten = response.content.strip()

    return {"rewritten_query": rewritten or question}


@log_node_time("tool_node")
def tool_node(state: GraphState) -> dict:
    tool_name = state.get("tool_name")
    tool_args = state.get("tool_args", {})

    if tool_name == "calculator":
        result = calculator_tool(CalculatorInput(**tool_args))
        return {"tool_result": result}

    if tool_name == "search_kb":
        docs = chroma_service.search(
            query=tool_args["query"],
            user_id=state["user_id"],
            top_k=5,
        )
        return {"tool_result": json.dumps(docs, ensure_ascii=False)}

    return {"tool_result": f"未知工具：{tool_name}"}


@log_node_time("retrieve_chroma")
def retrieve_chroma(state: GraphState) -> dict[str, Any]:
    query = state.get("rewritten_query") or state["question"]  # use rewitten question

    try:
        docs = chroma_service.search(
            query=query,
            user_id=state["user_id"],
            top_k=5,
        )
        return {"chroma_docs": docs}
    except Exception as e:
        return {"chroma_docs": [], "error": f"Chroma 检索失败：{e}"}


@log_node_time("retrieve_bm25")
def retrieve_bm25(state: GraphState) -> dict[str, Any]:
    query = state.get("rewritten_query") or state["question"]

    try:
        docs = search_bm25(
            query=query,
            user_id=state["user_id"],
            top_k=5,
        )
        return {"bm25_docs": docs}
    except Exception as e:
        return {"bm25_docs": [], "error": f"BM25 检索失败：{e}"}


@log_node_time("merge_results")
def merge_results(state: GraphState) -> dict:
    fused_docs = reciprocal_rank_fusion(
        [
            state.get("chroma_docs", []),
            state.get("bm25_docs", []),
        ],
        top_n=6,
    )

    context_parts = []
    citations = []

    for i, doc in enumerate(fused_docs, start=1):
        source = doc.get("source_filename", "unknown")
        chunk_index = doc.get("chunk_index", -1)

        context_parts.append(
            f"[{i}] 来源文件：{source}，chunk_index：{chunk_index}\n{doc['content']}"
        )

        citations.append(
            {
                "index": i,
                "source_filename": source,
                "chunk_id": doc.get("chunk_id"),
                "chunk_index": chunk_index,
                "retriever": doc.get("retriever"),
                "fusion_score": doc.get("fusion_score"),
            }
        )

    return {
        "retrieved_docs": fused_docs,
        "context": "\n\n".join(context_parts),
        "citations": citations,
    }


@log_node_time("generate_answer")
def generate_answer(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    context = state.get("context", "")

    if context:
        system_prompt = (
            "你是一个严谨的知识库问答助手。"
            "请只根据给定上下文回答问题。"
            "如果上下文中没有答案，请明确说：根据当前文档无法回答。"
            "回答时尽量标注引用编号，例如 [1]、[2]。"
        )

        user_prompt = f"""
用户问题：
{question}

检索到的上下文：
{context}

请基于上下文回答，并保留必要引用编号。
"""
    else:
        system_prompt = "你是一个简洁、可靠的 AI 助手。"
        user_prompt = question

    response = llm.invoke(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )

    return {"answer": response.content}


async def stream_llm_answer(question: str, context: str):
    if context:
        system_prompt = (
            "你是一个严谨的知识库问答助手。"
            "请只根据给定上下文回答。"
            "如果上下文中没有答案，请说：根据当前文档无法回答。"
        )

        user_prompt = f"""
用户问题：
{question}

上下文：
{context}

请基于上下文回答，并保留引用编号。
"""
    else:
        system_prompt = "你是一个简洁、可靠的 AI 助手。"
        user_prompt = question

    async for chunk in llm.astream(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    ):
        if chunk.content:
            yield chunk.content


@log_node_time("validate_answer_node")
def validate_answer_node(state: GraphState) -> dict:
    result = validate_citations(
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
    )

    return {
        "citation_check": result,
    }


@log_node_time("save_message_node")
def save_message_node(state: GraphState) -> dict[str, Any]:
    user_id = state["user_id"]
    session_id = ensure_session(user_id, state.get("session_id"))

    save_message(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=state["question"],
    )

    answer = state.get("answer")
    citations = state.get("citations", [])

    save_message(
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=answer,
        citations=json.dumps(state.get("citations", []), ensure_ascii=False),
    )

    citation_check = validate_citations(
        answer=answer,
        citations=citations,
    )

    retrieved_contexts = [
        doc.get("content", "")
        for doc in state.get("retrieved_docs", [])
    ]

    save_eval_record(
        user_id=user_id,
        question=state["question"],
        answer=state.get("answer", ""),
        retrieved_contexts=retrieved_contexts,
        citations=state.get("citations", []),
        latency_ms=None,
        ground_truth=None,
    )

    return {"session_id": session_id, "citation_check": citation_check}
