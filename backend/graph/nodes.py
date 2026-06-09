import hashlib
import json
import re
from typing import Any

from langchain_ollama import ChatOllama

from backend.db.sqlite import ensure_session, save_eval_record, save_message
from backend.evaluation.citation_validator import validate_citations
from backend.graph.state import GraphState
from backend.retrieval.bm25_store import search_bm25
from backend.retrieval.chroma_store import chroma_service
from backend.retrieval.hybrid import reciprocal_rank_fusion
from backend.tools.basic_tools import CalculatorInput, calculator_tool
from backend.utils.logger import log_node_time


llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.2,
    base_url="http://127.0.0.1:11434",
)


def _extract_calculator_expression(question: str) -> str | None:
    """
    从问题中提取简单数学表达式。
    这里只支持 calculator_tool 当前能处理的数字、括号和四则运算。
    """
    candidates = re.findall(r"[0-9][0-9+\-*/().\s]*[0-9)]", question)

    for candidate in candidates:
        expr = candidate.strip()
        if any(op in expr for op in ["+", "-", "*", "/"]):
            return expr

    return None


@log_node_time("classify_question")
def classify_question(state: GraphState) -> dict[str, Any]:
    """
    路由节点。

    这里不只是返回 rag/chat，还会在判断需要工具时，把 tool_name 和 tool_args
    一起写入 state。这样后面的 tool_node 才知道应该调用哪个工具。
    """
    question = state["question"].strip()

    calculator_expr = _extract_calculator_expression(question)
    calculator_keywords = ["计算", "算一下", "等于多少", "calculator", "calculate"]
    if calculator_expr and any(keyword in question.lower() for keyword in calculator_keywords):
        return {
            "route": "tool",
            "route_reason": "检测到四则运算问题，交给 calculator 工具处理",
            "tool_name": "calculator",
            "tool_args": {"expression": calculator_expr},
            "tool_call_count": 0,
        }

    rag_keywords = [
        "文档",
        "知识库",
        "资料",
        "上传",
        "根据上下文",
        "根据材料",
        "根据文件",
        "README",
        "代码里",
        "项目中",
        "test.md",
        "LangGraph",
        "Chroma",
        "BM25",
    ]

    if any(keyword in question for keyword in rag_keywords):
        return {
            "route": "rag",
            "route_reason": "问题需要结合知识库或项目文档回答",
        }

    return {
        "route": "chat",
        "route_reason": "普通对话，不需要检索知识库或调用工具",
    }


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
    try:
        response = llm.invoke([("human", prompt)])
        rewritten = response.content.strip()

        return {"rewritten_query": rewritten or question}

    except Exception as e:
        return {
            "rewritten_query": question,
            "error": f"rewrite_query failed: {e}",
        }


@log_node_time("tool_node")
def tool_node(state: GraphState) -> dict[str, Any]:
    """
    工具调用节点。

    classify_question 负责判断是否需要工具，并写入 tool_name/tool_args；
    tool_node 负责真正执行工具函数，并把结果写回 tool_result。
    """
    tool_name = state.get("tool_name")
    tool_args = state.get("tool_args", {})
    tool_call_count = state.get("tool_call_count", 0) + 1

    try:
        if tool_name == "calculator":
            result = calculator_tool(CalculatorInput(**tool_args))
            return {
                "tool_result": result,
                "tool_call_count": tool_call_count,
            }

        if tool_name == "search_kb":
            query = tool_args.get("query") or state.get("question", "")
            docs = chroma_service.search(
                query=query,
                user_id=state["user_id"],
                kb_id=state.get("kb_id", "default"),
                top_k=int(tool_args.get("top_k", 5)),
            )
            return {
                "tool_result": json.dumps(docs, ensure_ascii=False),
                "tool_call_count": tool_call_count,
            }

        return {
            "tool_result": f"未知工具：{tool_name}",
            "tool_call_count": tool_call_count,
        }

    except Exception as e:
        return {
            "tool_result": f"工具调用失败：{e}",
            "tool_call_count": tool_call_count,
            "error": f"tool_node failed: {e}",
        }


@log_node_time("retrieve_chroma")
def retrieve_chroma(state: GraphState) -> dict[str, Any]:
    query = state.get("rewritten_query") or state["question"]

    try:
        docs = chroma_service.search(
            query=query,
            user_id=state["user_id"],
            kb_id=state.get("kb_id", "default"),
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
            kb_id=state.get("kb_id", "default"),
            top_k=5,
        )
        return {"bm25_docs": docs}
    except Exception as e:
        return {"bm25_docs": [], "error": f"BM25 检索失败：{e}"}


@log_node_time("merge_results")
def merge_results(state: GraphState) -> dict[str, Any]:
    fused_docs = reciprocal_rank_fusion(
        [
            state.get("chroma_docs", []),
            state.get("bm25_docs", []),
        ],
        top_n=6,
    )

    seen_contents = set()
    deduped_docs = []

    for doc in fused_docs:
        content = doc.get("content", "")
        normalized_content = " ".join(content.split())

        if normalized_content:
            content_key = hashlib.md5(
                normalized_content.encode("utf-8")
            ).hexdigest()
        else:
            content_key = doc.get("chunk_id", "")

        if content_key in seen_contents:
            continue

        seen_contents.add(content_key)
        deduped_docs.append(doc)

    context_parts = []
    citations = []

    for idx, doc in enumerate(deduped_docs, start=1):
        content = doc.get("content", "")
        source_filename = doc.get("source_filename", "")
        chunk_id = doc.get("chunk_id", "")
        chunk_index = doc.get("chunk_index", 0)
        retriever = doc.get("retriever", "")

        context_parts.append(
            f"[{idx}] 来源文件：{source_filename}\n"
            f"知识库：{doc.get('kb_id', 'default')}\n"
            f"内容：{content}"
        )

        citations.append(
            {
                "index": idx,
                "source_filename": source_filename,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "kb_id": doc.get("kb_id", state.get("kb_id", "default")),
                "retriever": retriever,
                "fusion_score": doc.get("fusion_score"),
            }
        )

    context = "\n\n".join(context_parts)

    return {
        "retrieved_docs": deduped_docs,
        "context": context,
        "citations": citations,
    }


@log_node_time("generate_answer")
def generate_answer(state: GraphState) -> dict[str, Any]:
    question = state["question"]
    context = state.get("context", "")
    citations = state.get("citations", [])
    route = state.get("route", "chat")

    if route == "tool":
        tool_name = state.get("tool_name", "未知工具")
        tool_result = state.get("tool_result", "")
        return {
            "answer": f"工具 {tool_name} 的调用结果：{tool_result}",
            "citations": [],
            "retrieved_docs": [],
        }

    if route == "rag":
        if not context.strip():
            return {
                "answer": "根据当前文档无法回答，因为没有检索到相关内容。",
                "citations": [],
                "retrieved_docs": [],
            }

        system_prompt = """
你是一个严格的 RAG 知识库问答助手。

你必须遵守以下规则：
1. 只能根据【已知文档内容】回答问题。
2. 不允许使用你自己的常识补充答案。
3. 不允许编造文档中没有的信息。
4. 回答必须是自然语言，不能只输出引用编号。
5. 如果文档中没有答案，请回答：根据当前文档无法回答。
6. 如果使用了文档内容，请在句子末尾标注引用编号，例如 [1]。
"""

        user_prompt = f"""
【已知文档内容】
{context}

【用户问题】
{question}

请严格根据【已知文档内容】回答。
"""

        response = llm.invoke(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )

        answer = getattr(response, "content", str(response)).strip()

        if not answer:
            answer = "根据当前文档无法回答。"

        return {
            "answer": answer,
            "citations": citations,
        }

    system_prompt = """
你是一个本地运行的 LangGraph RAG Agent 助手。
你可以进行普通对话，也可以在用户明确询问文档、知识库、上传资料时，基于检索到的文档回答问题。
如果用户问你是谁或你是什么，请简洁介绍自己。
"""

    response = llm.invoke(
        [
            ("system", system_prompt),
            ("human", question),
        ]
    )

    answer = getattr(response, "content", str(response)).strip()

    return {
        "answer": answer,
        "citations": [],
        "retrieved_docs": [],
    }


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
def validate_answer_node(state: GraphState) -> dict[str, Any]:
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
    kb_id = state.get("kb_id", "default")
    session_id = ensure_session(
        user_id=user_id,
        kb_id=kb_id,
        session_id=state.get("session_id"),
    )
    answer = state.get("answer", "") or ""
    citations = state.get("citations", [])

    save_message(
        session_id=session_id,
        user_id=user_id,
        kb_id=kb_id,
        role="user",
        content=state["question"],
    )

    save_message(
        session_id=session_id,
        user_id=user_id,
        kb_id=kb_id,
        role="assistant",
        content=answer,
        citations=json.dumps(citations, ensure_ascii=False),
    )

    citation_check = validate_citations(
        answer=answer,
        citations=citations,
    )

    retrieved_contexts = [
        doc.get("content", "")
        for doc in state.get("retrieved_docs", [])
        if doc.get("content", "")
    ]

    save_eval_record(
        user_id=user_id,
        kb_id=kb_id,
        question=state["question"],
        answer=answer,
        retrieved_contexts=retrieved_contexts,
        citations=citations,
        latency_ms=state.get("latency_ms"),
        ground_truth=None,
    )

    return {"session_id": session_id, "citation_check": citation_check}
