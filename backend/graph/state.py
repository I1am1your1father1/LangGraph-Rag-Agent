from typing import Any, TypedDict


class RetrievedDoc(TypedDict, total=False):
    content: str
    chunk_id: str
    document_id: str
    user_id: str
    kb_id: str
    source_filename: str
    chunk_index: int
    score: float | None
    retriever: str


class GraphState(TypedDict, total=False):
    user_id: str
    kb_id: str
    session_id: str
    question: str

    rewritten_query: str
    route: str
    route_reason: str

    tool_call_count: int
    tool_name: str
    tool_args: dict[str, Any]
    tool_result: str

    chroma_docs: list[RetrievedDoc]
    bm25_docs: list[RetrievedDoc]
    retrieved_docs: list[RetrievedDoc]

    context: str
    citations: list[dict[str, Any]]

    answer: str
    citation_check: dict[str, Any]

    latency_ms: float | None
    error: str | None
