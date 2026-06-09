from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from backend.db.sqlite import list_chunks_for_user


def search_bm25(
    query: str,
    user_id: str,
    kb_id: str = "default",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    rows = list_chunks_for_user(user_id=user_id, kb_id=kb_id)

    if not rows:
        return []

    documents = [
        Document(
            page_content=row["content"],
            metadata={
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "user_id": row["user_id"],
                "kb_id": row["kb_id"],
                "source_filename": row["source_filename"],
                "chunk_index": row["chunk_index"],
            },
        )
        for row in rows
    ]

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = top_k

    result_docs = retriever.invoke(query)

    return [
        {
            "content": doc.page_content,
            "chunk_id": doc.metadata.get("chunk_id"),
            "document_id": doc.metadata.get("document_id"),
            "user_id": doc.metadata.get("user_id"),
            "kb_id": doc.metadata.get("kb_id"),
            "source_filename": doc.metadata.get("source_filename"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "score": None,
            "retriever": "bm25",
        }
        for doc in result_docs
    ]
