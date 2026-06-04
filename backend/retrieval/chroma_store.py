from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import CHROMA_DIR


class ChromaService:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name="/root/autodl-tmp/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = Chroma(
            collection_name="knowledge_base",
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    def add_chunks(self, chunk_rows: list[dict[str, Any]]) -> None:
        documents: list[Document] = []
        ids: list[str] = []

        for row in chunk_rows:
            documents.append(
                Document(
                    page_content=row["content"],
                    metadata={
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "user_id": row["user_id"],
                        "kb_id": row["kb_id"],
                        "source_filename": row["source_filename"],
                        "chunk_index": row["chunk_index"],
                    },
                )
            )
            ids.append(row["chunk_id"])

        if documents:
            self.vectorstore.add_documents(documents=documents, ids=ids)

    def search(
        self,
        query: str,
        user_id: str,
        kb_id: str = "default",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k,
            filter={
                "$and": [
                    {"user_id": user_id},
                    {"kb_id": kb_id},
                ]
            },
        )

        docs: list[dict[str, Any]] = []

        for doc, score in results:
            docs.append(
                {
                    "content": doc.page_content,
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "document_id": doc.metadata.get("document_id"),
                    "source_filename": doc.metadata.get("source_filename"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "score": float(score),
                    "retriever": "chroma",
                }
            )

        return docs


chroma_service = ChromaService()