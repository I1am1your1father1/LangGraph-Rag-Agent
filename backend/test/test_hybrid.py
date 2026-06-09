from backend.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_deduplicate_and_rank():
    chroma_docs = [
        {"chunk_id": "a", "content": "A", "retriever": "chroma"},
        {"chunk_id": "b", "content": "B", "retriever": "chroma"},
    ]

    bm25_docs = [
        {"chunk_id": "b", "content": "B", "retriever": "bm25"},
        {"chunk_id": "c", "content": "C", "retriever": "bm25"},
    ]

    result = reciprocal_rank_fusion(
        [chroma_docs, bm25_docs],
        top_n=3,
    )

    ids = [doc["chunk_id"] for doc in result]

    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert "b" in ids

    assert ids[0] == "b"

    b_doc = result[0]
    assert b_doc["fusion_score"] > 0
    assert "chroma" in b_doc["retriever"]
    assert "bm25" in b_doc["retriever"]