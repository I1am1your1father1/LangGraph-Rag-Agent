from backend.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_deduplicate_and_rank():
    chroma_docs = [
        {"chunk_id": "a", "content": "A"},
        {"chunk_id": "b", "content": "B"},
    ]

    bm25_docs = [
        {"chunk_id": "b", "content": "B"},
        {"chunk_id": "c", "content": "C"},
    ]

    result = reciprocal_rank_fusion(
        [chroma_docs, bm25_docs],
        top_n=3,
    )

    ids = [doc["chunk_id"] for doc in result]

    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert "b" in ids