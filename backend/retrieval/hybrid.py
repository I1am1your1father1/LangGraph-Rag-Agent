from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
    top_n: int = 6,
) -> list[dict[str, Any]]:
    """
    RRF 融合多个检索器结果。

    score = sum(1 / (k + rank))

    k 通常取 60，防止排名靠后的结果影响过大。
    """
    score_map: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict[str, Any]] = {}

    for docs in result_lists:
        for rank, doc in enumerate(docs, start=1):
            chunk_id = doc.get("chunk_id")
            if not chunk_id:
                continue

            score_map[chunk_id] += 1.0 / (k + rank)

            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc

    sorted_items = sorted(
        score_map.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    fused_docs = []
    for chunk_id, score in sorted_items[:top_n]:
        doc = doc_map[chunk_id]
        doc["fusion_score"] = score
        fused_docs.append(doc)

    return fused_docs