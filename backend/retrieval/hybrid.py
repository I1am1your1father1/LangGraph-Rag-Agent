from typing import Any


def reciprocal_rank_fusion(
    results: list[list[dict[str, Any]]],
    top_n: int = 5,
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion, RRF.

    作用：
    将多个检索器的结果按照排名进行融合。

    参数：
    results:
        多个检索器返回的结果列表，例如：
        [
            chroma_docs,
            bm25_docs,
        ]

    top_n:
        最终返回前多少条结果。

    k:
        平滑参数，常用默认值是 60。
        k 越大，不同排名之间的分数差距越小。

    返回：
    融合、去重、排序后的文档列表。
    """

    fused: dict[str, dict[str, Any]] = {}

    for docs in results:
        for rank, doc in enumerate(docs, start=1):
            chunk_id = doc.get("chunk_id")

            if not chunk_id:
                chunk_id = doc.get("content", "")[:100]

            if not chunk_id:
                continue

            score = 1.0 / (k + rank)

            if chunk_id not in fused:
                new_doc = dict(doc)
                new_doc["fusion_score"] = 0.0
                fused[chunk_id] = new_doc

            fused[chunk_id]["fusion_score"] += score

            old_retriever = fused[chunk_id].get("retriever", "")
            new_retriever = doc.get("retriever", "")

            if new_retriever and new_retriever not in old_retriever:
                if old_retriever:
                    fused[chunk_id]["retriever"] = old_retriever + "+" + new_retriever
                else:
                    fused[chunk_id]["retriever"] = new_retriever

    fused_docs = sorted(
        fused.values(),
        key=lambda x: x.get("fusion_score", 0.0),
        reverse=True,
    )

    return fused_docs[:top_n]