import json
import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from backend.db.sqlite import get_conn


def load_eval_records(limit: int = 50, only_with_context: bool = True) -> list[dict]:
    sql = """
    SELECT id, user_id, question, answer, ground_truth,
           retrieved_contexts, citations, latency_ms, created_at
    FROM eval_records
    ORDER BY created_at DESC
    LIMIT ?
    """

    records = []

    with get_conn() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()

    for row in rows:
        contexts = json.loads(row["retrieved_contexts"] or "[]")

        if only_with_context and len(contexts) == 0:
            continue

        records.append(
            {
                "record_id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "contexts": contexts,
                "ground_truth": row["ground_truth"] or "",
            }
        )

    return records


def build_metrics(has_ground_truth: bool):
    metrics = [
        faithfulness,
        answer_relevancy,
    ]

    if has_ground_truth:
        try:
            from ragas.metrics import context_recall, answer_correctness

            metrics.extend(
                [
                    context_recall,
                    answer_correctness,
                ]
            )
        except Exception as e:
            print("加载 context_recall / answer_correctness 失败，跳过：", e)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--model", type=str, default="qwen2.5:3b")
    parser.add_argument("--output", type=str, default="data/ragas_eval_results.csv")
    args = parser.parse_args()

    records = load_eval_records(limit=args.limit)

    if not records:
        print("没有可评估的数据。请先通过 /chat 或 /chat/stream 问几个 RAG 问题。")
        return

    has_ground_truth = any(r.get("ground_truth") for r in records)

    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in records
        ]
    )

    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=args.model,
            temperature=0,
            base_url="http://127.0.0.1:11434",
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="/root/autodl-tmp/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    metrics = build_metrics(has_ground_truth)

    print("开始 RAGAS 评估...")
    print("样本数量:", len(records))
    print("指标:", [m.name for m in metrics])

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print("\n整体结果：")
    print(result)

    try:
        df = result.to_pandas()
    except Exception:
        df = pd.DataFrame(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n详细结果已保存到：{output_path}")


if __name__ == "__main__":
    main()