from __future__ import annotations

from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import EMBEDDING_MODEL


SplitterType = Literal["semantic", "recursive", "fixed"]


def _fallback_recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    按段落、标题、句号等语义边界优先切分。
    它不是简单按字符硬切，而是尽量先在自然段和句子处分开。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ". ",
            "! ",
            "? ",
            "；",
            "; ",
            "，",
            ", ",
            " ",
            "",
        ],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def _semantic_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    基于 embedding 的语义切片。

    真正参与语义切片计算的是句子/段落 embedding 之间的相似度；
    SemanticChunker 只是负责调用 embedding 模型、比较相邻片段语义差异，
    然后在语义变化较大的位置断开。

    如果环境中没有安装 langchain-experimental，或者 embedding 模型不可用，
    会自动退回到 RecursiveCharacterTextSplitter，避免上传接口直接崩溃。
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80,
        )

        semantic_chunks = [
            doc.page_content.strip()
            for doc in splitter.create_documents([text])
            if doc.page_content.strip()
        ]

        if not semantic_chunks:
            return _fallback_recursive_split(text, chunk_size, chunk_overlap)

        # SemanticChunker 可能产生过长片段，这里只对过长片段做二次递归切分。
        final_chunks: list[str] = []
        for chunk in semantic_chunks:
            if len(chunk) <= int(chunk_size * 1.5):
                final_chunks.append(chunk)
            else:
                final_chunks.extend(
                    _fallback_recursive_split(chunk, chunk_size, chunk_overlap)
                )

        return final_chunks

    except Exception:
        return _fallback_recursive_split(text, chunk_size, chunk_overlap)


def _fixed_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = end - chunk_overlap

    return chunks


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    splitter_type: SplitterType = "semantic",
) -> list[str]:
    """
    文档切片入口。

    splitter_type="semantic"：优先使用 embedding 语义切片，失败时自动退回递归切片。
    splitter_type="recursive"：按标题、段落、句子等自然边界递归切片。
    splitter_type="fixed"：保留原来的固定长度切片，主要用于对比实验。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    text = text.strip()
    if not text:
        return []

    if splitter_type == "semantic":
        return _semantic_split(text, chunk_size, chunk_overlap)

    if splitter_type == "recursive":
        return _fallback_recursive_split(text, chunk_size, chunk_overlap)

    if splitter_type == "fixed":
        return _fixed_split(text, chunk_size, chunk_overlap)

    raise ValueError(f"未知切片类型：{splitter_type}")
