import json
import shutil
from pathlib import Path
from typing import Any
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import UPLOAD_DIR
from backend.db.sqlite import create_chunks, create_document, init_db, init_eval_table
from backend.documents.loader import load_text_file
from backend.documents.splitter import split_text
from backend.graph.workflow import rag_graph
from backend.retrieval.chroma_store import chroma_service
from backend.graph.nodes import (
    classify_question,
    rewrite_query,
    retrieve_chroma,
    retrieve_bm25,
    merge_results,
    stream_llm_answer,
    save_message_node,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="LangGraph RAG Agent")
app.mount("/web", StaticFiles(directory="frontend", html=True), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # My address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()
    init_eval_table()
    


class ChatRequest(BaseModel):
    question: str
    user_id: str = "demo"
    session_id: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = "demo",
    kb_id: str = "default",
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in {".txt", ".md"}:
        raise HTTPException(status_code=400, detail="当前只支持 .txt 和 .md 文件")

    save_path = UPLOAD_DIR / file.filename

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = load_text_file(save_path)
        chunks = split_text(text, chunk_size=800, chunk_overlap=120)

        document_id = create_document(
            user_id=user_id,
            filename=file.filename or "unknown",
        )

        chunk_rows = create_chunks(
            document_id=document_id,
            user_id=user_id,
            filename=file.filename or "unknown",
            chunk_texts=chunks,
        )

        chroma_service.add_chunks(chunk_rows)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败：{e}")

    return {
        "message": "upload success",
        "document_id": document_id,
        "filename": file.filename,
        "chunk_count": len(chunks),
    }


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    session_id = req.session_id or f"{req.user_id}_default"

    result = rag_graph.invoke(
        {
            "user_id": req.user_id,
            "session_id": req.session_id,
            "question": req.question,
        },
        config={
            "configurable": {
                "thread_id": session_id
            }
        },
    )

    return {
        "session_id": result.get("session_id"),
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "retrieved_docs": result.get("retrieved_docs", []),
        "error": result.get("error"),
    }


@app.get("/chat/stream")
async def chat_stream(
    question: str,
    user_id: str = "demo",
    session_id: str | None = None,
):
    async def event_generator():
        state = {
            "user_id": user_id,
            "session_id": session_id,
            "question": question,
        }

        state.update(classify_question(state))

        if state.get("route") == "rag":
            state.update(rewrite_query(state))
            state.update(retrieve_chroma(state))
            state.update(retrieve_bm25(state))
            state.update(merge_results(state))

        answer_parts = []

        async for token in stream_llm_answer(
            question=question,
            context=state.get("context", ""),
        ):
            answer_parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

        answer = "".join(answer_parts)

        state["answer"] = answer
        state.update(save_message_node(state))

        yield f"data: {json.dumps({'type': 'done', 'citations': state.get('citations', [])}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )