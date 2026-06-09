import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.auth.dependencies import CurrentUser, get_current_user
from backend.auth.jwt import create_access_token
from backend.config import UPLOAD_DIR
from backend.db.sqlite import create_chunks, create_document, init_db, init_eval_table
from backend.documents.splitter import split_text
from backend.graph.nodes import (
    classify_question,
    generate_answer,
    merge_results,
    retrieve_bm25,
    retrieve_chroma,
    rewrite_query,
    save_message_node,
    stream_llm_answer,
    tool_node,
)
from backend.graph.workflow import rag_graph
from backend.retrieval.chroma_store import chroma_service
from backend.utils.document_parser import parse_document


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="LangGraph RAG Agent")
app.mount("/web", StaticFiles(directory="frontend", html=True), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()
    init_eval_table()


class TokenRequest(BaseModel):
    user_id: str = Field(default="demo", description="开发阶段用于签发 token 的用户 ID")
    expires_minutes: int = Field(default=60 * 24, ge=1, le=60 * 24 * 30)


class ChatRequest(BaseModel):
    question: str
    kb_id: str = "default"
    session_id: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/token")
def issue_dev_token(req: TokenRequest) -> dict[str, Any]:
    """
    开发阶段的 JWT 签发接口。

    正式系统中这里应该替换成用户名/密码登录、OAuth 登录或企业 SSO，
    不能让任意 user_id 直接换 token。
    """
    user_id = req.user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 不能为空")

    token = create_access_token(
        user_id=user_id,
        expires_minutes=req.expires_minutes,
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user_id,
        "expires_minutes": req.expires_minutes,
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    kb_id: str = "default",
    current_user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()

    if suffix not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(
            status_code=400,
            detail="当前只支持 .txt、.md、.pdf、.docx 文件",
        )

    user_id = current_user.user_id
    kb_id = kb_id.strip() or "default"

    user_upload_dir = UPLOAD_DIR / user_id / kb_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = user_upload_dir / safe_name

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = parse_document(save_path)

        chunks = split_text(
            text,
            chunk_size=800,
            chunk_overlap=120,
            splitter_type="semantic",
        )

        document_id = create_document(
            user_id=user_id,
            kb_id=kb_id,
            filename=filename,
        )

        chunk_rows = create_chunks(
            document_id=document_id,
            user_id=user_id,
            kb_id=kb_id,
            filename=filename,
            chunk_texts=chunks,
        )

        chroma_service.add_chunks(chunk_rows)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文档处理失败：{e}",
        ) from e

    return {
        "message": "upload success",
        "user_id": user_id,
        "kb_id": kb_id,
        "document_id": document_id,
        "filename": filename,
        "file_type": suffix,
        "chunk_count": len(chunks),
        "splitter_type": "semantic",
    }


@app.post("/chat")
def chat(
    req: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    user_id = current_user.user_id
    kb_id = req.kb_id.strip() or "default"
    session_id = req.session_id or f"{user_id}_{kb_id}_default"

    result = rag_graph.invoke(
        {
            "user_id": user_id,
            "kb_id": kb_id,
            "session_id": session_id,
            "question": req.question,
        },
        config={
            "configurable": {
                "thread_id": session_id,
            }
        },
    )

    return {
        "user_id": user_id,
        "kb_id": kb_id,
        "session_id": result.get("session_id"),
        "route": result.get("route"),
        "route_reason": result.get("route_reason"),
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "retrieved_docs": result.get("retrieved_docs", []),
        "tool_name": result.get("tool_name"),
        "tool_result": result.get("tool_result"),
        "error": result.get("error"),
    }


@app.get("/chat/stream")
async def chat_stream(
    question: str,
    kb_id: str = "default",
    session_id: str | None = None,
    current_user: CurrentUser = Depends(get_current_user),
):
    async def event_generator():
        user_id = current_user.user_id
        current_kb_id = kb_id.strip() or "default"
        current_session_id = session_id or f"{user_id}_{current_kb_id}_default"

        state: dict[str, Any] = {
            "user_id": user_id,
            "kb_id": current_kb_id,
            "session_id": current_session_id,
            "question": question,
        }

        try:
            state.update(classify_question(state))

            if state.get("route") == "rag":
                state.update(rewrite_query(state))
                state.update(retrieve_chroma(state))
                state.update(retrieve_bm25(state))
                state.update(merge_results(state))

            if state.get("route") == "tool":
                state.update(tool_node(state))
                state.update(generate_answer(state))

                payload = {
                    "type": "token",
                    "content": state.get("answer", ""),
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            else:
                answer_parts = []

                async for token in stream_llm_answer(
                    question=question,
                    context=state.get("context", ""),
                ):
                    answer_parts.append(token)

                    payload = {
                        "type": "token",
                        "content": token,
                    }

                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                answer = "".join(answer_parts)
                state["answer"] = answer

            state.update(save_message_node(state))

            final_payload = {
                "type": "done",
                "user_id": user_id,
                "kb_id": current_kb_id,
                "session_id": state.get("session_id"),
                "route": state.get("route"),
                "route_reason": state.get("route_reason"),
                "citations": state.get("citations", []),
            }

            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_payload = {
                "type": "error",
                "content": str(e),
            }

            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
