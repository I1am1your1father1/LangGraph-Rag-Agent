import sqlite3
import uuid
from datetime import datetime
from typing import Any

from backend.config import DB_PATH


def get_conn() -> sqlite3.Connection:
    """
    Get connection with SQlite
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Init the database
    """
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                source_filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
            )
            """
        )


def create_document(user_id: str, filename: str) -> str:
    document_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, user_id, filename, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, user_id, filename, now),
        )

    return document_id


def create_chunks(
    document_id: str,
    user_id: str,
    filename: str,
    chunk_texts: list[str],
) -> list[dict[str, Any]]:
    now = datetime.utcnow().isoformat()
    rows: list[dict[str, Any]] = []

    with get_conn() as conn:
        for index, content in enumerate(chunk_texts):
            chunk_id = uuid.uuid4().hex

            conn.execute(
                """
                INSERT INTO chunks (
                    id, document_id, user_id, content,
                    source_filename, chunk_index, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    document_id,
                    user_id,
                    content,
                    filename,
                    index,
                    now,
                ),
            )

            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "user_id": user_id,
                    "content": content,
                    "source_filename": filename,
                    "chunk_index": index,
                    "created_at": now,
                }
            )

    return rows


def list_chunks_for_user(user_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, document_id, user_id, content, source_filename, chunk_index
            FROM chunks
            WHERE user_id = ?
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def ensure_session(user_id: str, session_id: str | None = None) -> str:
    if session_id:
        return session_id

    new_session_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (id, user_id, title, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (new_session_id, user_id, "New Chat", now),
        )

    return new_session_id


def save_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    citations: str | None = None,
) -> str:
    message_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                id, session_id, user_id, role, content, citations, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                user_id,
                role,
                content,
                citations,
                now,
            ),
        )

    return message_id


def init_eval_table() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_records (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                ground_truth TEXT,
                retrieved_contexts TEXT NOT NULL,
                citations TEXT,
                latency_ms REAL,
                created_at TEXT NOT NULL
            )
            """
        )


import uuid
import json
from datetime import datetime


def save_eval_record(
    user_id: str,
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    citations: list[dict],
    latency_ms: float | None = None,
    ground_truth: str | None = None,
) -> str:
    record_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO eval_records (
                id, user_id, question, answer, ground_truth,
                retrieved_contexts, citations, latency_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                user_id,
                question,
                answer,
                ground_truth,
                json.dumps(retrieved_contexts, ensure_ascii=False),
                json.dumps(citations, ensure_ascii=False),
                latency_ms,
                now,
            ),
        )

    return record_id
