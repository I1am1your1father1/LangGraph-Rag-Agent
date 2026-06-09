import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from backend.config import DB_PATH


def get_conn() -> sqlite3.Connection:
    """
    获取 SQLite 连接。
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_def: str,
) -> None:
    """
    简单的 SQLite 迁移工具：如果旧表缺少字段，就补充字段。
    这样已有 data/app.db 不需要手动删除也能继续使用。
    """
    columns = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }

    if column_name not in columns:
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"
        )


def init_db() -> None:
    """
    初始化数据库表。
    """
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                kb_id TEXT DEFAULT 'default',
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
                kb_id TEXT DEFAULT 'default',
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
                kb_id TEXT DEFAULT 'default',
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
                kb_id TEXT DEFAULT 'default',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
            )
            """
        )

        # 兼容旧版本数据库。
        _ensure_column(conn, "documents", "kb_id", "TEXT DEFAULT 'default'")
        _ensure_column(conn, "chunks", "kb_id", "TEXT DEFAULT 'default'")
        _ensure_column(conn, "chat_sessions", "kb_id", "TEXT DEFAULT 'default'")
        _ensure_column(conn, "messages", "kb_id", "TEXT DEFAULT 'default'")

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_user_kb
            ON documents(user_id, kb_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_user_kb
            ON chunks(user_id, kb_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_user_kb
            ON chat_sessions(user_id, kb_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_session_user_kb
            ON messages(session_id, user_id, kb_id)
            """
        )


def create_document(user_id: str, filename: str, kb_id: str = "default") -> str:
    document_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, user_id, kb_id, filename, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, user_id, kb_id, filename, now),
        )

    return document_id


def create_chunks(
    document_id: str,
    user_id: str,
    filename: str,
    chunk_texts: list[str],
    kb_id: str = "default",
) -> list[dict[str, Any]]:
    now = datetime.utcnow().isoformat()
    rows: list[dict[str, Any]] = []

    with get_conn() as conn:
        for index, content in enumerate(chunk_texts):
            chunk_id = uuid.uuid4().hex

            conn.execute(
                """
                INSERT INTO chunks (
                    id, document_id, user_id, kb_id, content,
                    source_filename, chunk_index, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    document_id,
                    user_id,
                    kb_id,
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
                    "kb_id": kb_id,
                    "content": content,
                    "source_filename": filename,
                    "chunk_index": index,
                    "created_at": now,
                }
            )

    return rows


def list_chunks_for_user(
    user_id: str,
    kb_id: str = "default",
) -> list[dict[str, Any]]:
    """
    只返回当前 user_id + kb_id 下的切片。
    这是 BM25 检索侧的知识库隔离入口。
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, document_id, user_id, kb_id, content,
                   source_filename, chunk_index
            FROM chunks
            WHERE user_id = ? AND kb_id = ?
            ORDER BY created_at ASC
            """,
            (user_id, kb_id),
        ).fetchall()

    return [dict(row) for row in rows]


def ensure_session(
    user_id: str,
    kb_id: str = "default",
    session_id: str | None = None,
) -> str:
    """
    确保会话存在，并且会话属于当前 user_id + kb_id。
    如果前端传入 session_id 但数据库中还不存在，也会创建该会话。
    """
    now = datetime.utcnow().isoformat()
    target_session_id = session_id or uuid.uuid4().hex

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM chat_sessions
            WHERE id = ? AND user_id = ? AND kb_id = ?
            """,
            (target_session_id, user_id, kb_id),
        ).fetchone()

        if row is None:
            conn.execute(
                """
                INSERT INTO chat_sessions (id, user_id, kb_id, title, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (target_session_id, user_id, kb_id, "New Chat", now),
            )

    return target_session_id


def save_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    citations: str | None = None,
    kb_id: str = "default",
) -> str:
    message_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                id, session_id, user_id, kb_id, role, content, citations, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                user_id,
                kb_id,
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
                kb_id TEXT DEFAULT 'default',
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

        _ensure_column(conn, "eval_records", "kb_id", "TEXT DEFAULT 'default'")

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_eval_records_user_kb
            ON eval_records(user_id, kb_id)
            """
        )


def save_eval_record(
    user_id: str,
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    citations: list[dict],
    latency_ms: float | None = None,
    ground_truth: str | None = None,
    kb_id: str = "default",
) -> str:
    record_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO eval_records (
                id, user_id, kb_id, question, answer, ground_truth,
                retrieved_contexts, citations, latency_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                user_id,
                kb_id,
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
