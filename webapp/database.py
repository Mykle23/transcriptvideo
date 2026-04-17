import html
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_MUTABLE_COLUMNS = frozenset({
    "name", "status", "progress", "started_at", "completed_at",
    "audio_duration", "language", "error_message", "keep_video",
    "video_filename",
})


def _row_to_dict(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


@contextmanager
def _get_conn(db_path: Path, row_factory=_row_to_dict):
    conn = sqlite3.connect(db_path)
    conn.row_factory = row_factory
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=3000")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    with _get_conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                folder_name TEXT NOT NULL UNIQUE,
                video_filename TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                progress REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                audio_duration REAL,
                language TEXT,
                error_message TEXT,
                keep_video INTEGER DEFAULT 1
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS transcriptions_fts USING fts5(
                job_id,
                name,
                content,
                tokenize='unicode61'
            );
        """)


def create_job(
    db_path: Path,
    *,
    name: str,
    folder_name: str,
    video_filename: str,
    status: str = "pending",
    job_id: str | None = None,
    created_at: str | None = None,
) -> dict:
    with _get_conn(db_path) as conn:
        job_id = job_id or uuid.uuid4().hex
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO jobs (id, name, folder_name, video_filename, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (job_id, name, folder_name, video_filename, status, created_at),
        )
        conn.commit()
        return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def get_job(db_path: Path, job_id: str) -> dict | None:
    with _get_conn(db_path) as conn:
        return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def get_all_jobs(db_path: Path) -> list[dict]:
    with _get_conn(db_path) as conn:
        return conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()


def get_next_pending(db_path: Path) -> dict | None:
    with _get_conn(db_path) as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()


def update_job(db_path: Path, job_id: str, **fields) -> None:
    if not fields:
        return
    invalid = set(fields) - _MUTABLE_COLUMNS
    if invalid:
        raise ValueError(f"Non-updatable columns: {invalid}")
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]
    with _get_conn(db_path) as conn:
        conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)
        conn.commit()


def cancel_job(db_path: Path, job_id: str) -> None:
    update_job(db_path, job_id, status="cancelled")


def delete_job(db_path: Path, job_id: str) -> None:
    with _get_conn(db_path) as conn:
        conn.execute("DELETE FROM transcriptions_fts WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()


def index_transcription(
    db_path: Path, *, job_id: str, name: str, content: str
) -> None:
    escaped_content = html.escape(content)
    with _get_conn(db_path) as conn:
        conn.execute(
            "DELETE FROM transcriptions_fts WHERE job_id = ?", (job_id,)
        )
        conn.execute(
            "INSERT INTO transcriptions_fts (job_id, name, content) VALUES (?, ?, ?)",
            (job_id, name, escaped_content),
        )
        conn.commit()


def search_transcriptions(db_path: Path, query: str) -> list[dict]:
    with _get_conn(db_path, row_factory=None) as conn:
        try:
            results = conn.execute(
                """SELECT job_id, name, snippet(transcriptions_fts, 2, '<mark>', '</mark>', '...', 40)
                   FROM transcriptions_fts
                   WHERE content MATCH ?
                   ORDER BY rank""",
                (query,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [
        {"job_id": r[0], "name": r[1], "snippet": r[2]}
        for r in results
    ]
