import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _row_to_dict(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = _row_to_dict
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
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
    conn.close()


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
    conn = _connect(db_path)
    job_id = job_id or uuid.uuid4().hex
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO jobs (id, name, folder_name, video_filename, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (job_id, name, folder_name, video_filename, status, created_at),
    )
    conn.commit()
    job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return job


def get_job(db_path: Path, job_id: str) -> dict | None:
    conn = _connect(db_path)
    job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return job


def get_all_jobs(db_path: Path) -> list[dict]:
    conn = _connect(db_path)
    jobs = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    conn.close()
    return jobs


def get_next_pending(db_path: Path) -> dict | None:
    conn = _connect(db_path)
    job = conn.execute(
        "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    conn.close()
    return job


def update_job(db_path: Path, job_id: str, **fields) -> None:
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]
    conn = _connect(db_path)
    conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()


def cancel_job(db_path: Path, job_id: str) -> None:
    update_job(db_path, job_id, status="cancelled")


def delete_job(db_path: Path, job_id: str) -> None:
    conn = _connect(db_path)
    conn.execute("DELETE FROM transcriptions_fts WHERE job_id = ?", (job_id,))
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()


def index_transcription(
    db_path: Path, *, job_id: str, name: str, content: str
) -> None:
    conn = _connect(db_path)
    conn.execute(
        "DELETE FROM transcriptions_fts WHERE job_id = ?", (job_id,)
    )
    conn.execute(
        "INSERT INTO transcriptions_fts (job_id, name, content) VALUES (?, ?, ?)",
        (job_id, name, content),
    )
    conn.commit()
    conn.close()


def search_transcriptions(db_path: Path, query: str) -> list[dict]:
    conn = _connect(db_path)
    conn.row_factory = None
    results = conn.execute(
        """SELECT job_id, name, snippet(transcriptions_fts, 2, '<mark>', '</mark>', '...', 40)
           FROM transcriptions_fts
           WHERE content MATCH ?
           ORDER BY rank""",
        (query,),
    ).fetchall()
    conn.close()
    return [
        {"job_id": r[0], "name": r[1], "snippet": r[2]}
        for r in results
    ]
