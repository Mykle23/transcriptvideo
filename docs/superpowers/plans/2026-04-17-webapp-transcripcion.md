# TranscriptVideo Webapp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web interface for the existing video transcription tool so the user can upload, queue, monitor, and search transcriptions from any device on their Tailscale VPN.

**Architecture:** FastAPI backend (same WSL venv) serves both the REST API and the HTML frontend. SQLite stores job metadata + full-text search index. A background worker thread loads the Whisper model once and processes jobs FIFO. SSE streams live progress to the browser. The frontend is a single-page Jinja2 template with Alpine.js for reactivity and Tailwind CSS (CDN) for styling — no Node.js, no build step.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, SQLite (FTS5), faster-whisper (existing), Jinja2, Alpine.js (CDN), Tailwind CSS (CDN)

---

## File Structure

```
transcriptvideo/
├── transcribe.py                 # existing CLI — unchanged, still works standalone
├── videos/                       # uploaded + existing videos
├── transcripciones/              # one subfolder per job
├── venv/                         # existing WSL Python venv
├── webapp/
│   ├── __init__.py
│   ├── app.py                    # FastAPI app: routes, SSE, lifespan
│   ├── config.py                 # paths, constants
│   ├── database.py               # SQLite schema + all queries
│   ├── transcriber.py            # transcription logic extracted from transcribe.py
│   ├── worker.py                 # background job processor thread
│   ├── templates/
│   │   └── index.html            # single-page UI
│   └── static/
│       └── app.js                # SSE, upload progress, search, fetch helpers
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # shared fixtures (tmp DB, test client)
│   ├── test_database.py
│   ├── test_transcriber.py
│   └── test_api.py
├── start-webapp.bat              # Windows desktop launcher
└── docs/
```

Key decisions:
- **No Node.js** — Alpine.js + Tailwind via CDN. Zero build step.
- **UUID folder names** for new transcriptions (`transcripciones/<uuid>/`). Avoids filesystem naming issues.
- **Existing transcriptions** imported on first startup — scanned from `transcripciones/`, added to DB with folder name as display name.
- **Whisper model loaded once** at startup in the worker thread, reused across jobs.
- **SSE polls DB** every second for active job progress — simple, reliable, no async/thread coordination complexity.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `webapp/__init__.py`
- Create: `webapp/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Install Python dependencies in WSL venv**

Run inside WSL:
```bash
cd /mnt/c/Development/transcriptvideo
source venv/bin/activate
pip install fastapi uvicorn[standard] python-multipart aiofiles sse-starlette jinja2 pytest httpx
```

- [ ] **Step 2: Create webapp package**

`webapp/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 3: Create config module**

`webapp/config.py`:
```python
from pathlib import Path

BASE_DIR = Path("/mnt/c/Development/transcriptvideo")
VIDEOS_DIR = BASE_DIR / "videos"
TRANSCRIPTIONS_DIR = BASE_DIR / "transcripciones"
DB_PATH = BASE_DIR / "webapp" / "transcriptvideo.db"

MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

ALLOWED_EXTENSIONS = {".mp4"}
```

- [ ] **Step 4: Create test conftest with shared fixtures**

`tests/conftest.py`:
```python
import tempfile
from pathlib import Path

import pytest

from webapp.config import TRANSCRIPTIONS_DIR, VIDEOS_DIR


@pytest.fixture
def tmp_dirs(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    transcriptions = tmp_path / "transcripciones"
    transcriptions.mkdir()
    return {"videos": videos, "transcriptions": transcriptions, "base": tmp_path}


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test.db"
```

- [ ] **Step 5: Verify structure**

Run:
```bash
cd /mnt/c/Development/transcriptvideo
source venv/bin/activate
python -c "from webapp.config import BASE_DIR; print(BASE_DIR)"
```
Expected: `/mnt/c/Development/transcriptvideo`

- [ ] **Step 6: Commit**

```bash
git init
git add webapp/__init__.py webapp/config.py tests/__init__.py tests/conftest.py
git commit -m "feat: project scaffolding for webapp"
```

---

## Task 2: Database Layer

**Files:**
- Create: `webapp/database.py`
- Create: `tests/test_database.py`

- [ ] **Step 1: Write failing tests for database operations**

`tests/test_database.py`:
```python
import sqlite3
from datetime import datetime, timezone

from webapp.database import (
    cancel_job,
    create_job,
    get_all_jobs,
    get_job,
    index_transcription,
    init_db,
    search_transcriptions,
    update_job,
)


def test_init_db_creates_tables(tmp_db):
    init_db(tmp_db)
    conn = sqlite3.connect(tmp_db)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {t[0] for t in tables}
    assert "jobs" in table_names
    assert "transcriptions_fts" in table_names
    conn.close()


def test_create_and_get_job(tmp_db):
    init_db(tmp_db)
    job = create_job(
        tmp_db,
        name="Test Video",
        folder_name="abc-123",
        video_filename="recording.mp4",
    )
    assert job["name"] == "Test Video"
    assert job["status"] == "pending"
    assert job["folder_name"] == "abc-123"

    fetched = get_job(tmp_db, job["id"])
    assert fetched["name"] == "Test Video"


def test_get_all_jobs_ordered(tmp_db):
    init_db(tmp_db)
    create_job(tmp_db, name="First", folder_name="f1", video_filename="a.mp4")
    create_job(tmp_db, name="Second", folder_name="f2", video_filename="b.mp4")
    jobs = get_all_jobs(tmp_db)
    assert len(jobs) == 2
    assert jobs[0]["name"] == "Second"  # newest first


def test_update_job(tmp_db):
    init_db(tmp_db)
    job = create_job(tmp_db, name="Old Name", folder_name="f1", video_filename="a.mp4")
    update_job(tmp_db, job["id"], name="New Name", progress=50.0, status="processing")
    updated = get_job(tmp_db, job["id"])
    assert updated["name"] == "New Name"
    assert updated["progress"] == 50.0
    assert updated["status"] == "processing"


def test_cancel_job(tmp_db):
    init_db(tmp_db)
    job = create_job(tmp_db, name="Cancel Me", folder_name="f1", video_filename="a.mp4")
    cancel_job(tmp_db, job["id"])
    cancelled = get_job(tmp_db, job["id"])
    assert cancelled["status"] == "cancelled"


def test_search_transcriptions(tmp_db):
    init_db(tmp_db)
    job = create_job(tmp_db, name="Meeting", folder_name="f1", video_filename="a.mp4")
    index_transcription(
        tmp_db,
        job_id=job["id"],
        name="Meeting",
        content="We discussed the quarterly budget and revenue targets.",
    )
    results = search_transcriptions(tmp_db, "budget")
    assert len(results) == 1
    assert results[0]["job_id"] == job["id"]
    assert "budget" in results[0]["snippet"].lower()


def test_search_no_results(tmp_db):
    init_db(tmp_db)
    results = search_transcriptions(tmp_db, "nonexistent")
    assert len(results) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /mnt/c/Development/transcriptvideo
source venv/bin/activate
pytest tests/test_database.py -v
```
Expected: FAIL — `cannot import name ... from 'webapp.database'`

- [ ] **Step 3: Implement database module**

`webapp/database.py`:
```python
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
    conn.row_factory = None  # We'll build dicts manually for snippet
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_database.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add webapp/database.py tests/test_database.py
git commit -m "feat: SQLite database layer with FTS5 search"
```

---

## Task 3: Transcription Core Extraction

**Files:**
- Create: `webapp/transcriber.py`
- Create: `tests/test_transcriber.py`
- Note: `transcribe.py` remains unchanged — the CLI still works standalone

- [ ] **Step 1: Write failing tests for post-processing logic**

`tests/test_transcriber.py`:
```python
from webapp.transcriber import Segment, clean_segments


def test_collapses_repeated_words():
    seg = Segment(0.0, 2.0, "no no no no no no no no")
    result, removed = clean_segments([seg])
    assert len(result) == 1
    assert result[0].text == "No."
    assert removed == 1


def test_keeps_normal_segment():
    seg = Segment(0.0, 5.0, "Today we discussed the quarterly budget and revenue targets.")
    result, removed = clean_segments([seg])
    assert len(result) == 1
    assert result[0].text == seg.text
    assert removed == 0


def test_merges_consecutive_identical_short_segments():
    segs = [
        Segment(0.0, 0.5, "Si."),
        Segment(0.5, 1.0, "Si."),
        Segment(1.0, 1.5, "Si."),
        Segment(1.5, 2.0, "Si."),
        Segment(2.0, 2.5, "Si."),
    ]
    result, removed = clean_segments(segs)
    assert len(result) == 1
    assert result[0].start == 0.0
    assert result[0].end == 2.5
    assert removed == 4


def test_does_not_merge_fewer_than_four():
    segs = [
        Segment(0.0, 0.5, "Si."),
        Segment(0.5, 1.0, "Si."),
        Segment(1.0, 1.5, "Si."),
    ]
    result, removed = clean_segments(segs)
    assert len(result) == 3
    assert removed == 0


def test_format_timestamp():
    from webapp.transcriber import format_timestamp

    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(3661.5) == "01:01:01,500"
    assert format_timestamp(59.999) == "00:00:59,999"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_transcriber.py -v
```
Expected: FAIL — `cannot import name 'Segment' from 'webapp.transcriber'`

- [ ] **Step 3: Implement transcriber module**

`webapp/transcriber.py`:
```python
"""Core transcription logic extracted from transcribe.py.

The original transcribe.py CLI remains unchanged and functional.
This module provides the same logic as an importable library with
progress callback support for the webapp worker.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Segment:
    start: float
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _clean_intra_segment_repetition(text: str) -> str:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 6:
        return text
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.25:
        return words[0].capitalize() + "."
    return text


def clean_segments(segments: list[Segment]) -> tuple[list[Segment], int]:
    """Remove hallucinated repetitions. Returns (cleaned_segments, removed_count)."""
    cleaned: list[Segment] = []
    removed = 0

    for seg in segments:
        new_text = _clean_intra_segment_repetition(seg.text)
        if new_text != seg.text:
            removed += 1
        cleaned.append(Segment(seg.start, seg.end, new_text))

    result: list[Segment] = []
    i = 0
    while i < len(cleaned):
        seg = cleaned[i]
        norm = re.sub(r"[^\w]", "", seg.text.lower())
        j = i + 1
        while j < len(cleaned):
            norm_j = re.sub(r"[^\w]", "", cleaned[j].text.lower())
            if (
                norm_j == norm
                and (cleaned[j].end - cleaned[j].start) <= 1.5
                and len(norm) <= 10
            ):
                j += 1
            else:
                break
        run_length = j - i
        if run_length >= 4:
            result.append(Segment(seg.start, cleaned[j - 1].end, seg.text))
            removed += run_length - 1
            i = j
        else:
            result.append(seg)
            i += 1

    return result, removed


def transcribe_video(
    model,
    video_path: Path,
    output_dir: Path,
    on_progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Run transcription and save .txt output.

    Args:
        model: A loaded WhisperModel instance.
        video_path: Path to the .mp4 file.
        output_dir: Directory to write transcripcion.txt into.
        on_progress: Callback(percent, current_segment_text) called per segment.
        should_cancel: Callback that returns True if the job should be cancelled.

    Returns:
        dict with keys: language, audio_duration, segment_count, removed_count
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    whisper_segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=400,
        ),
    )

    raw_segments: list[Segment] = []
    for segment in whisper_segments:
        if should_cancel and should_cancel():
            return {"cancelled": True}

        raw_segments.append(Segment(segment.start, segment.end, segment.text.strip()))

        if on_progress and info.duration > 0:
            pct = segment.end / info.duration * 100
            text_preview = segment.text.strip()[:80]
            on_progress(min(pct, 99.9), text_preview)

    cleaned, removed = clean_segments(raw_segments)

    txt_lines: list[str] = []
    for seg in cleaned:
        start_ts = format_timestamp(seg.start)
        end_ts = format_timestamp(seg.end)
        txt_lines.append(f"[{start_ts} --> {end_ts}] {seg.text}")

    txt_path = output_dir / "transcripcion.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    if on_progress:
        on_progress(100.0, "")

    return {
        "cancelled": False,
        "language": info.language,
        "audio_duration": info.duration,
        "segment_count": len(cleaned),
        "removed_count": removed,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_transcriber.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add webapp/transcriber.py tests/test_transcriber.py
git commit -m "feat: extract transcription core into importable module"
```

---

## Task 4: Background Worker

**Files:**
- Create: `webapp/worker.py`

- [ ] **Step 1: Implement the worker**

`webapp/worker.py`:
```python
"""Background worker thread that processes transcription jobs FIFO.

Loads the Whisper model once at startup. Polls the DB for pending jobs.
Updates progress in DB so the SSE endpoint can read it.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from webapp.config import (
    COMPUTE_TYPE,
    DEVICE,
    MODEL_SIZE,
    TRANSCRIPTIONS_DIR,
    VIDEOS_DIR,
)
from webapp.database import (
    get_job,
    get_next_pending,
    index_transcription,
    update_job,
)
from webapp.transcriber import transcribe_video

logger = logging.getLogger(__name__)


class TranscriptionWorker:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.model = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._current_job_id: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def cancel_current(self, job_id: str) -> bool:
        return self._current_job_id == job_id

    def _load_model(self):
        if self.model is not None:
            return
        logger.info("Loading Whisper model '%s' on %s (%s)...", MODEL_SIZE, DEVICE, COMPUTE_TYPE)
        from faster_whisper import WhisperModel

        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        logger.info("Model loaded.")

    def _run(self) -> None:
        self._load_model()
        while not self._stop_event.is_set():
            job = get_next_pending(self.db_path)
            if job is None:
                self._stop_event.wait(timeout=2)
                continue
            self._process_job(job)

    def _process_job(self, job: dict) -> None:
        job_id = job["id"]
        self._current_job_id = job_id
        now = datetime.now(timezone.utc).isoformat()
        update_job(self.db_path, job_id, status="processing", started_at=now)

        video_path = VIDEOS_DIR / job["video_filename"]
        output_dir = TRANSCRIPTIONS_DIR / job["folder_name"]

        if not video_path.exists():
            update_job(
                self.db_path,
                job_id,
                status="failed",
                error_message=f"Video file not found: {video_path.name}",
            )
            self._current_job_id = None
            return

        def on_progress(pct: float, text: str) -> None:
            update_job(self.db_path, job_id, progress=round(pct, 1))

        def should_cancel() -> bool:
            if self._stop_event.is_set():
                return True
            current = get_job(self.db_path, job_id)
            return current is not None and current["status"] == "cancelled"

        try:
            result = transcribe_video(
                model=self.model,
                video_path=video_path,
                output_dir=output_dir,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )

            if result.get("cancelled"):
                logger.info("Job %s cancelled.", job_id)
                self._current_job_id = None
                return

            completed_at = datetime.now(timezone.utc).isoformat()
            update_job(
                self.db_path,
                job_id,
                status="completed",
                progress=100.0,
                completed_at=completed_at,
                audio_duration=result["audio_duration"],
                language=result["language"],
            )

            # Index for full-text search
            txt_path = output_dir / "transcripcion.txt"
            if txt_path.exists():
                content = txt_path.read_text(encoding="utf-8")
                index_transcription(
                    self.db_path,
                    job_id=job_id,
                    name=job["name"],
                    content=content,
                )

            # Delete video if user chose not to keep it
            current = get_job(self.db_path, job_id)
            if current and not current["keep_video"]:
                video_path.unlink(missing_ok=True)

            logger.info("Job %s completed.", job_id)

        except Exception:
            logger.exception("Job %s failed.", job_id)
            update_job(
                self.db_path,
                job_id,
                status="failed",
                error_message="Transcription failed unexpectedly. Check server logs.",
            )
        finally:
            self._current_job_id = None
```

- [ ] **Step 2: Verify imports work**

Run:
```bash
cd /mnt/c/Development/transcriptvideo
source venv/bin/activate
python -c "from webapp.worker import TranscriptionWorker; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add webapp/worker.py
git commit -m "feat: background worker thread for FIFO transcription queue"
```

---

## Task 5: API — Job Management & File Upload

**Files:**
- Create: `webapp/app.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing tests for job API endpoints**

`tests/test_api.py`:
```python
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(tmp_dirs, tmp_db):
    """Create a test client with patched paths and no real worker."""
    with (
        patch("webapp.config.VIDEOS_DIR", tmp_dirs["videos"]),
        patch("webapp.config.TRANSCRIPTIONS_DIR", tmp_dirs["transcriptions"]),
        patch("webapp.config.DB_PATH", tmp_db),
        patch("webapp.app.DB_PATH", tmp_db),
        patch("webapp.app.VIDEOS_DIR", tmp_dirs["videos"]),
        patch("webapp.app.TRANSCRIPTIONS_DIR", tmp_dirs["transcriptions"]),
    ):
        from webapp.database import init_db

        init_db(tmp_db)

        from webapp.app import app

        # Disable worker for tests
        app.state.worker = MagicMock()

        client = TestClient(app)
        yield client


def test_upload_creates_job(app_client):
    fake_mp4 = io.BytesIO(b"fake video content")
    response = app_client.post(
        "/api/jobs",
        data={"name": "Test Recording"},
        files={"file": ("test.mp4", fake_mp4, "video/mp4")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Recording"
    assert data["status"] == "pending"
    assert data["video_filename"] is not None


def test_upload_rejects_non_mp4(app_client):
    fake_file = io.BytesIO(b"not a video")
    response = app_client.post(
        "/api/jobs",
        data={"name": "Bad File"},
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400


def test_list_jobs(app_client):
    fake_mp4 = io.BytesIO(b"fake")
    app_client.post(
        "/api/jobs",
        data={"name": "Job 1"},
        files={"file": ("a.mp4", fake_mp4, "video/mp4")},
    )
    response = app_client.get("/api/jobs")
    assert response.status_code == 200
    jobs = response.json()
    assert len(jobs) == 1
    assert jobs[0]["name"] == "Job 1"


def test_get_job(app_client):
    fake_mp4 = io.BytesIO(b"fake")
    create_resp = app_client.post(
        "/api/jobs",
        data={"name": "My Job"},
        files={"file": ("a.mp4", fake_mp4, "video/mp4")},
    )
    job_id = create_resp.json()["id"]
    response = app_client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "My Job"


def test_get_job_not_found(app_client):
    response = app_client.get("/api/jobs/nonexistent")
    assert response.status_code == 404


def test_update_job_name(app_client):
    fake_mp4 = io.BytesIO(b"fake")
    create_resp = app_client.post(
        "/api/jobs",
        data={"name": "Old Name"},
        files={"file": ("a.mp4", fake_mp4, "video/mp4")},
    )
    job_id = create_resp.json()["id"]
    response = app_client.patch(
        f"/api/jobs/{job_id}",
        json={"name": "New Name"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "New Name"


def test_cancel_pending_job(app_client):
    fake_mp4 = io.BytesIO(b"fake")
    create_resp = app_client.post(
        "/api/jobs",
        data={"name": "Cancel Me"},
        files={"file": ("a.mp4", fake_mp4, "video/mp4")},
    )
    job_id = create_resp.json()["id"]
    response = app_client.post(f"/api/jobs/{job_id}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_api.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp.app'`

- [ ] **Step 3: Implement the FastAPI app with job endpoints**

`webapp/app.py`:
```python
import json
import logging
import os
import signal
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from webapp.config import (
    ALLOWED_EXTENSIONS,
    DB_PATH,
    TRANSCRIPTIONS_DIR,
    VIDEOS_DIR,
)
from webapp.database import (
    cancel_job,
    create_job,
    delete_job,
    get_all_jobs,
    get_job,
    get_next_pending,
    index_transcription,
    init_db,
    search_transcriptions,
    update_job,
)
from webapp.worker import TranscriptionWorker

logger = logging.getLogger(__name__)

WEBAPP_DIR = Path(__file__).parent
TEMPLATES_DIR = WEBAPP_DIR / "templates"
STATIC_DIR = WEBAPP_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db(DB_PATH)
    _import_existing_transcriptions()
    worker = TranscriptionWorker(DB_PATH)
    worker.start()
    app.state.worker = worker
    logger.info("Webapp started. Worker running.")
    yield
    # Shutdown
    worker.stop()
    logger.info("Webapp stopped.")


app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _import_existing_transcriptions() -> None:
    """Scan transcripciones/ for folders not yet in the DB and import them."""
    if not TRANSCRIPTIONS_DIR.exists():
        return
    existing_jobs = get_all_jobs(DB_PATH)
    known_folders = {j["folder_name"] for j in existing_jobs}

    for folder in sorted(TRANSCRIPTIONS_DIR.iterdir()):
        if not folder.is_dir() or folder.name in known_folders:
            continue
        txt_path = folder / "transcripcion.txt"
        if not txt_path.exists():
            continue

        job_id = uuid.uuid4().hex
        job = create_job(
            DB_PATH,
            name=folder.name,
            folder_name=folder.name,
            video_filename="",
            status="completed",
            job_id=job_id,
            created_at=datetime.fromtimestamp(
                folder.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        )
        update_job(DB_PATH, job_id, progress=100.0, completed_at=job["created_at"])

        content = txt_path.read_text(encoding="utf-8")
        index_transcription(DB_PATH, job_id=job_id, name=folder.name, content=content)

    logger.info("Existing transcriptions imported.")


# ── HTML Page ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── Job CRUD ───────────────────────────────────────────────

@app.post("/api/jobs", status_code=201)
async def create_job_endpoint(
    name: str = Form(...),
    file: UploadFile = File(...),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Only {', '.join(ALLOWED_EXTENSIONS)} files allowed.")

    job_id = uuid.uuid4().hex
    video_filename = f"{job_id}{ext}"
    video_path = VIDEOS_DIR / video_filename

    async with aiofiles.open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            await f.write(chunk)

    job = create_job(
        DB_PATH,
        name=name.strip(),
        folder_name=job_id,
        video_filename=video_filename,
        job_id=job_id,
    )
    return job


@app.get("/api/jobs")
def list_jobs():
    return get_all_jobs(DB_PATH)


@app.get("/api/jobs/{job_id}")
def get_job_endpoint(job_id: str):
    job = get_job(DB_PATH, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@app.patch("/api/jobs/{job_id}")
def update_job_endpoint(job_id: str, body: dict):
    job = get_job(DB_PATH, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    allowed_fields = {"name", "keep_video"}
    updates = {k: v for k, v in body.items() if k in allowed_fields}
    if not updates:
        raise HTTPException(400, "No valid fields to update.")

    update_job(DB_PATH, job_id, **updates)

    # If name changed and job is completed, update FTS index
    if "name" in updates and job["status"] == "completed":
        txt_path = TRANSCRIPTIONS_DIR / job["folder_name"] / "transcripcion.txt"
        if txt_path.exists():
            content = txt_path.read_text(encoding="utf-8")
            index_transcription(
                DB_PATH, job_id=job_id, name=updates["name"], content=content
            )

    return get_job(DB_PATH, job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job_endpoint(job_id: str):
    job = get_job(DB_PATH, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] not in ("pending", "processing"):
        raise HTTPException(400, f"Cannot cancel job with status '{job['status']}'.")
    cancel_job(DB_PATH, job_id)
    return get_job(DB_PATH, job_id)


@app.delete("/api/jobs/{job_id}")
def delete_job_endpoint(job_id: str):
    job = get_job(DB_PATH, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] == "processing":
        raise HTTPException(400, "Cannot delete a job that is currently processing.")

    # Delete video file
    if job["video_filename"]:
        video_path = VIDEOS_DIR / job["video_filename"]
        video_path.unlink(missing_ok=True)

    # Delete transcription folder
    folder = TRANSCRIPTIONS_DIR / job["folder_name"]
    if folder.exists():
        import shutil
        shutil.rmtree(folder)

    delete_job(DB_PATH, job_id)
    return {"ok": True}


# ── Transcription & Download ──────────────────────────────

@app.get("/api/jobs/{job_id}/transcription")
def get_transcription(job_id: str):
    job = get_job(DB_PATH, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] != "completed":
        raise HTTPException(400, "Transcription not available yet.")

    txt_path = TRANSCRIPTIONS_DIR / job["folder_name"] / "transcripcion.txt"
    if not txt_path.exists():
        raise HTTPException(404, "Transcription file not found.")

    content = txt_path.read_text(encoding="utf-8")
    return {"content": content}


@app.get("/api/jobs/{job_id}/download")
def download_transcription(job_id: str):
    job = get_job(DB_PATH, job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    txt_path = TRANSCRIPTIONS_DIR / job["folder_name"] / "transcripcion.txt"
    if not txt_path.exists():
        raise HTTPException(404, "Transcription file not found.")

    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in job["name"])
    return FileResponse(
        txt_path,
        filename=f"{safe_name}.txt",
        media_type="text/plain",
    )


# ── Search ─────────────────────────────────────────────────

@app.get("/api/search")
def search(q: str = ""):
    if not q.strip():
        return []
    return search_transcriptions(DB_PATH, q.strip())


# ── SSE Progress ───────────────────────────────────────────

@app.get("/api/progress")
async def progress_stream(request: Request):
    async def generate():
        while True:
            if await request.is_disconnected():
                break
            jobs = get_all_jobs(DB_PATH)
            active = [
                j for j in jobs if j["status"] in ("pending", "processing")
            ]
            yield json.dumps(active)
            await asyncio.sleep(1)

    import asyncio

    return EventSourceResponse(generate())


# ── Shutdown ───────────────────────────────────────────────

@app.post("/api/shutdown")
def shutdown():
    """Gracefully stop the server."""
    logger.info("Shutdown requested via API.")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"ok": True}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_api.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 5: Fix any import/patch issues and re-run until green**

Common fix: the `lifespan` starts the real worker which tries to load the GPU model. The test fixture needs to prevent this. If tests fail because of lifespan, update `conftest.py` to add:

```python
@pytest.fixture(autouse=True)
def _no_lifespan():
    """Prevent lifespan from starting the real worker in tests."""
    pass
```

Or adjust the `app_client` fixture to properly mock the lifespan. The TestClient from FastAPI automatically runs lifespan events, so the mock patches on `DB_PATH` in `webapp.app` must be in place before the client is created.

- [ ] **Step 6: Commit**

```bash
git add webapp/app.py tests/test_api.py
git commit -m "feat: FastAPI app with job CRUD, upload, SSE, search, shutdown"
```

---

## Task 6: Frontend — Single-Page UI

**Files:**
- Create: `webapp/templates/index.html`
- Create: `webapp/static/app.js`

- [ ] **Step 1: Create the HTML template**

`webapp/templates/index.html`:
```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TranscriptVideo</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: { brand: '#6366f1' }
                }
            }
        }
    </script>
    <style>
        [x-cloak] { display: none !important; }
        mark { background-color: #fde68a; padding: 0 2px; border-radius: 2px; }
    </style>
</head>
<body class="bg-gray-50 text-gray-900 min-h-screen">

<div x-data="appData()" x-init="init()" x-cloak>

    <!-- Header -->
    <header class="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div class="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
            <h1 class="text-lg font-semibold text-gray-800">TranscriptVideo</h1>
            <div class="flex items-center gap-3">
                <button @click="searchOpen = true"
                    class="text-sm text-gray-500 hover:text-gray-800 flex items-center gap-1">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                    </svg>
                    Buscar
                </button>
                <button @click="if(confirm('Apagar el servidor?')) shutdown()"
                    class="text-sm text-red-500 hover:text-red-700">
                    Apagar
                </button>
            </div>
        </div>
    </header>

    <main class="max-w-4xl mx-auto px-4 py-6 space-y-8">

        <!-- Upload Section -->
        <section class="bg-white rounded-lg border border-gray-200 p-5">
            <h2 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-4">Subir video</h2>
            <form @submit.prevent="upload()" class="space-y-3">
                <div>
                    <input type="text" x-model="uploadName" placeholder="Nombre de la transcripcion"
                        class="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand focus:border-transparent"
                        required>
                </div>
                <div class="flex items-center gap-3">
                    <label class="flex-1 relative">
                        <input type="file" accept=".mp4" @change="uploadFile = $event.target.files[0]"
                            class="block w-full text-sm text-gray-500 file:mr-3 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-medium file:bg-gray-100 file:text-gray-700 hover:file:bg-gray-200 cursor-pointer">
                    </label>
                    <button type="submit" :disabled="!uploadFile || !uploadName.trim() || uploading"
                        class="px-4 py-2 bg-brand text-white text-sm font-medium rounded hover:bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed">
                        <span x-show="!uploading">Subir</span>
                        <span x-show="uploading" x-text="'Subiendo ' + uploadProgress + '%'"></span>
                    </button>
                </div>
                <div x-show="uploading" class="w-full bg-gray-200 rounded-full h-1.5">
                    <div class="bg-brand h-1.5 rounded-full transition-all duration-300"
                         :style="'width:' + uploadProgress + '%'"></div>
                </div>
            </form>
        </section>

        <!-- Active Queue -->
        <section x-show="activeJobs.length > 0">
            <h2 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-3">En proceso</h2>
            <div class="space-y-3">
                <template x-for="job in activeJobs" :key="job.id">
                    <div class="bg-white rounded-lg border border-gray-200 p-4">
                        <div class="flex items-center justify-between mb-2">
                            <span class="font-medium text-sm" x-text="job.name"></span>
                            <div class="flex items-center gap-2">
                                <span class="text-xs px-2 py-0.5 rounded-full"
                                    :class="job.status === 'processing' ? 'bg-blue-100 text-blue-700' : 'bg-yellow-100 text-yellow-700'"
                                    x-text="job.status === 'processing' ? 'Procesando' : 'En cola'"></span>
                                <button @click="cancelJob(job.id)"
                                    class="text-xs text-red-500 hover:text-red-700">Cancelar</button>
                            </div>
                        </div>
                        <div x-show="job.status === 'processing'" class="w-full bg-gray-200 rounded-full h-1.5">
                            <div class="bg-blue-500 h-1.5 rounded-full transition-all duration-500"
                                 :style="'width:' + job.progress + '%'"></div>
                        </div>
                        <div x-show="job.status === 'processing'" class="text-xs text-gray-400 mt-1"
                             x-text="Math.round(job.progress) + '%'"></div>
                    </div>
                </template>
            </div>
        </section>

        <!-- Completed -->
        <section>
            <h2 class="text-sm font-medium text-gray-500 uppercase tracking-wide mb-3">Transcripciones</h2>
            <div x-show="completedJobs.length === 0" class="text-sm text-gray-400 italic">
                No hay transcripciones todavia.
            </div>
            <div class="space-y-2">
                <template x-for="job in completedJobs" :key="job.id">
                    <div class="bg-white rounded-lg border border-gray-200 p-4 cursor-pointer hover:border-gray-300 transition"
                         @click="openTranscription(job)">
                        <div class="flex items-center justify-between">
                            <div>
                                <span class="font-medium text-sm" x-text="job.name"></span>
                                <div class="text-xs text-gray-400 mt-0.5">
                                    <span x-text="formatDate(job.completed_at || job.created_at)"></span>
                                    <span x-show="job.language" class="ml-2" x-text="job.language?.toUpperCase()"></span>
                                    <span x-show="job.audio_duration" class="ml-2" x-text="formatDuration(job.audio_duration)"></span>
                                </div>
                            </div>
                            <div class="flex items-center gap-2">
                                <span x-show="job.status === 'failed'"
                                    class="text-xs px-2 py-0.5 rounded-full bg-red-100 text-red-700">Error</span>
                                <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                                </svg>
                            </div>
                        </div>
                    </div>
                </template>
            </div>
        </section>
    </main>

    <!-- Transcription Detail Modal -->
    <div x-show="detailJob" x-cloak
         class="fixed inset-0 bg-black/40 z-20 flex items-start justify-center pt-12 px-4"
         @click.self="detailJob = null; detailContent = ''">
        <div class="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[80vh] flex flex-col" @click.stop>
            <!-- Modal header -->
            <div class="flex items-center justify-between p-4 border-b border-gray-200">
                <div class="flex items-center gap-2 flex-1 min-w-0">
                    <template x-if="!editingName">
                        <h3 class="font-medium text-sm truncate cursor-pointer hover:text-brand"
                            @click="editingName = true; editNameVal = detailJob.name"
                            x-text="detailJob?.name"></h3>
                    </template>
                    <template x-if="editingName">
                        <form @submit.prevent="saveName()" class="flex items-center gap-2 flex-1">
                            <input x-model="editNameVal" x-ref="nameInput"
                                class="flex-1 border border-gray-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-brand"
                                @keydown.escape="editingName = false">
                            <button type="submit" class="text-xs text-brand hover:underline">Guardar</button>
                            <button type="button" @click="editingName = false" class="text-xs text-gray-400">Cancelar</button>
                        </form>
                    </template>
                </div>
                <div class="flex items-center gap-2 ml-3">
                    <a :href="'/api/jobs/' + detailJob?.id + '/download'"
                       class="text-xs text-brand hover:underline">Descargar</a>
                    <select x-show="detailJob?.video_filename"
                        :value="detailJob?.keep_video ? '1' : '0'"
                        @change="toggleKeepVideo(detailJob.id, $event.target.value === '1')"
                        class="text-xs border border-gray-300 rounded px-1 py-0.5">
                        <option value="1">Conservar video</option>
                        <option value="0">Borrar video</option>
                    </select>
                    <button @click="if(confirm('Eliminar esta transcripcion?')) deleteJob(detailJob.id)"
                        class="text-xs text-red-500 hover:text-red-700">Eliminar</button>
                    <button @click="detailJob = null; detailContent = ''"
                        class="text-gray-400 hover:text-gray-600 ml-1">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                        </svg>
                    </button>
                </div>
            </div>
            <!-- Modal body -->
            <div class="p-4 overflow-y-auto flex-1">
                <div x-show="!detailContent" class="text-sm text-gray-400">Cargando...</div>
                <pre class="text-sm leading-relaxed whitespace-pre-wrap font-mono text-gray-700"
                     x-text="detailContent"></pre>
            </div>
        </div>
    </div>

    <!-- Search Modal -->
    <div x-show="searchOpen" x-cloak
         class="fixed inset-0 bg-black/40 z-20 flex items-start justify-center pt-12 px-4"
         @click.self="searchOpen = false"
         @keydown.escape.window="searchOpen = false">
        <div class="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[70vh] flex flex-col" @click.stop>
            <div class="p-4 border-b border-gray-200">
                <input type="text" x-model="searchQuery" @input.debounce.300ms="doSearch()"
                    x-ref="searchInput"
                    placeholder="Buscar en todas las transcripciones..."
                    class="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand">
            </div>
            <div class="p-4 overflow-y-auto flex-1">
                <div x-show="searchResults === null" class="text-sm text-gray-400">
                    Escribe para buscar.
                </div>
                <div x-show="searchResults !== null && searchResults.length === 0"
                     class="text-sm text-gray-400">
                    Sin resultados.
                </div>
                <div class="space-y-3">
                    <template x-for="r in (searchResults || [])" :key="r.job_id">
                        <div class="border border-gray-200 rounded p-3 cursor-pointer hover:border-gray-300"
                             @click="openTranscriptionById(r.job_id); searchOpen = false">
                            <div class="text-sm font-medium" x-text="r.name"></div>
                            <div class="text-xs text-gray-500 mt-1" x-html="r.snippet"></div>
                        </div>
                    </template>
                </div>
            </div>
        </div>
    </div>

</div>

<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create the JavaScript file**

`webapp/static/app.js`:
```javascript
function appData() {
    return {
        // State
        jobs: [],
        activeJobs: [],
        completedJobs: [],
        uploadName: '',
        uploadFile: null,
        uploading: false,
        uploadProgress: 0,
        detailJob: null,
        detailContent: '',
        editingName: false,
        editNameVal: '',
        searchOpen: false,
        searchQuery: '',
        searchResults: null,
        eventSource: null,

        init() {
            this.loadJobs();
            this.connectSSE();
        },

        // ── Data loading ──────────────────────────────

        async loadJobs() {
            const resp = await fetch('/api/jobs');
            this.jobs = await resp.json();
            this.categorizeJobs();
        },

        categorizeJobs() {
            this.activeJobs = this.jobs.filter(
                j => j.status === 'pending' || j.status === 'processing'
            );
            this.completedJobs = this.jobs.filter(
                j => j.status !== 'pending' && j.status !== 'processing'
            );
        },

        // ── SSE ───────────────────────────────────────

        connectSSE() {
            this.eventSource = new EventSource('/api/progress');
            this.eventSource.onmessage = (event) => {
                const active = JSON.parse(event.data);
                // Merge active job updates into our state
                for (const updated of active) {
                    const idx = this.jobs.findIndex(j => j.id === updated.id);
                    if (idx >= 0) {
                        this.jobs[idx] = { ...this.jobs[idx], ...updated };
                    }
                }
                // Check if any previously active job is now missing from active list
                const activeIds = new Set(active.map(j => j.id));
                const wasActive = this.activeJobs.some(j => !activeIds.has(j.id));
                if (wasActive) {
                    // A job finished/cancelled — reload all jobs to get final state
                    this.loadJobs();
                }
                this.categorizeJobs();
            };
            this.eventSource.onerror = () => {
                // Reconnect after a delay
                setTimeout(() => this.connectSSE(), 3000);
            };
        },

        // ── Upload ────────────────────────────────────

        upload() {
            if (!this.uploadFile || !this.uploadName.trim()) return;

            this.uploading = true;
            this.uploadProgress = 0;

            const formData = new FormData();
            formData.append('name', this.uploadName.trim());
            formData.append('file', this.uploadFile);

            const xhr = new XMLHttpRequest();
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    this.uploadProgress = Math.round((e.loaded / e.total) * 100);
                }
            };
            xhr.onload = () => {
                this.uploading = false;
                if (xhr.status === 201) {
                    this.uploadName = '';
                    this.uploadFile = null;
                    // Reset file input
                    const fileInput = document.querySelector('input[type="file"]');
                    if (fileInput) fileInput.value = '';
                    this.loadJobs();
                } else {
                    alert('Error al subir: ' + xhr.responseText);
                }
            };
            xhr.onerror = () => {
                this.uploading = false;
                alert('Error de conexion al subir el archivo.');
            };
            xhr.open('POST', '/api/jobs');
            xhr.send(formData);
        },

        // ── Job actions ───────────────────────────────

        async cancelJob(jobId) {
            await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
            this.loadJobs();
        },

        async deleteJob(jobId) {
            await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
            this.detailJob = null;
            this.detailContent = '';
            this.loadJobs();
        },

        async toggleKeepVideo(jobId, keep) {
            await fetch(`/api/jobs/${jobId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ keep_video: keep ? 1 : 0 }),
            });
            this.loadJobs();
        },

        // ── Transcription detail ──────────────────────

        async openTranscription(job) {
            this.detailJob = job;
            this.detailContent = '';
            this.editingName = false;
            if (job.status === 'completed') {
                const resp = await fetch(`/api/jobs/${job.id}/transcription`);
                if (resp.ok) {
                    const data = await resp.json();
                    this.detailContent = data.content;
                } else {
                    this.detailContent = 'Error al cargar la transcripcion.';
                }
            } else if (job.status === 'failed') {
                this.detailContent = 'Error: ' + (job.error_message || 'desconocido');
            }
        },

        async openTranscriptionById(jobId) {
            const resp = await fetch(`/api/jobs/${jobId}`);
            if (resp.ok) {
                const job = await resp.json();
                this.openTranscription(job);
            }
        },

        async saveName() {
            if (!this.editNameVal.trim() || !this.detailJob) return;
            await fetch(`/api/jobs/${this.detailJob.id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: this.editNameVal.trim() }),
            });
            this.detailJob.name = this.editNameVal.trim();
            this.editingName = false;
            this.loadJobs();
        },

        // ── Search ────────────────────────────────────

        async doSearch() {
            if (!this.searchQuery.trim()) {
                this.searchResults = null;
                return;
            }
            const resp = await fetch(`/api/search?q=${encodeURIComponent(this.searchQuery)}`);
            this.searchResults = await resp.json();
        },

        // ── Shutdown ──────────────────────────────────

        async shutdown() {
            await fetch('/api/shutdown', { method: 'POST' });
        },

        // ── Helpers ───────────────────────────────────

        formatDate(iso) {
            if (!iso) return '';
            const d = new Date(iso);
            return d.toLocaleDateString('es-ES', {
                day: '2-digit', month: 'short', year: 'numeric',
                hour: '2-digit', minute: '2-digit',
            });
        },

        formatDuration(seconds) {
            if (!seconds) return '';
            const m = Math.floor(seconds / 60);
            return m + ' min';
        },
    };
}
```

- [ ] **Step 3: Verify template renders**

Run:
```bash
cd /mnt/c/Development/transcriptvideo
source venv/bin/activate
python -c "
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile, os
tmp = tempfile.mkdtemp()
db = os.path.join(tmp, 'test.db')
with patch('webapp.app.DB_PATH', db), \
     patch('webapp.config.DB_PATH', db), \
     patch('webapp.app.TranscriptionWorker') as MockWorker:
    MockWorker.return_value = MagicMock()
    from webapp.app import app
    client = TestClient(app)
    resp = client.get('/')
    print('Status:', resp.status_code)
    print('Has Alpine:', 'alpinejs' in resp.text)
    print('Has Tailwind:', 'tailwindcss' in resp.text)
"
```
Expected: Status 200, Has Alpine: True, Has Tailwind: True

- [ ] **Step 4: Commit**

```bash
git add webapp/templates/index.html webapp/static/app.js
git commit -m "feat: single-page frontend with Alpine.js + Tailwind"
```

---

## Task 7: Windows Launcher

**Files:**
- Create: `start-webapp.bat`

- [ ] **Step 1: Create the .bat launcher**

`start-webapp.bat`:
```batch
@echo off
echo Starting TranscriptVideo Webapp...
echo.
echo The server will be available at:
echo   http://localhost:8000
echo.
echo Press Ctrl+C or use the Shutdown button in the UI to stop.
echo.

start http://localhost:8000

wsl -d Ubuntu bash -c "cd /mnt/c/Development/transcriptvideo && source venv/bin/activate && python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000"

pause
```

Note: `--host 0.0.0.0` so it's accessible from other devices on Tailscale. The `start` command opens the browser before the server is ready — the user may need to refresh once. This is acceptable for a personal tool.

- [ ] **Step 2: Verify .bat syntax is valid**

Open the file and confirm it looks correct. No programmatic test needed — this is a simple batch file.

- [ ] **Step 3: Commit**

```bash
git add start-webapp.bat
git commit -m "feat: Windows .bat launcher for desktop"
```

---

## Task 8: Logging Configuration

**Files:**
- Modify: `webapp/app.py` (add logging setup at module level)

- [ ] **Step 1: Add logging config to app.py**

Add at the top of `webapp/app.py`, after imports:
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
```

- [ ] **Step 2: Commit**

```bash
git add webapp/app.py
git commit -m "feat: configure logging for webapp"
```

---

## Task 9: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write an integration test that exercises the full flow (without GPU)**

`tests/test_integration.py`:
```python
"""Integration test: upload → queue → check status → read transcription.

Uses a mock transcriber so no GPU is needed.
"""

import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def full_app(tmp_dirs, tmp_db):
    """Create app with a fake worker that 'transcribes' instantly."""
    with (
        patch("webapp.config.VIDEOS_DIR", tmp_dirs["videos"]),
        patch("webapp.config.TRANSCRIPTIONS_DIR", tmp_dirs["transcriptions"]),
        patch("webapp.config.DB_PATH", tmp_db),
        patch("webapp.app.DB_PATH", tmp_db),
        patch("webapp.app.VIDEOS_DIR", tmp_dirs["videos"]),
        patch("webapp.app.TRANSCRIPTIONS_DIR", tmp_dirs["transcriptions"]),
        patch("webapp.worker.VIDEOS_DIR", tmp_dirs["videos"]),
        patch("webapp.worker.TRANSCRIPTIONS_DIR", tmp_dirs["transcriptions"]),
    ):
        from webapp.database import init_db

        init_db(tmp_db)

        from webapp.app import app

        # Replace the worker with one that completes jobs instantly
        app.state.worker = MagicMock()

        client = TestClient(app)
        yield client, tmp_db, tmp_dirs


def test_full_flow(full_app):
    client, tmp_db, tmp_dirs = full_app

    # 1. Upload a video
    fake_mp4 = io.BytesIO(b"fake video data")
    resp = client.post(
        "/api/jobs",
        data={"name": "Integration Test"},
        files={"file": ("test.mp4", fake_mp4, "video/mp4")},
    )
    assert resp.status_code == 201
    job = resp.json()
    job_id = job["id"]
    assert job["status"] == "pending"

    # 2. Verify the video was saved
    video_path = tmp_dirs["videos"] / job["video_filename"]
    assert video_path.exists()

    # 3. Check job list
    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    # 4. Simulate transcription completion by writing output files
    #    and updating the DB manually (since the real worker is mocked)
    from webapp.database import index_transcription, update_job

    output_dir = tmp_dirs["transcriptions"] / job["folder_name"]
    output_dir.mkdir(parents=True)
    txt_content = "[00:00:00,000 --> 00:00:05,000] Hello world"
    (output_dir / "transcripcion.txt").write_text(txt_content, encoding="utf-8")
    update_job(tmp_db, job_id, status="completed", progress=100.0)
    index_transcription(
        tmp_db, job_id=job_id, name="Integration Test", content=txt_content
    )

    # 5. Read transcription via API
    resp = client.get(f"/api/jobs/{job_id}/transcription")
    assert resp.status_code == 200
    assert "Hello world" in resp.json()["content"]

    # 6. Download
    resp = client.get(f"/api/jobs/{job_id}/download")
    assert resp.status_code == 200
    assert b"Hello world" in resp.content

    # 7. Rename
    resp = client.patch(
        f"/api/jobs/{job_id}",
        json={"name": "Renamed Test"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "Renamed Test"

    # 8. Search
    resp = client.get("/api/search?q=Hello")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert results[0]["job_id"] == job_id

    # 9. Delete
    resp = client.delete(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    assert not output_dir.exists()

    resp = client.get("/api/jobs")
    assert len(resp.json()) == 0
```

- [ ] **Step 2: Run all tests**

Run:
```bash
cd /mnt/c/Development/transcriptvideo
source venv/bin/activate
pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration smoke test for full upload-transcribe-search flow"
```

---

## Task 10: Manual End-to-End Verification

- [ ] **Step 1: Start the server manually in WSL**

```bash
cd /mnt/c/Development/transcriptvideo
source venv/bin/activate
python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

Watch the console output. Expected:
- "Loading Whisper model 'large-v3' on cuda..." (takes ~10s)
- "Model loaded."
- "Existing transcriptions imported."
- Uvicorn shows "Application startup complete"

- [ ] **Step 2: Open browser and verify UI**

Open `http://localhost:8000`. Verify:
- Header shows "TranscriptVideo" with "Buscar" and "Apagar" buttons
- Upload section with name input + file picker + "Subir" button
- "Transcripciones" section shows the 2 existing transcriptions (imported from disk)

- [ ] **Step 3: Test upload with a small video**

Upload a short test .mp4. Verify:
- Upload progress bar fills
- Job appears in "En proceso" with "En cola" badge
- Status changes to "Procesando" with progress bar
- On completion, moves to "Transcripciones" list

- [ ] **Step 4: Test transcription viewer**

Click on a completed transcription. Verify:
- Modal opens with transcription text
- "Descargar" link works
- Click name to edit — type new name, save — name updates
- "Conservar video" / "Borrar video" dropdown works

- [ ] **Step 5: Test search**

Click "Buscar". Type a word from one of the transcriptions. Verify:
- Results appear with highlighted snippets
- Click a result → transcription detail opens

- [ ] **Step 6: Test cancel**

Upload a long video. While processing, click "Cancelar". Verify:
- Job status changes to cancelled
- Worker moves on (or waits for next job)

- [ ] **Step 7: Test the .bat launcher**

Double-click `start-webapp.bat` from Windows. Verify:
- Console window opens with startup messages
- Browser opens to `http://localhost:8000`
- UI works correctly

- [ ] **Step 8: Test shutdown**

Click "Apagar" in the UI. Confirm the dialog. Verify:
- Server stops
- Console window shows the shutdown

---

## Summary of Dependencies

**Python (install in WSL venv):**
```
pip install fastapi uvicorn[standard] python-multipart aiofiles sse-starlette jinja2 pytest httpx
```

**Frontend (CDN, no install needed):**
- Tailwind CSS: `https://cdn.tailwindcss.com`
- Alpine.js: `https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js`

**No Node.js required.**
