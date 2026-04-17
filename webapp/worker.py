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
