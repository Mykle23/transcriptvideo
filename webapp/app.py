import asyncio
import json
import logging
import os
import shutil
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
    index_transcription,
    init_db,
    search_transcriptions,
    update_job,
)
from webapp.worker import TranscriptionWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
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
if STATIC_DIR.exists():
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


# -- HTML Page --

@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -- Job CRUD --

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
        shutil.rmtree(folder)

    delete_job(DB_PATH, job_id)
    return {"ok": True}


# -- Transcription & Download --

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


# -- Search --

@app.get("/api/search")
def search(q: str = ""):
    if not q.strip():
        return []
    return search_transcriptions(DB_PATH, q.strip())


# -- SSE Progress --

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

    return EventSourceResponse(generate())


# -- Shutdown --

@app.post("/api/shutdown")
def shutdown():
    """Gracefully stop the server."""
    logger.info("Shutdown requested via API.")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"ok": True}
