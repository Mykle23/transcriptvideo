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
