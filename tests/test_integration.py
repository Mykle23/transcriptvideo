"""Integration test: upload → queue → check status → read transcription.

Uses a mock transcriber so no GPU is needed.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def full_app(tmp_dirs, tmp_db):
    """Create app with mocked worker."""
    with (
        patch("webapp.app.DB_PATH", tmp_db),
        patch("webapp.app.VIDEOS_DIR", tmp_dirs["videos"]),
        patch("webapp.app.TRANSCRIPTIONS_DIR", tmp_dirs["transcriptions"]),
        patch("webapp.app.TranscriptionWorker") as MockWorker,
    ):
        MockWorker.return_value = MagicMock()

        from webapp.database import init_db
        init_db(tmp_db)

        from webapp.app import app
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
    assert len(resp.json()) >= 1

    # 4. Simulate transcription completion
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
    assert len(results) >= 1
    assert any(r["job_id"] == job_id for r in results)

    # 9. Delete
    resp = client.delete(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    assert not output_dir.exists()

    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 404
