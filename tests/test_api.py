import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(tmp_dirs, tmp_db):
    """Create a test client with patched paths and no real worker."""
    mock_worker_cls = MagicMock()
    mock_worker_instance = MagicMock()
    mock_worker_cls.return_value = mock_worker_instance

    with (
        patch("webapp.app.DB_PATH", tmp_db),
        patch("webapp.app.VIDEOS_DIR", tmp_dirs["videos"]),
        patch("webapp.app.TRANSCRIPTIONS_DIR", tmp_dirs["transcriptions"]),
        patch("webapp.app.TranscriptionWorker", mock_worker_cls),
    ):
        from webapp.database import init_db
        init_db(tmp_db)

        from webapp.app import app

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
    assert len(jobs) >= 1


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
