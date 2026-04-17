from pathlib import Path

import pytest


@pytest.fixture
def tmp_dirs(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    transcriptions = tmp_path / "transcriptions"
    transcriptions.mkdir()
    return {"videos": videos, "transcriptions": transcriptions, "base": tmp_path}


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test.db"
