from pathlib import Path

BASE_DIR = Path("/mnt/c/Development/transcriptvideo")
VIDEOS_DIR = BASE_DIR / "videos"
TRANSCRIPTIONS_DIR = BASE_DIR / "transcriptions"
DB_PATH = BASE_DIR / "webapp" / "transcriptvideo.db"

MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

ALLOWED_EXTENSIONS = {".mp4"}
