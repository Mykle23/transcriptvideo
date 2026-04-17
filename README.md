# TranscriptVideo

<p align="left">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/tests-21%20passing-green.svg" alt="Tests passing">
  <img src="https://img.shields.io/badge/GPU-CUDA-76B900.svg" alt="CUDA">
  <a href="docs/README.es.md"><img src="https://img.shields.io/badge/lang-ES-yellow.svg" alt="Espanol"></a>
</p>

> Self-hosted video transcription that runs entirely on your own GPU. Upload a video, get an accurate transcript. No cloud, no API keys, no data leaving your machine.

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with a minimal FastAPI + Alpine.js web UI. Designed for personal use over Tailscale/VPN — upload from any device, process on the machine with the GPU.

<!-- Add a screenshot at docs/screenshot.png -->
<!-- ![TranscriptVideo UI](docs/screenshot.png) -->

## Features

- **Local only** — Whisper runs on your GPU, videos never leave the machine
- **Web UI** — upload, queue, monitor progress live, read + download transcripts
- **Dark / light theme** — Linear/Vercel-inspired minimal interface, dark by default
- **FIFO queue** with cancellation support (one job at a time, GPU-bound)
- **Live progress via SSE** — real-time updates as segments are transcribed
- **Full-text search** across all past transcripts (SQLite FTS5)
- **Editable names** — rename transcripts after the fact
- **Keep or delete** the source video per job
- **Auto language detection** — primarily Spanish + English, others supported
- **Hallucination cleanup** — deterministic post-processing strips the typical `no no no no` / `Yes. Yes. Yes.` Whisper artifacts
- **CLI fallback** — `transcribe.py` still works standalone for scripted use
- **No Node.js** — frontend is a single Jinja2 template + Alpine.js + Tailwind via CDN

## Why this exists

Cloud transcription services are fast and convenient, but they have real downsides: you send your audio to someone else's servers, pay per minute, and trust them with recordings you may not want floating around. For personal content (meetings, voice notes, OBS recordings) a local GPU can transcribe faster than real-time with `large-v3` quality — you just need a reasonable UI around it.

This project wraps the standard faster-whisper workflow in a small webapp so you can upload from your laptop/phone over your VPN, queue jobs, and get searchable transcripts without ever hitting a third-party API.

## Requirements

- **Windows 11** with **WSL2** (Ubuntu distro)
- **NVIDIA GPU** with drivers + CUDA (WSL2 uses the Windows driver via NVIDIA's CUDA passthrough)
- **Python 3.12** inside WSL (`sudo apt install python3.12 python3.12-venv`)
- **Git** inside WSL
- **Tailscale** (optional, for access from other devices)

> Note: the launcher targets WSL2 on Windows but the backend is plain FastAPI — it runs on any Linux with CUDA. Swap the `.bat` for a shell script if you want to run it natively.

## Installation (from scratch)

All commands run **inside WSL** (not CMD/PowerShell).

### 1. Clone

```bash
cd /mnt/c/Development
git clone https://github.com/Mykle23/transcriptvideo.git
cd transcriptvideo
```

### 2. Create the venv

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> First install pulls ~2 GB (PyTorch + CUDA + ctranslate2). Takes several minutes.

### 4. Verify CUDA

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cuda'); print('CUDA OK')"
```

### 5. (Optional) Pre-download the model

The first real run with `large-v3` pulls ~3 GB. Pre-download it now to avoid waiting later:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16')"
```

> Always pass `HF_HUB_ENABLE_HF_TRANSFER=0`. Without it, the parallel downloader can crash WiFi on some PCs (see Troubleshooting).

### 6. (Optional) Run the tests

```bash
pytest tests/ -v
```

21 tests should pass.

## Usage

### Webapp

Double-click **`start-webapp.bat`** from the Windows desktop.

- Opens `http://localhost:8000` in your browser
- Launches the FastAPI server inside WSL
- Loads the Whisper model into GPU (~10 s once cached)

If the browser shows a connection error, wait a few seconds and refresh (the server needs a moment to start).

**Access from other devices (via Tailscale):**

```
http://<tailscale-ip-of-the-pc>:8000
```

The server binds to `0.0.0.0:8000`, so any device on the same Tailscale net can reach it.

### CLI (direct use)

```bash
source venv/bin/activate
python transcribe.py "my_video.mp4"
```

Videos must live under `videos/`. Output goes to `transcriptions/<video-name>/transcription.txt`.

## Project structure

```
transcriptvideo/
├── transcribe.py              # standalone CLI (legacy, still works)
├── start-webapp.bat           # Windows launcher (WSL + uvicorn + browser)
├── videos/                    # .mp4 files (ignored by git)
├── transcriptions/            # output folder per job (ignored by git)
│   └── <name>/
│       └── transcription.txt
├── webapp/
│   ├── app.py                 # FastAPI app: routes + lifespan + SSE
│   ├── config.py              # paths and constants
│   ├── database.py            # SQLite + FTS5 search
│   ├── transcriber.py         # transcription + hallucination cleanup
│   ├── worker.py              # background job processor (FIFO, GPU-bound)
│   ├── templates/index.html   # single-page UI (Alpine.js + Tailwind)
│   └── static/app.js          # frontend logic (SSE, upload, search, theme)
├── tests/
│   ├── conftest.py
│   ├── test_api.py            # FastAPI endpoint tests
│   ├── test_database.py       # DB CRUD + FTS5
│   ├── test_integration.py    # full upload-to-delete flow
│   └── test_transcriber.py    # segment cleanup logic
├── requirements.txt
├── LICENSE
└── README.md
```

## Tech stack

| Layer | Tech |
|-------|------|
| Transcription | faster-whisper, `large-v3`, CUDA float16 |
| Backend | FastAPI, uvicorn, SQLite (WAL + FTS5) |
| Frontend | Alpine.js + Tailwind CSS via CDN (no Node.js, no build) |
| Live progress | Server-Sent Events (SSE) |
| Runtime | WSL2 Ubuntu + NVIDIA GPU |

## Troubleshooting

### WiFi disconnects when starting the server for the first time

**Symptom:** when the `.bat` launches, WiFi drops 1-2 min later (no networks visible in Windows).

**Cause:** `faster-whisper` pulls the model via `huggingface_hub`, which enables `hf_transfer` by default — an aggressive Rust downloader using many parallel connections. On some PCs (Realtek-based WiFi adapters common in WSL2 setups), this crashes the Windows network stack.

**Fix:** the `start-webapp.bat` already forces `HF_HUB_ENABLE_HF_TRANSFER=0` (serial download — slower but stable). If you download the model manually, export the same variable:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16')"
```

### "Unable to open file 'model.bin'" when the server starts

The cache is corrupted or incomplete (`.incomplete` blob left behind). Wipe and retry:

```bash
wsl bash -c "rm -rf /root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3"
```

Relaunch the `.bat` — it will re-download cleanly.

### "Internal Server Error" on the home page

If `/` returns 500 but `/api/jobs` works, you have an outdated Starlette. The code uses the modern `templates.TemplateResponse(request, "index.html")` signature. Upgrade:

```bash
python -m pip install --upgrade starlette fastapi
```

## Notes

- The Whisper model loads **once** at server startup and is reused across all jobs.
- Language is auto-detected per video.
- Post-processing removes typical Whisper hallucinations: segments where one word dominates >75% (`no no no no ...`) collapse to that word, and runs of 4+ consecutive identical short segments (`Yes. Yes. Yes. Yes.`) merge into one.
- Existing transcriptions already on disk (created with the CLI) are auto-imported into the DB on first launch.
- No Node.js or build step required for the frontend.

## License

[MIT](LICENSE) — use it, fork it, ship it.

---

<sub>Versión en español: [docs/README.es.md](docs/README.es.md)</sub>
