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
