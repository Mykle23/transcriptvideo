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
        output_dir: Directory to write transcription.txt into.
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

    txt_path = output_dir / "transcription.txt"
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
