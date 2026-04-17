import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from faster_whisper import WhisperModel

BASE_DIR = Path("/mnt/c/Development/transcriptvideo")
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUT_BASE = BASE_DIR / "transcripciones"

MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"


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


def clean_intra_segment_repetition(text: str) -> str:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 6:
        return text
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.25:
        first_word = words[0]
        return first_word.capitalize() + "."
    return text


def remove_hallucinations(segments: list[Segment]) -> tuple[list[Segment], int]:
    cleaned: list[Segment] = []
    removed = 0

    for seg in segments:
        new_text = clean_intra_segment_repetition(seg.text)
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
            if norm_j == norm and (cleaned[j].end - cleaned[j].start) <= 1.5 and len(norm) <= 10:
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


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python transcribe.py <nombre_del_video>")
        print(f"\nVideos disponibles en {VIDEOS_DIR}:")
        for f in sorted(VIDEOS_DIR.glob("*")):
            if f.is_file():
                print(f"  - {f.name}")
        sys.exit(1)

    video_name = sys.argv[1]
    video_path = VIDEOS_DIR / video_name

    if not video_path.exists():
        print(f"Error: No se encontró el video '{video_name}' en {VIDEOS_DIR}")
        sys.exit(1)

    output_dir = OUTPUT_BASE / video_path.stem
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Error: Ya existe una transcripción en {output_dir}")
        print("Si deseas re-transcribir, elimina la carpeta primero.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando modelo '{MODEL_SIZE}' en GPU ({COMPUTE_TYPE})...")
    start_load = time.time()

    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    load_time = time.time() - start_load
    print(f"Modelo cargado en {load_time:.1f}s")
    print(f"Transcribiendo: {video_path}")
    print("-" * 60)

    start_transcribe = time.time()

    whisper_segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=400,
        ),
    )

    print(f"Audio detectado: {info.duration:.1f}s ({info.duration / 60:.1f} min)")
    print(f"Idioma detectado: {info.language} (probabilidad: {info.language_probability:.2%})")
    print("-" * 60)

    raw_segments: list[Segment] = []
    for segment in whisper_segments:
        raw_segments.append(Segment(segment.start, segment.end, segment.text.strip()))
        elapsed = time.time() - start_transcribe
        progress = segment.end / info.duration * 100 if info.duration > 0 else 0
        speed = segment.end / elapsed if elapsed > 0 else 0
        print(f"  [{progress:5.1f}%] {format_timestamp(segment.start)} --> {format_timestamp(segment.end)}  {segment.text.strip()[:80]}{'...' if len(segment.text.strip()) > 80 else ''}")
        if len(raw_segments) % 50 == 0:
            eta = (info.duration - segment.end) / speed if speed > 0 else 0
            print(f"         --- Velocidad: {speed:.1f}x | ETA: {eta / 60:.1f} min ---")

    total_time = time.time() - start_transcribe
    speed_ratio = info.duration / total_time if total_time > 0 else 0

    print("-" * 60)
    print("Aplicando post-procesado (eliminando alucinaciones)...")
    clean_segments, removed = remove_hallucinations(raw_segments)
    print(f"  Segmentos originales: {len(raw_segments)}")
    print(f"  Segmentos eliminados: {removed}")
    print(f"  Segmentos finales: {len(clean_segments)}")

    txt_lines: list[str] = []
    srt_lines: list[str] = []
    for idx, seg in enumerate(clean_segments, start=1):
        start_ts = format_timestamp(seg.start)
        end_ts = format_timestamp(seg.end)
        txt_lines.append(f"[{start_ts} --> {end_ts}] {seg.text}")
        srt_lines.append(str(idx))
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(seg.text)
        srt_lines.append("")

    txt_path = output_dir / "transcripcion.txt"
    srt_path = output_dir / "transcripcion.srt"

    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Transcripcion completada!")
    print(f"  Tiempo total: {total_time / 60:.1f} min")
    print(f"  Velocidad: {speed_ratio:.1f}x tiempo real")
    print(f"  Archivos generados:")
    print(f"    - {txt_path}")
    print(f"    - {srt_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
