# Proyecto: Transcripción de videos

## Estado actual

Herramienta local de transcripción de videos usando **faster-whisper + GPU CUDA**. Funcional y en uso.

### Stack técnico
- **Modelo**: `large-v3` (faster-whisper)
- **Hardware**: GPU NVIDIA con CUDA, WSL2 Ubuntu
- **Compute type**: float16
- **Entorno**: `venv/` en la raíz del proyecto
- **Ejecución**: `python transcribe.py "<nombre_video>.mp4"`

### Estructura de archivos
```
transcriptvideo/
├── transcribe.py           # script principal
├── videos/                 # todos los .mp4 aquí
├── transcripciones/        # una carpeta por video (no se sobreescribe)
│   └── <nombre_video>/
│       ├── transcripcion.txt   # formato [HH:MM:SS,mmm --> ...] texto
│       └── transcripcion.srt   # formato SRT estándar
└── venv/
```

### Decisiones de diseño tomadas
- **Detección automática de idioma** (principalmente español, ocasionalmente inglés) — no se fuerza `language`
- **Sin `initial_prompt`** — cada video es de tema distinto, no tiene sentido
- **No sobreescribir transcripciones** — si existe, error. Hay que borrar la carpeta para re-transcribir
- **Post-procesado determinista** contra alucinaciones de Whisper (en el propio script):
  - Colapsa segmentos con palabra repetida >75% (patrón "no, no, no, no...")
  - Fusiona 4+ segmentos cortos idénticos consecutivos ("Sí. Sí. Sí..." o "Chau. Chau...")
- **Parámetros del modelo** que SÍ usamos: `beam_size=5`, `vad_filter=True` con `min_silence_duration_ms=500`, `speech_pad_ms=400`
- **Parámetros que probamos y descartamos** (degradan calidad):
  - `no_repeat_ngram_size=3` → rompía nombres propios (Mews → Mew)
  - `repetition_penalty=1.1` + `hallucination_silence_threshold=1.0` → generaban nuevas alucinaciones tipo "Sí. Sí. Sí."

## Próxima fase: Frontend + API web

El usuario quiere una webapp para:
- Subir videos desde otros dispositivos vía VPN (el procesamiento se hace en este PC con GPU)
- Cola de procesamiento (1 a la vez por limitación de GPU)
- Ver estado/progreso en vivo
- Listar y descargar transcripciones completadas

### Decisiones ya tomadas sobre la webapp
- **Stack propuesto**: FastAPI (backend, reusa el venv) + Next.js/Astro con Tailwind (frontend bonito)
- **Progreso en vivo**: SSE o WebSocket
- **Cola**: simple (sqlite o RQ)
- **Lanzamiento**: `.bat` en el escritorio de Windows → lanza WSL + API + abre navegador
- **Sin autostart al iniciar Windows** — el usuario quiere control manual
- **Parar el servidor**: botón "Apagar" dentro de la UI (no un segundo .bat)
- **Planteamiento de desarrollo**: el usuario pidió un "equipo" (PM = Claude, + devs especialistas + QA + diseñador) con fases claras

### Estado de la nueva fase
El usuario pidió comenzar con una ronda de preguntas del Product Manager antes de arrancar el desarrollo. La sesión anterior terminó justo cuando iba a empezar esa ronda de preguntas.

## Hardware del usuario
- **PC principal** (donde se ejecuta todo): tiene GPU NVIDIA con CUDA. WSL2.
- **Homelab**:
  - Ryzen 7 5800H, 64 GB RAM — sin GPU
  - Intel i5-13500T, 16 GB RAM — sin GPU
- Los homelabs NO sirven para transcripción rápida (solo CPU), la GPU del PC principal es la que hace todo el trabajo pesado.

## Preferencias de colaboración
- Responder en **español**, tono directo y conciso
- Para preguntas exploratorias: recomendación + tradeoff principal en 2-3 frases, NO implementar sin confirmación
- Estructura organizada (carpetas separadas, nada de sobreescribir)
