# TranscriptVideo

Herramienta local para transcribir videos usando **faster-whisper** con GPU NVIDIA (CUDA). Incluye una webapp para subir videos, gestionar la cola de transcripcion y buscar en transcripciones completadas desde cualquier dispositivo via Tailscale.

## Requisitos

- **Windows 11** con **WSL2** (Ubuntu)
- **GPU NVIDIA** con soporte CUDA
- **Python 3.12** (dentro de WSL)
- **Tailscale** (opcional, para acceso desde otros dispositivos)

## Instalacion

### 1. Clonar el proyecto

```bash
cd /mnt/c/Development
git clone <repo-url> transcriptvideo
cd transcriptvideo
```

### 2. Crear el entorno virtual y las dependencias

```bash
python3 -m venv venv
source venv/bin/activate
pip install faster-whisper
pip install fastapi uvicorn[standard] python-multipart aiofiles sse-starlette jinja2
```

### 3. Verificar CUDA

```bash
python3 -c "from faster_whisper import WhisperModel; m = WhisperModel('tiny', device='cuda'); print('CUDA OK')"
```

## Uso

### Opcion A: Webapp (recomendado)

Doble clic en **`start-webapp.bat`** desde el escritorio de Windows.

Esto:
1. Abre el navegador en `http://localhost:8000`
2. Arranca el servidor FastAPI dentro de WSL
3. Carga el modelo Whisper en GPU (~10 segundos la primera vez)

> Si el navegador muestra error de conexion, esperar unos segundos y refrescar.

#### Acceso desde otros dispositivos

El servidor escucha en `0.0.0.0:8000`. Desde otro dispositivo en la misma red Tailscale, abrir:

```
http://<ip-tailscale-del-pc>:8000
```

#### Funciones de la webapp

- **Subir video**: Elegir un `.mp4`, darle un nombre, y subirlo. Se muestra progreso de subida.
- **Cola**: Los videos se procesan en orden (FIFO). Se puede cancelar cualquier job en cola o en proceso.
- **Progreso en vivo**: La barra de progreso se actualiza en tiempo real mientras se transcribe.
- **Ver transcripcion**: Clic en una transcripcion completada para leerla directamente en el navegador.
- **Descargar**: Descargar el `.txt` con la transcripcion.
- **Renombrar**: Clic en el nombre de una transcripcion para editarlo.
- **Buscar**: Buscar texto en todas las transcripciones desde el boton "Buscar".
- **Borrar video**: Elegir si conservar o borrar el video original despues de transcribir.
- **Apagar**: Boton "Apagar" en la UI para detener el servidor.

### Opcion B: CLI (uso directo)

```bash
cd /mnt/c/Development/transcriptvideo
source venv/bin/activate
python transcribe.py "nombre_del_video.mp4"
```

Los videos deben estar en `videos/`. Las transcripciones se guardan en `transcripciones/<nombre>/`.

## Estructura del proyecto

```
transcriptvideo/
├── transcribe.py              # CLI original (standalone)
├── start-webapp.bat           # Lanzador Windows
├── videos/                    # Videos .mp4
├── transcripciones/           # Carpeta por transcripcion
│   └── <nombre>/
│       └── transcripcion.txt
├── webapp/
│   ├── app.py                 # FastAPI (API + servidor web)
│   ├── config.py              # Configuracion (rutas, modelo)
│   ├── database.py            # SQLite + busqueda full-text (FTS5)
│   ├── transcriber.py         # Logica de transcripcion (libreria)
│   ├── worker.py              # Hilo background (cola FIFO)
│   ├── templates/
│   │   └── index.html         # UI (Alpine.js + Tailwind CSS)
│   └── static/
│       └── app.js             # Logica frontend (SSE, subida, busqueda)
├── tests/                     # Tests automatizados
└── venv/                      # Entorno virtual Python
```

## Stack tecnico

| Componente | Tecnologia |
|------------|------------|
| Transcripcion | faster-whisper, modelo `large-v3`, CUDA float16 |
| Backend | FastAPI, SQLite (WAL + FTS5), uvicorn |
| Frontend | Alpine.js + Tailwind CSS (CDN, sin Node.js) |
| Progreso en vivo | Server-Sent Events (SSE) |
| Ejecucion | WSL2 Ubuntu, GPU NVIDIA |

## Tests

```bash
cd /mnt/c/Development/transcriptvideo
source venv/bin/activate
pytest tests/ -v
```

## Notas

- El modelo Whisper se carga **una sola vez** al iniciar el servidor y se reutiliza para todos los jobs.
- La deteccion de idioma es automatica (principalmente espanol, ocasionalmente ingles).
- El post-procesado elimina alucinaciones tipicas de Whisper (repeticiones tipo "no, no, no, no...").
- Las transcripciones existentes (creadas con el CLI) se importan automaticamente al iniciar la webapp.
- No requiere Node.js ni ningun paso de build para el frontend.
