
import os
import uuid
import shutil
import threading
from pathlib import Path
from typing import Optional

import soundfile as sf
from fastapi import (
    FastAPI, UploadFile, File, HTTPException,
    BackgroundTasks, Depends, Security
)
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "audio"
RESULT_DIR = BASE_DIR / "results"

for directory in (UPLOAD_DIR, AUDIO_DIR, RESULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="N-ATLAS Multilingual Video Localizer",
    version="1.0.0"
)

ai_services = {
    "transcribe": None,
    "translate": None,
    "speech": None,
    "render": None
}

videos = {}
audio_files = {}
jobs = {}
jobs_lock = threading.Lock()

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)

def authenticate(key: Optional[str] = Security(api_key_header)):
    import secrets
    expected = os.environ.get("DEV_API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured"
        )
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")

protected = [Depends(authenticate)]

def configure_ai_services(transcribe=None, translate=None, speech=None):
    for name, handler in {
        "transcribe": transcribe,
        "translate": translate,
        "speech": speech
    }.items():
        if handler is not None:
            ai_services[name] = handler

def configure_video_renderer(renderer):
    ai_services["render"] = renderer

def require_service(name):
    handler = ai_services.get(name)
    if handler is None:
        raise HTTPException(
            status_code=503,
            detail=f"{name} service unavailable"
        )
    return handler

class TranslationRequest(BaseModel):
    text: str
    language: str

class SpeechRequest(BaseModel):
    text: str
    language: str

class LocalizationRequest(BaseModel):
    video_id: str
    audio_id: str
    translated_text: str
    language: str

@app.get("/health", dependencies=protected)
def health():
    return {
        "status": "ok",
        "ai_models": (
            "connected"
            if all(callable(x) for x in ai_services.values())
            else "partially_connected"
        ),
        "services": {
            k: callable(v) for k, v in ai_services.items()
        }
    }

@app.post("/upload", dependencies=protected)
async def upload(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in {".mp4", ".mov", ".webm"}:
        raise HTTPException(400, "Unsupported video format")

    video_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{video_id}{extension}"

    size = 0
    try:
        with path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 100 * 1024 * 1024:
                    raise HTTPException(413, "Video exceeds 100MB")
                output.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    videos[video_id] = str(path)
    return {"video_id": video_id, "filename": file.filename}

@app.get("/videos/{video_id}", dependencies=protected)
def get_video(video_id: str):
    path = videos.get(video_id)
    if not path:
        raise HTTPException(404, "Video not found")
    return {
        "video_id": video_id,
        "filename": Path(path).name,
        "exists": Path(path).is_file()
    }

@app.post("/transcribe/{video_id}", dependencies=protected)
def transcribe_video(video_id: str):
    path = videos.get(video_id)
    if not path:
        raise HTTPException(404, "Video not found")
    text = require_service("transcribe")(path)
    return {"video_id": video_id, "transcription": text}

@app.post("/translate", dependencies=protected)
def translate(request: TranslationRequest):
    result = require_service("translate")(
        request.text, request.language
    )
    return {"language": request.language, "translated_text": result}

@app.post("/generate-speech", dependencies=protected)
def generate_speech(request: SpeechRequest):
    audio, rate = require_service("speech")(
        request.text, request.language
    )
    audio_id = uuid.uuid4().hex
    path = AUDIO_DIR / f"{audio_id}.wav"
    sf.write(str(path), audio, rate)
    audio_files[audio_id] = str(path)
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=f"{audio_id}.wav",
        headers={"X-Audio-ID": audio_id}
    )

@app.post("/upload-audio", dependencies=protected)
async def upload_audio(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() != ".wav":
        raise HTTPException(400, "Only WAV files supported")

    audio_id = uuid.uuid4().hex
    path = AUDIO_DIR / f"{audio_id}.wav"

    size = 0
    try:
        with path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 25 * 1024 * 1024:
                    raise HTTPException(413, "Audio exceeds 25MB")
                output.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    audio_files[audio_id] = str(path)
    return {"audio_id": audio_id}

def process_localization(job_id, request):
    try:
        result = ai_services["render"](
            videos[request.video_id],
            audio_files[request.audio_id],
            request.translated_text,
            request.language
        )

        with jobs_lock:
            jobs[job_id] = {
                "job_id": job_id,
                "status": "completed",
                "result_path": str(result)
            }
    except Exception as exc:
        with jobs_lock:
            jobs[job_id] = {
                "job_id": job_id,
                "status": "failed",
                "error": str(exc)
            }

@app.post("/localize", dependencies=protected)
def localize(request: LocalizationRequest, background_tasks: BackgroundTasks):
    if request.video_id not in videos:
        raise HTTPException(404, "Video not found")
    if request.audio_id not in audio_files:
        raise HTTPException(404, "Audio not found")
    if request.language.title() not in {"English", "Yoruba", "Hausa", "Igbo"}:
        raise HTTPException(400, "Unsupported language")

    require_service("render")

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"job_id": job_id, "status": "queued"}

    background_tasks.add_task(process_localization, job_id, request)
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}", dependencies=protected)
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.get("/results/{job_id}", dependencies=protected)
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(409, "Result not ready")

    path = Path(job["result_path"])
    if not path.is_file():
        raise HTTPException(404, "Result file missing")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4"
    )


# Permanent Bring Your Own API endpoints.
from byoa_routes import router as byoa_router
app.include_router(byoa_router, dependencies=protected)
