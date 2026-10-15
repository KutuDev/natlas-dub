"""/dub/* job endpoints, mounted onto the handover FastAPI app (api.app).

Mounted with the handover's own `protected` dependency, so every route here
requires X-API-Key == DEV_API_KEY (401 on a bad key, 503 fail-closed if unset).
"""

import logging
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import asr, config, handlers, jobs, media, pipeline, session_log

log = logging.getLogger("natlas_dub.routes")
router = APIRouter(prefix="/dub", tags=["N-ATLAS Dub"])

ALLOWED_EXT = {".mp4", ".mov", ".webm", ".m4v"}  # handover allow-list (.mp4/.mov/.webm) + .m4v
_gpu_lock = threading.Lock()  # one GPU-heavy stage at a time


def _serial(fn, job_id, *args):
    jobs.update(job_id, message="Queued (waiting for the GPU)")
    with _gpu_lock:
        fn(job_id, *args)


def _job_or_404(job_id: str) -> dict:
    job = jobs.get(job_id)  # validates uuid4-hex format; anything else -> None
    if not job:
        raise HTTPException(404, "Job not found (jobs expire; please upload again)")
    return job


def _not_running(job: dict):
    if job["status"] == "running":
        raise HTTPException(409, "This job is busy; wait for the current step to finish")


@router.get("/health")
def dub_health():
    return {
        "status": "ok",
        "ffmpeg": media.ffmpeg_available(),
        "asr": {"default_model": config.ASR_MODEL, "hf_token_set": bool(config.HF_TOKEN),
                "loaded_models": asr.loaded_models()},
        "models": handlers.status(),
        "limits": {"max_video_seconds": config.MAX_VIDEO_SECONDS, "max_video_mb": config.MAX_VIDEO_MB},
    }


# ---------------------------------------------------------------- upload + duration gate
@router.post("/upload")
async def upload(file: UploadFile = File(...), consent: bool = Form(False),
                 tester_name: str = Form("", max_length=120), tester_email: str = Form("", max_length=200),
                 trim_to_seconds: float = Form(0, ge=0, le=600)):
    """trim_to_seconds > 0: if the video is longer than the limit, keep only its first N seconds
    (N is capped at MAX_VIDEO_SECONDS). 0 = reject over-length videos (default)."""
    if not consent:
        raise HTTPException(400, "Please confirm you own or have rights to this video and its voice.")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "Unsupported video format")
    jobs.cleanup_expired()
    max_bytes = int(config.MAX_VIDEO_MB * 1024 * 1024)
    job = jobs.new_job(filename=Path(file.filename or "").name[:200],
                       tester_name=tester_name.strip(), tester_email=tester_email.strip())
    path = Path(job["dir"]) / f"source{ext}"
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"Video exceeds {config.MAX_VIDEO_MB:g}MB")
                out.write(chunk)
        try:
            info = media.probe(str(path))
        except media.MediaError:
            raise HTTPException(400, "Could not read this video file.")
        if not info["has_video"]:
            raise HTTPException(400, "No video stream found in the file.")
        if not info["has_audio"]:
            raise HTTPException(400, "This video has no audio track to transcribe.")
        if info["duration"] <= 0:
            raise HTTPException(400, "Could not determine video duration.")
        trimmed_from = None
        if info["duration"] > config.MAX_VIDEO_SECONDS:
            if trim_to_seconds <= 0:
                raise HTTPException(413, f"Video is {info['duration']:.0f}s; the limit is {config.MAX_VIDEO_SECONDS:g}s. "
                                         "Tick 'use only the first 60 seconds' to trim it.")
            keep = min(trim_to_seconds, config.MAX_VIDEO_SECONDS)
            trimmed = Path(job["dir"]) / "source_trimmed.mp4"
            try:
                media.trim_video(str(path), str(trimmed), keep)
                path.unlink(missing_ok=True)
                path = trimmed
                trimmed_from = round(info["duration"], 2)
                info = media.probe(str(path))
            except media.MediaError:
                raise HTTPException(400, "Could not trim this video.")
    except Exception:
        jobs.discard(job["job_id"])
        raise
    jobs.update(job["job_id"], video_path=str(path), duration=info["duration"],
                has_audio=info["has_audio"], size_bytes=size, trimmed_from=trimmed_from)
    session_log.log_event(jobs.get(job["job_id"]), "upload_received", success=True)
    log.info("upload job=%s duration=%.1fs size=%dB trimmed_from=%s", job["job_id"], info["duration"], size, trimmed_from)
    return {"job_id": job["job_id"], "duration": round(info["duration"], 2),
            "size_mb": round(size / 1048576, 2), "trimmed_from": trimmed_from}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    return jobs.public_view(_job_or_404(job_id))


@router.get("/jobs/{job_id}/source")
def source_video(job_id: str):
    job = _job_or_404(job_id)
    return FileResponse(job["video_path"])


# ---------------------------------------------------------------- transcribe + translate
class ProcessRequest(BaseModel):
    target_language: str = Field(min_length=2, max_length=2)
    source_language: str = Field("en", min_length=2, max_length=2)
    voice: str = Field("female", max_length=10)
    protected_terms: List[str] = Field(default_factory=list, max_length=100)


@router.post("/jobs/{job_id}/process")
def process(job_id: str, req: ProcessRequest, bg: BackgroundTasks):
    job = _job_or_404(job_id)
    _not_running(job)
    if req.target_language not in config.TARGET_LANGUAGES:
        raise HTTPException(400, "Unsupported target language")
    if req.source_language not in config.SOURCE_LANGUAGES:
        raise HTTPException(400, "Unsupported source language")
    if req.voice not in config.VOICES:
        raise HTTPException(400, "Voice must be female or male")
    terms = [t.strip()[:100] for t in req.protected_terms if t and t.strip()]
    jobs.update(job_id, target_language=req.target_language, source_language=req.source_language,
                voice=req.voice, protected_terms=terms, segments=[], track_path=None,
                output_path=None, translation_edited=False, warnings=[], status="running",
                stage="processing", message="Queued", timings={})
    session_log.log_event(jobs.get(job_id), "process_started", success=True)
    bg.add_task(_serial, pipeline.run_process, job_id)
    return {"job_id": job_id, "status": "running"}


# ---------------------------------------------------------------- review (segment IR)
class SegmentEdit(BaseModel):
    edited_text: str = Field(max_length=2000)


def _seg(job: dict, idx: int) -> dict:
    if idx < 0 or idx >= len(job["segments"]):
        raise HTTPException(404, "Segment not found")
    return job["segments"][idx]


@router.put("/jobs/{job_id}/segments/{idx}")
def edit_segment(job_id: str, idx: int, body: SegmentEdit):
    _not_running(_job_or_404(job_id))

    def apply(job):
        seg = _seg(job, idx)
        text = body.edited_text.strip()
        if text == (seg["translated_text"] or "").strip():
            seg["edited_text"], seg["status"] = None, "ai_generated"
        else:
            seg["edited_text"], seg["status"] = text, "edited"
            job["translation_edited"] = True
        # Edits invalidate generated audio/video. translated_text is never overwritten.
        job.update(track_path=None, output_path=None, stage="review")
        return {**seg, "final_text": jobs.final_text(seg)}
    return jobs.mutate(job_id, apply)


@router.post("/jobs/{job_id}/segments/{idx}/approve")
def approve_segment(job_id: str, idx: int):
    _not_running(_job_or_404(job_id))

    def apply(job):
        seg = _seg(job, idx)
        if not jobs.final_text(seg):
            raise HTTPException(400, "Cannot approve an empty segment")
        seg["status"] = "approved"
        return {**seg, "final_text": jobs.final_text(seg)}
    return jobs.mutate(job_id, apply)


@router.post("/jobs/{job_id}/approve-all")
def approve_all(job_id: str):
    _not_running(_job_or_404(job_id))

    def apply(job):
        for seg in job["segments"]:
            if jobs.final_text(seg):
                seg["status"] = "approved"
    jobs.mutate(job_id, apply)
    return jobs.public_view(jobs.get(job_id))


@router.get("/jobs/{job_id}/srt")
def srt(job_id: str):
    job = _job_or_404(job_id)
    if not job["segments"]:
        raise HTTPException(409, "Result not ready")
    body = media.build_srt([{"start": s["start"], "end": s["end"], "text": jobs.final_text(s)}
                            for s in job["segments"]])
    return PlainTextResponse(body, media_type="application/x-subrip",
                             headers={"Content-Disposition": f'attachment; filename="natlas_dub_{job_id[:8]}.srt"'})


# ---------------------------------------------------------------- voice
class TTSRequest(BaseModel):
    speed: float = Field(1.0, ge=0.65, le=1.25)
    voice: Optional[str] = Field(None, max_length=10)


@router.post("/jobs/{job_id}/tts")
def tts(job_id: str, req: TTSRequest, bg: BackgroundTasks):
    job = _job_or_404(job_id)
    _not_running(job)
    if not job["segments"]:
        raise HTTPException(409, "Transcribe and translate first")
    pending = [s["index"] + 1 for s in job["segments"] if s["status"] != "approved" and jobs.final_text(s)]
    if pending:
        raise HTTPException(409, f"Approve all segments first (pending: {pending[:10]})")
    if req.voice:
        if req.voice not in config.VOICES:
            raise HTTPException(400, "Voice must be female or male")
        jobs.update(job_id, voice=req.voice)
    jobs.update(job_id, status="running", stage="tts", message="Queued")
    session_log.log_event(jobs.get(job_id), "tts_started", success=True)
    bg.add_task(_serial, pipeline.run_tts, job_id, req.speed)
    return {"job_id": job_id, "status": "running"}


@router.get("/jobs/{job_id}/audio")
def audio(job_id: str):
    job = _job_or_404(job_id)
    if not job.get("track_path"):
        raise HTTPException(409, "Result not ready")
    return FileResponse(job["track_path"], media_type="audio/wav")


# ---------------------------------------------------------------- render / export
class RenderRequest(BaseModel):
    subtitles: bool = True
    background: bool = False
    sub_position: str = Field("bottom", pattern="^(bottom|top)$")
    mask: bool = False                      # cover burned-in source captions
    mask_height: float = Field(0.14, ge=0.05, le=0.4)


@router.post("/jobs/{job_id}/render")
def render(job_id: str, req: RenderRequest, bg: BackgroundTasks):
    job = _job_or_404(job_id)
    _not_running(job)
    if not job.get("track_path"):
        raise HTTPException(409, "Generate the voice first")
    jobs.update(job_id, status="running", stage="rendering", message="Queued")
    session_log.log_event(jobs.get(job_id), "render_started", success=True)
    bg.add_task(pipeline.run_render, job_id, req.subtitles, req.background,
                req.sub_position, req.mask, req.mask_height)
    return {"job_id": job_id, "status": "running"}


@router.get("/jobs/{job_id}/video")
def video(job_id: str, download: bool = False):
    job = _job_or_404(job_id)
    if job["status"] == "running" or not job.get("output_path"):
        raise HTTPException(409, "Result not ready")
    path = Path(job["output_path"])
    if not path.is_file():
        raise HTTPException(404, "Result file missing")
    if download:
        session_log.log_event(job, "export_downloaded", success=True)
        return FileResponse(str(path), media_type="video/mp4",
                            filename=f"natlas_dub_{job['target_language']}_{job_id[:8]}.mp4")
    return FileResponse(str(path), media_type="video/mp4")


# ---------------------------------------------------------------- session log (read by Render admin)
@router.get("/admin/sessions")
def sessions():
    entries = session_log.read_all()
    return {"count": len(entries), "entries": entries}
