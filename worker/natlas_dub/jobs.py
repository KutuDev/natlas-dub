"""In-memory job store + segment intermediate representation (IR).

Segment IR fields: index, start, end, original_text, target_language,
translated_text (AI output, never overwritten after approval),
edited_text, status (ai_generated | edited | approved), flags, warnings.

final_text = edited_text if present else translated_text. It is the single
source of truth for TTS and subtitles.
"""

import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from . import config

_jobs: Dict[str, dict] = {}
_lock = threading.Lock()


def new_job(**fields) -> dict:
    job_id = uuid.uuid4().hex
    job_dir = config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "job_id": job_id,
        "dir": str(job_dir),
        "created": time.time(),
        "stage": "uploaded",       # uploaded|processing|review|tts|voiced|rendering|rendered
        "status": "idle",          # idle|running|done|error
        "error": None,
        "message": "",
        "segments": [],
        "warnings": [],
        "timings": {},
        "translation_edited": False,
        **fields,
    }
    with _lock:
        _jobs[job_id] = job
    return job


_ID_RE = re.compile(r"[0-9a-f]{32}")


def valid_id(job_id: str) -> bool:
    return bool(_ID_RE.fullmatch(job_id or ""))


def get(job_id: str) -> Optional[dict]:
    if not valid_id(job_id):
        return None
    with _lock:
        return _jobs.get(job_id)


def mutate(job_id: str, fn):
    """Run fn(job) under the store lock (used for segment edits/approvals)."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return fn(job)


def discard(job_id: str) -> None:
    with _lock:
        job = _jobs.pop(job_id, None)
    if job:
        shutil.rmtree(job["dir"], ignore_errors=True)


def update(job_id: str, **fields) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def final_text(seg: dict) -> str:
    edited = seg.get("edited_text")
    return (edited if edited is not None and edited.strip() else seg.get("translated_text") or "").strip()


def public_view(job: dict) -> dict:
    hidden = {"dir", "video_path", "audio_path", "track_path", "output_path", "srt_path", "clip_paths"}
    view = {k: v for k, v in job.items() if k not in hidden}
    view["segments"] = [{**s, "final_text": final_text(s), "char_count": len(final_text(s))}
                        for s in job.get("segments", [])]
    view["has_audio_preview"] = bool(job.get("track_path"))
    view["has_video"] = bool(job.get("output_path"))
    return view


def cleanup_expired() -> int:
    cutoff = time.time() - config.JOB_TTL_HOURS * 3600
    removed = 0
    with _lock:
        expired = [jid for jid, j in _jobs.items() if j["created"] < cutoff and j["status"] != "running"]
        for jid in expired:
            shutil.rmtree(_jobs[jid]["dir"], ignore_errors=True)
            del _jobs[jid]
            removed += 1
        known = set(_jobs)
    # Orphaned job dirs (e.g. left over from a previous worker process).
    if config.JOBS_DIR.exists():
        for d in config.JOBS_DIR.iterdir():
            try:
                if d.is_dir() and d.name not in known and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    return removed
