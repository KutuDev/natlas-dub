"""Append-only JSONL validation log. Entries are written only from real job events."""

import json
import threading
from datetime import datetime, timezone
from typing import List

from . import config

_lock = threading.Lock()


def log_event(job: dict, event: str, success: bool, error: str = None) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "worker",
        "event": event,
        "job_id": job.get("job_id"),
        "tester_name": job.get("tester_name") or None,
        "tester_email": job.get("tester_email") or None,
        "source_language": job.get("source_language"),
        "target_language": job.get("target_language"),
        "voice": job.get("voice"),
        "video_duration_s": round(float(job.get("duration") or 0), 2),
        "trimmed_from_s": job.get("trimmed_from"),
        "segments": len(job.get("segments") or []),
        "timings_s": job.get("timings", {}),
        "success": success,
        "error": error,
        "translation_edited": bool(job.get("translation_edited")),
    }
    config.SESSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(config.SESSION_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_all() -> List[dict]:
    if not config.SESSION_LOG_PATH.exists():
        return []
    out = []
    with open(config.SESSION_LOG_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out
