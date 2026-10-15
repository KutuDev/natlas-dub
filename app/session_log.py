"""Render-side JSONL log for events only the proxy can see (worker offline, rejected uploads,
rate limits). Pipeline events (timings, edits) are logged by the worker and merged in
/admin/sessions. Ephemeral on Render free (no disk): download it via /admin/sessions."""

import json
import threading
from datetime import datetime, timezone

from . import config

_lock = threading.Lock()


def log_event(event: str, success: bool, **fields) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "source": "render", "event": event, "success": success, **fields}
    config.SESSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(config.SESSION_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_all():
    if not config.SESSION_LOG_PATH.exists():
        return []
    out = []
    with _lock, open(config.SESSION_LOG_PATH, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out
