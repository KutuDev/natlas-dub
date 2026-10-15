"""Render (thin layer) configuration. No ML, no FFmpeg here."""

import os
from pathlib import Path


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# Fallback worker URL; the live Cloudflare tunnel URL is normally registered at runtime
# via POST /admin/worker and held in memory (lost when the free instance sleeps).
GPU_WORKER_URL = os.environ.get("GPU_WORKER_URL", "").rstrip("/")
GPU_WORKER_KEY = os.environ.get("GPU_WORKER_KEY", "")  # must equal the worker's DEV_API_KEY
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")
# true when the worker is serverless (Modal): health polls must not wake the GPU.
WORKER_ON_DEMAND = os.environ.get("WORKER_ON_DEMAND", "").lower() in ("1", "true", "yes")
# Optional: when set, /api/* requires X-API-Key. Leave unset so evaluators can use the UI.
PUBLIC_API_KEY = os.environ.get("PUBLIC_API_KEY", "")

MAX_VIDEO_SECONDS = _float("MAX_VIDEO_SECONDS", 90)   # enforced on the worker (ffprobe)
MAX_VIDEO_MB = _float("MAX_VIDEO_MB", 100)            # enforced here while streaming AND on the worker
MAX_JSON_BYTES = int(_float("MAX_JSON_BYTES", 256 * 1024))

# Timeouts (s). Handover BYOA used 60/90; uploads/media get longer.
TIMEOUT_DEFAULT = _float("WORKER_TIMEOUT", 60)
TIMEOUT_MEDIA = _float("WORKER_MEDIA_TIMEOUT", 300)
WORKER_RETRIES = int(_float("WORKER_RETRIES", 1))

# Per-IP rate limit on job creation (uploads).
RATE_LIMIT_UPLOADS = int(_float("RATE_LIMIT_UPLOADS", 6))
RATE_LIMIT_WINDOW_S = _float("RATE_LIMIT_WINDOW_S", 600)

SESSION_LOG_PATH = Path(os.environ.get("RENDER_SESSION_LOG_PATH", "/tmp/natlas_render_sessions.jsonl"))

# CORS: only the Render origin (Render injects RENDER_EXTERNAL_URL) plus optional extras.
ALLOWED_ORIGINS = [o for o in [os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")]
                   + [x.strip().rstrip("/") for x in os.environ.get("EXTRA_ALLOWED_ORIGINS", "").split(",")] if o]
