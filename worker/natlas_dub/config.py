"""Worker configuration (runs on the Kaggle GPU box), read from environment variables."""

import os
from pathlib import Path


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_HOME = os.environ.get("HF_HOME", "") or None

# Default ASR model (Nigerian-accented English). Override with ASR_MODEL.
ASR_MODEL = os.environ.get("ASR_MODEL", "NCAIR1/NigerianAccentedEnglish")
# Source-language specific N-ATLAS ASR models.
ASR_MODELS_BY_LANGUAGE = {
    "en": ASR_MODEL,
    "yo": os.environ.get("ASR_MODEL_YO", "NCAIR1/Yoruba-ASR"),
    "ha": os.environ.get("ASR_MODEL_HA", "NCAIR1/Hausa-ASR"),
    "ig": os.environ.get("ASR_MODEL_IG", "NCAIR1/Igbo-ASR"),
}

# ASR device: "auto" -> cuda:1 when two GPUs (N-ATLaS sits on cuda:0), else cuda:0, else CPU.
ASR_DEVICE = os.environ.get("ASR_DEVICE", "auto")

# Registration with the Render app (see register.py).
RENDER_APP_URL = os.environ.get("RENDER_APP_URL", "").rstrip("/")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")
REGISTER_INTERVAL_S = _float("REGISTER_INTERVAL_S", 120)

MAX_VIDEO_SECONDS = _float("MAX_VIDEO_SECONDS", 90)
MAX_VIDEO_MB = _float("MAX_VIDEO_MB", 100)

JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/kaggle/working/natlas_jobs" if Path("/kaggle/working").exists() else "/tmp/natlas_jobs"))
JOB_TTL_HOURS = _float("JOB_TTL_HOURS", 6)
SESSION_LOG_PATH = Path(os.environ.get("SESSION_LOG_PATH", "/kaggle/working/natlas_sessions.jsonl" if Path("/kaggle/working").exists() else "/tmp/natlas_sessions.jsonl"))

# Max speed-up when fitting a TTS clip into its time slot.
MAX_FIT_TEMPO = _float("MAX_FIT_TEMPO", 1.25)
BACKGROUND_VOLUME = _float("BACKGROUND_VOLUME", 0.12)

TARGET_LANGUAGES = {
    "yo": "Yoruba",
    "ig": "Igbo",
    "ha": "Hausa",
    "en": "English",  # Nigerian English
}
SOURCE_LANGUAGES = {"en": "English", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}
VOICES = {"female", "male"}
