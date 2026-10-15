"""Test setup. ALL model calls are mocked here (test-only): no gated ASR download,
no GPU. The fake TTS is a synthetic sine tone and is never used outside tests."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HANDOVER = ROOT.parent / "handover" / "N_ATLAS_Developer_Handover"
TMP = Path(tempfile.mkdtemp(prefix="natlas_test_"))

os.environ.update({
    "DEV_API_KEY": "test-worker-key",
    "GPU_WORKER_KEY": "test-worker-key",
    "ADMIN_KEY": "test-admin-key",
    "JOBS_DIR": str(TMP / "jobs"),
    "SESSION_LOG_PATH": str(TMP / "worker_sessions.jsonl"),
    "RENDER_SESSION_LOG_PATH": str(TMP / "render_sessions.jsonl"),
    "MAX_VIDEO_SECONDS": "90",
    "MAX_VIDEO_MB": "100",
    "HF_TOKEN": "",
})

# Copy the untouched handover api.py/byoa_routes.py into a temp dir so their
# module-level mkdirs don't write into the handover folder.
HO_COPY = TMP / "handover"
HO_COPY.mkdir()
for name in ("api.py", "byoa_routes.py"):
    shutil.copy(HANDOVER / name, HO_COPY / name)
sys.path.insert(0, str(HO_COPY))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT))


def make_clip(path: Path, seconds: float, with_audio: bool = True):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=15:duration={seconds}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


def fake_tone(text, language, voice=None):
    """TEST-ONLY fake TTS: 0.08 s of 440 Hz tone per word, 24 kHz."""
    import numpy as np
    sr = 24000
    dur = max(0.4, 0.08 * len(text.split()))
    t = np.arange(int(sr * dur)) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr


def fake_translate(text, language):
    """TEST-ONLY fake translator: keeps placeholders, prefixes language."""
    return f"[{language}] " + text


FAKE_SEGMENTS = [
    {"start": 0.0, "end": 1.4, "text": "Buy MTN data at www.mtn.ng for ₦500 today", "flags": []},
    {"start": 1.5, "end": 2.8, "text": "Call 0803 123 4567 now", "flags": []},
]
