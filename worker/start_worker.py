"""N-ATLAS Dub worker entry point (Kaggle GPU notebook).

Copy this file and the natlas_dub/ package into the handover backend folder
(N_ATLAS_Developer_Handover/), next to api.py. It does NOT modify api.py or
byoa_routes.py: it mounts the /dub router onto the existing api.app with the
handover's own X-API-Key dependency and wires the restored models in.

Usage (see kaggle_cells.md):
    import start_worker
    app = start_worker.wire_models(namespace)        # namespace from restore_models / restore_yarngpt
    url = start_worker.serve(port=8000)               # uvicorn thread + cloudflared + Render registration
"""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MMS_LICENSE_NOTE = "MMS-TTS/SpeechT5 fallback (facebook/mms-tts-* is CC BY-NC 4.0)"


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"t": self.formatTime(record), "level": record.levelname,
                           "logger": record.name, "msg": record.getMessage()})


def setup_logging():
    root = logging.getLogger("natlas_dub")
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(_JsonFormatter())
        root.addHandler(h)
        root.setLevel(logging.INFO)


def attach():
    """Mount /dub routes on the handover app, protected by its X-API-Key dependency."""
    import api
    from natlas_dub import asr, routes

    if not os.environ.get("DEV_API_KEY"):
        raise RuntimeError("Set DEV_API_KEY securely before starting the worker.")
    if not getattr(api.app.state, "natlas_dub_attached", False):
        api.app.include_router(routes.router, dependencies=api.protected)
        api.app.state.natlas_dub_attached = True
    # Keep the original /transcribe/{video_id} endpoint working, now backed by N-ATLAS ASR.
    api.configure_ai_services(transcribe=asr.transcribe_video_text)
    return api.app


def wire_models(namespace: dict):
    """Connect models restored by restore_models.load_models() (+ optional YarnGPT)."""
    setup_logging()
    import api
    import multilingual_ai as ai
    from natlas_dub import handlers

    for name, value in namespace.items():
        setattr(ai, name, value)

    def fallback(text, language):
        return ai.four_language_speech(text, language)

    yarn_keys = ("yarngpt_audio_tokenizer", "yarngpt_model", "yarngpt2_audio_tokenizer", "yarngpt2_model")
    if all(namespace.get(k) is not None for k in yarn_keys):
        import yarngpt_service as ys
        ys.configure_yarngpt(namespace["yarngpt_audio_tokenizer"], namespace["yarngpt_model"],
                             namespace["yarngpt2_audio_tokenizer"], namespace["yarngpt2_model"], fallback)
        speech = ys.final_speech_handler           # (text, language, voice) -> (wave, 24000)
        backend, supports_voice = "YarnGPT (local + YarnGPT2 English)", True
    else:
        def speech(text, language, voice=None):  # voice ignored by MMS/SpeechT5
            return fallback(text, language)
        backend, supports_voice = MMS_LICENSE_NOTE, False

    handlers.configure(translate=ai.multilingual_translate, speech=speech,
                       speech_backend=backend, speech_supports_voice=supports_voice)
    # Original handover endpoints (/translate, /generate-speech, /localize) keep working.
    api.configure_ai_services(translate=ai.multilingual_translate,
                              speech=lambda text, language: speech(text, language, None))
    api.configure_video_renderer(ai.multilingual_render)
    logging.getLogger("natlas_dub").info("models wired; speech backend: %s", backend)
    return attach()


def serve(port: int = 8000, tunnel: bool = True, register: bool = True) -> str:
    """Start uvicorn in a background thread (works inside a running notebook loop)."""
    import uvicorn
    from natlas_dub import register as reg

    setup_logging()
    app = attach()
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info"))
    threading.Thread(target=server.run, daemon=True, name="uvicorn").start()
    for _ in range(60):
        if server.started:
            break
        time.sleep(0.5)
    url = f"http://localhost:{port}"
    if tunnel:
        url = reg.start_tunnel(port)
        print("Worker public URL:", url)
    if register:
        reg.start_heartbeat(url)
    return url
