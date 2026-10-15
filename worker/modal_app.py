"""N-ATLAS Dub GPU worker on Modal (alternative to the Kaggle notebook).

Deploy:  modal deploy worker/modal_app.py
Needs Modal secret "natlas-dub-worker" with HF_TOKEN, DEV_API_KEY (= Render GPU_WORKER_KEY).
The web endpoint URL is stable, so Render uses it as GPU_WORKER_URL; no tunnel, no registration.
One container (jobs live in memory), scales to zero after SCALEDOWN seconds idle.
"""

import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
HANDOVER = Path(os.environ.get("HANDOVER_DIR", HERE.parent.parent / "handover" / "N_ATLAS_Developer_Handover"))
REMOTE = "/root/app"
SCALEDOWN = int(os.environ.get("WORKER_SCALEDOWN_S", "300"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core", "fonts-noto-core")
    .pip_install(
        "torch==2.5.1", "transformers==4.46.3", "accelerate", "bitsandbytes", "huggingface_hub",
        "sentencepiece", "openai-whisper", "soundfile", "numpy<2.2", "librosa",
        "fastapi", "uvicorn[standard]", "python-multipart", "httpx", "pydantic", "python-dotenv",
        "gradio>=5.49,<6",
    )
    .env({"HF_HOME": "/cache/hf", "JOBS_DIR": "/tmp/natlas_jobs", "SESSION_LOG_PATH": "/cache/natlas_sessions.jsonl",
          "NATLAS_RUNTIME": "modal"})
    .add_local_dir(str(HANDOVER), REMOTE, ignore=["__pycache__", "*.pyc"])
    .add_local_dir(str(HERE / "natlas_dub"), f"{REMOTE}/natlas_dub", ignore=["__pycache__"])
    .add_local_file(str(HERE / "start_worker.py"), f"{REMOTE}/start_worker.py")
)

cache = modal.Volume.from_name("natlas-dub-cache", create_if_missing=True)
app = modal.App("natlas-dub-worker")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name("natlas-dub-worker")],
    timeout=1800,
    scaledown_window=SCALEDOWN,
    max_containers=1,
    memory=24576,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(label="natlas-dub-worker")
def web():
    import sys

    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)
    import start_worker
    from restore_models import load_models

    ns = load_models()
    try:
        from restore_yarngpt import load_yarngpt_models
        ns.update(load_yarngpt_models())
    except Exception as exc:  # documented fallback
        print("YarnGPT not restored; MMS/SpeechT5 fallback:", exc)
    asgi = start_worker.wire_models(ns)
    cache.commit()
    return asgi
