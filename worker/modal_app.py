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
HANDOVER = Path(os.environ.get("HANDOVER_DIR", HERE / "handover"))
REMOTE = "/root/app"
SCALEDOWN = int(os.environ.get("WORKER_SCALEDOWN_S", "300"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core", "fonts-noto-core", "git", "wget")
    # YarnGPT (Nigerian TTS, required): repo for the audio tokenizer + its WavTokenizer decoder files,
    # exactly as the YarnGPT-local model card instructs (outetts==0.2.3, uroman, gdown checkpoint).
    .pip_install("outetts==0.2.3", "uroman", "inflect", "gdown", "pydub")
    .run_commands(
        "git clone --depth 1 https://github.com/saheedniyi02/yarngpt.git /opt/yarngpt",
        "mkdir -p /opt/yarngpt_decoder",
        "wget -q -O /opt/yarngpt_decoder/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml "
        "https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/"
        "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        "cd /opt/yarngpt_decoder && gdown 1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt && ls -la /opt/yarngpt_decoder",
    )
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1", "transformers==4.46.3", "accelerate", "bitsandbytes", "huggingface_hub",
        "sentencepiece", "openai-whisper", "soundfile", "numpy<2.2", "librosa",
        "fastapi", "uvicorn[standard]", "python-multipart", "httpx", "pydantic", "python-dotenv",
        "gradio>=5.49,<6",
    )
    .env({"HF_HOME": "/cache/hf", "JOBS_DIR": "/tmp/natlas_jobs", "SESSION_LOG_PATH": "/cache/natlas_sessions.jsonl",
          "NATLAS_RUNTIME": "modal", "PYTHONPATH": "/opt", "MAX_VIDEO_SECONDS": "180"})
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
        ns.update(load_yarngpt_models(decoder_dir="/opt/yarngpt_decoder"))
        print("YarnGPT restored")
    except Exception as exc:  # fallback stays available, but health reports it
        import traceback
        traceback.print_exc()
        os.environ["YARNGPT_ERROR"] = f"{type(exc).__name__}: {exc}"[:300]
    asgi = start_worker.wire_models(ns)
    cache.commit()
    return asgi
