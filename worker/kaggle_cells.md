# Kaggle worker: notebook cells

Kaggle settings: **GPU T4 x2**, **Internet on**. Add the handover folder (`N_ATLAS_Developer_Handover/`) and this repo's `worker/` folder as Kaggle Datasets (or upload them). Add these as **Kaggle Secrets**: `DEV_API_KEY`, `HF_TOKEN`, `ADMIN_KEY`, `RENDER_APP_URL`.

`DEV_API_KEY` must equal Render's `GPU_WORKER_KEY`, and `ADMIN_KEY` must equal Render's `ADMIN_KEY`.

## Cell 1: secrets and Hugging Face login
```python
import os
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
for k in ("DEV_API_KEY", "HF_TOKEN", "ADMIN_KEY", "RENDER_APP_URL"):
    os.environ[k] = s.get_secret(k)
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])   # account must have accepted the NCAIR1 model licences
```

## Cell 2: install
```python
!pip -q install -r /kaggle/input/<handover-dataset>/N_ATLAS_Developer_Handover/requirements.txt
!pip -q install -r /kaggle/input/<worker-dataset>/worker/requirements-worker.txt
!which ffmpeg ffprobe || (apt-get -qq update && apt-get -qq install -y ffmpeg)
!apt-get -qq install -y fonts-dejavu-core fonts-noto-core >/dev/null 2>&1 || true   # subtitle fonts
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
# YarnGPT (optional but needed for Male/Female voices). Follow https://github.com/saheedniyi02/yarngpt
# for the exact install; at the time of writing it is roughly:
# !pip -q install outetts uroman && pip -q install git+https://github.com/saheedniyi02/yarngpt.git
```

## Cell 3: copy the handover backend and apply the worker patch
```python
import shutil, sys
HO = "/kaggle/working/N_ATLAS_Developer_Handover"
shutil.copytree("/kaggle/input/<handover-dataset>/N_ATLAS_Developer_Handover", HO, dirs_exist_ok=True)
shutil.copytree("/kaggle/input/<worker-dataset>/worker/natlas_dub", f"{HO}/natlas_dub", dirs_exist_ok=True)
shutil.copy("/kaggle/input/<worker-dataset>/worker/start_worker.py", HO)
sys.path.insert(0, HO); os.chdir(HO)
```

## Cell 4 (optional): WavTokenizer decoder files for YarnGPT
These files were **not** in the handover. Verified on Hugging Face on 2026-10-15: the YAML is in
`novateur/WavTokenizer-medium-speech-75token`. The repo `novateur/WavTokenizer-large-speech-75token` has
`wavtokenizer_large_speech_320_v2.ckpt`, not the `..._24k.ckpt` name that `restore_yarngpt.py` expects.
Use the checkpoint the YarnGPT README points to. The v2 file is **untested** with YarnGPT.
```python
!mkdir -p /kaggle/working/yarngpt_decoder
!wget -q -O /kaggle/working/yarngpt_decoder/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml
# Checkpoint: save it as /kaggle/working/yarngpt_decoder/wavtokenizer_large_speech_320_24k.ckpt
```

## Cell 5: restore models (once per fresh runtime)
```python
from restore_models import load_models
ns = load_models()                       # N-ATLaS 8B (4-bit), MMS/SpeechT5 fallback TTS, whisper tiny (unused by /dub)
try:
    from restore_yarngpt import load_yarngpt_models
    ns.update(load_yarngpt_models())     # enables YarnGPT with Female/Male voices
except Exception as e:
    print("YarnGPT not restored, falling back to MMS-TTS/SpeechT5 (no voice choice; mms-tts is CC BY-NC 4.0):", e)
```

## Cell 6: wire, serve, tunnel, register with Render
```python
import start_worker
start_worker.wire_models(ns)
url = start_worker.serve(port=8000)      # uvicorn thread, cloudflared quick tunnel, registration every 120 s
print(url)
```

## Cell 7 (optional): pre-load the ASR model so the first evaluator does not wait for the ~1 GB download
```python
from natlas_dub import asr
asr._get_pipeline(asr.model_for("en"))   # NCAIR1/NigerianAccentedEnglish, on cuda:1 when two GPUs
```

## Cell 8: check
```python
import httpx
print(httpx.get(f"{url}/dub/health", headers={"X-API-Key": os.environ["DEV_API_KEY"]}).json())
print(httpx.get(f"{os.environ['RENDER_APP_URL']}/health", timeout=90).json())   # gpu_worker.online should be True
from natlas_dub import register; print(register.status())
```

Keep the notebook session running. Kaggle sessions end after about 12 hours, or sooner when idle. Quick-tunnel URLs change on every restart. The heartbeat re-registers the new URL automatically.
