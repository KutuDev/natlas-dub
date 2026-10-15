# N-ATLAS Dub

**Developer infrastructure for localising video into Nigerian languages.** N-ATLAS Dub turns short English videos into Yoruba, Hausa and Igbo dubs. N-ATLaS transcribes and translates, a person reviews every line, YarnGPT gives it a Nigerian voice, and FFmpeg returns a subtitled MP4. Built by OPEX Consulting Ltd for NAIC 2026 (Problem Statement 1: Developer Infrastructure).

> N-ATLAS Dub, created by OPEX Consulting. Powered by Awarri.
> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.

| | |
|---|---|
| **Live app** | https://natlas-dub.onrender.com |
| **Developer documentation portal (interactive API)** | https://natlas-dub.onrender.com/docs |
| **API reference (ReDoc)** | https://natlas-dub.onrender.com/redoc |
| **OpenAPI spec** | https://natlas-dub.onrender.com/openapi.json |
| **Real-run evidence** | [`evidence/`](evidence/) |

## Models

| Step | Model | Notes |
|---|---|---|
| Speech recognition | `NCAIR1/NigerianAccentedEnglish` (N-ATLaS ASR) | `NCAIR1/Yoruba-ASR`, `Hausa-ASR`, `Igbo-ASR` for source videos in those languages |
| Translation | `NCAIR1/N-ATLaS` (8B) | 4-bit NF4 (bitsandbytes) on one NVIDIA L4 |
| Voice | `saheedniyi/YarnGPT-local` (yo/ha/ig), `saheedniyi/YarnGPT2` (en) | WavTokenizer `wavtokenizer_large_speech_320_24k.ckpt`; six voices |
| Render | FFmpeg | Clip fitting, mixing, caption mask, burned subtitles, SRT |

## Quick start: call the API

No GPU, model downloads or API key are needed to use the hosted service. The first call after idle wakes the GPU worker, which takes a few minutes.

```bash
BASE=https://natlas-dub.onrender.com

# 1. Upload (you must own the video or have rights to it)
JOB=$(curl -s -F file=@advert.mp4 -F consent=true -F tester_name="Your Name" \
      $BASE/api/upload | python3 -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

# 2. N-ATLaS ASR + translation into Yoruba (yo), Hausa (ha) or Igbo (ig)
curl -s -X POST $BASE/api/jobs/$JOB/process -H 'Content-Type: application/json' \
     -d '{"target_language":"yo","voice":"female","protected_terms":["Moniepoint"]}'

# 3. Poll until status is "done", then read the segments
curl -s $BASE/api/jobs/$JOB

# 4. Review: edit a line, then approve everything
curl -s -X PUT $BASE/api/jobs/$JOB/segments/0 -H 'Content-Type: application/json' -d '{"edited_text":"..."}'
curl -s -X POST $BASE/api/jobs/$JOB/approve-all

# 5. YarnGPT voice, then render (poll GET /api/jobs/$JOB between steps)
curl -s -X POST $BASE/api/jobs/$JOB/tts -H 'Content-Type: application/json' -d '{"speed":1.0}'
curl -s -X POST $BASE/api/jobs/$JOB/render -H 'Content-Type: application/json' \
     -d '{"subtitles":true,"sub_position":"bottom","mask":true,"mask_height":0.14}'

# 6. Download
curl -s -o dubbed.mp4 $BASE/api/jobs/$JOB/video
curl -s -o dubbed.srt $BASE/api/jobs/$JOB/srt
```

## Public API

Full request and response schemas are at [/docs](https://natlas-dub.onrender.com/docs).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service and worker status, limits |
| POST | `/api/upload` | Multipart `file`, `consent=true`, optional `tester_name`, `tester_email`, `trim_to_seconds`. Returns `job_id`, `duration`, `size_mb`, `trimmed_from` |
| POST | `/api/jobs/{id}/process` | `target_language` (yo/ha/ig/en), `source_language` (default en), `voice` (female/male), `protected_terms[]` |
| GET | `/api/jobs/{id}` | `status` (idle/running/done/error), `stage`, `segments[]` (with `final_text`), `warnings[]`, `timings` |
| PUT | `/api/jobs/{id}/segments/{i}` | `edited_text` |
| POST | `/api/jobs/{id}/segments/{i}/approve`, `/api/jobs/{id}/approve-all` | Approve |
| POST | `/api/jobs/{id}/tts` | `speed` 0.65–1.25, optional `voice`. 409 until all segments are approved |
| POST | `/api/jobs/{id}/render` | `subtitles`, `background`, `sub_position` (bottom/top), `mask`, `mask_height` 0.05–0.4 |
| GET | `/api/jobs/{id}/{video,audio,srt,source}` | Downloads (MP4, WAV, SRT, original) |

Errors: 400 bad input, 401 bad API key (only if `PUBLIC_API_KEY` is set), 404 unknown job, 409 step not ready, 413 file over 100 MB or video over 180 s, 429 rate limit, 502/503 GPU worker offline.

## Architecture

```
Browser / Flutter app / your code
        │ HTTPS
        ▼
Render web service (FastAPI, free plan)            natlas-dub.onrender.com
  static UI · /docs · upload size and rate limits · streaming proxy /api/* → worker /dub/*
        │ HTTPS + X-API-Key (server-side secret, never sent to clients)
        ▼
Modal GPU worker (NVIDIA L4, 24 GB RAM, 1 container, scales to zero after 300 s idle)
  N-ATLaS ASR → entity masking → N-ATLaS translation → human review
  → YarnGPT TTS → FFmpeg render · JSONL session log on volume natlas-dub-cache
```

Render runs no ML and no FFmpeg. If the worker cannot be reached, the UI says the GPU worker is offline. No output is ever faked.

## Pipeline and segment model
1. **Upload** (consent required) is streamed to the worker. Videos over 100 MB get a 413. ffprobe enforces the 180 s limit (MAX_VIDEO_SECONDS), also with a 413. With `trim_to_seconds` set, a longer video is cut to its first 180 s instead.
2. **ASR** runs on the worker GPU. It uses the transformers ASR pipeline with `return_timestamps=True` and produces timed segments. Segments are flagged by heuristic (empty, implausible speech rate, repetition) because the pipeline does not expose log-probs.
3. **Masking, then translation.** URLs, emails, phone numbers, currency amounts, @handles/#tags, ALL-CAPS and CamelCase brand tokens, and the user's protected terms become `[[Pn]]` placeholders. N-ATLaS translates the masked text and the placeholders are then restored. If the model drops a placeholder, the segment gets a warning; nothing is silently invented.
4. **Review.** Each segment holds `{start, end, original_text, target_language, translated_text, edited_text, status: ai_generated|edited|approved, flags, warnings}`. `final_text = edited_text or translated_text` is the single source of truth for TTS and subtitles. Approval never re-translates. TTS is blocked (409) until every segment is approved.
5. **Voice.** The worker generates one TTS clip per segment, then applies the speed slider (0.65x–1.25x). Each clip is placed at its segment start. A clip longer than its slot is atempo-fitted up to a 1.25x cap; beyond that the clip overlaps and the UI warns.
6. **Render.** The worker replaces the audio with the dub track. Optional extras: the original audio at 12% as "background" (there is no stem separation, so the original voice is faintly audible), SRT subtitles burned at the bottom or top, and a solid lower band (default 14% of frame height) to cover captions already burned into the source.


## Deploy your own

1. **Worker (Modal).** Add repository secrets `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN` (with access to the gated NCAIR1 models) and `DEV_API_KEY`. Every push to `main` runs [`.github/workflows/deploy-modal.yml`](.github/workflows/deploy-modal.yml), which deploys [`worker/modal_app.py`](worker/modal_app.py) and records the worker URL as the GitHub deployment environment `modal-worker`.
2. **Web layer (Render).** New → Blueprint picks up `render.yaml` (Docker, free plan, health check `/health`). Set `GPU_WORKER_URL` to the Modal URL, `GPU_WORKER_KEY` to the same value as `DEV_API_KEY`, `ADMIN_KEY`, and `WORKER_ON_DEMAND=true`.

## Environment variables
| Where | Var | Purpose |
|---|---|---|
| Render | `GPU_WORKER_KEY` | Sent as `X-API-Key` to the worker; must equal the worker's `DEV_API_KEY` |
| Render | `ADMIN_KEY` | Protects `/admin/*` (fails closed with 503 if unset); the worker uses it to register |
| Render | `GPU_WORKER_URL` | Modal worker URL (used when no worker has self-registered) |
| Render | `WORKER_ON_DEMAND` | `true`: `/health` does not wake the GPU; the first job does |
| Render | `MAX_VIDEO_MB` / `MAX_VIDEO_SECONDS` | Size cap enforced while streaming; seconds is informational here |
| Render | `PUBLIC_API_KEY` | Optional; when set, `/api/*` requires `X-API-Key`. Leave empty for evaluators |
| Render | `RATE_LIMIT_UPLOADS`, `RATE_LIMIT_WINDOW_S`, `WORKER_TIMEOUT`, `WORKER_MEDIA_TIMEOUT`, `EXTRA_ALLOWED_ORIGINS`, `RENDER_SESSION_LOG_PATH` | Tuning |
| Worker | `DEV_API_KEY` | Handover auth key; the worker refuses to start without it |
| Worker | `HF_TOKEN` | Gated NCAIR1 models (N-ATLaS, ASR) |
| Worker | `RENDER_APP_URL`, `ADMIN_KEY` | Optional self-registration (tunnel deployments only; Modal uses Render's `GPU_WORKER_URL`) |
| Worker | `ASR_MODEL`, `ASR_MODEL_YO/HA/IG`, `ASR_DEVICE`, `MAX_VIDEO_SECONDS`, `MAX_VIDEO_MB`, `JOB_TTL_HOURS`, `REGISTER_INTERVAL_S`, `SESSION_LOG_PATH`, `JOBS_DIR` | Tuning |


## Run locally

```bash
pip install -r requirements-dev.txt
pytest -q                                   # 19 tests
ADMIN_KEY=dev GPU_WORKER_KEY=k uvicorn app.main:app --port 10000   # UI at http://localhost:10000, docs at /docs
```

The automated tests replace the ASR, translation and TTS models with stand-ins. They check routes, proxying, authentication, limits, job state, rendering, caption masking and errors, not model quality.

## Validation and evidence

Real end-to-end runs on the Modal L4 worker, 15 October 2026, using two public Nigerian adverts:

| Run | Video | ASR | N-ATLaS translation | YarnGPT voice | Render |
|---|---|---|---|---|---|
| Financial advert → Yoruba | 62.47 s | 11.29 s | 40.23 s | 269.14 s | 10.82 s |
| Engineering advert → Hausa | 56.47 s | 11.47 s | 25.14 s | 271.10 s | 12.51 s |
| Engineering advert → Igbo | 56.47 s | 4.35 s | 32.54 s | 297.21 s | 12.34 s |

Transcripts, translations, warnings and subtitles are in [`evidence/`](evidence/). The worker logs every upload, stage start and completion, edit and error. Admins read the log at `GET /admin/sessions` with the `X-Admin-Key` header.

## Licences and notices
- N-ATLaS and the NCAIR1 ASR models: follow the N-ATLaS licence, which requires the attribution above, caps use at 1,000 active users without a separate agreement, and prohibits impersonation and synthetic fraud. The UI requires a rights and consent checkbox.
- **Fallback TTS `facebook/mms-tts-yor` / `mms-tts-hau` is CC BY-NC 4.0 (non-commercial).** It is used only when YarnGPT is not restored, and the UI then warns that voice choice is unavailable. `Shinzmann/soro-tts-ibo` and SpeechT5 are also fallback-only; check their licences before any commercial use.
- YarnGPT: see saheedniyi/YarnGPT-local and YarnGPT2. WavTokenizer: novateur (MIT).


## Known limitations

- ASR can mishear brand and fintech terms (for example "Moniepoint" as "money points"), and N-ATLaS can choose the wrong sense of a word (for example "settlement"). Human review of every line is therefore required before voicing.
- YarnGPT voicing takes about 4.5 to 5 times the video's length on an L4.
- Some dubbed lines run longer than their slot even after a 1.25x speed-up. The user is warned for each one.
- One GPU container: concurrent jobs queue. Jobs are temporary and are lost when the container scales down, so download outputs in the same session.
- Translation quality has not yet been benchmarked against professional reference translations.
- YarnGPT-local is CC BY-NC-SA 4.0: commercial use needs the authors' permission.
