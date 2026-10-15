# N-ATLAS Dub

N-ATLAS Dub dubs short videos into Yoruba, Igbo, Hausa and Nigerian English. A person reviews the translation before any voice is generated. It is the NAIC 2026 submission from OPEX Consulting.

> N-ATLAS Dub, created by OPEX Consulting. Powered by Awarri.
> N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.

## Architecture

```
Browser ──► Render (free, 512 MB)                         ──►  Kaggle GPU worker (T4 x2)
            app/  FastAPI thin layer                           handover api.py  (unchanged)
            • static UI (static/)                              + worker/natlas_dub  (/dub/* router)
            • authenticated proxy /api/* → worker /dub/*         • N-ATLAS ASR  NCAIR1/NigerianAccentedEnglish,
            • streamed upload pass-through (size cap)              NCAIR1/Yoruba-ASR, Hausa-ASR, Igbo-ASR
            • /health (worker reachability)                      • brand/entity masking → N-ATLaS 8B translation
            • /admin/worker (tunnel registration)                • segment review / approve
            • /admin/sessions (validation log)                   • YarnGPT TTS (MMS/SpeechT5 fallback)
                                                                 • FFmpeg: fit, mix, subtitles, mask, MP4
                                                                 • duration gate, trim helper, JSONL log
                 ◄──── POST /admin/worker {url} every 120 s ────  cloudflared quick tunnel
```

Render runs no ML and no FFmpeg. If the worker is not registered or cannot be reached, the UI shows **"GPU worker offline — processing unavailable"**. No output is ever faked.

### Pipeline and segment IR
1. **Upload** (consent required) is streamed to the worker. Videos over 100 MB get a 413. ffprobe enforces the 180 s limit (MAX_VIDEO_SECONDS), also with a 413. An optional *trim to first 60 s* re-encodes the first 180 s.
2. **ASR** runs on the worker GPU. It uses the transformers ASR pipeline with `return_timestamps=True` and produces timed segments. Segments are flagged by heuristic (empty, implausible speech rate, repetition) because the pipeline does not expose log-probs.
3. **Masking, then translation.** URLs, emails, phone numbers, currency amounts, @handles/#tags, ALL-CAPS and CamelCase brand tokens, and the user's protected terms become `[[Pn]]` placeholders. N-ATLaS translates the masked text and the placeholders are then restored. If the model drops a placeholder, the segment gets a warning; nothing is silently invented.
4. **Review.** Each segment holds `{start, end, original_text, target_language, translated_text, edited_text, status: ai_generated|edited|approved, flags, warnings}`. `final_text = edited_text or translated_text` is the single source of truth for TTS and subtitles. Approval never re-translates. TTS is blocked (409) until every segment is approved.
5. **Voice.** The worker generates one TTS clip per segment, then applies the speed slider (0.65x–1.25x). Each clip is placed at its segment start. A clip longer than its slot is atempo-fitted up to a 1.25x cap; beyond that the clip overlaps and the UI warns.
6. **Render.** The worker replaces the audio with the dub track. Optional extras: the original audio at 12% as "background" (there is no stem separation, so the original voice is faintly audible), SRT subtitles burned at the bottom or top, and a solid lower band (default 14% of frame height) to cover captions already burned into the source.

## Environment variables
| Where | Var | Purpose |
|---|---|---|
| Render | `GPU_WORKER_KEY` | Sent as `X-API-Key` to the worker; must equal the worker's `DEV_API_KEY` |
| Render | `ADMIN_KEY` | Protects `/admin/*` (fails closed with 503 if unset); the worker uses it to register |
| Render | `GPU_WORKER_URL` | Optional fallback when nothing is registered |
| Render | `MAX_VIDEO_MB` / `MAX_VIDEO_SECONDS` | Size cap enforced while streaming; seconds is informational here |
| Render | `PUBLIC_API_KEY` | Optional; when set, `/api/*` requires `X-API-Key`. Leave empty for evaluators |
| Render | `RATE_LIMIT_UPLOADS`, `RATE_LIMIT_WINDOW_S`, `WORKER_TIMEOUT`, `WORKER_MEDIA_TIMEOUT`, `EXTRA_ALLOWED_ORIGINS`, `RENDER_SESSION_LOG_PATH` | Tuning |
| Worker | `DEV_API_KEY` | Handover auth key; the worker refuses to start without it |
| Worker | `HF_TOKEN` | Gated NCAIR1 models (N-ATLaS, ASR) |
| Worker | `RENDER_APP_URL`, `ADMIN_KEY` | Self-registration |
| Worker | `ASR_MODEL`, `ASR_MODEL_YO/HA/IG`, `ASR_DEVICE`, `MAX_VIDEO_SECONDS`, `MAX_VIDEO_MB`, `JOB_TTL_HOURS`, `REGISTER_INTERVAL_S`, `SESSION_LOG_PATH`, `JOBS_DIR` | Tuning |

## Run locally
```bash
pip install -r requirements-dev.txt
pytest -q                                   # 19 tests; models are mocked (test-only)
ADMIN_KEY=dev GPU_WORKER_KEY=k uvicorn app.main:app --port 10000   # UI at http://localhost:10000
```
To run a worker locally without a GPU, mount the router onto the handover `api.app` with `start_worker.attach()`. Every processing step will then report "model not loaded". That is the intended behaviour.

## Deploy
**Render.** Push the repo, then choose New → Blueprint so Render picks up `render.yaml` (Docker, plan `free`, health check `/health`). Set `GPU_WORKER_KEY` and `ADMIN_KEY`. The free plan sleeps after about 15 minutes idle and takes about a minute to wake. The UI retries during wake-up. While the worker is running, its 120 s heartbeat keeps Render awake.

**Worker.** Follow `worker/kaggle_cells.md`: install, copy the handover backend, add `worker/natlas_dub` and `start_worker.py`, restore models, then `start_worker.wire_models(ns)` and `start_worker.serve(8000)`.

## GPU worker contract (Render → worker, all with `X-API-Key: <DEV_API_KEY>`)
| Method | Path | Body / result |
|---|---|---|
| GET | `/dub/health` | `{status, ffmpeg, asr{...}, models{translate, speech, speech_backend, speech_supports_voice}, limits}` |
| POST | `/dub/upload` | multipart: `file`, `consent=true`, `tester_name?`, `tester_email?`, `trim_to_seconds?` → `{job_id, duration, size_mb, trimmed_from}` |
| GET | `/dub/jobs/{id}` | job + segments (`final_text`, `char_count`), `status: idle|running|done|error`, `error_code`, `warnings` |
| POST | `/dub/jobs/{id}/process` | `{source_language: en|yo|ha|ig, target_language: yo|ig|ha|en, voice: female|male, protected_terms: []}` |
| PUT | `/dub/jobs/{id}/segments/{i}` | `{edited_text}` |
| POST | `/dub/jobs/{id}/segments/{i}/approve`, `/dub/jobs/{id}/approve-all` | |
| GET | `/dub/jobs/{id}/srt` | SRT from `final_text` |
| POST | `/dub/jobs/{id}/tts` | `{speed: 0.65–1.25, voice?}` (409 until all approved) |
| GET | `/dub/jobs/{id}/audio` | WAV dub track (409 `Result not ready`) |
| POST | `/dub/jobs/{id}/render` | `{subtitles, background, sub_position: bottom|top, mask, mask_height: 0.05–0.4}` |
| GET | `/dub/jobs/{id}/video[?download=true]` | MP4 (409 `Result not ready`), supports Range |
| GET | `/dub/jobs/{id}/source` | original upload |
| GET | `/dub/admin/sessions` | worker JSONL entries |
| Worker → Render | `POST {RENDER_APP_URL}/admin/worker` | header `X-Admin-Key`, `{url}`; https only |

The handover's original endpoints (`/health`, `/upload`, `/transcribe/{id}`, `/translate`, `/generate-speech`, `/localize`, `/jobs`, `/results`, BYOA routes) are unchanged and still served. `/transcribe` now uses N-ATLAS ASR.

## Validation log
The worker appends JSONL to `/kaggle/working/natlas_sessions.jsonl` for each real event: transcribe/translate, voice, render, export, and failures. Each entry has timestamp, job id, optional tester name/email, languages, voice, duration, `trimmed_from`, per-stage timings, success/error and `translation_edited`. Render logs only proxy-level events (worker offline, rejected or rate-limited uploads). `GET /admin/sessions` (header `X-Admin-Key`) merges both. Render free has no persistent disk, so Render's part is lost on restart. Download the worker file from Kaggle output.

## Licences and notices
- N-ATLaS and the NCAIR1 ASR models: follow the N-ATLaS licence, which prohibits impersonation and synthetic fraud. The UI requires a rights/consent checkbox.
- **Fallback TTS `facebook/mms-tts-yor` / `mms-tts-hau` is CC BY-NC 4.0 (non-commercial).** It is used only when YarnGPT is not restored, and the UI then warns that voice choice is unavailable. `Shinzmann/soro-tts-ibo` and SpeechT5 are also fallback-only; check their licences before any commercial use.
- YarnGPT: see saheedniyi/YarnGPT-local and YarnGPT2. WavTokenizer: novateur (MIT).

## Known limitations (honest)
- No model inference has been run in development. The gated ASR models, N-ATLaS and YarnGPT were mocked in tests. The first real run happens on Kaggle.
- The handover's `multilingual_translate` prompt says "Translate the following English text" with `max_new_tokens=250`. Non-English source videos and long segments are therefore unverified. Placeholders like `[[P0]]` may be altered by the model; the restore step is tolerant and dropped placeholders are flagged.
- YarnGPT needs WavTokenizer files that were not in the handover (see `worker/kaggle_cells.md`). Without them the worker falls back to MMS/SpeechT5, where Male/Female voice choice has no effect.
- The "Background music" option is the original soundtrack at low volume, not separated music.
- Jobs are kept in memory on the worker (files under `/kaggle/working/natlas_jobs`, TTL 6 h). A worker restart loses in-flight jobs.
- Quick-tunnel URLs change on every restart, and Cloudflare caps request bodies at about 100 MB. Render free sleeps, so the first visit can take about a minute.
- Subtitle font is DejaVu Sans. Yoruba stacked tone marks on dotted vowels (ọ̀, ẹ́) rendered in a spot check but are small at 720p. Install `fonts-noto-core` on the worker if they look wrong.
