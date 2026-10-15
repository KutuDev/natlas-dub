# N-ATLAS Multilingual Video Localizer
## Flutter API Integration Guide

### Backend

Framework: FastAPI
Development port: 8000
Authentication: X-API-Key header

The Cloudflare HTTPS URL is temporary. Obtain the current
base URL and API key privately from the backend developer.

### Supported Languages

- English
- Yoruba
- Hausa
- Igbo

Use these language names in API requests.

### API Workflow

All requests require the X-API-Key header.

#### 1. Health Check

GET /health

Expected success: HTTP 200

#### 2. Upload Video

POST /upload

Content-Type: multipart/form-data
Field name: file

Response example:
{"video_id":"VIDEO_ID","filename":"video.mp4"}

#### 3. Transcribe Video

POST /transcribe/{video_id}

Response:
{"video_id":"VIDEO_ID","transcription":"Recognized speech"}

#### 4. Translate Text

POST /translate

JSON:
{"text":"Welcome to our store.","language":"Yoruba"}

Response:
{"language":"Yoruba","translated_text":"Translated text"}

#### 5. Generate Speech

POST /generate-speech

JSON:
{"text":"Translated text","language":"Yoruba"}

Response:
- Binary WAV audio
- Content-Type: audio/wav
- X-Audio-ID response header

IMPORTANT: Save the X-Audio-ID header value.
It is required for video localization.

#### 6. Localize Video

POST /localize

JSON:
{
  "video_id":"VIDEO_ID",
  "audio_id":"AUDIO_ID",
  "translated_text":"Translated text",
  "language":"Yoruba"
}

Response example:
{"job_id":"JOB_ID","status":"queued"}

#### 7. Check Job Status

GET /jobs/{job_id}

Poll until status is completed or failed.

#### 8. Download Result

GET /results/{job_id}

Successful response:
- Binary MP4 video
- Content-Type: video/mp4

### Additional Endpoints

GET /videos/{video_id}
POST /upload-audio

### Flutter / Dio Notes

- Set baseUrl to the current HTTPS tunnel address.
- Send X-API-Key in request headers.
- Use FormData and MultipartFile for uploads.
- Use ResponseType.bytes for WAV and MP4 downloads.
- Read X-Audio-ID from the speech response headers.
- Poll the job endpoint instead of assuming immediate completion.
- Handle HTTP 401, 404, 422 and 500 responses.
- Do not hardcode the API key in a production mobile app.

### Deployment Limitations

- Cloudflare Quick Tunnel URLs change between sessions.
- Kaggle must remain running for the API to remain accessible.
- Jobs and uploaded-file references may be lost on restart.
- Model weights are not included in this backup.
- Fresh sessions require dependency installation and model loading.
- This deployment is intended for development and integration testing.

### Verified Tests

- External HTTPS connectivity: PASS
- API authentication: PASS
- Video upload: PASS
- Whisper transcription: PASS
- Yoruba, Hausa and Igbo translation requests: PASS
- English and Yoruba speech generation: PASS
- Yoruba video rendering and MP4 download: PASS

Translation quality and real-world video synchronization
still require further evaluation.
