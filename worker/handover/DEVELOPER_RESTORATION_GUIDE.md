# N-ATLAS Video Localizer — Developer Restoration Guide

## Project overview

FastAPI backend for multilingual video localization using:
- NCAIR1/N-ATLaS for language processing
- OpenAI Whisper tiny.en for speech transcription
- YarnGPT-local for Yoruba, Hausa and Igbo speech
- YarnGPT2 for English speech
- WavTokenizer for YarnGPT audio decoding
- FFmpeg for video/audio processing

The Flutter app communicates with this backend over HTTP.

## Backend files

- api.py — original endpoints and BYOA router integration
- byoa_routes.py — custom translation and speech endpoints
- startup.py — connects loaded AI models to API services
- restore_models.py — restores transcription and language models
- restore_yarngpt.py — restores YarnGPT models and decoder
- services/ — transcription, translation, speech and video logic
- requirements.txt — Python dependencies
- .env.example — safe environment template

## Environment setup

1. Install Python dependencies:

   pip install -r requirements.txt

2. Install FFmpeg on the host system.

3. Copy .env.example to .env.

4. Set DEV_API_KEY and HF_TOKEN in .env.

5. Review restore_models.py and restore_yarngpt.py
   for model-loading instructions and device requirements.

6. Restore the model checkpoints before starting the
   model-dependent localization pipeline.

## Model sources

N-ATLAS:
https://huggingface.co/NCAIR1/N-ATLaS

YarnGPT-local:
https://huggingface.co/saheedniyi/YarnGPT-local

YarnGPT2:
https://huggingface.co/saheedniyi/YarnGPT2

Whisper:
https://github.com/openai/whisper

## WavTokenizer dependency

The original YarnGPT restoration workflow expects:

- wavtokenizer_large_speech_320_24k.ckpt
- wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml

The checkpoint files are NOT included in this handover.

The previous Colab runtime was reset, and the original
checkpoint location could not be recovered from Google Drive.

Obtain compatible files from the original trusted
WavTokenizer/YarnGPT setup before attempting YarnGPT restoration.

The original restoration code references:
  /kaggle/working/yarngpt_decoder

Update the decoder directory for the deployment environment.

## API endpoints

Original endpoints:
GET  /health
POST /upload
GET  /videos/{video_id}
POST /transcribe/{video_id}
POST /translate
POST /generate-speech
POST /upload-audio
POST /localize
GET  /jobs/{job_id}
GET  /results/{job_id}

Bring Your Own API endpoints:
POST /translate-custom
POST /generate-speech-custom
GET  /custom-speech-results/{audio_id}

The BYOA endpoints require the X-API-Key developer header.

## Custom translation request

POST /translate-custom

{
  "text": "Good morning",
  "language": "Yoruba",
  "api_url": "https://api.openai.com/v1/chat/completions",
  "api_key": "YOUR_PROVIDER_API_KEY",
  "model": "YOUR_SUPPORTED_MODEL"
}

Custom translation supports approved OpenAI-compatible
chat-completion endpoints.

## Custom speech request

POST /generate-speech-custom

{
  "text": "Good morning",
  "language": "English",
  "api_url": "https://api.openai.com/v1/audio/speech",
  "api_key": "YOUR_PROVIDER_API_KEY",
  "model": "YOUR_SUPPORTED_TTS_MODEL",
  "voice": "alloy"
}

The response includes audio_id and audio_url.
Retrieve the WAV using:

GET /custom-speech-results/{audio_id}

Include X-API-Key when downloading the audio.

## Security

- Never expose DEV_API_KEY in a public Flutter application.
- Never commit real provider credentials or Hugging Face tokens.
- Use a trusted backend or secure gateway to authenticate clients.
- Provider keys should be treated as sensitive request data.
- HTTPS is required for external provider endpoints.
- Restrict public access to the backend before production deployment.

## Deployment limitations

- Model checkpoints are not bundled.
- A Colab runtime is temporary and unsuitable as permanent hosting.
- Model-dependent endpoints require a configured model runtime.
- The BYOA integration has passed static syntax/route checks.
- Live end-to-end validation is still required.
- Model restoration, GPU compatibility and deployment environment
  must be verified by the developer.
