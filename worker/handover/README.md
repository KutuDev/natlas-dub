# N-ATLAS Multilingual Video Localizer

## 1. Project Overview

An AI-powered application that converts English marketing videos
into localized Nigerian-language videos.

The current prototype supports English-to-Yoruba localization.

## 2. Core Features

1. Upload an English marketing video.
2. Extract and transcribe English speech using Whisper.
3. Translate English text into Yoruba using N-ATLAS.
4. Allow human review and correction of translations.
5. Generate Yoruba narration using Meta MMS-TTS.
6. Create Yoruba subtitles.
7. Render and download the localized MP4 video.

## 3. AI Models

Speech Recognition:
- OpenAI Whisper tiny.en

Translation:
- NCAIR1/N-ATLaS
- 8B parameter language model
- Loaded using 4-bit quantization
- Requires authorized Hugging Face access

Speech Generation:
- facebook/mms-tts-yor
- 16,000 Hz audio output
- License: CC BY-NC 4.0 (noncommercial)

## 4. Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- BitsAndBytes
- Whisper
- Meta MMS-TTS
- FFmpeg
- Gradio

## 5. Processing Architecture

English Video
    |
    v
FFmpeg Audio Extraction
    |
    v
Whisper Transcription
    |
    v
N-ATLAS Translation
    |
    v
Human Translation Review
    |
    v
Yoruba Voice Generation
    |
    v
Subtitle Generation
    |
    v
FFmpeg Video Rendering
    |
    v
Localized MP4 Download

## 6. Project Structure

services/transcription.py
    English speech recognition.

services/translation.py
    N-ATLAS translation.

services/speech_generation.py
    Yoruba text-to-speech generation.

services/video_processing.py
    Subtitle creation and video rendering.

requirements.txt
    Initial Python dependencies.

README.md
    Project documentation.

## 7. Development Environment

The prototype was developed and tested in Kaggle
using two NVIDIA Tesla T4 GPUs.

N-ATLAS was loaded using 4-bit quantization.

Whisper and Yoruba TTS were executed on CPU.

FFmpeg must be installed on the host system.

## 8. Deployment Requirements

A production developer should:

- Separate model initialization from request handling.
- Build reusable backend services.
- Provide an API for the frontend.
- Implement secure video uploads.
- Validate file types and upload sizes.
- Implement background processing for longer videos.
- Handle concurrent users.
- Manage temporary files and cleanup.
- Secure Hugging Face credentials.
- Configure appropriate GPU infrastructure.
- Test installation in a clean environment.

The exported Python service files are not yet
standalone, independently executable modules.

## 9. Known Limitations

- Yoruba is the only implemented target language.
- N-ATLAS translations require human review.
- Subtitle timing is estimated from sentence lengths.
- Long narration may require video-duration adjustment.
- The current prototype uses temporary Gradio hosting.
- The application has not undergone production load testing.
- Commercial deployment requires licensing review.
- The current TTS model has a noncommercial license.

## 10. Future Development

- Hausa localization.
- Igbo localization.
- Improved subtitle synchronization.
- Automatic narration-duration handling.
- Professional web interface.
- Mobile application integration.
- User accounts and processing history.
- Scalable GPU-backed deployment.

## 11. Project Status

Functional English-to-Yoruba prototype completed.

The full pipeline has been tested successfully
with a short marketing video.

Production deployment and additional language
support remain future development tasks.
