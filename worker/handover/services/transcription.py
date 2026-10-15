# SERVICE IMPORTS

import os
import subprocess
import tempfile
import gradio as gr

"""
Transcription Service
DEVELOPER DOCUMENTATION

Purpose: Extracts English speech from an uploaded video using FFmpeg and Whisper.

Note: This module was exported from the working Kaggle prototype.
Its model objects, imports, and configuration may still depend
on the notebook environment. Integration work is required
before using it as a standalone application service.
"""

def transcribe_video(video_path):
    if video_path is None:
        raise gr.Error("Please upload a video first.")

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.wav")

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                audio_path
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise gr.Error(
                "Could not extract audio. Check that the video contains speech."
            )

        transcription = asr_model.transcribe(
            audio_path,
            fp16=False
        )

    return transcription["text"].strip()



asr_model = None

def configure_transcription(whisper_model):
    """Register the loaded Whisper speech-recognition model."""
    global asr_model
    asr_model = whisper_model

