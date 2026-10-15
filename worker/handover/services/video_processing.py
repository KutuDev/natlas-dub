# SERVICE IMPORTS

import os
import re
import uuid
import subprocess
import soundfile as sf
import gradio as gr

"""
Video Processing Service
DEVELOPER DOCUMENTATION

Purpose: Combines Yoruba narration, subtitles, and the original video using FFmpeg.

Note: This module was exported from the working Kaggle prototype.
Its model objects, imports, and configuration may still depend
on the notebook environment. Integration work is required
before using it as a standalone application service.
"""

def create_localized_video(video_path, audio_path, yoruba_text):
    if not video_path or not audio_path:
        raise gr.Error("Upload a video and generate Yoruba voice first.")

    if not yoruba_text or not yoruba_text.strip():
        raise gr.Error("The approved Yoruba translation is missing.")

    output_dir = "/kaggle/working/localized_videos"
    os.makedirs(output_dir, exist_ok=True)

    job_id = uuid.uuid4().hex[:8]
    subtitle_path = os.path.join(output_dir, f"{job_id}.srt")
    output_path = os.path.join(output_dir, f"{job_id}_yoruba.mp4")

    audio_duration = sf.info(audio_path).duration

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", yoruba_text.strip())
        if s.strip()
    ]

    if not sentences:
        raise gr.Error("No subtitle text found.")

    # Approximate subtitle timing based on sentence length.
    # Precise speech alignment can be added later.
    weights = [max(len(s), 1) for s in sentences]
    total_weight = sum(weights)

    with open(subtitle_path, "w", encoding="utf-8") as f:
        start = 0.0

        for i, (sentence, weight) in enumerate(
            zip(sentences, weights), start=1
        ):
            end = min(
                audio_duration,
                start + audio_duration * weight / total_weight
            )

            f.write(
                f"{i}\n"
                f"{srt_time(start)} --> {srt_time(end)}\n"
                f"{sentence}\n\n"
            )

            start = end

    subtitle_filter = (
        f"subtitles={subtitle_path}:"
        "force_style='FontName=DejaVu Sans,"
        "FontSize=19,Bold=1,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Shadow=1,"
        "Alignment=2,MarginV=20'"
    )

    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex", f"[0:v]{subtitle_filter}[v]",
        "-map", "[v]",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", "apad",
        "-t", str(max(audio_duration, 1)),
        "-movflags", "+faststart",
        output_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise gr.Error(
            "Video rendering failed: " + result.stderr[-1200:]
        )

    return output_path


def srt_time(seconds):
    """Convert seconds into SRT subtitle timestamp format."""
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

