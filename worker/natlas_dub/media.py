"""FFmpeg helpers: probe, extract audio, fit TTS clips, build dub track, SRT, render."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

from . import config

MIX_SR = 48000


class MediaError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd, cwd=None, timeout=600):
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    if proc.returncode != 0:
        raise MediaError(f"{cmd[0]} failed: {proc.stderr[-800:]}")
    return proc


def probe(path: str) -> dict:
    proc = _run(["ffprobe", "-v", "error", "-print_format", "json",
                 "-show_format", "-show_streams", str(path)], timeout=60)
    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams", [])
    duration = float(info.get("format", {}).get("duration") or 0)
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return {"duration": duration, "has_video": has_video, "has_audio": has_audio}


def extract_audio(video: str, out_wav: str) -> None:
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1",
          "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)])


def wav_duration(path: str) -> float:
    return sf.info(str(path)).duration


def _atempo_chain(factor: float) -> str:
    # atempo supports 0.5..2.0 per instance; chain if needed.
    parts = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor /= 0.5
    parts.append(f"atempo={factor:.4f}")
    return ",".join(parts)


def normalize_clip(src: str, dst: str, tempo: float = 1.0) -> float:
    """Resample to 48k mono, apply tempo. Returns resulting duration."""
    af = f"{_atempo_chain(tempo)}" if abs(tempo - 1.0) > 1e-3 else "anull"
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", af, "-ac", "1",
          "-ar", str(MIX_SR), "-c:a", "pcm_s16le", str(dst)])
    return wav_duration(dst)


def fit_and_place(segments: List[dict], clip_paths: List[str], work_dir: Path,
                  video_duration: float, speed: float = 1.0) -> dict:
    """Fit each clip to its slot (start..next start) and place it on a timeline.

    segments: [{start, end}] in order; clip_paths: raw TTS wavs (same order).
    speed: user speech speed (0.65-1.25), applied before fitting.
    Returns {"track": path, "warnings": [...], "clips": [...]}.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    total = int(np.ceil(video_duration * MIX_SR)) + MIX_SR
    track = np.zeros(total, dtype=np.float32)
    warnings, clips_info = [], []

    for i, (seg, raw) in enumerate(zip(segments, clip_paths)):
        start = float(seg["start"])
        next_start = float(segments[i + 1]["start"]) if i + 1 < len(segments) else video_duration
        slot = max(next_start - start, 0.2)

        base = work_dir / f"clip_{i:03d}_speed.wav"
        dur = normalize_clip(raw, base, tempo=speed)
        fit = 1.0
        if dur > slot:
            need = dur / slot
            fit = min(need, config.MAX_FIT_TEMPO)
            if need > config.MAX_FIT_TEMPO:
                warnings.append(
                    f"Segment {i + 1}: voice is {dur:.1f}s for a {slot:.1f}s slot; "
                    f"sped up {fit:.2f}x (cap) and it will overlap the next segment. "
                    "Shorten the text or raise speech speed.")
        final = base
        if fit > 1.0:
            final = work_dir / f"clip_{i:03d}_fit.wav"
            dur = normalize_clip(base, final, tempo=fit)

        data, sr = sf.read(str(final), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        offset = int(round(start * MIX_SR))
        end_idx = min(offset + len(data), total)
        track[offset:end_idx] += data[: end_idx - offset]
        if offset + len(data) > int(video_duration * MIX_SR):
            warnings.append(f"Segment {i + 1}: voice runs past the end of the video and is cut.")
        clips_info.append({"index": i, "start": start, "slot": round(slot, 2),
                           "duration": round(dur, 2), "fit_tempo": round(fit, 3)})

    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0.99:
        track = track / peak * 0.99
    track = track[: int(np.ceil(video_duration * MIX_SR))]
    out = work_dir / "dub_track.wav"
    sf.write(str(out), track, MIX_SR, subtype="PCM_16")
    return {"track": str(out), "warnings": warnings, "clips": clips_info}


def srt_time(seconds: float) -> str:
    ms = int(round(max(seconds, 0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def build_srt(entries: List[dict]) -> str:
    """entries: [{start, end, text}] -> SRT string."""
    lines = []
    n = 0
    for e in entries:
        text = (e.get("text") or "").strip()
        if not text:
            continue
        n += 1
        end = max(float(e["end"]), float(e["start"]) + 0.5)
        lines.append(f"{n}\n{srt_time(float(e['start']))} --> {srt_time(end)}\n{text}\n")
    return "\n".join(lines)


SUB_STYLE = ("FontName=DejaVu Sans,FontSize=18,Bold=1,PrimaryColour=&H00FFFFFF,"
             "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=22")


def render_video(video: str, dub_track: str, out_mp4: str, duration: float,
                 srt_path: str = None, background: bool = False,
                 background_volume: float = None, has_audio: bool = True) -> None:
    """Replace the audio with the dub track; optionally mix original audio low and burn subs.

    Note: "background" mixes the ORIGINAL audio track at low volume. There is no
    vocal/music separation, so the original voice is audible faintly underneath.
    """
    vol = config.BACKGROUND_VOLUME if background_volume is None else background_volume
    work = Path(out_mp4).parent
    vfilters = ["scale='min(1280,iw)':-2"]
    if srt_path:
        # Run with cwd = srt dir and a bare filename to avoid filter-path escaping issues.
        srt_name = Path(srt_path).name
        if Path(srt_path).parent.resolve() != work.resolve():
            shutil.copy(srt_path, work / srt_name)
        vfilters.append(f"subtitles={srt_name}:force_style='{SUB_STYLE}'")
    vf = ",".join(vfilters)

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(Path(video).resolve()),
           "-i", str(Path(dub_track).resolve())]
    if background and has_audio:
        fc = (f"[0:v]{vf}[v];[0:a]volume={vol}[bg];[1:a]volume=1.0[dub];"
              f"[dub][bg]amix=inputs=2:duration=longest:normalize=0,apad[a]")
    else:
        fc = f"[0:v]{vf}[v];[1:a]apad[a]"
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-t", f"{duration:.3f}",
            "-movflags", "+faststart", str(Path(out_mp4).resolve())]
    _run(cmd, cwd=str(work), timeout=900)
