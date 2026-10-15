"""Job stages run as background tasks. Each stage records timings and logs real outcomes."""

import logging
import time
from pathlib import Path

import soundfile as sf

from . import asr, config, handlers, jobs, media, protect, session_log

log = logging.getLogger("natlas_dub.pipeline")


def _fail(job_id: str, stage: str, exc: Exception):
    msg = str(exc)
    if isinstance(exc, handlers.ServiceUnavailable):
        code = "model_unavailable"
    elif isinstance(exc, asr.ASRError):
        code = "asr_error"
    elif isinstance(exc, media.MediaError):
        code = "ffmpeg_error"
    else:
        code = "internal_error"
        msg = f"{type(exc).__name__}: {msg}"[:500]
    log.exception("job %s stage %s failed (%s)", job_id, stage, code)
    jobs.update(job_id, status="error", error=msg, error_code=code, message=f"{stage} failed")
    job = jobs.get(job_id)
    if job:
        session_log.log_event(job, f"{stage}_failed", success=False, error=f"{code}: {msg[:300]}")


def translate_segment_text(text: str, target_name: str, source_code: str, target_code: str,
                           protected_terms) -> dict:
    """Mask -> translate with N-ATLaS (in-process) -> restore. Returns {translated_text, warnings}."""
    if source_code == target_code:
        return {"translated_text": text, "warnings": []}
    masked = protect.mask(text, protected_terms)
    residual = protect.PLACEHOLDER_RE.sub("", masked.masked_text).strip(" .,!?;:-")
    if not residual:  # nothing translatable (e.g. only a brand name / URL)
        return {"translated_text": text, "warnings": []}
    raw = handlers.translate(masked.masked_text, target_name)
    restored, missing = protect.restore(raw, masked.mapping)
    warnings = []
    if missing:
        warnings.append("Protected term(s) dropped by the model, please re-insert: " + ", ".join(missing))
    return {"translated_text": restored, "warnings": warnings}


def run_process(job_id: str):
    """Extract audio -> ASR (Render CPU) -> translate per segment (GPU worker)."""
    job = jobs.get(job_id)
    jobs.update(job_id, stage="processing", status="running", error=None, message="Extracting audio")
    try:
        t0 = time.time()
        wav = str(Path(job["dir"]) / "audio_16k.wav")
        media.extract_audio(job["video_path"], wav)
        jobs.update(job_id, audio_path=wav)
        job["timings"]["extract_audio"] = round(time.time() - t0, 2)

        if job["source_language"] != job["target_language"] and not handlers.status()["translate"]:
            raise handlers.ServiceUnavailable("Translation model (N-ATLaS) is not loaded on the worker.")

        jobs.update(job_id, message=f"Transcribing with {asr.model_for(job['source_language'])} (first run downloads the model)")
        t0 = time.time()
        raw_segments = asr.transcribe(wav, job["source_language"], job["duration"])
        job["timings"]["asr"] = round(time.time() - t0, 2)
        if not raw_segments:
            raise asr.ASRError("No speech was recognised in this video.")

        target_code = job["target_language"]
        target_name = config.TARGET_LANGUAGES[target_code]
        segments = []
        t0 = time.time()
        for i, rs in enumerate(raw_segments):
            jobs.update(job_id, message=f"Translating segment {i + 1}/{len(raw_segments)} with N-ATLaS")
            tr = translate_segment_text(rs["text"], target_name, job["source_language"], target_code,
                                        job.get("protected_terms", []))
            segments.append({
                "index": i,
                "start": rs["start"],
                "end": rs["end"],
                "original_text": rs["text"],
                "target_language": target_name,
                "translated_text": tr["translated_text"],
                "edited_text": None,
                "status": "ai_generated",
                "flags": rs.get("flags", []),
                "warnings": tr["warnings"],
            })
        job["timings"]["translation"] = round(time.time() - t0, 2)
        jobs.update(job_id, segments=segments, stage="review", status="done",
                    message="Review and approve each segment")
        session_log.log_event(jobs.get(job_id), "transcribe_translate_done", success=True)
    except Exception as exc:
        _fail(job_id, "transcribe_translate", exc)


def run_tts(job_id: str, speed: float):
    """TTS per approved segment on GPU worker, then fit + place clips on a timeline."""
    job = jobs.get(job_id)
    jobs.update(job_id, stage="tts", status="running", error=None, message="Generating voice")
    try:
        if not handlers.status()["speech"]:
            raise handlers.ServiceUnavailable("Speech model is not loaded on the worker.")
        t0 = time.time()
        tts_dir = Path(job["dir"]) / "tts"
        tts_dir.mkdir(exist_ok=True)
        lang_name = config.TARGET_LANGUAGES[job["target_language"]]
        voiced_segments, clip_paths = [], []
        for seg in job["segments"]:
            text = jobs.final_text(seg)
            if not text:
                continue
            jobs.update(job_id, message=f"Generating voice {seg['index'] + 1}/{len(job['segments'])}")
            waveform, rate = handlers.speech(text, lang_name, job["voice"])
            p = tts_dir / f"raw_{seg['index']:03d}.wav"
            sf.write(str(p), waveform, int(rate))
            voiced_segments.append(seg)
            clip_paths.append(str(p))
        job["timings"]["tts"] = round(time.time() - t0, 2)
        if not clip_paths:
            raise RuntimeError("No approved text to voice.")

        t0 = time.time()
        placed = media.fit_and_place(voiced_segments, clip_paths, tts_dir, job["duration"], speed=speed)
        job["timings"]["audio_fit"] = round(time.time() - t0, 2)
        warnings = list(placed["warnings"])
        if not handlers.status()["speech_supports_voice"]:
            warnings.insert(0, f"Voice selection not available with the current speech backend "
                               f"({handlers.status()['speech_backend']}); default voice used.")
        jobs.update(job_id, track_path=placed["track"], clip_paths=clip_paths, speed=speed,
                    warnings=warnings, clips=placed["clips"],
                    output_path=None, stage="voiced", status="done", message="Voice ready")
        session_log.log_event(jobs.get(job_id), "voice_done", success=True)
    except Exception as exc:
        _fail(job_id, "voice", exc)


def run_render(job_id: str, subtitles: bool, background: bool, sub_position: str = "bottom",
               mask: bool = False, mask_height: float = 0.14):
    job = jobs.get(job_id)
    jobs.update(job_id, stage="rendering", status="running", error=None, message="Rendering video")
    try:
        t0 = time.time()
        d = Path(job["dir"])
        srt = d / "subtitles.srt"
        srt.write_text(media.build_srt([
            {"start": s["start"], "end": s["end"], "text": jobs.final_text(s)} for s in job["segments"]
        ]), encoding="utf-8")
        out = d / f"natlas_dub_{job['target_language']}.mp4"
        media.render_video(job["video_path"], job["track_path"], str(out), job["duration"],
                           srt_path=str(srt) if subtitles else None, background=background,
                           has_audio=job.get("has_audio", True), sub_position=sub_position,
                           mask=mask, mask_height=mask_height)
        job["timings"]["render"] = round(time.time() - t0, 2)
        jobs.update(job_id, output_path=str(out), srt_path=str(srt), stage="rendered", status="done",
                    render_options={"subtitles": subtitles, "background": background,
                                    "sub_position": sub_position, "mask": mask, "mask_height": mask_height},
                    message="Video ready")
        session_log.log_event(jobs.get(job_id), "render_done", success=True)
    except Exception as exc:
        _fail(job_id, "render", exc)
