"""Speech recognition with official N-ATLAS ASR models (runs on the Kaggle GPU worker).

Uses the Hugging Face transformers ASR pipeline (Whisper-small fine-tunes,
gated: requires HF_TOKEN with access granted on each model page).
Models are lazy-loaded once per source language and cached in HF_HOME.
"""

import threading
from typing import Dict, List

import numpy as np
import soundfile as sf

from . import config

_pipelines: Dict[str, object] = {}
_load_lock = threading.Lock()
_infer_lock = threading.Lock()  # one CPU inference at a time


class ASRError(RuntimeError):
    pass


def model_for(source_language: str) -> str:
    return config.ASR_MODELS_BY_LANGUAGE.get(source_language, config.ASR_MODEL)


def loaded_models() -> List[str]:
    return list(_pipelines.keys())


def _pick_device(torch):
    if config.ASR_DEVICE != "auto":
        return config.ASR_DEVICE
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    return "cpu"


def _get_pipeline(model_id: str):
    if model_id in _pipelines:
        return _pipelines[model_id]
    with _load_lock:
        if model_id in _pipelines:
            return _pipelines[model_id]
        if not config.HF_TOKEN:
            raise ASRError("HF_TOKEN is not set; N-ATLAS ASR models are gated on Hugging Face.")
        try:
            import torch
            from transformers import pipeline

            device = _pick_device(torch)
            kwargs = {}
            if config.HF_HOME:
                kwargs["model_kwargs"] = {"cache_dir": config.HF_HOME}
            asr = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                token=config.HF_TOKEN,
                device=device,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                **kwargs,
            )
        except Exception as exc:  # gated access, network, OOM...
            raise ASRError(f"Could not load ASR model {model_id}: {exc}") from exc
        _pipelines[model_id] = asr
        return asr


def flag_low_confidence(text: str, start: float, end: float) -> List[str]:
    """Heuristic flags (the HF pipeline does not expose token log-probs)."""
    flags = []
    words = text.split()
    dur = max(end - start, 0.01)
    if not text.strip():
        flags.append("empty")
    elif len(words) <= 1 and dur > 2.0:
        flags.append("very_short_for_duration")
    if words and len(words) / dur > 6.0:
        flags.append("speech_rate_implausible")
    if len(words) >= 6 and len(set(w.lower() for w in words)) / len(words) < 0.4:
        flags.append("repetition_possible_hallucination")
    return flags


def transcribe(wav_path: str, source_language: str = "en", duration: float = None) -> List[dict]:
    """Return segments [{start, end, text, flags}] from a 16 kHz mono WAV."""
    model_id = model_for(source_language)
    asr = _get_pipeline(model_id)
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if duration is None:
        duration = len(audio) / sr
    with _infer_lock:
        result = asr(
            {"raw": np.ascontiguousarray(audio), "sampling_rate": sr},
            return_timestamps=True,
            chunk_length_s=30,
            batch_size=1,
        )
    return chunks_to_segments(result, duration)


def chunks_to_segments(result: dict, duration: float) -> List[dict]:
    chunks = result.get("chunks") or []
    segments = []
    if not chunks and result.get("text", "").strip():
        chunks = [{"timestamp": (0.0, duration), "text": result["text"]}]
    for i, ch in enumerate(chunks):
        ts = ch.get("timestamp") or (None, None)
        start = ts[0] if ts[0] is not None else (segments[-1]["end"] if segments else 0.0)
        end = ts[1]
        if end is None:  # last chunk can be open-ended
            nxt = chunks[i + 1]["timestamp"][0] if i + 1 < len(chunks) else None
            end = nxt if nxt is not None else duration
        start = float(max(0.0, min(start, duration)))
        end = float(max(start, min(end, duration)))
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text,
            "flags": flag_low_confidence(text, start, end),
        })
    return segments


def transcribe_video_text(video_path: str) -> str:
    """Drop-in for the handover api 'transcribe' service: video path -> plain text."""
    import tempfile
    from . import media
    with tempfile.TemporaryDirectory() as tmp:
        wav = f"{tmp}/audio.wav"
        media.extract_audio(video_path, wav)
        segs = transcribe(wav, "en")
    return " ".join(s["text"] for s in segs).strip()
