"""In-process model handlers, wired by start_worker.py after models are restored.

translate(text, language_name) -> str          e.g. multilingual_ai.multilingual_translate
speech(text, language_name, voice) -> (np.ndarray, sample_rate)
    e.g. yarngpt_service.final_speech_handler (honours voice) or a wrapper around
    multilingual_ai.four_language_speech (ignores voice).
"""

state = {
    "translate": None,
    "speech": None,
    "speech_backend": None,       # e.g. "yarngpt" | "mms/speecht5 (voice not selectable)"
    "speech_supports_voice": False,
}


class ServiceUnavailable(RuntimeError):
    pass


def configure(translate=None, speech=None, speech_backend=None, speech_supports_voice=False):
    if translate is not None:
        state["translate"] = translate
    if speech is not None:
        state["speech"] = speech
        state["speech_backend"] = speech_backend
        state["speech_supports_voice"] = speech_supports_voice


def translate(text: str, language: str) -> str:
    fn = state["translate"]
    if fn is None:
        raise ServiceUnavailable("Translation model (N-ATLaS) is not loaded on the worker.")
    out = fn(text, language)
    if not isinstance(out, str) or not out.strip():
        raise RuntimeError("Translation model returned empty output.")
    return out.strip()


def speech(text: str, language: str, voice: str):
    fn = state["speech"]
    if fn is None:
        raise ServiceUnavailable("Speech model (YarnGPT) is not loaded on the worker.")
    return fn(text, language, voice)


def status() -> dict:
    return {
        "translate": callable(state["translate"]),
        "speech": callable(state["speech"]),
        "speech_backend": state["speech_backend"],
        "speech_supports_voice": state["speech_supports_voice"],
        "yarngpt_error": __import__("os").environ.get("YARNGPT_ERROR") or None,
    }
