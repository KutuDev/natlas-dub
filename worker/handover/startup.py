
"""Connect the N-ATLAS API to AI models loaded in the Kaggle notebook."""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def connect_services(namespace):
    """Connect existing notebook models without loading them again."""
    import api
    from services import transcription, translation

    required = [
        "whisper_model",
        "model",
        "tokenizer",
        "multilingual_translate",
        "four_language_speech",
        "multilingual_render",
    ]

    missing = [name for name in required if name not in namespace]

    if missing:
        raise RuntimeError(
            "Missing notebook components: " + ", ".join(missing)
        )

    transcription.configure_transcription(
        namespace["whisper_model"]
    )

    translation.configure_translation(
        namespace["model"],
        namespace["tokenizer"]
    )

    api.configure_ai_services(
        transcribe=transcription.transcribe_video,
        translate=namespace["multilingual_translate"],
        speech=namespace["four_language_speech"]
    )

    api.configure_video_renderer(
        namespace["multilingual_render"]
    )

    if not os.environ.get("DEV_API_KEY"):
        raise RuntimeError(
            "Set DEV_API_KEY securely before starting the API."
        )

    return api.app
