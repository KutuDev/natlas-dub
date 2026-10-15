
"""Start N-ATLAS services after a fresh Kaggle runtime.

Usage in a Kaggle notebook:

    from run_project import initialize
    app = initialize()

Set DEV_API_KEY securely before calling initialize().
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def initialize():
    if not os.environ.get("DEV_API_KEY"):
        raise RuntimeError("DEV_API_KEY is not configured")

    from restore_models import load_models
    import multilingual_ai as ai
    from startup import connect_services

    # Load models once, only after a fresh runtime.
    namespace = load_models()

    # Supply model objects to the saved multilingual functions.
    for name, value in namespace.items():
        setattr(ai, name, value)

    namespace.update({
        "multilingual_translate": ai.multilingual_translate,
        "four_language_speech": ai.four_language_speech,
        "multilingual_render": ai.multilingual_render,
    })

    # multilingual_speech is called by four_language_speech.
    ai.multilingual_speech = ai.multilingual_speech

    return connect_services(namespace)
