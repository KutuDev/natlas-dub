
"""Restore YarnGPT models after a fresh Kaggle session."""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM


def load_yarngpt_models(
    decoder_dir="/kaggle/working/yarngpt_decoder"
):
    from yarngpt.audiotokenizer import (
        AudioTokenizerForLocal,
        AudioTokenizerV2
    )

    decoder = Path(decoder_dir)

    config = decoder / (
        "wavtokenizer_mediumdata_frame75_3s_nq1_"
        "code4096_dim512_kmeans200_attn.yaml"
    )

    checkpoint = decoder / (
        "wavtokenizer_large_speech_320_24k.ckpt"
    )

    if not config.exists() or not checkpoint.exists():
        raise FileNotFoundError(
            "YarnGPT decoder files missing. "
            "Restore the decoder files before loading."
        )

    print("Restoring YarnGPT-local...")

    local_tokenizer = AudioTokenizerForLocal(
        "saheedniyi/YarnGPT-local",
        str(checkpoint),
        str(config)
    )

    local_model = AutoModelForCausalLM.from_pretrained(
        "saheedniyi/YarnGPT-local",
        torch_dtype="auto"
    ).to(local_tokenizer.device).eval()

    print("Restoring YarnGPT2 English...")

    english_tokenizer = AudioTokenizerV2(
        "saheedniyi/YarnGPT2",
        str(checkpoint),
        str(config)
    )

    english_model = AutoModelForCausalLM.from_pretrained(
        "saheedniyi/YarnGPT2",
        torch_dtype="auto"
    ).to(english_tokenizer.device).eval()

    print("Both YarnGPT models restored.")

    return {
        "yarngpt_audio_tokenizer": local_tokenizer,
        "yarngpt_model": local_model,
        "yarngpt2_audio_tokenizer": english_tokenizer,
        "yarngpt2_model": english_model
    }


def restore_complete_system():
    """Restore models and connect all FastAPI services.

    Call only in a fresh Kaggle session after dependencies,
    decoder files, and environment variables are ready.
    """
    from restore_models import load_models
    from startup import connect_services
    import multilingual_ai

    print("Restoring original AI models...")
    namespace = load_models()

    # Supply restored model objects to existing speech functions.
    for name, value in namespace.items():
        setattr(multilingual_ai, name, value)

    print("Restoring YarnGPT models...")
    namespace.update(load_yarngpt_models())

    # Existing notebook functions must be available.
    namespace.update({
        "multilingual_translate": multilingual_ai.multilingual_translate,
        "four_language_speech": multilingual_ai.four_language_speech,
        "multilingual_render": multilingual_ai.multilingual_render
    })

    app = connect_services(namespace)

    print("All available speech services connected.")
    return app, namespace
