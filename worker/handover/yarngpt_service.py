"""YarnGPT multilingual speech integration.

Models are supplied by the Kaggle notebook.
This module does not load models automatically.
"""

import torch
import numpy as np

VOICE_OPTIONS = {
    "Yoruba": {
        "female": "yoruba_female1",
        "male": "yoruba_male2"
    },
    "Igbo": {
        "female": "igbo_female1",
        "male": "igbo_male2"
    },
    "Hausa": {
        "female": "hausa_female1",
        "male": "hausa_male1"
    }
}

ENGLISH_VOICES = {
    "female": "idera",
    "male": "jude"
}

yarngpt_audio_tokenizer = None
yarngpt_model = None
yarngpt2_audio_tokenizer = None
yarngpt2_model = None
original_speech_handler = None
previous_speech_handler = None


def configure_yarngpt(
    local_tokenizer,
    local_model,
    english_tokenizer,
    english_model,
    fallback_handler
):
    global yarngpt_audio_tokenizer, yarngpt_model
    global yarngpt2_audio_tokenizer, yarngpt2_model
    global original_speech_handler, previous_speech_handler

    yarngpt_audio_tokenizer = local_tokenizer
    yarngpt_model = local_model
    yarngpt2_audio_tokenizer = english_tokenizer
    yarngpt2_model = english_model
    original_speech_handler = fallback_handler
    previous_speech_handler = combined_speech_handler



def yarngpt_speech(text, language, voice=None):
    language = language.strip().title()

    if language not in VOICE_OPTIONS:
        raise ValueError(f"YarnGPT does not support {language}")

    if not text.strip():
        raise ValueError("Speech text cannot be empty")

    options = VOICE_OPTIONS[language]

    # Default to female when no voice is specified.
    selected_voice = voice or "female"

    # Accept either gender or exact YarnGPT voice name.
    selected_voice = options.get(selected_voice.lower(), selected_voice)

    if selected_voice not in options.values():
        raise ValueError(
            f"Invalid voice '{voice}' for {language}. "
            f"Available: {options}"
        )

    prompt = yarngpt_audio_tokenizer.create_prompt(
        text,
        lang=language.lower(),
        speaker_name=selected_voice
    )

    input_ids = yarngpt_audio_tokenizer.tokenize_prompt(prompt)

    with torch.inference_mode():
        output = yarngpt_model.generate(
            input_ids=input_ids,
            temperature=0.1,
            repetition_penalty=1.1,
            num_beams=4,
            max_length=4000
        )

    codes = yarngpt_audio_tokenizer.get_codes(output)

    if len(codes) == 0:
        raise RuntimeError("YarnGPT generated no audio codes")

    audio = yarngpt_audio_tokenizer.get_audio(codes)

    waveform = np.asarray(
        audio.detach().cpu().numpy().squeeze(),
        dtype=np.float32
    )

    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError("YarnGPT generated invalid audio")

    if np.max(np.abs(waveform)) < 0.001:
        raise RuntimeError("YarnGPT generated silent audio")

    return waveform, 24000

def combined_speech_handler(text, language, voice=None):
    language = language.strip().title()

    if language == "English":
        return original_speech_handler(text, language)

    if language in VOICE_OPTIONS:
        try:
            return yarngpt_speech(text, language, voice)
        except Exception as e:
            print(f"YarnGPT warning ({language}): {e}")

            # Invalid voice requests should not silently switch voices.
            if voice is not None:
                raise

            print("Using original TTS fallback.")
            return original_speech_handler(text, language)

    return original_speech_handler(text, language)

def english_yarngpt_speech(text, voice=None):
    if not text.strip():
        raise ValueError("Speech text cannot be empty")

    selected = (voice or "female").strip().lower()
    selected = ENGLISH_VOICES.get(selected, selected)

    if selected not in ENGLISH_VOICES.values():
        raise ValueError(f"Invalid English voice: {voice}")

    prompt = yarngpt2_audio_tokenizer.create_prompt(
        text,
        lang="english",
        speaker_name=selected
    )

    input_ids = yarngpt2_audio_tokenizer.tokenize_prompt(prompt)

    with torch.inference_mode():
        output = yarngpt2_model.generate(
            input_ids=input_ids,
            temperature=0.1,
            repetition_penalty=1.1,
            max_length=4000
        )

    codes = yarngpt2_audio_tokenizer.get_codes(output)

    if len(codes) == 0:
        raise RuntimeError("YarnGPT2 generated no audio codes")

    audio = yarngpt2_audio_tokenizer.get_audio(codes)

    waveform = np.asarray(
        audio.detach().cpu().numpy().squeeze(),
        dtype=np.float32
    )

    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError("Invalid English audio")

    if np.max(np.abs(waveform)) < 0.001:
        raise RuntimeError("Silent English audio")

    return waveform, 24000

def final_speech_handler(text, language, voice=None):
    language = language.strip().title()

    if language == "English":
        try:
            return english_yarngpt_speech(text, voice)
        except Exception as e:
            print("YarnGPT2 English warning:", e)

            if voice is not None:
                raise

            print("Using SpeechT5 fallback.")
            return original_speech_handler(text, "English")

    return previous_speech_handler(text, language, voice)
