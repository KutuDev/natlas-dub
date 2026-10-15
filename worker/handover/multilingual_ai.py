"""Multilingual AI functions for N-ATLAS Video Localizer.

Model objects are supplied by the active Kaggle notebook.
Do not import this module expecting models to load automatically.
"""

import torch
import numpy as np
import shutil
from pathlib import Path
from uuid import uuid4
from services import video_processing

SUPPORTED_LANGUAGES = ("Yoruba", "Hausa", "Igbo", "English")
SUPPORTED_VIDEO_LANGUAGES = set(SUPPORTED_LANGUAGES)

def multilingual_translate(text, language):
    language = language.strip().title()
    text = text.strip()

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    if not text:
        raise ValueError("Translation text cannot be empty.")

    if language == "English":
        return text

    messages = [{
        "role": "user",
        "content": (
            f"Translate the following English text into natural {language}. "
            "Preserve the original meaning and tone. "
            f"Return only the {language} translation:\n\n{text}"
        )
    }]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to("cuda:0")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ).strip()


def multilingual_speech(text, language):
    if language not in speech_models:
        raise ValueError(f"TTS unavailable for {language}")

    if not text.strip():
        raise ValueError("Speech text cannot be empty")

    inputs = speech_tokenizers[language](
        text,
        return_tensors="pt"
    )

    with torch.inference_mode():
        waveform = speech_models[language](**inputs).waveform

    return waveform.squeeze().cpu().numpy(), 16000


def four_language_speech(text, language):
    language = language.strip().title()

    if not text.strip():
        raise ValueError("Speech text cannot be empty")

    if language in ("Yoruba", "Hausa", "Igbo"):
        return multilingual_speech(text, language)

    if language == "English":
        inputs = english_processor(
            text=text,
            return_tensors="pt"
        )

        with torch.inference_mode():
            audio = english_model.generate_speech(
                inputs["input_ids"],
                speaker_embedding,
                vocoder=english_vocoder
            )

        return audio.cpu().numpy(), 16000

    raise ValueError(f"Unsupported language: {language}")


def multilingual_render(video_path, audio_path, translated_text, language):
    language = language.strip().title()

    if language not in SUPPORTED_VIDEO_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    video_path = Path(video_path)
    audio_path = Path(audio_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    original_output = Path(
        video_processing.create_localized_video(
            str(video_path),
            str(audio_path),
            translated_text
        )
    )

    if not original_output.is_file():
        raise FileNotFoundError(
            f"Rendered video not found: {original_output}"
        )

    final_output = original_output.parent / (
        f"{uuid4().hex[:8]}_{language.lower()}.mp4"
    )

    shutil.move(str(original_output), str(final_output))

    return str(final_output)

