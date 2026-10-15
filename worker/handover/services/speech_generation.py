# SERVICE IMPORTS

import torch
import numpy as np
import soundfile as sf
import tempfile
import gradio as gr

"""
Speech Generation Service
DEVELOPER DOCUMENTATION

Purpose: Converts reviewed Yoruba text into spoken audio using Meta MMS-TTS.

Note: This module was exported from the working Kaggle prototype.
Its model objects, imports, and configuration may still depend
on the notebook environment. Integration work is required
before using it as a standalone application service.
"""

def generate_yoruba_voice(yoruba_text):
    if not yoruba_text or not yoruba_text.strip():
        raise gr.Error("Please enter or approve the Yoruba translation first.")

    inputs = tts_tokenizer(
        yoruba_text.strip(),
        return_tensors="pt"
    )

    with torch.inference_mode():
        waveform = tts_model(**inputs).waveform

    audio = waveform.squeeze().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)

    output_path = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ).name

    sf.write(output_path, audio, sampling_rate)

    return output_path



tts_model = None
tts_tokenizer = None
sampling_rate = 16000

def configure_speech(speech_model, speech_tokenizer, sample_rate=16000):
    """Register the Yoruba text-to-speech model and tokenizer."""
    global tts_model, tts_tokenizer, sampling_rate
    tts_model = speech_model
    tts_tokenizer = speech_tokenizer
    sampling_rate = sample_rate

