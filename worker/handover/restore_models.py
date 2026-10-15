
"""Restore AI models after a fresh Kaggle session.

Requires:
- Kaggle GPU enabled (two T4 GPUs recommended)
- Hugging Face authentication completed
- Dependencies installed

Call load_models() only after a runtime reset.
"""

import torch
import whisper
import numpy as np
import zipfile

from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    VitsModel,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan
)


def load_models():
    print("Restoring Whisper...")
    whisper_model = whisper.load_model("tiny.en", device="cpu")

    print("Restoring N-ATLAS...")
    model_id = "NCAIR1/N-ATLaS"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=True,
        low_cpu_mem_usage=True
    )

    print("Restoring African-language TTS...")

    repositories = {
        "Yoruba": "facebook/mms-tts-yor",
        "Hausa": "facebook/mms-tts-hau",
        "Igbo": "Shinzmann/soro-tts-ibo"
    }

    speech_models = {}
    speech_tokenizers = {}

    for language, repository in repositories.items():
        speech_tokenizers[language] = (
            AutoTokenizer.from_pretrained(repository)
        )

        speech_models[language] = (
            VitsModel.from_pretrained(repository)
            .to("cpu")
            .eval()
        )

    print("Restoring English SpeechT5...")

    english_processor = SpeechT5Processor.from_pretrained(
        "microsoft/speecht5_tts"
    )

    english_model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts"
    ).to("cpu").eval()

    english_vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan"
    ).to("cpu").eval()

    archive = hf_hub_download(
        repo_id="Matthijs/cmu-arctic-xvectors",
        filename="spkrec-xvect.zip",
        repo_type="dataset"
    )

    with zipfile.ZipFile(archive) as z:
        candidates = [
            name for name in z.namelist()
            if name.endswith(".npy")
            and not name.startswith("__MACOSX")
        ]

        if not candidates:
            raise RuntimeError("Speaker embedding missing")

        with z.open(candidates[0]) as f:
            embedding = np.load(f)

    speaker_embedding = torch.tensor(
        embedding,
        dtype=torch.float32
    ).reshape(1, -1)

    assert speaker_embedding.shape == (1, 512)

    print("All AI models restored.")

    return {
        "whisper_model": whisper_model,
        "model": model,
        "tokenizer": tokenizer,
        "speech_models": speech_models,
        "speech_tokenizers": speech_tokenizers,
        "english_processor": english_processor,
        "english_model": english_model,
        "english_vocoder": english_vocoder,
        "speaker_embedding": speaker_embedding
    }
