
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


router = APIRouter(tags=["Custom AI"])

TRANSLATION_HOSTS = {
    "api.openai.com",
    "openrouter.ai",
    "api.groq.com",
    "api.together.xyz",
}

TTS_URL = "https://api.openai.com/v1/audio/speech"
CUSTOM_AUDIO_DIR = Path(__file__).resolve().parent / "custom_audio"


class CustomTranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    language: str = Field(min_length=2, max_length=50)
    api_url: str
    api_key: str = Field(min_length=1)
    model: str = Field(min_length=1, max_length=200)


class CustomSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4096)
    language: str = Field(min_length=2, max_length=50)
    api_url: str
    api_key: str = Field(min_length=1)
    model: str = Field(min_length=1, max_length=200)
    voice: str = Field(default="alloy", min_length=1, max_length=100)


def validate_translation_url(url):
    try:
        parsed = urlparse(url)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname in TRANSLATION_HOSTS
            and parsed.port in (None, 443)
            and parsed.path.endswith("/chat/completions")
            and not parsed.username
            and not parsed.password
            and not parsed.fragment
        )
    except ValueError:
        valid = False

    if not valid:
        raise HTTPException(
            status_code=400,
            detail="Unsupported translation API URL."
        )
    return url


@router.post("/translate-custom")
async def translate_custom(request: CustomTranslationRequest):
    url = validate_translation_url(request.api_url)

    payload = {
        "model": request.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Translate the user's text into {request.language}. "
                    "Return only the translation."
                )
            },
            {"role": "user", "content": request.text}
        ]
    }

    try:
        async with httpx.AsyncClient(
            timeout=60,
            follow_redirects=False,
            trust_env=False
        ) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {request.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            response.raise_for_status()
            result = response.json()

        return {
            "language": request.language,
            "translated_text": result["choices"][0]["message"]["content"],
            "provider": "custom"
        }

    except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="External translation provider failed."
        )


@router.post("/generate-speech-custom")
async def generate_speech_custom(request: CustomSpeechRequest):
    if request.api_url != TTS_URL:
        raise HTTPException(
            status_code=400,
            detail="Unsupported speech API URL."
        )

    try:
        async with httpx.AsyncClient(
            timeout=90,
            follow_redirects=False,
            trust_env=False
        ) as client:
            response = await client.post(
                TTS_URL,
                headers={
                    "Authorization": f"Bearer {request.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": request.model,
                    "input": request.text,
                    "voice": request.voice,
                    "response_format": "wav"
                }
            )
            response.raise_for_status()

        audio = response.content

        if (
            len(audio) > 25 * 1024 * 1024
            or not audio.startswith(b"RIFF")
            or audio[8:12] != b"WAVE"
        ):
            raise ValueError("Invalid WAV response")

        audio_id = uuid.uuid4().hex
        CUSTOM_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        output_path = CUSTOM_AUDIO_DIR / f"custom_{audio_id}.wav"
        output_path.write_bytes(audio)

        return {
            "audio_id": audio_id,
            "language": request.language,
            "provider": "custom",
            "audio_url": f"/custom-speech-results/{audio_id}"
        }

    except (httpx.HTTPError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="External speech provider failed."
        )


@router.get("/custom-speech-results/{audio_id}")
def get_custom_speech_result(audio_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", audio_id):
        raise HTTPException(status_code=400, detail="Invalid audio ID.")

    path = CUSTOM_AUDIO_DIR / f"custom_{audio_id}.wav"

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found.")

    return FileResponse(
        str(path),
        media_type="audio/wav",
        filename=f"custom_{audio_id}.wav"
    )
