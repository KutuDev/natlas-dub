"""FFmpeg tests on the real Moniepoint clips (not in git). Skipped when the clips are absent.
TEST-ONLY: synthetic tones stand in for TTS; no model is called."""
import os
import subprocess
import time
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from conftest import ROOT, fake_tone, fake_translate, FAKE_SEGMENTS
from app_imports import media, handlers, asr

CLIPS = Path(os.environ.get("TESTCLIPS_DIR", ROOT.parent / "testclips"))
ENG = CLIPS / "moniepoint__engineering__poweringdreams___moniepo.mp4"
FIN = CLIPS / "moniepoint__poweringdreams__financialhappiness___.mp4"
LONG = CLIPS / "nigeria_s_informal_economy__yiteovie_s_story___mon.mp4"
ART = ROOT / "test_artifacts"
pytestmark = pytest.mark.skipif(not ENG.exists(), reason="real test clips not present")
KEY = {"X-API-Key": "test-worker-key"}

SAMPLE_SRT = [
    {"start": 0.5, "end": 4.0, "text": "Ẹ kú àárọ̀ o, ẹ jẹ́ ká sọ̀rọ̀ nípa Moniepoint"},
    {"start": 4.2, "end": 8.0, "text": "Àwọn onímọ̀ ẹ̀rọ wa ń ṣiṣẹ́ takuntakun"},
    {"start": 8.2, "end": 12.0, "text": "ɓ ɗ ƙ ƴ · ị ọ ụ ṅ (diacritic check)"},
]


def _frame(video, t, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1", str(out)], check=True)


def _render(src, tmp_path, name, **kw):
    info = media.probe(str(src))
    wav = tmp_path / "a16k.wav"
    t0 = time.time()
    media.extract_audio(str(src), str(wav))
    t_extract = time.time() - t0
    assert sf.info(str(wav)).samplerate == 16000
    assert abs(sf.info(str(wav)).duration - info["duration"]) < 0.2
    clips = []
    for i, seg in enumerate(SAMPLE_SRT):
        w, sr = fake_tone(seg["text"], "Yoruba")
        p = tmp_path / f"tone{i}.wav"
        sf.write(str(p), w, sr)
        clips.append(str(p))
    placed = media.fit_and_place(SAMPLE_SRT, clips, tmp_path / "fit", info["duration"])
    srt = tmp_path / "subs.srt"
    srt.write_text(media.build_srt(SAMPLE_SRT), encoding="utf-8")
    out = tmp_path / f"{name}.mp4"
    t0 = time.time()
    media.render_video(str(src), placed["track"], str(out), info["duration"], srt_path=str(srt), **kw)
    t_render = time.time() - t0
    oi = media.probe(str(out))
    assert oi["has_video"] and oi["has_audio"]
    assert abs(oi["duration"] - info["duration"]) < 0.3
    ART.mkdir(exist_ok=True)
    _frame(out, 2.0, ART / f"{name}_t2.png")
    _frame(src, 2.0, ART / f"{name}_src_t2.png")
    print(f"{name}: src {info['duration']:.1f}s extract {t_extract:.1f}s render {t_render:.1f}s "
          f"out {out.stat().st_size/1e6:.1f}MB")
    return out


def test_engineering_bottom_subs_with_mask(tmp_path):
    _render(ENG, tmp_path, "engineering_mask_bottom", mask=True, mask_height=0.14, sub_position="bottom")


def test_financial_top_subs_no_mask_with_background(tmp_path):
    _render(FIN, tmp_path, "financial_top_bg", mask=False, sub_position="top", background=True)


@pytest.mark.skipif(not LONG.exists(), reason="long clip absent")
def test_long_clip_gate_and_trim():
    import start_worker
    app = start_worker.attach()
    handlers.configure(translate=fake_translate, speech=fake_tone, speech_backend="TEST", speech_supports_voice=True)
    c = TestClient(app)
    with open(LONG, "rb") as fh:
        r = c.post("/dub/upload", headers=KEY, files={"file": (LONG.name, fh, "video/mp4")}, data={"consent": "true"})
    assert r.status_code == 413 and "304s" in r.json()["detail"], r.text
    t0 = time.time()
    with open(LONG, "rb") as fh:
        r = c.post("/dub/upload", headers=KEY, files={"file": (LONG.name, fh, "video/mp4")},
                   data={"consent": "true", "trim_to_seconds": "60"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert abs(body["duration"] - 60) < 0.5 and body["trimmed_from"] > 300
    print(f"long clip trimmed {body['trimmed_from']}s -> {body['duration']}s in {time.time()-t0:.1f}s")
