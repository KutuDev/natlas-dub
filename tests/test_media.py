import soundfile as sf
from pathlib import Path

from conftest import TMP, fake_tone, make_clip
from app_imports import media


def test_srt_generation():
    srt = media.build_srt([{"start": 0, "end": 1.25, "text": "Ẹ kú àárọ̀"},
                           {"start": 61.5, "end": 61.6, "text": "Ọjà"},
                           {"start": 70, "end": 71, "text": "  "}])
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,250\nẸ kú àárọ̀\n")
    assert "2\n00:01:01,500 --> 00:01:02,000\nỌjà" in srt  # min 0.5 s display
    assert "3\n" not in srt  # empty skipped


def test_probe_extract_fit_and_render(tmp_path):
    clip = make_clip(tmp_path / "in.mp4", 3)
    info = media.probe(str(clip))
    assert info["has_video"] and info["has_audio"] and 2.9 < info["duration"] < 3.2
    wav = tmp_path / "a.wav"
    media.extract_audio(str(clip), str(wav))
    i = sf.info(str(wav))
    assert i.samplerate == 16000 and i.channels == 1

    # TEST-ONLY synthetic tones as fake TTS clips. Clip 2 is too long for its slot.
    clips = []
    for n, words in enumerate([5, 40]):
        w, sr = fake_tone(" ".join(["w"] * words), "Yoruba")
        p = tmp_path / f"tts{n}.wav"
        sf.write(str(p), w, sr)
        clips.append(str(p))
    segs = [{"start": 0.0, "end": 1.0}, {"start": 1.2, "end": 2.9}]
    placed = media.fit_and_place(segs, clips, tmp_path / "fit", info["duration"], speed=1.0)
    assert Path(placed["track"]).is_file()
    assert abs(sf.info(placed["track"]).duration - info["duration"]) < 0.05
    assert placed["clips"][1]["fit_tempo"] == 1.25
    assert any("overlap" in w or "cut" in w for w in placed["warnings"])

    srt = tmp_path / "subs.srt"
    srt.write_text(media.build_srt([{"start": 0, "end": 1, "text": "Ẹ kú àárọ̀ ṣé"}]), encoding="utf-8")
    out = tmp_path / "out.mp4"
    media.render_video(str(clip), placed["track"], str(out), info["duration"], srt_path=str(srt), background=True)
    oi = media.probe(str(out))
    assert oi["has_video"] and oi["has_audio"] and abs(oi["duration"] - info["duration"]) < 0.3
