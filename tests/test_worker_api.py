"""Worker /dub flow on the real handover api.app, with mocked models (test-only)."""
import time

import pytest
from fastapi.testclient import TestClient

import conftest
from conftest import FAKE_SEGMENTS, fake_tone, fake_translate, make_clip
from app_imports import asr, config, handlers

KEY = {"X-API-Key": "test-worker-key"}


@pytest.fixture(scope="module")
def client():
    import start_worker
    app = start_worker.attach()
    handlers.configure(translate=fake_translate, speech=fake_tone,
                       speech_backend="TEST fake tone", speech_supports_voice=True)
    asr.transcribe = lambda wav, lang="en", duration=None: [dict(s) for s in FAKE_SEGMENTS]
    return TestClient(app)


def wait(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/dub/jobs/{job_id}", headers=KEY).json()
        if j["status"] != "running":
            return j
        time.sleep(0.2)
    raise AssertionError("timeout")


def upload(client, path, consent="true"):
    with open(path, "rb") as fh:
        return client.post("/dub/upload", headers=KEY, files={"file": ("clip.mp4", fh, "video/mp4")},
                           data={"consent": consent, "tester_name": "Test", "tester_email": "t@example.com"})


def test_auth_kept_from_handover(client):
    assert client.get("/dub/health").status_code == 401
    assert client.get("/dub/health", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/health", headers=KEY).status_code == 200  # original handover route intact


def test_upload_gates(client, tmp_path):
    clip = make_clip(tmp_path / "c.mp4", 3)
    assert upload(client, clip, consent="false").status_code == 400
    with open(clip, "rb") as fh:
        r = client.post("/dub/upload", headers=KEY, files={"file": ("x.avi", fh)}, data={"consent": "true"})
    assert r.status_code == 400
    old = config.MAX_VIDEO_SECONDS
    config.MAX_VIDEO_SECONDS = 2
    try:
        r = upload(client, clip)
        assert r.status_code == 413 and "limit" in r.json()["detail"]
    finally:
        config.MAX_VIDEO_SECONDS = old
    old_mb = config.MAX_VIDEO_MB
    config.MAX_VIDEO_MB = 0.001
    try:
        assert upload(client, clip).status_code == 413
    finally:
        config.MAX_VIDEO_MB = old_mb
    assert client.get("/dub/jobs/../../etc", headers=KEY).status_code == 404
    assert client.get("/dub/jobs/" + "0" * 32, headers=KEY).status_code == 404


def test_full_flow(client, tmp_path):
    clip = make_clip(tmp_path / "c.mp4", 3)
    r = upload(client, clip)
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    assert client.get(f"/dub/jobs/{jid}/video", headers=KEY).status_code == 409  # Result not ready

    r = client.post(f"/dub/jobs/{jid}/process", headers=KEY,
                    json={"source_language": "en", "target_language": "yo", "voice": "male",
                          "protected_terms": ["data"]})
    assert r.status_code == 200
    j = wait(client, jid)
    assert j["status"] == "done", j
    s0 = j["segments"][0]
    assert s0["status"] == "ai_generated" and s0["target_language"] == "Yoruba"
    for term in ("MTN", "www.mtn.ng", "₦500", "data"):
        assert term in s0["translated_text"]
    assert "0803 123 4567" in j["segments"][1]["translated_text"]

    # TTS blocked until everything is approved.
    assert client.post(f"/dub/jobs/{jid}/tts", headers=KEY, json={"speed": 1.0}).status_code == 409
    r = client.put(f"/dub/jobs/{jid}/segments/0", headers=KEY, json={"edited_text": "Ẹ ra MTN data lónìí"})
    assert r.json()["status"] == "edited" and r.json()["final_text"] == "Ẹ ra MTN data lónìí"
    assert client.post(f"/dub/jobs/{jid}/segments/0/approve", headers=KEY).json()["status"] == "approved"
    assert client.post(f"/dub/jobs/{jid}/tts", headers=KEY, json={"speed": 2.0}).status_code == 422
    client.post(f"/dub/jobs/{jid}/approve-all", headers=KEY)
    j = client.get(f"/dub/jobs/{jid}", headers=KEY).json()
    assert j["segments"][0]["translated_text"].startswith("[Yoruba]")  # AI text preserved
    assert j["translation_edited"] is True

    assert client.post(f"/dub/jobs/{jid}/tts", headers=KEY, json={"speed": 0.9}).status_code == 200
    j = wait(client, jid)
    assert j["status"] == "done" and j["has_audio_preview"], j
    a = client.get(f"/dub/jobs/{jid}/audio", headers=KEY)
    assert a.status_code == 200 and a.content[:4] == b"RIFF"

    srt = client.get(f"/dub/jobs/{jid}/srt", headers=KEY).text
    assert "Ẹ ra MTN data lónìí" in srt

    assert client.post(f"/dub/jobs/{jid}/render", headers=KEY, json={"subtitles": True, "background": True}).status_code == 200
    j = wait(client, jid, 120)
    assert j["status"] == "done" and j["has_video"], j
    v = client.get(f"/dub/jobs/{jid}/video?download=true", headers=KEY)
    assert v.status_code == 200 and b"ftyp" in v.content[:64]

    log = client.get("/dub/admin/sessions", headers=KEY).json()["entries"]
    mine = [e for e in log if e.get("job_id") == jid]
    assert [e["event"] for e in mine] == ["transcribe_translate_done", "voice_done", "render_done", "export_downloaded"]
    assert mine[-1]["translation_edited"] is True and mine[0]["tester_name"] == "Test"
    assert "asr" in mine[-1]["timings_s"] and "render" in mine[-1]["timings_s"]


def test_models_missing_is_reported_not_faked(client, tmp_path):
    clip = make_clip(tmp_path / "c.mp4", 2)
    jid = upload(client, clip).json()["job_id"]
    saved = handlers.state["translate"]
    handlers.state["translate"] = None
    try:
        client.post(f"/dub/jobs/{jid}/process", headers=KEY, json={"target_language": "ha"})
        j = wait(client, jid)
        assert j["status"] == "error" and j["error_code"] == "model_unavailable" and not j["segments"]
    finally:
        handlers.state["translate"] = saved
