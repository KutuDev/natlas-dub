"""Render thin layer: offline behaviour, admin auth, registration, and an end-to-end
run through the proxy against a real local worker process (mocked models, test-only)."""
import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from conftest import FAKE_SEGMENTS, fake_tone, fake_translate, make_clip

ADMIN = {"X-Admin-Key": "test-admin-key"}


@pytest.fixture(scope="module")
def render():
    from app import main
    return TestClient(main.app)


def test_offline_until_registered(render, tmp_path):
    from app import worker_link
    worker_link._state.update(url=None, registered_at=None)
    h = render.get("/health")
    assert h.status_code == 200 and h.json()["gpu_worker"]["online"] is False
    assert render.get("/").status_code == 200 and "N-ATLAS Dub" in render.get("/").text
    clip = make_clip(tmp_path / "c.mp4", 2)
    with open(clip, "rb") as fh:
        r = render.post("/api/upload", files={"file": ("c.mp4", fh, "video/mp4")}, data={"consent": "true"})
    assert r.status_code == 503 and r.json()["code"] == "worker_offline"
    assert "GPU worker offline — processing unavailable" in r.json()["detail"]
    assert render.get("/api/jobs/" + "a" * 32).status_code == 503
    assert render.get("/api/jobs/not-a-job").status_code == 404


def test_admin_auth(render):
    assert render.get("/admin/sessions").status_code == 401
    assert render.post("/admin/worker", json={"url": "https://x.trycloudflare.com"},
                       headers={"X-Admin-Key": "nope"}).status_code == 401
    r = render.post("/admin/worker", json={"url": "http://evil.example.com"}, headers=ADMIN)
    assert r.status_code == 400
    r = render.post("/admin/worker", json={"url": "https://user:pw@x.trycloudflare.com"}, headers=ADMIN)
    assert r.status_code == 400


def test_registered_but_unreachable(render):
    r = render.post("/admin/worker", json={"url": "https://127.0.0.1:1"}, headers=ADMIN)
    assert r.status_code == 200
    h = render.get("/health").json()["gpu_worker"]
    assert h["online"] is False and h["reason"] == "unreachable"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def worker_url():
    import start_worker
    from natlas_dub import asr, handlers
    app = start_worker.attach()
    handlers.configure(translate=fake_translate, speech=fake_tone, speech_backend="TEST", speech_supports_voice=True)
    asr.transcribe = lambda wav, lang="en", duration=None: [dict(s) for s in FAKE_SEGMENTS]
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


def wait(render, jid, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = render.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.3)
    raise AssertionError("timeout")


def test_end_to_end_through_proxy(render, worker_url, tmp_path):
    from app import config
    assert render.post("/admin/worker", json={"url": worker_url}, headers=ADMIN).status_code == 200
    h = render.get("/health").json()["gpu_worker"]
    assert h["online"] is True and h["ready"] is True, h

    clip = make_clip(tmp_path / "c.mp4", 3)
    with open(clip, "rb") as fh:
        r = render.post("/api/upload", files={"file": ("c.mp4", fh, "video/mp4")},
                        data={"consent": "true", "tester_name": "Proxy Tester"})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    assert render.post(f"/api/jobs/{jid}/process", json={"target_language": "ig", "voice": "female"}).status_code == 200
    j = wait(render, jid)
    assert j["status"] == "done" and len(j["segments"]) == 2
    assert render.post(f"/api/jobs/{jid}/approve-all").status_code == 200
    assert render.post(f"/api/jobs/{jid}/tts", json={"speed": 1.1}).status_code == 200
    assert wait(render, jid)["has_audio_preview"]
    a = render.get(f"/api/jobs/{jid}/audio")
    assert a.status_code == 200 and a.content[:4] == b"RIFF"
    assert render.post(f"/api/jobs/{jid}/render", json={"subtitles": True}).status_code == 200
    assert wait(render, jid, 120)["has_video"]
    v = render.get(f"/api/jobs/{jid}/video", headers={"Range": "bytes=0-99"})
    assert v.status_code == 206 and len(v.content) == 100
    full = render.get(f"/api/jobs/{jid}/video?download=true")
    assert full.status_code == 200 and "attachment" in full.headers.get("content-disposition", "")

    # Oversized JSON body rejected at the proxy.
    big = "x" * (config.MAX_JSON_BYTES + 10)
    assert render.put(f"/api/jobs/{jid}/segments/0", content=big,
                      headers={"Content-Type": "application/json"}).status_code == 413

    s = render.get("/admin/sessions", headers=ADMIN).json()
    assert s["worker_log_error"] is None
    events = [e["event"] for e in s["entries"] if e.get("job_id") == jid]
    assert events == ["upload_received", "process_started", "transcribe_translate_done", "tts_started", "voice_done", "render_started", "render_done", "export_downloaded"]
    assert any(e["event"] == "upload_worker_offline" for e in s["entries"])  # from the offline test


def test_upload_size_and_rate_limit(render, worker_url, tmp_path):
    from app import config, main
    render.post("/admin/worker", json={"url": worker_url}, headers=ADMIN)
    old = config.MAX_VIDEO_MB
    config.MAX_VIDEO_MB = 0.1
    try:
        clip = make_clip(tmp_path / "c.mp4", 2)
        big = b"\0" * 300_000
        r = render.post("/api/upload", files={"file": ("c.mp4", big)}, data={"consent": "true"})
        assert r.status_code == 413
    finally:
        config.MAX_VIDEO_MB = old
    main._hits.clear()
    old_rl = config.RATE_LIMIT_UPLOADS
    config.RATE_LIMIT_UPLOADS = 2
    try:
        codes = []
        for _ in range(3):
            with open(clip, "rb") as fh:
                codes.append(render.post("/api/upload", files={"file": ("c.mp4", fh)},
                                         data={"consent": "true"}).status_code)
        assert codes == [200, 200, 429]
    finally:
        config.RATE_LIMIT_UPLOADS = old_rl
        main._hits.clear()


def test_worker_rejects_wrong_key_without_echo(render, worker_url):
    from app import config
    old = config.GPU_WORKER_KEY
    config.GPU_WORKER_KEY = "wrong-key-value"
    try:
        r = render.get("/api/jobs/" + "b" * 32)
        assert r.status_code == 502 and "wrong-key-value" not in r.text and "test-worker-key" not in r.text
    finally:
        config.GPU_WORKER_KEY = old
