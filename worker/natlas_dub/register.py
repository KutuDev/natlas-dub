"""Cloudflare quick tunnel + registration of the live tunnel URL with the Render app.

The Render app (free plan) keeps the worker URL only in memory and loses it when
it sleeps/restarts, so registration repeats every REGISTER_INTERVAL_S seconds.
Note: these periodic requests also count as traffic and keep the free Render
instance awake while the worker is running.
"""

import logging
import re
import subprocess
import threading
import time
from typing import Optional

import httpx

from . import config

log = logging.getLogger("natlas_dub.register")
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_state = {"url": None, "last_ok": None, "last_error": None}


def start_tunnel(port: int = 8000, binary: str = "cloudflared", timeout: float = 60) -> str:
    """Start `cloudflared tunnel --url http://localhost:PORT` and return the public URL."""
    proc = subprocess.Popen(
        [binary, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    found = {}

    def _reader():
        for line in proc.stdout:
            m = TUNNEL_RE.search(line)
            if m and "url" not in found:
                found["url"] = m.group(0)
    threading.Thread(target=_reader, daemon=True).start()
    deadline = time.time() + timeout
    while time.time() < deadline and "url" not in found:
        if proc.poll() is not None:
            raise RuntimeError("cloudflared exited before printing a tunnel URL")
        time.sleep(0.5)
    if "url" not in found:
        raise RuntimeError("Timed out waiting for the cloudflared tunnel URL")
    return found["url"]


def register_once(worker_url: str) -> bool:
    if not (config.RENDER_APP_URL and config.ADMIN_KEY):
        _state["last_error"] = "RENDER_APP_URL or ADMIN_KEY not set; skipping registration"
        log.warning(_state["last_error"])
        return False
    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=60.0), trust_env=False) as client:
            # Generous timeout: a sleeping free Render instance can take ~1 min to wake.
            resp = client.post(f"{config.RENDER_APP_URL}/admin/worker",
                               json={"url": worker_url},
                               headers={"X-Admin-Key": config.ADMIN_KEY})
        if resp.status_code == 200:
            _state.update(last_ok=time.time(), last_error=None)
            return True
        _state["last_error"] = f"Render returned HTTP {resp.status_code}"
    except httpx.HTTPError as exc:
        _state["last_error"] = f"{type(exc).__name__}"
    log.warning("worker registration failed: %s", _state["last_error"])  # never logs the key
    return False


def start_heartbeat(worker_url: str, interval: Optional[float] = None) -> threading.Thread:
    _state["url"] = worker_url
    interval = interval or config.REGISTER_INTERVAL_S

    def _loop():
        while True:
            ok = register_once(worker_url)
            log.info("registered worker with Render: %s", ok)
            time.sleep(interval)
    t = threading.Thread(target=_loop, daemon=True, name="render-register")
    t.start()
    return t


def status() -> dict:
    return dict(_state)
