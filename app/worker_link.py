"""Where the GPU worker lives and whether it is reachable.

The worker registers its live Cloudflare tunnel URL via POST /admin/worker.
It is kept in memory only; when absent (e.g. after the free instance wakes
from sleep) the env GPU_WORKER_URL is used until the worker's next heartbeat.
"""

import time
from urllib.parse import urlparse

import httpx

from . import config

_state = {"url": None, "registered_at": None}
_health_cache = {"t": 0.0, "value": None}


def set_url(url: str) -> str:
    url = url.strip().rstrip("/")
    p = urlparse(url)
    allow_http = p.hostname in ("localhost", "127.0.0.1")
    if p.scheme not in ("https",) + (("http",) if allow_http else ()) or not p.hostname \
            or p.username or p.password or p.query or p.fragment:
        raise ValueError("Worker URL must be an https URL without credentials, query or fragment")
    _state.update(url=url, registered_at=time.time())
    _health_cache.update(t=0.0, value=None)
    return url


def current_url() -> str:
    return _state["url"] or config.GPU_WORKER_URL


def info() -> dict:
    src = "registered" if _state["url"] else ("env" if config.GPU_WORKER_URL else None)
    return {"source": src, "registered_at": _state["registered_at"]}


def headers() -> dict:
    return {"X-API-Key": config.GPU_WORKER_KEY} if config.GPU_WORKER_KEY else {}


async def health(max_age: float = 15.0) -> dict:
    url = current_url()
    if not url:
        return {"online": False, "reason": "not_registered",
                "detail": "No GPU worker has registered yet."}
    now = time.time()
    if _health_cache["value"] is not None and now - _health_cache["t"] < max_age:
        return _health_cache["value"]
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=5.0), trust_env=False) as c:
            r = await c.get(f"{url}/dub/health", headers=headers())
        if r.status_code == 200:
            body = r.json()
            models = body.get("models", {})
            value = {"online": True, "models": models,
                     "ready": bool(models.get("translate") and models.get("speech")),
                     "limits": body.get("limits"), **info()}
        elif r.status_code in (401, 503):
            value = {"online": False, "reason": "auth",
                     "detail": "Worker rejected Render's API key (GPU_WORKER_KEY must equal worker DEV_API_KEY)."}
        else:
            value = {"online": False, "reason": "http_error", "detail": f"Worker returned HTTP {r.status_code}"}
    except (httpx.HTTPError, ValueError) as exc:
        value = {"online": False, "reason": "unreachable", "detail": type(exc).__name__}
    _health_cache.update(t=now, value=value)
    return value
