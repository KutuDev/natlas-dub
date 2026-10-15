"""N-ATLAS Dub, Render thin layer: static UI + authenticated proxy to the Kaggle GPU worker.

No ML and no FFmpeg run here (Render free plan, 512 MB RAM).
"""

import json
import logging
import re
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import config, session_log, worker_link

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
JOB_RE = re.compile(r"[0-9a-f]{32}")
OFFLINE_MSG = "GPU worker offline — processing unavailable"


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"t": self.formatTime(record), "level": record.levelname,
                           "logger": record.name, "msg": record.getMessage()})


log = logging.getLogger("natlas_render")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(_JsonFormatter())
    log.addHandler(_h)
    log.setLevel(logging.INFO)

app = FastAPI(title="N-ATLAS Dub", version="1.0.0", docs_url=None, redoc_url=None)
if config.ALLOWED_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS,
                       allow_methods=["GET", "POST", "PUT"], allow_headers=["Content-Type", "X-API-Key"])


# ---------------------------------------------------------------- auth
def optional_api_key(x_api_key: Optional[str] = Header(default=None)):
    if config.PUBLIC_API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, config.PUBLIC_API_KEY)):
        raise HTTPException(401, "Invalid API key")


def admin_key(x_admin_key: Optional[str] = Header(default=None)):
    if not config.ADMIN_KEY:
        raise HTTPException(503, "Admin access is not configured")  # fail closed
    if not x_admin_key or not secrets.compare_digest(x_admin_key, config.ADMIN_KEY):
        raise HTTPException(401, "Invalid admin key")


api = [Depends(optional_api_key)]


# ---------------------------------------------------------------- rate limit (job creation)
_hits = defaultdict(deque)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def rate_limited(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > config.RATE_LIMIT_WINDOW_S:
        q.popleft()
    if len(q) >= config.RATE_LIMIT_UPLOADS:
        return True
    q.append(now)
    return False


# ---------------------------------------------------------------- proxy helpers
def _check_job(job_id: str):
    if not JOB_RE.fullmatch(job_id):
        raise HTTPException(404, "Job not found")


def _offline(detail: str = "", status: int = 503):
    return JSONResponse({"detail": OFFLINE_MSG + (f" ({detail})" if detail else ""), "code": "worker_offline"},
                        status_code=status)


def _map_worker_status(r: httpx.Response) -> JSONResponse:
    if r.status_code in (401, 403) or (r.status_code == 503 and "authentication" in r.text.lower()):
        log.error("worker rejected Render credentials (HTTP %s)", r.status_code)
        return JSONResponse({"detail": "GPU worker refused the connection (server configuration).",
                             "code": "worker_auth"}, status_code=502)
    try:
        body = r.json()
    except ValueError:
        body = {"detail": f"Worker returned HTTP {r.status_code}"}
    return JSONResponse(body, status_code=r.status_code)


async def _forward_json(method: str, path: str, request: Request = None, timeout: float = None):
    url = worker_link.current_url()
    if not url:
        return _offline("no worker registered")
    body = None
    if request is not None:
        body = await request.body()
        if len(body) > config.MAX_JSON_BYTES:
            raise HTTPException(413, "Request too large")
    last = None
    for attempt in range(config.WORKER_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout or config.TIMEOUT_DEFAULT, connect=10.0),
                                         trust_env=False) as c:
                r = await c.request(method, f"{url}{path}", content=body,
                                    headers={**worker_link.headers(), "Content-Type": "application/json"})
            return _map_worker_status(r)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last = exc
            continue
        except httpx.ReadTimeout:
            return JSONResponse({"detail": "GPU worker timed out; please retry.", "code": "worker_timeout"},
                                status_code=504)
        except httpx.HTTPError as exc:
            last = exc
            break
    log.warning("worker unreachable on %s %s: %s", method, path, type(last).__name__)
    return _offline()


async def _stream_media(path: str, request: Request):
    url = worker_link.current_url()
    if not url:
        return _offline("no worker registered")
    client = httpx.AsyncClient(timeout=httpx.Timeout(config.TIMEOUT_MEDIA, connect=10.0), trust_env=False)
    hdrs = worker_link.headers()
    if "range" in request.headers:
        hdrs["Range"] = request.headers["range"]
    try:
        req = client.build_request("GET", f"{url}{path}", headers=hdrs, params=dict(request.query_params))
        r = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        log.warning("worker unreachable on GET %s: %s", path, type(exc).__name__)
        return _offline()
    if r.status_code >= 400:
        await r.aread()
        await client.aclose()
        return _map_worker_status(r)
    keep = {k: v for k, v in r.headers.items()
            if k.lower() in ("content-type", "content-length", "content-range", "accept-ranges",
                             "content-disposition", "last-modified", "etag")}

    async def close():
        await r.aclose()
        await client.aclose()
    return StreamingResponse(r.aiter_raw(), status_code=r.status_code, headers=keep,
                             background=BackgroundTask(close))


# ---------------------------------------------------------------- health / admin
@app.get("/health")
async def health():
    """Always 200 for Render's health check; reports worker reachability."""
    return {"status": "ok", "gpu_worker": await worker_link.health(),
            "limits": {"max_video_seconds": config.MAX_VIDEO_SECONDS, "max_video_mb": config.MAX_VIDEO_MB}}


class WorkerRegistration(BaseModel):
    url: str = Field(min_length=8, max_length=300)


@app.post("/admin/worker", dependencies=[Depends(admin_key)])
async def register_worker(body: WorkerRegistration):
    try:
        url = worker_link.set_url(body.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    log.info("worker registered: %s", url)
    return {"ok": True, "url": url}


@app.get("/admin/worker", dependencies=[Depends(admin_key)])
async def worker_status():
    return {"url": worker_link.current_url() or None, **worker_link.info(),
            "health": await worker_link.health(max_age=0)}


@app.get("/admin/sessions", dependencies=[Depends(admin_key)])
async def admin_sessions():
    render_entries = session_log.read_all()
    worker_entries, worker_error = [], None
    url = worker_link.current_url()
    if url:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0), trust_env=False) as c:
                r = await c.get(f"{url}/dub/admin/sessions", headers=worker_link.headers())
            if r.status_code == 200:
                worker_entries = r.json().get("entries", [])
            else:
                worker_error = f"worker returned HTTP {r.status_code}"
        except (httpx.HTTPError, ValueError) as exc:
            worker_error = type(exc).__name__
    else:
        worker_error = "no worker registered"
    entries = sorted(render_entries + worker_entries, key=lambda e: e.get("timestamp", ""))
    return {"count": len(entries), "worker_log_error": worker_error, "entries": entries}


# ---------------------------------------------------------------- public API (proxied)
@app.post("/api/upload", dependencies=api)
async def upload(request: Request):
    ip = client_ip(request)
    if rate_limited(ip):
        session_log.log_event("upload_rate_limited", False)
        raise HTTPException(429, "Too many uploads from your network; please wait a few minutes.")
    max_bytes = int(config.MAX_VIDEO_MB * 1024 * 1024) + 64 * 1024  # + multipart overhead
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        session_log.log_event("upload_rejected_size", False)
        raise HTTPException(413, f"Video exceeds {config.MAX_VIDEO_MB:g}MB")
    url = worker_link.current_url()
    if not url:
        session_log.log_event("upload_worker_offline", False)
        return _offline("no worker registered")

    seen = {"n": 0}

    async def body():
        async for chunk in request.stream():
            seen["n"] += len(chunk)
            if seen["n"] > max_bytes:
                raise HTTPException(413, f"Video exceeds {config.MAX_VIDEO_MB:g}MB")
            yield chunk

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(config.TIMEOUT_MEDIA, connect=10.0),
                                     trust_env=False) as c:
            r = await c.post(f"{url}/dub/upload", content=body(),
                             headers={**worker_link.headers(),
                                      "Content-Type": request.headers.get("content-type", "")})
    except HTTPException:
        session_log.log_event("upload_rejected_size", False)
        raise
    except httpx.HTTPError as exc:
        log.warning("upload proxy failed: %s", type(exc).__name__)
        session_log.log_event("upload_worker_offline", False, error=type(exc).__name__)
        return _offline()
    resp = _map_worker_status(r)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = None
        session_log.log_event("upload_rejected", False, status=r.status_code, error=str(detail)[:200])
    return resp


@app.get("/api/jobs/{job_id}", dependencies=api)
async def job_status(job_id: str):
    _check_job(job_id)
    return await _forward_json("GET", f"/dub/jobs/{job_id}")


@app.post("/api/jobs/{job_id}/process", dependencies=api)
async def process(job_id: str, request: Request):
    _check_job(job_id)
    return await _forward_json("POST", f"/dub/jobs/{job_id}/process", request)


@app.put("/api/jobs/{job_id}/segments/{idx}", dependencies=api)
async def edit_segment(job_id: str, idx: int, request: Request):
    _check_job(job_id)
    return await _forward_json("PUT", f"/dub/jobs/{job_id}/segments/{idx}", request)


@app.post("/api/jobs/{job_id}/segments/{idx}/approve", dependencies=api)
async def approve_segment(job_id: str, idx: int):
    _check_job(job_id)
    return await _forward_json("POST", f"/dub/jobs/{job_id}/segments/{idx}/approve")


@app.post("/api/jobs/{job_id}/approve-all", dependencies=api)
async def approve_all(job_id: str):
    _check_job(job_id)
    return await _forward_json("POST", f"/dub/jobs/{job_id}/approve-all")


@app.post("/api/jobs/{job_id}/tts", dependencies=api)
async def tts(job_id: str, request: Request):
    _check_job(job_id)
    return await _forward_json("POST", f"/dub/jobs/{job_id}/tts", request)


@app.post("/api/jobs/{job_id}/render", dependencies=api)
async def render(job_id: str, request: Request):
    _check_job(job_id)
    return await _forward_json("POST", f"/dub/jobs/{job_id}/render", request)


@app.get("/api/jobs/{job_id}/{kind}", dependencies=api)
async def media(job_id: str, kind: str, request: Request):
    _check_job(job_id)
    if kind not in ("source", "audio", "video", "srt"):
        raise HTTPException(404, "Not found")
    return await _stream_media(f"/dub/jobs/{job_id}/{kind}", request)


# ---------------------------------------------------------------- static UI
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
