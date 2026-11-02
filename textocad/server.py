from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from textocad import keystore
from textocad.core.config import settings
from textocad.generate import MODES, generate, rebuild
from textocad.services import cancellation, llm_models
from textocad.services.providers import ProviderKeyError, redact

BUILDS = Path("builds").resolve()
WEB = Path(__file__).resolve().parent / "web"
_ID = re.compile(r"^[0-9a-f]{12}$")

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

keystore.load()
app = FastAPI(title="TextoCAD engine")


@app.middleware("http")
async def local_only(request: Request, call_next):
    host = request.headers.get("host", "")
    origin = request.headers.get("origin")
    # Stops other websites open in the same browser from driving this server.
    if request.method not in ("GET", "HEAD", "OPTIONS") and origin and urlsplit(origin).netloc != host:
        return JSONResponse({"detail": "Cross-site request refused."}, status_code=403)
    if request.url.path.startswith("/settings") and host.rsplit(":", 1)[0] not in _LOCAL_HOSTS:
        return JSONResponse({"detail": "Open this page at http://127.0.0.1 to manage the API key."}, status_code=403)
    return await call_next(request)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    mode: str | None = None
    extraDetails: bool = False
    kernel: str = "openscad"
    parentId: str | None = None
    requestId: str | None = None


class KeyRequest(BaseModel):
    apiKey: str = Field(min_length=1, max_length=400)
    provider: str | None = None
    model: str | None = None


class ExportRequest(BaseModel):
    parts: list[str] = Field(min_length=1, max_length=200)
    format: str
    layout: str = "merged"
    colors: dict[str, str] = {}


class RebuildRequest(BaseModel):
    params: dict[str, float | int | bool | str] = {}


def _build_dir(build_id: str) -> Path:
    if not _ID.match(build_id):
        raise HTTPException(404)
    return BUILDS / build_id


def _publish(result, build_id: str, prompt: str = "") -> dict[str, Any]:
    data = result.to_dict()
    data["prompt"] = prompt
    _build_dir(build_id).joinpath("result.json").write_text(json.dumps(data), encoding="utf-8")
    data["id"] = build_id
    data["files"] = {k: f"/builds/{build_id}/{Path(v).name}" for k, v in result.files.items()}
    return data


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/logo.png", include_in_schema=False)
def logo() -> FileResponse:
    return FileResponse(WEB / "logo.png", media_type="image/png")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "providers": llm_models.list_providers()}


@app.get("/settings")
def get_settings() -> dict[str, Any]:
    return keystore.status()


@app.post("/settings/key")
def save_key(req: KeyRequest) -> dict[str, Any]:
    try:
        return keystore.save(req.apiKey, req.provider, req.model)
    except ProviderKeyError as exc:
        raise HTTPException(exc.status if 400 <= exc.status < 600 else 400, redact(str(exc), req.apiKey))


@app.delete("/settings/key")
def remove_key() -> dict[str, Any]:
    return keystore.clear()


@app.post("/generate")
def generate_model(req: GenerateRequest) -> dict[str, Any]:
    if not keystore.status()["hasKey"]:
        raise HTTPException(400, "Add an API key first.")
    if len(req.prompt) > settings.max_prompt_chars:
        raise HTTPException(413, "Prompt too long.")
    if req.mode and req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {', '.join(MODES)}")
    mode, prior_script, prior_prompt = req.mode, None, None
    if req.parentId:
        stored = _build_dir(req.parentId) / "result.json"
        if stored.is_file():
            prev = json.loads(stored.read_text(encoding="utf-8"))
            mode = mode or prev.get("mode")
            if prev.get("mode") == mode:
                prior_script, prior_prompt = prev.get("script") or None, prev.get("prompt") or None
    build_id = uuid.uuid4().hex[:12]
    token = cancellation.begin(req.requestId, "local")
    try:
        result = generate(
            req.prompt, mode, out_dir=_build_dir(build_id), prior_script=prior_script,
            prior_prompt=prior_prompt, extra_details=req.extraDetails, kernel=req.kernel,
        )
    except cancellation.GenerationCancelled:
        raise HTTPException(409, "Generation stopped.")
    finally:
        cancellation.end(token)
    if not result.ok:
        raise HTTPException(422, result.error)
    return _publish(result, build_id, req.prompt)


@app.post("/generate/{request_id}/stop")
def stop_generation(request_id: str) -> dict[str, bool]:
    return {"stopped": cancellation.request_cancel(request_id, "local")}


@app.post("/builds/{build_id}/rebuild")
def rebuild_model(build_id: str, req: RebuildRequest) -> dict[str, Any]:
    """Re-run a model this server generated, with new parameter values."""
    stored = _build_dir(build_id) / "result.json"
    if not stored.is_file():
        raise HTTPException(404)
    prev = json.loads(stored.read_text(encoding="utf-8"))
    if prev.get("mode") not in ("printing", "precision"):
        raise HTTPException(400, "Parameter rebuilds work for printing and precision models.")
    known = {p.get("key") for p in prev.get("params", [])}
    params = {k: v for k, v in req.params.items() if k in known}
    new_id = uuid.uuid4().hex[:12]
    result = rebuild(prev["script"], params, prev["mode"], param_schema=prev.get("params"), out_dir=_build_dir(new_id))
    if not result.ok:
        raise HTTPException(422, result.error)
    result.title = prev.get("title", "")
    return _publish(result, new_id, prev.get("prompt", ""))


@app.post("/builds/{build_id}/export")
def export_parts(build_id: str, req: ExportRequest) -> FileResponse:
    """Chosen parts in one format, as one file or a zip with a file per part."""
    from textocad.services.part_export import SELECTION_FORMATS, export_selection, stamp_exports

    build = _build_dir(build_id)
    if not (build / "result.json").is_file():
        raise HTTPException(404)
    if req.format not in SELECTION_FORMATS or req.layout not in ("merged", "separate"):
        raise HTTPException(400, "Unsupported format or layout.")
    try:
        path = export_selection(str(build), req.parts, req.format, req.layout, req.colors)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    stamp_exports([path])
    return FileResponse(path, filename=Path(path).name)


@app.get("/builds/{build_id}/{name}")
def download(build_id: str, name: str) -> FileResponse:
    path = (_build_dir(build_id) / name).resolve()
    if BUILDS not in path.parents or not path.is_file() or name == "result.json":
        raise HTTPException(404)
    return FileResponse(path, filename=name)
