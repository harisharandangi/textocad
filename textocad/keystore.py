"""The API key saved from the web page, kept in ~/.textocad/credentials.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

from textocad.core.config import settings
from textocad.services import providers

_ENV = {
    "key": settings.textocad_api_key,
    "provider": settings.textocad_provider,
    "model": settings.textocad_model,
    "deepseek": settings.deepseek_api_key,
}


def path() -> Path:
    return Path(os.getenv("TEXTOCAD_HOME", "").strip() or Path.home() / ".textocad") / "credentials.json"


def _read() -> dict:
    try:
        data = json.loads(path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("api_key") else {}
    except (OSError, ValueError):
        return {}


def _apply(key: str | None, provider: str | None, model: str | None) -> None:
    settings.textocad_api_key = key
    settings.textocad_provider = provider
    settings.textocad_model = model
    settings.deepseek_api_key = key if provider == "deepseek" else _ENV["deepseek"]


def load() -> None:
    """Use the saved key if there is one, otherwise whatever .env set."""
    saved = _read()
    if saved:
        _apply(saved["api_key"], saved.get("provider"), saved.get("model"))


def save(api_key: str, provider: str | None = None, model: str | None = None) -> dict:
    """Check the key with its provider, then store it. Raises ProviderKeyError."""
    pid, models = providers.identify(api_key, provider)
    key = api_key.strip()
    model = (model or "").strip() or None
    if model and models and model not in models:
        raise providers.ProviderKeyError(f"{model} is not available with this key.")
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"api_key": key, "provider": pid, "model": model}, fh)
    _apply(key, pid, model)
    return status()


def clear() -> dict:
    try:
        path().unlink()
    except FileNotFoundError:
        pass
    _apply(_ENV["key"], _ENV["provider"], _ENV["model"])
    return status()


def status() -> dict:
    key = settings.textocad_api_key or ""
    label = providers.PROVIDERS.get(settings.textocad_provider or "", {}).get("label")
    return {
        "hasKey": bool(key or settings.deepseek_api_key or settings.anthropic_api_key),
        "saved": bool(_read()),
        "provider": label or settings.textocad_provider or ("DeepSeek" if settings.deepseek_api_key else None),
        "model": settings.textocad_model or None,
        "hint": f"…{key[-4:]}" if len(key) >= 8 else None,
        "providers": [{"id": pid, "label": spec["label"], "console": spec.get("console")}
                      for pid, spec in providers.PROVIDERS.items()],
    }
