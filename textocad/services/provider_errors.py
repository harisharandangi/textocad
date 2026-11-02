"""What to do when the AI provider itself refuses a call."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


_MAX_MODELS_PER_CALL = 4


class ProviderError(RuntimeError):
    """The provider refused the call; retrying the same request will not help."""

    def __init__(self, message: str, *, kind: str, model: str = "", detail: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.model = model
        self.detail = detail


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


_QUOTA_WORDS = (
    "quota", "insufficient_quota", "billing", "credit", "balance", "limit: 0",
    "payment", "exceeded your current", "per day", "free tier", "free_tier",
)
_MODEL_WORDS = (
    "no longer available", "not found", "does not exist", "not supported",
    "unknown model", "invalid model", "model_not_found", "is not available",
    "deprecated", "decommissioned", "not have access to model", "no access to model",
)
_AUTH_WORDS = (
    "api key not valid", "invalid api key", "invalid x-api-key", "incorrect api key",
    "api_key_invalid", "unauthorized", "authentication", "permission_denied",
    "invalid_api_key", "api key expired",
)


def classify(exc: BaseException) -> str | None:
    """auth / model / quota / rate for a provider refusal, else None."""
    if isinstance(exc, ProviderError):
        return exc.kind
    module = type(exc).__module__ or ""
    if not (module.startswith("openai") or module.startswith("anthropic") or module.startswith("httpx")):
        return None


    status = _status(exc)
    text = str(exc).lower()
    if status == 401:
        return "auth"
    if status == 402:
        return "quota"
    if status == 404:
        return "model"
    if status == 429:
        return "quota" if any(w in text for w in _QUOTA_WORDS) else "rate"
    if status == 403:
        if any(w in text for w in _MODEL_WORDS) or "model" in text:
            return "model"
        return "auth"
    if status == 400:
        if any(w in text for w in _AUTH_WORDS):
            return "auth"
        if "model" in text and any(w in text for w in _MODEL_WORDS):
            return "model"
        if any(w in text for w in ("quota", "billing", "credit balance")):
            return "quota"
        return None
    return None


_RETRY_RE = re.compile(r"retry in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", re.IGNORECASE)


def _retry_after_seconds(text: str) -> float | None:
    match = _RETRY_RE.search(text)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _when(seconds: float | None) -> str:
    if not seconds:
        return ""
    if seconds >= 5400:
        return f" It resets in about {round(seconds / 3600)} hours."
    if seconds >= 90:
        return f" It resets in about {round(seconds / 60)} minutes."
    return " Try again in a minute."


def _own_provider(active: Any) -> str | None:
    """The saved-key provider id an active selection runs on, if it is one."""
    provider = str(getattr(active, "provider", "") or "")
    if provider.startswith("own-"):
        return provider[4:]
    from textocad.services import llm_models

    if llm_models.byok_active():
        return llm_models.BYOK_KEY_SOURCE.get(provider)
    return None


def _provider_label(own: str | None) -> str:
    if not own:
        return "The AI service"
    from textocad.services.providers import PROVIDERS

    label = str(PROVIDERS.get(own, {}).get("label") or own)
    return label.split(" (")[0]


def friendly(kind: str, *, own: str | None, model: str, raw: str) -> str:
    """The sentence a user reads. Never contains the key."""
    if not own:
        if kind == "rate":
            return "The AI service is busy right now. Please try again in a minute."
        return "The AI service is not answering right now. Please try again in a few minutes."

    label = _provider_label(own)
    where = "Check TEXTOCAD_API_KEY / TEXTOCAD_MODEL."
    low = raw.lower()
    if kind == "auth":
        return (
            f"{label} did not accept your API key. It may have been deleted or "
            f"revoked. Check TEXTOCAD_API_KEY."
        )
    if kind == "model":
        return (
            f"The model {model} is not available on your {label} key, and no other model "
            f"on the key answered. Pick a different model for the key. {where}"
        )
    if kind == "quota":
        free = "free" in low and ("tier" in low or "daily" in low or "per day" in low)
        what = "used its free daily limit" if free else "run out of quota or credit"
        return (
            f"Your {label} key has {what}, on every model we could try."
            f"{_when(_retry_after_seconds(raw))} To keep going now, add billing to the key "
            f"with {label} or use another key. {where}"
        )
    return f"{label} is limiting how fast your key can be used. Wait a minute and try again."


_dead: dict[tuple[str, str, str], float] = {}
_dead_lock = threading.Lock()


def _key_id(base_url: str, api_key: str) -> tuple[str, str]:
    digest = hashlib.sha256(str(api_key or "").encode()).hexdigest()[:24]
    return str(base_url or ""), digest


def _is_dead(base_url: str, api_key: str, model: str) -> bool:
    key = (*_key_id(base_url, api_key), model)
    with _dead_lock:
        until = _dead.get(key)
        if until is None:
            return False
        if until < time.time():
            _dead.pop(key, None)
            return False
        return True


def _mark_dead(base_url: str, api_key: str, model: str, kind: str, raw: str) -> None:
    if kind == "model":
        ttl = 6 * 3600.0
    else:
        ttl = min(max(_retry_after_seconds(raw) or 3600.0, 60.0), 24 * 3600.0)
    with _dead_lock:
        if len(_dead) > 5000:
            _dead.clear()
        _dead[(*_key_id(base_url, api_key), model)] = time.time() + ttl


_switches: ContextVar[dict | None] = ContextVar("textocad_model_switches", default=None)


def start_tracking():
    return _switches.set({})


def stop_tracking(token) -> None:
    _switches.reset(token)


def switched_model() -> dict | None:
    """{"from", "to", "kind"} when this generation moved to another model."""
    record = _switches.get()
    return dict(record) if record else None


def _note_switch(original: str, used: str, kind: str) -> None:
    record = _switches.get()
    if record is not None and "from" not in record:
        record.update({"from": original, "to": used, "kind": kind})
    elif record is not None:
        record["to"] = used


def call_with_fallback(
    send: Callable[[str], T],
    model: str,
    *,
    base_url: str,
    api_key: str,
) -> T:
    """Run ``send(model)``; on a provider refusal, try the key's next models."""
    from textocad.services import llm_models
    from textocad.services.providers import redact

    active = llm_models.get_active()
    own = _own_provider(active)
    can_switch = bool(own) and str(getattr(active, "provider", "")).startswith("own-")

    candidates: list[str] = [model]
    expanded = False
    tried: list[str] = []
    last_kind = ""
    last_raw = ""
    first_kind = ""

    while True:
        current = next(
            (m for m in candidates if m not in tried and not _is_dead(base_url, api_key, m)),
            None,
        )
        if current is None and can_switch and not expanded:
            expanded = True
            from textocad.services.providers import fallback_models

            try:
                extra = fallback_models(own, api_key, exclude=set(tried) | {model})
            except Exception:
                extra = []
            candidates.extend(m for m in extra if m not in candidates)
            continue
        if current is None or len(tried) >= _MAX_MODELS_PER_CALL:
            kind = last_kind or "model"
            raise ProviderError(
                friendly(kind, own=own, model=model, raw=last_raw),
                kind=kind,
                model=model,
                detail=redact(last_raw, api_key)[:1500],
            )

        tried.append(current)
        try:
            result = send(current)
        except Exception as exc:
            kind = classify(exc)
            if kind is None:
                raise
            raw = str(exc)
            last_kind, last_raw = kind, raw
            first_kind = first_kind or kind
            log.warning(
                "provider refused %s on %s (%s): %s",
                current, getattr(active, "provider", "?"), kind, redact(raw, api_key)[:300],
            )
            if kind in ("model", "quota") and can_switch:
                _mark_dead(base_url, api_key, current, kind, raw)
                continue
            raise ProviderError(
                friendly(kind, own=own, model=current, raw=raw),
                kind=kind,
                model=current,
                detail=redact(raw, api_key)[:1500],
            ) from None

        if current != model:
            _note_switch(model, current, first_kind)

            if active is not None and getattr(active, "model", None) == model:
                llm_models.set_active(replace(active, model=current))
        return result
