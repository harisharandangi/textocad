"""Multi-model registry + request-scoped model selection."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any

from textocad.core.config import setting_or_default, settings

SPEEDS = ("fast", "standard", "deepthink")


@dataclass(frozen=True)
class ResolvedLlm:
    provider: str
    label: str
    base_url: str
    api_key: str
    model: str


    protocol: str = "anthropic"


    vision_model: str = ""


    supports_images: bool = False


_cfg = setting_or_default


_VISION_OFF = {"off", "none", "disabled", "false"}


def _optional_model(name: str) -> str:
    """A model id, or "" when explicitly switched off."""
    value = _cfg(name)
    return "" if value.strip().lower() in _VISION_OFF else value


def _deepseek_key() -> str | None:
    if settings.deepseek_api_key:
        return settings.deepseek_api_key


    if "deepseek" in (settings.anthropic_base_url or "").lower():
        return settings.anthropic_api_key
    return None


def _providers() -> dict[str, dict[str, Any]]:


    openrouter_base = _cfg("openrouter_base_url")
    defs = {
        "deepseek": {
            "label": "DeepSeek",
            "base_url": _cfg("deepseek_base_url"),
            "api_key": _deepseek_key(),

            "vision_model": _optional_model("deepseek_model_vision"),
            "models": {
                "fast": _cfg("deepseek_model_fast"),
                "standard": _cfg("deepseek_model_standard"),
                "deepthink": _cfg("deepseek_model_deepthink"),
            },
        },
        "glm": {
            "label": "GLM",
            "base_url": settings.glm_base_url or openrouter_base,
            "api_key": settings.glm_api_key or settings.openrouter_api_key,
            "models": {
                "fast": _cfg("glm_model_fast"),
                "standard": _cfg("glm_model_standard"),
                "deepthink": _cfg("glm_model_deepthink"),
            },
        },
        "kimi": {
            "label": "Kimi",
            "base_url": settings.kimi_base_url or openrouter_base,
            "api_key": settings.kimi_api_key or settings.openrouter_api_key,
            "models": {
                "fast": _cfg("kimi_model_fast"),
                "standard": _cfg("kimi_model_standard"),
                "deepthink": _cfg("kimi_model_deepthink"),
            },
        },
        "minimax": {
            "label": "MiniMax",
            "base_url": settings.minimax_base_url or openrouter_base,
            "api_key": settings.minimax_api_key or settings.openrouter_api_key,
            "models": {
                "fast": _cfg("minimax_model_fast"),
                "standard": _cfg("minimax_model_standard"),
                "deepthink": _cfg("minimax_model_deepthink"),
            },
        },
    }

    byok = _byok_keys.get()
    if byok is None:
        return defs


    for pid, p in defs.items():
        source = BYOK_KEY_SOURCE.get(pid)
        user_key = byok.get(source) if source else None
        p["api_key"] = user_key or None
        if user_key and source == "openrouter":
            p["base_url"] = openrouter_base
    return defs


BYOK_KEY_SOURCE: dict[str, str] = {
    "deepseek": "deepseek",
    "glm": "openrouter",
    "kimi": "openrouter",
    "minimax": "openrouter",
}


BYOK_DEFAULT_ENGINE: dict[str, str] = {"deepseek": "deepseek", "openrouter": "glm"}

_byok_keys: ContextVar[dict[str, str] | None] = ContextVar("textocad_byok_keys", default=None)


@contextmanager
def byok_scope(keys: dict[str, str]):
    """Serve every LLM call in this block from the user's own keys only."""
    token = _byok_keys.set(dict(keys))
    try:
        yield
    finally:
        _byok_keys.reset(token)


def byok_active() -> bool:
    return _byok_keys.get() is not None


def normalize_speed(value: str | None) -> str:
    clean = str(value or "").strip().lower()
    return clean if clean in SPEEDS else "standard"


def _image_provider_ids() -> set[str]:
    return {
        p.strip().lower()
        for p in (settings.llm_image_providers or "").split(",")
        if p.strip()
    }


def available_speeds(models: dict[str, str]) -> list[str]:
    """Speed tiers that map to DISTINCT API models."""
    by_model: dict[str, list[str]] = {}
    for s in SPEEDS:
        by_model.setdefault(models[s], []).append(s)

    kept: set[str] = set()
    for speeds in by_model.values():
        if "standard" in speeds:
            kept.add("standard")
        elif "deepthink" in speeds:
            kept.add("deepthink")
        else:
            kept.add("fast")
    return [s for s in SPEEDS if s in kept]


def list_providers() -> list[dict[str, Any]]:
    """Public list for the frontend model menu: id, label, whether the provider is usable (key configured), which."""
    image_ids = _image_provider_ids()
    return [
        {
            "id": pid,
            "label": p["label"],
            "configured": bool(p["api_key"]),
            "speeds": available_speeds(p["models"]),
            "supportsImages": pid in image_ids or bool(p.get("vision_model")),
        }
        for pid, p in _providers().items()
    ]


def resolve(provider: str | None, speed: str | None) -> ResolvedLlm | None:
    """Resolve a provider+speed to a concrete endpoint/model."""
    pid = str(provider or "").strip().lower()
    defs = _providers()
    if pid not in defs:
        return None
    p = defs[pid]
    if not p["api_key"]:
        return None
    s = normalize_speed(speed)
    return ResolvedLlm(
        provider=pid,
        label=p["label"],
        base_url=p["base_url"],
        api_key=p["api_key"],
        model=p["models"][s],
        protocol=str(p.get("protocol") or "anthropic"),
        vision_model=str(p.get("vision_model") or ""),


        supports_images=(
            pid in _image_provider_ids() or bool(p.get("vision_model"))
        ),
    )


def resolve_utility(provider: str, model_id: str) -> ResolvedLlm | None:
    """Resolve a UTILITY model (router, classifier) by id."""
    pid = str(provider or "").strip().lower()
    wanted = str(model_id or "").strip()
    defs = _providers()
    if pid not in defs or not wanted:
        return None
    p = defs[pid]
    if not p["api_key"]:
        return None
    return ResolvedLlm(
        provider=pid,
        label=p["label"],
        base_url=p["base_url"],
        api_key=p["api_key"],
        model=wanted,
        protocol=str(p.get("protocol") or "anthropic"),
        supports_images=pid in _image_provider_ids(),
    )


def resolve_own_key() -> ResolvedLlm | None:
    """The model TEXTOCAD_PROVIDER / TEXTOCAD_API_KEY / TEXTOCAD_MODEL name, or None when they are unset."""
    from textocad.services import providers

    pid = str(settings.textocad_provider or "").strip().lower()
    key = str(settings.textocad_api_key or "").strip()
    if not key:
        return None
    if not pid or pid == "auto":
        pid, models = providers.identify(key)
    else:
        models = [] if settings.textocad_model else providers.check_with_provider(pid, key)
    if providers.PROVIDERS.get(pid, {}).get("engine"):
        return resolve(providers.PROVIDERS[pid]["engine"], settings.llm_default_speed)
    model = settings.textocad_model or providers.default_model(pid, models)
    return providers.selection_for(pid, key, model)


def resolve_default() -> ResolvedLlm | None:
    return resolve(settings.llm_default_provider, settings.llm_default_speed)


_active: ContextVar[ResolvedLlm | None] = ContextVar("textocad_active_llm", default=None)


def set_active(selection: ResolvedLlm | None) -> Token:
    return _active.set(selection)


def reset_active(token: Token) -> None:
    _active.reset(token)


def get_active() -> ResolvedLlm | None:
    return _active.get()


@contextmanager
def fast_tier():
    """Run HELPER agents (intent classification, assembly planning, requirement-spec extraction) on the."""
    active = get_active()
    if active is None:
        yield
        return
    fast = resolve(active.provider, "fast")
    if fast is None or fast.model == active.model:
        yield
        return
    token = set_active(fast)
    try:
        yield
    finally:
        reset_active(token)


def _helper_selection(role: str, *, has_image: bool = False) -> ResolvedLlm | None:
    """Resolve the model configured for a helper ROLE."""
    spec = setting_or_default(f"helper_model_{role}")
    if has_image:


        active_img = get_active()
        if active_img is not None and active_img.vision_model:
            return replace(
                active_img,
                model=active_img.vision_model,
                label=f"{active_img.label} vision",
                supports_images=True,
            )
        spec = setting_or_default("helper_model_vision") or spec
    provider, _, model_id = spec.partition(":")
    provider, model_id = provider.strip().lower(), model_id.strip()
    if not provider or not model_id:
        return None

    active = get_active()
    if not has_image and active is not None and provider != active.provider:


        return resolve(active.provider, "fast")

    if model_id in SPEEDS:


        return resolve(provider, model_id)

    if model_id == "vision":


        base = resolve(provider, "fast")
        if base is None or not base.vision_model:
            return None
        return replace(
            base,
            model=base.vision_model,
            label=f"{base.label} vision",
            supports_images=True,
        )

    return resolve_utility(provider, model_id)


@contextmanager
def helper_tier(role: str, *, has_image: bool = False):
    """Run a HELPER call on a model sized to that helper's job."""
    selection = None
    try:
        selection = _helper_selection(role, has_image=has_image)
    except Exception:
        selection = None

    if selection is None:
        with fast_tier():
            yield
        return

    token = set_active(selection)
    try:
        yield
    finally:
        reset_active(token)


_llm_seconds: ContextVar[list[float] | None] = ContextVar(
    "textocad_llm_seconds", default=None
)


def start_llm_timing() -> Token:
    return _llm_seconds.set([])


def add_llm_seconds(elapsed: float) -> None:
    bucket = _llm_seconds.get()
    if bucket is not None and elapsed > 0:
        bucket.append(float(elapsed))


def read_llm_seconds() -> float:
    return float(sum(_llm_seconds.get() or []))


def stop_llm_timing(token: Token) -> None:
    _llm_seconds.reset(token)
