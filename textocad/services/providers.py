"""AI providers you can run TextoCAD on with your own API key."""

from __future__ import annotations

import re
import time

import httpx


PROVIDERS: dict[str, dict] = {
    "openai": {
        "label": "OpenAI",
        "protocol": "openai",
        "base_url": "https://api.openai.com/v1",
        "console": "https://platform.openai.com/api-keys",
        "prefixes": ("sk-proj-", "sk-svcacct-", "sk-admin-"),
        "images": True,
        "prefer": (r"^gpt-5(\.\d+)?$", r"^gpt-5", r"^o\d+$", r"^gpt-4\.1$", r"^gpt-4o$"),
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "protocol": "anthropic",
        "base_url": "https://api.anthropic.com",
        "models_url": "https://api.anthropic.com/v1/models",
        "console": "https://console.anthropic.com/settings/keys",
        "prefixes": ("sk-ant-",),
        "images": True,
        "prefer": (r"sonnet", r"opus", r"haiku"),
    },
    "gemini": {
        "label": "Google Gemini",
        "protocol": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "console": "https://aistudio.google.com/apikey",
        "prefixes": ("AIza",),
        "images": True,


        "prefer": (
            r"^gemini-[\d.]+-pro(-preview)?$",
            r"^gemini-[\d.]+-flash(-preview)?$",
            r"^gemini-[\d.]+-pro",
            r"^gemini-[\d.]+-flash(?!-lite)",
            r"^gemini",
        ),
    },
    "xai": {
        "label": "xAI (Grok)",
        "protocol": "openai",
        "base_url": "https://api.x.ai/v1",
        "console": "https://console.x.ai",
        "prefixes": ("xai-",),
        "images": True,
        "prefer": (r"^grok-\d+(\.\d+)?$", r"^grok-\d", r"^grok"),
    },
    "deepseek": {
        "label": "DeepSeek",
        "protocol": "openai",
        "base_url": "https://api.deepseek.com",
        "console": "https://platform.deepseek.com/api_keys",
        "prefixes": (),
        "engine": "deepseek",
        "images": True,
        "prefer": (),
    },
    "openrouter": {
        "label": "OpenRouter (any model)",
        "protocol": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "console": "https://openrouter.ai/settings/keys",
        "prefixes": ("sk-or-",),
        "images": True,
        "prefer": (
            r"^deepseek/deepseek-v4",
            r"^anthropic/claude-sonnet",
            r"^openai/gpt-5",
            r"^google/gemini-[\d.]+-pro",
            r"^qwen/qwen3-coder",
        ),
    },
    "mistral": {
        "label": "Mistral",
        "protocol": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "console": "https://console.mistral.ai/api-keys",
        "prefixes": (),
        "images": False,
        "prefer": (r"^mistral-large-latest$", r"^codestral-latest$", r"^mistral-medium-latest$", r"^mistral-large"),
    },
    "groq": {
        "label": "Groq",
        "protocol": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "console": "https://console.groq.com/keys",
        "prefixes": ("gsk_",),
        "images": False,
        "prefer": (r"gpt-oss-120b", r"llama-4", r"llama-3\.3-70b", r"qwen"),
    },
    "together": {
        "label": "Together AI",
        "protocol": "openai",
        "base_url": "https://api.together.xyz/v1",
        "console": "https://api.together.ai/settings/api-keys",
        "prefixes": ("tgp_",),
        "images": False,
        "prefer": (r"Qwen3-Coder", r"DeepSeek-V", r"Llama-4", r"Llama-3\.3-70B"),
    },
    "fireworks": {
        "label": "Fireworks AI",
        "protocol": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "console": "https://fireworks.ai/account/api-keys",
        "prefixes": ("fw_",),
        "images": False,
        "prefer": (r"deepseek-v", r"qwen3-coder", r"llama4"),
    },
}


_NOT_CHAT = re.compile(
    r"embed|whisper|tts|audio|dall-e|image|imagen|moderation|realtime|transcri|"
    r"rerank|guard|davinci|babbage|search|computer-use|sora|aqa|veo|lyria",
    re.IGNORECASE,
)


class ProviderKeyError(Exception):
    """A user-facing problem with a key. The message never contains the key."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def redact(text: str, secret: str | None) -> str:
    """Strip a key (and anything shaped like one) from text that may be shown."""
    out = str(text or "")
    if secret:
        out = out.replace(secret, "[redacted]")
    out = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-[redacted]", out)
    return re.sub(r"(xai-|gsk_|fw_|tgp_|AIza)[A-Za-z0-9_\-]{8,}", r"\1[redacted]", out)


def _clean_key(api_key: str) -> str:
    key = str(api_key or "").strip()
    if not re.fullmatch(r"[\x21-\x7e]{16,400}", key):
        raise ProviderKeyError("That does not look like an API key. Copy it again from your provider.")
    return key


def _candidates(api_key: str) -> list[str]:
    """Providers a pasted key could belong to, most likely first."""
    for pid, spec in PROVIDERS.items():
        if any(api_key.startswith(p) for p in spec["prefixes"]):
            return [pid]
    if api_key.startswith("sk-"):


        if re.fullmatch(r"sk-[0-9a-f]{32}", api_key):
            return ["deepseek", "openai"]
        return ["openai", "deepseek"]

    return ["mistral", "together"]


def _auth_headers(provider: str, api_key: str) -> dict[str, str]:
    if PROVIDERS[provider]["protocol"] == "anthropic":
        return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {api_key}"}


def _natural_key(value: str) -> list:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", value.lower())]


def fetch_models(provider: str, api_key: str) -> list[str]:
    """The chat models this key can use, from the provider's model list."""
    spec = PROVIDERS[provider]
    url = spec.get("models_url") or f"{spec['base_url'].rstrip('/')}/models"
    try:
        response = httpx.get(url, headers=_auth_headers(provider, api_key), timeout=12.0)
    except Exception:
        raise ProviderKeyError(
            f"Could not reach {spec['label']} to check the key. Try again in a minute.",
            status=502,
        )
    if response.status_code in (400, 401, 403):
        raise ProviderKeyError(f"{spec['label']} rejected this key. Check it is active.")
    if response.status_code >= 400:
        raise ProviderKeyError(
            f"{spec['label']} returned an error ({response.status_code}) while checking the key.",
            status=502,
        )
    try:
        data = response.json()
    except Exception:
        data = {}
    items = data.get("data") if isinstance(data, dict) else data
    ids: list[str] = []
    for item in items or []:
        mid = item.get("id") if isinstance(item, dict) else item
        if not isinstance(mid, str):
            continue
        mid = mid.removeprefix("models/")
        if mid and not _NOT_CHAT.search(mid):
            ids.append(mid)
    return sorted(set(ids), key=_natural_key, reverse=True)[:500]


def default_model(provider: str, models: list[str]) -> str | None:
    """A strong general model the key can reach, newest first within a family."""
    if PROVIDERS[provider].get("engine"):
        return None
    ranked = rank_models(provider, models)
    return ranked[0] if ranked else None


def rank_models(provider: str, models: list[str]) -> list[str]:
    """Every model, best first: the provider's preferred families in order, then the rest as listed (newest first)."""
    ranked: list[str] = []
    for pattern in PROVIDERS[provider]["prefer"]:
        rx = re.compile(pattern, re.IGNORECASE)
        ranked.extend(m for m in models if rx.search(m) and m not in ranked)
    ranked.extend(m for m in models if m not in ranked)

    return [m for m in ranked if ":free" not in m] + [m for m in ranked if ":free" in m]


_MODEL_CACHE: dict[tuple[str, str], tuple[float, list[str]]] = {}
_MODEL_CACHE_SECONDS = 600.0


def fallback_models(provider: str, api_key: str, *, exclude: set[str] | None = None) -> list[str]:
    """Other models this key can use, best first - for when its model fails."""
    if provider not in PROVIDERS or PROVIDERS[provider].get("engine"):
        return []
    import hashlib

    cache_key = (provider, hashlib.sha256(api_key.encode()).hexdigest()[:24])
    hit = _MODEL_CACHE.get(cache_key)
    if hit and time.time() - hit[0] < _MODEL_CACHE_SECONDS:
        models = hit[1]
    else:
        models = fetch_models(provider, api_key)
        _MODEL_CACHE[cache_key] = (time.time(), models)
    skip = exclude or set()
    return [m for m in rank_models(provider, models) if m not in skip]


def check_with_provider(provider: str, api_key: str) -> list[str]:
    """Validate a key; returns the chat models it can use."""
    return fetch_models(provider, api_key)


def identify(api_key: str, provider: str | None = None) -> tuple[str, list[str]]:
    """Which provider a key belongs to, proved by a successful model listing."""
    key = _clean_key(api_key)
    pid = str(provider or "").strip().lower()
    if pid and pid != "auto":
        if pid not in PROVIDERS:
            raise ProviderKeyError("Unsupported provider.")
        return pid, check_with_provider(pid, key)
    unreachable: ProviderKeyError | None = None
    for candidate in _candidates(key):
        try:
            return candidate, check_with_provider(candidate, key)
        except ProviderKeyError as exc:
            if exc.status == 502:
                unreachable = exc
    if unreachable is not None:
        raise unreachable
    raise ProviderKeyError(
        "This key was not accepted by any provider we tried. Pick its provider "
        "from the list and save again."
    )

def selection_for(provider: str, api_key: str, model: str | None):
    """The model a generation runs on for a saved key, or None for a provider that runs as TextoCAD's own engine."""
    from textocad.services.llm_models import ResolvedLlm

    spec = PROVIDERS.get(provider)
    if not spec or spec.get("engine") or not model:
        return None
    return ResolvedLlm(
        provider=f"own-{provider}",
        label=f"Own · {spec['label']}",
        base_url=spec["base_url"],
        api_key=api_key,
        model=model,
        protocol=spec["protocol"],
        supports_images=bool(spec.get("images")),
    )
