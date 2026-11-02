"""OpenAI-protocol stand-in for the Anthropic adapter."""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from textocad.core.config import settings


class _Response:
    """Matches the Anthropic adapter's response object."""

    def __init__(self, text: str) -> None:
        self.text = text


class _Models:
    def __init__(
        self,
        client: OpenAI,
        *,
        model: str | None = None,
        vision_model: str | None = None,
        supports_images: bool | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._vision_model = vision_model
        self._supports_images = supports_images

    def generate_content(
        self,
        *,
        model: str | None = None,
        contents: Any,
        config: Any = None,
        image: dict[str, Any] | None = None,
    ) -> _Response:
        system_instruction = getattr(config, "system_instruction", None) if config else None
        temperature = getattr(config, "temperature", None) if config else None
        wants_json = (
            getattr(config, "response_mime_type", "") == "application/json"
            if config
            else False
        )

        has_image = bool(image and isinstance(image, dict) and image.get("data"))
        model_for_call = self._model
        vision_ok = bool(self._supports_images)
        if has_image and self._vision_model:
            model_for_call = self._vision_model
            vision_ok = True

        if has_image and vision_ok:


            mime = str(image.get("mime_type") or "image/png")
            user_content: Any = [
                {"type": "text", "text": str(contents)},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image['data']}"},
                },
            ]
        else:
            user_content = str(contents)

        messages: list[dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": str(system_instruction)})
        messages.append({"role": "user", "content": user_content})

        kwargs: dict[str, Any] = {
            "model": model_for_call,
            "messages": messages,
            "max_tokens": settings.anthropic_max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if wants_json:


            kwargs["response_format"] = {"type": "json_object"}


        try:
            kwargs["timeout"] = float(max(int(settings.generation_budget_seconds), 60))
        except Exception:
            pass

        from textocad.services.provider_errors import call_with_fallback

        started = time.perf_counter()
        try:


            completion = call_with_fallback(
                lambda m: _create_tolerant(self._client, {**kwargs, "model": m}),
                model_for_call,
                base_url=str(getattr(self._client, "base_url", "") or ""),
                api_key=str(getattr(self._client, "api_key", "") or ""),
            )
        finally:
            try:
                from textocad.services.llm_models import add_llm_seconds

                add_llm_seconds(time.perf_counter() - started)
            except Exception:
                pass

        text = ""
        try:
            text = completion.choices[0].message.content or ""
        except Exception:
            text = ""
        return _Response(text)


def _create_tolerant(client: OpenAI, kwargs: dict[str, Any]) -> Any:
    """Create a completion, adapting to what this provider and model accept."""
    from openai import BadRequestError

    kwargs = dict(kwargs)
    adjusted: set[str] = set()
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            message = str(exc).lower()
            token_key = "max_tokens" if "max_tokens" in kwargs else "max_completion_tokens"
            too_many = any(w in message for w in ("less than", "<=", "too large", "exceed", "maximum"))
            if ("max_tokens" in message or "max_completion_tokens" in message) and too_many and (
                int(kwargs.get(token_key) or 0) > 4096
            ):

                kwargs[token_key] = 8192 if int(kwargs[token_key]) > 8192 else 4096
            elif "max_tokens" in message and "max_tokens" in kwargs and "max_tokens" not in adjusted:
                adjusted.add("max_tokens")
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            elif "temperature" in message and "temperature" in kwargs:
                adjusted.add("temperature")
                kwargs.pop("temperature")
            elif "response_format" in message and "response_format" in kwargs:
                adjusted.add("response_format")
                kwargs.pop("response_format")
            else:
                raise


class OpenAICompatClient:
    """The shared client surface, over an OpenAI-protocol endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
        supports_images: bool | None = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.models = _Models(
            self._client,
            model=model,
            vision_model=vision_model,
            supports_images=supports_images,
        )
