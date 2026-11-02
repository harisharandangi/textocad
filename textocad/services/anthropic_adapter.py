"""Anthropic-protocol adapter behind TextoCAD's single LLM call shape."""

from __future__ import annotations

from typing import Any

import anthropic

from textocad.core.config import settings


def _vision_supported() -> bool:
    """Whether the statically-configured endpoint accepts image content blocks."""
    if not settings.anthropic_supports_images:
        return False
    base_url = (settings.anthropic_base_url or "").lower()
    if "deepseek" in base_url:
        return False
    return True


_CALL_DEADLINE_FRACTION = 0.6
_CALL_DEADLINE_FLOOR_SECONDS = 60.0


class _Response:
    """The response object every call site reads (only ``.text`` is used)."""

    def __init__(self, text: str) -> None:
        self.text = text


class _Models:
    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        model: str | None = None,
        vision_model: str | None = None,
        supports_images: bool | None = None,
        prefill: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._vision_model = vision_model
        self._supports_images = supports_images
        self._prefill = prefill

    def generate_content(
        self,
        *,
        model: str | None = None,
        contents: Any,
        config: Any = None,
        image: dict[str, Any] | None = None,
    ) -> _Response:
        system_instruction = getattr(config, "system_instruction", None)


        vision_ok = (
            self._supports_images
            if self._supports_images is not None
            else _vision_supported()
        )


        has_image = bool(image and isinstance(image, dict) and image.get("data"))
        model_for_call = self._model or settings.anthropic_model
        if has_image and self._vision_model:
            model_for_call = self._vision_model
            vision_ok = True
        if image and isinstance(image, dict) and image.get("data") and vision_ok:
            user_content: Any = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": str(image.get("mime_type") or "image/png"),
                        "data": str(image["data"]),
                    },
                },
                {"type": "text", "text": str(contents)},
            ]
        else:
            user_content = str(contents)

        kwargs: dict[str, Any] = {
            "model": model_for_call,
            "max_tokens": settings.anthropic_max_tokens,
            "messages": [{"role": "user", "content": user_content}],
        }
        if system_instruction:
            kwargs["system"] = str(system_instruction)


        try:
            kwargs["timeout"] = float(max(int(settings.generation_budget_seconds), 60))
        except Exception:
            pass


        wants_json = (
            getattr(config, "response_mime_type", "") == "application/json"
            if config
            else False
        )


        wants_json = wants_json and self._prefill
        if wants_json:
            kwargs["messages"].append({"role": "assistant", "content": "{"})


        import time


        deadline = max(
            float(settings.generation_budget_seconds) * _CALL_DEADLINE_FRACTION,
            _CALL_DEADLINE_FLOOR_SECONDS,
        )

        from textocad.services.provider_errors import call_with_fallback

        _t0 = time.perf_counter()

        def _stream_once(request: dict[str, Any]) -> tuple[list[str], bool]:
            parts: list[str] = []
            truncated = False
            with self._client.messages.stream(**request) as stream:
                for chunk in stream.text_stream:
                    parts.append(chunk)
                    if time.perf_counter() - _t0 > deadline:


                        truncated = True
                        break
                if not truncated:
                    message = stream.get_final_message()
                    parts = [
                        block.text
                        for block in message.content
                        if getattr(block, "type", None) == "text"
                    ]
            return parts, truncated

        def _send(model_id: str) -> tuple[list[str], bool]:
            request = {**kwargs, "model": model_id}
            while True:
                try:
                    return _stream_once(request)
                except anthropic.BadRequestError as exc:


                    tokens = int(request["max_tokens"])
                    if "max_tokens" not in str(exc) or tokens <= 4096:
                        raise
                    request["max_tokens"] = 8192 if tokens > 8192 else 4096

        try:


            parts, truncated = call_with_fallback(
                _send,
                model_for_call,
                base_url=str(getattr(self._client, "base_url", "") or ""),
                api_key=str(getattr(self._client, "api_key", "") or ""),
            )
        finally:


            try:
                from textocad.services.llm_models import add_llm_seconds

                add_llm_seconds(time.perf_counter() - _t0)
            except Exception:
                pass

        if truncated:
            _log_truncated(deadline, len("".join(parts)))

        text = "".join(parts)

        if wants_json and text and not text.lstrip().startswith("{"):
            text = "{" + text
        return _Response(text)


def _log_truncated(deadline: float, chars: int) -> None:
    """Say so when a stream is cut at the deadline."""
    print(
        f"[textocad.llm] stream cut at {deadline:.0f}s wall clock, keeping {chars} chars for salvage",
        flush=True,
    )

class AnthropicClient:
    """Client exposing ``.models.generate_content``."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
        supports_images: bool | None = None,
    ) -> None:
        client_kwargs: dict[str, Any] = {"api_key": api_key}


        resolved_base = base_url or settings.anthropic_base_url
        if resolved_base:
            client_kwargs["base_url"] = resolved_base
        self._client = anthropic.Anthropic(**client_kwargs)
        self.models = _Models(
            self._client,
            model=model,
            vision_model=vision_model,
            supports_images=supports_images,
            prefill="api.anthropic.com" not in str(resolved_base or "").lower(),
        )
