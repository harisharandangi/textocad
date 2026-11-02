from typing import Annotated, Any, ClassVar

from pydantic import AliasChoices, BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def EnvFloat(default: float) -> Any:
    """A float setting that survives being forwarded as an empty string."""

    def _coerce(value: Any) -> Any:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return value

    return Annotated[float, BeforeValidator(_coerce)]


class Settings(BaseSettings):


    llm_provider: str = Field(
        "anthropic",
        validation_alias=AliasChoices("LLM_PROVIDER", "TEXTOCAD_LLM_PROVIDER"),
    )
    anthropic_api_key: str | None = Field(
        None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "TEXTOCAD_ANTHROPIC_API_KEY"),
    )
    anthropic_model: str = Field(
        "claude-sonnet-4-6",
        validation_alias=AliasChoices("ANTHROPIC_MODEL", "TEXTOCAD_ANTHROPIC_MODEL"),
    )
    anthropic_max_tokens: int = Field(
        16000,
        validation_alias=AliasChoices("ANTHROPIC_MAX_TOKENS", "TEXTOCAD_ANTHROPIC_MAX_TOKENS"),
    )


    anthropic_base_url: str | None = Field(
        None,
        validation_alias=AliasChoices("ANTHROPIC_BASE_URL", "TEXTOCAD_ANTHROPIC_BASE_URL"),
    )


    anthropic_supports_images: bool = Field(
        True,
        validation_alias=AliasChoices(
            "ANTHROPIC_SUPPORTS_IMAGES", "TEXTOCAD_ANTHROPIC_SUPPORTS_IMAGES"
        ),
    )


    llm_default_provider: str = Field(
        "deepseek",
        validation_alias=AliasChoices("LLM_DEFAULT_PROVIDER", "TEXTOCAD_LLM_DEFAULT_PROVIDER"),
    )
    llm_default_speed: str = Field(
        "standard",
        validation_alias=AliasChoices("LLM_DEFAULT_SPEED", "TEXTOCAD_LLM_DEFAULT_SPEED"),
    )


    openrouter_api_key: str | None = Field(
        None, validation_alias=AliasChoices("OPENROUTER_API_KEY"),
    )


    openrouter_base_url: str = Field(
        "https://openrouter.ai/api",
        validation_alias=AliasChoices("OPENROUTER_BASE_URL"),
    )


    llm_image_providers: str = Field(
        "",
        validation_alias=AliasChoices("LLM_IMAGE_PROVIDERS"),
    )

    deepseek_api_key: str | None = Field(
        None, validation_alias=AliasChoices("DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = Field(
        "https://api.deepseek.com/anthropic",
        validation_alias=AliasChoices("DEEPSEEK_BASE_URL"),
    )


    deepseek_model_fast: str = Field(
        "deepseek-v4-flash", validation_alias=AliasChoices("DEEPSEEK_MODEL_FAST"),
    )
    deepseek_model_standard: str = Field(
        "deepseek-v4-flash", validation_alias=AliasChoices("DEEPSEEK_MODEL_STANDARD"),
    )
    deepseek_model_deepthink: str = Field(
        "deepseek-v4-flash", validation_alias=AliasChoices("DEEPSEEK_MODEL_DEEPTHINK"),
    )


    deepseek_model_vision: str = Field(
        "deepseek-v4-flash-vision-exp",
        validation_alias=AliasChoices("DEEPSEEK_MODEL_VISION"),
    )


    glm_api_key: str | None = Field(
        None, validation_alias=AliasChoices("GLM_API_KEY", "ZAI_API_KEY"),
    )
    glm_base_url: str | None = Field(
        None,
        validation_alias=AliasChoices("GLM_BASE_URL"),
    )
    glm_model_fast: str = Field(
        "z-ai/glm-4.7", validation_alias=AliasChoices("GLM_MODEL_FAST"),
    )
    glm_model_standard: str = Field(
        "z-ai/glm-4.7", validation_alias=AliasChoices("GLM_MODEL_STANDARD"),
    )
    glm_model_deepthink: str = Field(
        "z-ai/glm-5.2", validation_alias=AliasChoices("GLM_MODEL_DEEPTHINK"),
    )

    kimi_api_key: str | None = Field(
        None, validation_alias=AliasChoices("KIMI_API_KEY", "MOONSHOT_API_KEY"),
    )
    kimi_base_url: str | None = Field(
        None,
        validation_alias=AliasChoices("KIMI_BASE_URL"),
    )
    kimi_model_fast: str = Field(
        "moonshotai/kimi-k2.5", validation_alias=AliasChoices("KIMI_MODEL_FAST"),
    )
    kimi_model_standard: str = Field(
        "moonshotai/kimi-k2.5", validation_alias=AliasChoices("KIMI_MODEL_STANDARD"),
    )
    kimi_model_deepthink: str = Field(
        "moonshotai/kimi-k2.7-code", validation_alias=AliasChoices("KIMI_MODEL_DEEPTHINK"),
    )

    minimax_api_key: str | None = Field(
        None, validation_alias=AliasChoices("MINIMAX_API_KEY"),
    )
    minimax_base_url: str | None = Field(
        None,
        validation_alias=AliasChoices("MINIMAX_BASE_URL"),
    )
    minimax_model_fast: str = Field(
        "minimax/minimax-m3", validation_alias=AliasChoices("MINIMAX_MODEL_FAST"),
    )
    minimax_model_standard: str = Field(
        "minimax/minimax-m3", validation_alias=AliasChoices("MINIMAX_MODEL_STANDARD"),
    )
    minimax_model_deepthink: str = Field(
        "minimax/minimax-m3", validation_alias=AliasChoices("MINIMAX_MODEL_DEEPTHINK"),
    )


    verification_gate_enabled: bool = Field(
        False,
        validation_alias=AliasChoices(
            "TEXTOCAD_VERIFICATION_GATE",
            "VERIFICATION_GATE_ENABLED",
        ),
    )
    verification_max_attempts: int = Field(
        3,
        validation_alias=AliasChoices("TEXTOCAD_VERIFICATION_MAX_ATTEMPTS"),
    )


    generation_budget_seconds: int = Field(
        420,
        validation_alias=AliasChoices(
            "TEXTOCAD_GENERATION_BUDGET_SECONDS", "GENERATION_BUDGET_SECONDS"
        ),
    )


    router_timeout_seconds: EnvFloat(6.0) = Field(
        6.0, validation_alias=AliasChoices("ROUTER_TIMEOUT_SECONDS"),
    )


    helper_model_classify: str = Field(
        "deepseek:fast", validation_alias=AliasChoices("HELPER_MODEL_CLASSIFY"),
    )


    helper_model_plan: str = Field(
        "deepseek:fast", validation_alias=AliasChoices("HELPER_MODEL_PLAN"),
    )


    helper_model_extract: str = Field(
        "deepseek:fast", validation_alias=AliasChoices("HELPER_MODEL_EXTRACT"),
    )


    helper_model_vision: str = Field(
        "deepseek:vision",
        validation_alias=AliasChoices("HELPER_MODEL_VISION"),
    )


    escalation_model: str = Field(
        "deepseek:deepseek-v4-pro",
        validation_alias=AliasChoices("ESCALATION_MODEL"),
    )


    route_target_fast: str = Field(
        "deepseek:fast", validation_alias=AliasChoices("ROUTE_TARGET_FAST"),
    )
    route_target_standard: str = Field(
        "deepseek:standard", validation_alias=AliasChoices("ROUTE_TARGET_STANDARD"),
    )
    route_target_deepthink: str = Field(
        "deepseek:deepthink", validation_alias=AliasChoices("ROUTE_TARGET_DEEPTHINK"),
    )

    max_prompt_chars: int = Field(
        3_000,
        validation_alias=AliasChoices("TEXTOCAD_MAX_PROMPT_CHARS", "MAX_PROMPT_CHARS"),
    )
    max_operations: int = Field(
        200,
        validation_alias=AliasChoices("TEXTOCAD_MAX_OPERATIONS", "MAX_OPERATIONS"),
    )
    max_params: int = Field(
        100,
        validation_alias=AliasChoices("TEXTOCAD_MAX_PARAMS", "MAX_PARAMS"),
    )
    max_build_seconds: int = Field(
        120,
        validation_alias=AliasChoices("TEXTOCAD_MAX_BUILD_SECONDS", "MAX_BUILD_SECONDS"),
    )
    build_retention_hours: int = Field(
        24,
        validation_alias=AliasChoices(
            "TEXTOCAD_BUILD_RETENTION_HOURS",
            "BUILD_RETENTION_HOURS",
        ),
    )

    allow_openscad_exec: bool = Field(
        False,
        validation_alias=AliasChoices(
            "TEXTOCAD_ALLOW_OPENSCAD_EXEC",
            "ALLOW_OPENSCAD_EXEC",
        ),
    )
    allow_cadquery_exec: bool = Field(
        False,
        validation_alias=AliasChoices(
            "TEXTOCAD_ALLOW_CADQUERY_EXEC",
            "ALLOW_CADQUERY_EXEC",
        ),
    )


    sandbox_mem_mb: int = Field(
        6144,
        validation_alias=AliasChoices(
            "TEXTOCAD_SANDBOX_MEM_MB",
            "SANDBOX_MEM_MB",
        ),
    )
    sandbox_uid: int = Field(
        0,
        validation_alias=AliasChoices("TEXTOCAD_SANDBOX_UID", "SANDBOX_UID"),
    )


    router_model: str = Field(
        "off", validation_alias=AliasChoices("ROUTER_MODEL"),
    )


    textocad_provider: str | None = Field(
        None, validation_alias=AliasChoices("TEXTOCAD_PROVIDER"),
    )
    textocad_api_key: str | None = Field(
        None, validation_alias=AliasChoices("TEXTOCAD_API_KEY"),
    )
    textocad_model: str | None = Field(
        None, validation_alias=AliasChoices("TEXTOCAD_MODEL"),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()


def setting_or_default(name: str) -> str:
    """A string setting's value, with EMPTY-STRING protection."""
    value = getattr(settings, name, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    default = type(settings).model_fields[name].default
    return str(default) if default is not None else ""
