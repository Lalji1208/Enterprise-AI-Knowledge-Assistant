"""Phase 8 - LLM client for any OpenAI-compatible chat API (OpenAI, Ollama, ...).

Only the Chat Completions endpoint is used, because it is the one that
OpenAI-compatible servers implement. Errors are converted into LLMError with a
message that is safe to show to users (it never contains the API key).

Usage:
    from rag.llm import get_llm
    reply = get_llm().complete([{"role": "user", "content": "Hi"}])
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class LLMError(Exception):
    """A user-presentable LLM failure (configuration, network, provider error)."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    truncated: bool = False  # True when the answer hit the max-token limit


class LLMClient(Protocol):
    """Anything with this method can generate answers (real client or a test fake)."""

    model: str

    def complete(self, messages: list[Message]) -> LLMResponse: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout: float = 60.0,
        use_max_completion_tokens: bool = True,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # OpenAI's newer models require `max_completion_tokens`; many compatible
        # servers (e.g. Ollama) only understand the older `max_tokens`.
        self._max_tokens_param = "max_completion_tokens" if use_max_completion_tokens else "max_tokens"
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=2)

    def complete(self, messages: list[Message]) -> LLMResponse:
        import openai

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                **{self._max_tokens_param: self.max_tokens},
            )
        except openai.AuthenticationError as exc:
            raise LLMError("The LLM provider rejected the API key (check OPENAI_API_KEY).") from exc
        except openai.NotFoundError as exc:
            raise LLMError(
                f"Model '{self.model}' was not found at the provider (check LLM_MODEL)."
            ) from exc
        except openai.RateLimitError as exc:
            # HTTP 429 means either "no credit left" (won't fix itself) or a
            # temporary rate limit (retry later); tell the user which.
            if exc.type == "insufficient_quota" or exc.code == "insufficient_quota":
                raise LLMError(
                    "The LLM account has no credits or quota left. Add credits in the "
                    "provider's billing settings (OpenAI: platform.openai.com > Billing)."
                ) from exc
            raise LLMError("The LLM provider's rate limit was hit; try again shortly.") from exc
        except openai.APITimeoutError as exc:
            raise LLMError("The LLM provider did not respond in time.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not connect to the LLM provider (check the base URL).") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"The LLM provider returned an error (HTTP {exc.status_code}).") from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            model=response.model or self.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            truncated=choice.finish_reason == "length",
        )


def build_llm(settings: Settings) -> OpenAICompatibleClient:
    """Create the client for LLM_PROVIDER ("openai" or "ollama")."""
    provider = settings.llm_provider.strip().lower()
    common = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "timeout": settings.llm_timeout_seconds,
    }
    if provider == "openai":
        api_key = settings.openai_api_key.get_secret_value().strip()
        if not api_key:
            raise LLMError("OPENAI_API_KEY is empty. Add your key to the .env file.")
        return OpenAICompatibleClient(base_url=settings.openai_base_url, api_key=api_key, **common)
    if provider == "ollama":
        # Ollama ignores the key, but the SDK requires a non-empty value.
        return OpenAICompatibleClient(base_url=settings.ollama_base_url, api_key="ollama",
                                      use_max_completion_tokens=False, **common)
    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}' (use 'openai' or 'ollama').")


@lru_cache
def get_llm() -> OpenAICompatibleClient:
    return build_llm(get_settings())
