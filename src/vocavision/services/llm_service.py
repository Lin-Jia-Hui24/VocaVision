"""DashScope-backed LLM and VLM wrappers."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from vocavision.config import VocavisionSettings
from vocavision.exceptions import ConfigurationError, ExternalServiceError
from vocavision.utils.json_utils import parse_json_object


class DashScopeLLMService:
    def __init__(self, settings: VocavisionSettings) -> None:
        if not settings.dashscope_api_key:
            raise ConfigurationError("DASHSCOPE_API_KEY is required to call DashScope models.")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            timeout=settings.request_timeout_sec,
        )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=model or self.settings.llm_model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ExternalServiceError("DashScope returned an empty completion response.")
        return parse_json_object(content)
