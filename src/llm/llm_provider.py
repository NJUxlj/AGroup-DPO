"""LLMProvider - OpenAI-compatible LLM 调用入口。

为 PairBuilder（LLM-as-Judge）等模块提供统一的 LLM 调用封装，
同时支持第三方 API（OpenAI / 通义千问等）和本地部署模型（vLLM / xinference）。
"""

from __future__ import annotations

import json
from utils.logger import CustomLogger
import re
from typing import Any

from openai import OpenAI

log = CustomLogger.get_logger(__name__)


class LLMProvider:
    """OpenAI-compatible LLM 调用入口。

    统一封装 OpenAI-compatible API 调用，支持：
    - 第三方 API（需要 api_key，如 OpenAI / 通义千问 / DeepSeek）
    - 本地部署模型（api_key 为空，如 vLLM / xinference）
    - 自动 JSON 提取（处理 <thinking> 标签 / markdown code fences）
    - 内置指数退避重试（由 openai SDK max_retries 参数控制）

    使用方式：
        provider = LLMProvider(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-xxx",
        )
        text = provider.chat([{"role": "user", "content": "Hello"}])

    或本地模型：
        provider = LLMProvider(
            model="qwen2.5-7b-instruct",
            base_url="http://127.0.0.1:8001/v1",
        )
        text = provider.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Args:
            model: 模型名。
            base_url: LLM API 地址（含 /v1 后缀），如 https://api.openai.com/v1。
            api_key: API 密钥。本地模型留空即可。
            timeout: 请求超时秒数。
            max_retries: 最大重试次数（仅对 429/5xx 等可重试状态码生效）。
        """
        self._model = model
        self._base_url = base_url
        self._client = OpenAI(
            # 本地模型可能不需要 key，传空字符串；openai SDK 会发 Authorization: Bearer
            # 本地 vLLM/xinference 服务忽略此头
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    # ---------- 公开接口 ----------

    @property
    def model(self) -> str:
        """当前使用的模型名。"""
        return self._model

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """发送 chat 请求，返回响应文本。

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            temperature: 采样温度，0 表示确定性输出。
            max_tokens: 最大输出 token 数。
            response_format: 响应格式约束，如 {"type": "json_object"}。

        Returns:
            LLM 响应的文本内容。

        Raises:
            openai.APIError: API 调用失败（已包含内置重试后的最终失败）。
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        log.debug(
            "LLMProvider chat: model=%s, messages=%d, temperature=%.2f, max_tokens=%d",
            self._model,
            len(messages),
            temperature,
            max_tokens,
        )

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    @staticmethod
    def extract_json(text: str) -> str:
        """从 LLM 响应中提取 JSON（过滤 thinking tags / code fences / 其他杂质）。

        某些模型（如 MiniMax）会在返回内容前包含 <thinking>...</thinking> 标签，
        此方法会移除这些标签并提取纯 JSON 文本。

        Args:
            text: LLM 返回的原始文本。

        Returns:
            提取后的 JSON 字符串。
        """
        # 移除 <thinking>...</thinking> 标签和内容
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = text.strip()
        # 移除 markdown code fences
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```", "", text)
        text = text.strip()
        # 找到第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
        return text
