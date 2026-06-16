"""测试 LLMProvider (llm/llm_provider.py)"""

import json
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from llm.llm_provider import LLMProvider


# ------------------------------------------------------------------
# Mock helpers
# ------------------------------------------------------------------


def _make_mock_completion(content: str) -> ChatCompletion:
    """构造一个模拟的 OpenAI ChatCompletion 对象。"""
    return ChatCompletion(
        id="chatcmpl-mock",
        created=0,
        model="test-model",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=content,
                    role="assistant",
                ),
            )
        ],
        usage=CompletionUsage(
            completion_tokens=10,
            prompt_tokens=5,
            total_tokens=15,
        ),
    )


# ------------------------------------------------------------------
# extract_json
# ------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self):
        text = '{"winner": "A", "score_a": 0.9, "score_b": 0.3}'
        assert LLMProvider.extract_json(text) == text

    def test_json_with_thinking_tags(self):
        text = '<thinking>我需要分析两个答案</thinking>\n{"winner": "A", "score_a": 0.9, "score_b": 0.3}'
        result = LLMProvider.extract_json(text)
        assert "thinking" not in result
        assert "winner" in result

    def test_json_with_code_fence(self):
        text = '```json\n{"winner": "B", "score_a": 0.2, "score_b": 0.8}\n```'
        result = LLMProvider.extract_json(text)
        assert "```" not in result
        parsed = json.loads(result)
        assert parsed["winner"] == "B"

    def test_json_with_extra_text(self):
        text = '这是分析结果：{"winner": "A", "score_a": 0.95, "score_b": 0.1}，请确认。'
        result = LLMProvider.extract_json(text)
        assert result.startswith("{")
        assert result.endswith("}")
        parsed = json.loads(result)
        assert parsed["winner"] == "A"

    def test_no_json_returns_stripped(self):
        text = "纯文本，没有 JSON"
        result = LLMProvider.extract_json(text)
        assert result == text.strip()

    def test_multiline_json(self):
        text = """
        {
            "winner": "A",
            "score_a": 0.85,
            "score_b": 0.15,
            "reason": "答案A更合规"
        }
        """
        result = LLMProvider.extract_json(text)
        parsed = json.loads(result)
        assert parsed["reason"] == "答案A更合规"


# ------------------------------------------------------------------
# chat
# ------------------------------------------------------------------


class TestChat:
    def test_chat_returns_text(self):
        provider = LLMProvider(
            model="test-model",
            base_url="http://127.0.0.1:8000/v1",
            api_key="",
        )
        mock_completion = _make_mock_completion("你好，这是回复。")

        with patch.object(provider._client.chat.completions, "create", return_value=mock_completion):
            result = provider.chat([{"role": "user", "content": "你好"}])
            assert result == "你好，这是回复。"

    def test_chat_passes_temperature_and_max_tokens(self):
        provider = LLMProvider(
            model="test-model",
            base_url="http://127.0.0.1:8000/v1",
        )
        mock_completion = _make_mock_completion("ok")

        with patch.object(provider._client.chat.completions, "create", return_value=mock_completion) as mock_create:
            provider.chat(
                [{"role": "user", "content": "test"}],
                temperature=0.3,
                max_tokens=512,
            )
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.3
            assert call_kwargs["max_tokens"] == 512

    def test_chat_passes_response_format(self):
        provider = LLMProvider(
            model="test-model",
            base_url="http://127.0.0.1:8000/v1",
        )
        mock_completion = _make_mock_completion('{"key": "value"}')

        with patch.object(provider._client.chat.completions, "create", return_value=mock_completion) as mock_create:
            provider.chat(
                [{"role": "user", "content": "test"}],
                response_format={"type": "json_object"},
            )
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_chat_uses_model_from_init(self):
        provider = LLMProvider(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
        mock_completion = _make_mock_completion("ok")

        with patch.object(provider._client.chat.completions, "create", return_value=mock_completion) as mock_create:
            provider.chat([{"role": "user", "content": "hi"}])
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o"

    def test_chat_empty_content_returns_empty_string(self):
        provider = LLMProvider(
            model="test-model",
            base_url="http://127.0.0.1:8000/v1",
        )
        mock_completion = _make_mock_completion(None)  # content is None

        with patch.object(provider._client.chat.completions, "create", return_value=mock_completion):
            result = provider.chat([{"role": "user", "content": "hi"}])
            assert result == ""

    def test_chat_api_error_propagates(self):
        provider = LLMProvider(
            model="test-model",
            base_url="http://127.0.0.1:8000/v1",
            api_key="sk-bad",
            max_retries=0,  # 不重试
        )
        from openai import AuthenticationError

        with patch.object(
            provider._client.chat.completions, "create",
            side_effect=AuthenticationError("Invalid API key", response=MagicMock(), body=None),
        ):
            with pytest.raises(AuthenticationError):
                provider.chat([{"role": "user", "content": "hi"}])


# ------------------------------------------------------------------
# model property
# ------------------------------------------------------------------


class TestModelProperty:
    def test_model_returns_init_value(self):
        provider = LLMProvider(
            model="qwen2.5-7b-instruct",
            base_url="http://127.0.0.1:8001/v1",
        )
        assert provider.model == "qwen2.5-7b-instruct"


# ------------------------------------------------------------------
# 本地 vs 第三方 API
# ------------------------------------------------------------------


class TestLocalVsThirdParty:
    def test_local_model_no_api_key(self):
        """本地模型：api_key 为空，应正常工作。"""
        provider = LLMProvider(
            model="qwen2.5-7b-instruct",
            base_url="http://127.0.0.1:8001/v1",
            api_key="",
        )
        assert provider.model == "qwen2.5-7b-instruct"
        # 验证 client 被正确创建
        assert provider._client is not None

    def test_third_party_with_api_key(self):
        """第三方 API：带 api_key。"""
        provider = LLMProvider(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-test-key-123",
        )
        assert provider.model == "gpt-4o"
        assert provider._client is not None

    def test_dashscope_compatible_mode(self):
        """通义千问 DashScope compatible-mode。"""
        provider = LLMProvider(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-dashscope-key",
        )
        mock_completion = _make_mock_completion("通义千问回复")

        with patch.object(provider._client.chat.completions, "create", return_value=mock_completion):
            result = provider.chat([{"role": "user", "content": "你好"}])
            assert result == "通义千问回复"
