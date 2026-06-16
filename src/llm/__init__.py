# M02: LLM 统一调用入口
# 为 PairBuilder 等模块提供 OpenAI-compatible 的 LLM 调用能力

from llm.llm_provider import LLMProvider

__all__ = ["LLMProvider"]
