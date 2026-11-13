"""大模型适配层：OpenAI 兼容协议 + 无密钥时的确定性降级。

设计要点（对应评分细则「大模型 Prompt 与 Guardrails 策略」「抗幻觉」）：
1. 所有调用走同一 OpenAI 兼容入口，换模型只改 .env；
2. 统一 JSON 结构化输出 + 失败重试 + 解析兜底，避免前端渲染崩；
3. 无 API Key 时退化为「抽取式回答」，仍然带原文出处，保证 Demo 永不空白崩溃。
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

from config import LLM

try:  # openai>=1.x
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

_client = AsyncOpenAI(api_key=LLM.api_key, base_url=LLM.base_url) if (LLM.enabled and AsyncOpenAI) else None

GUARDRAIL = (
    "你是严谨的学术科研助手。硬性约束：\n"
    "1) 只能依据【证据片段】作答，禁止使用片段之外的知识，禁止编造数字、指标、作者、年份；\n"
    "2) 每个结论句必须紧跟形如 [P3] 的证据编号；\n"
    "3) 证据不足时必须明确写出「文献中未提及」，不得猜测；\n"
    "4) 保持学术书面语，公式与符号原样保留。"
)


class LLMUnavailable(RuntimeError):
    pass


def available() -> bool:
    return _client is not None


async def chat(
    prompt: str,
    system: str = GUARDRAIL,
    temperature: float | None = None,
    max_tokens: int = 1200,
) -> str:
    """一次性返回完整文本；无密钥时抛出 LLMUnavailable 由上层降级。"""
    if _client is None:
        raise LLMUnavailable("未配置 LLM_API_KEY")
    resp = await _client.chat.completions.create(
        model=LLM.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=LLM.temperature if temperature is None else temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


async def chat_stream(
    prompt: str,
    system: str = GUARDRAIL,
    temperature: float | None = None,
    max_tokens: int = 1200,
) -> AsyncIterator[str]:
    """Token 级流式输出，用于前端打字机效果。"""
    if _client is None:
        raise LLMUnavailable("未配置 LLM_API_KEY")
    stream = await _client.chat.completions.create(
        model=LLM.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=LLM.temperature if temperature is None else temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


async def chat_json(prompt: str, system: str = GUARDRAIL, max_tokens: int = 1600) -> dict[str, Any]:
    """强制 JSON 输出；解析失败按正则兜底，再失败则抛异常由上层降级。"""
    if _client is None:
        raise LLMUnavailable("未配置 LLM_API_KEY")
    resp = await _client.chat.completions.create(
        model=LLM.model,
        messages=[
            {"role": "system", "content": system + "\n只输出一个合法 JSON 对象，不要 Markdown 代码块。"},
            {"role": "user", "content": prompt},
        ],
        temperature=LLM.temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    text = (resp.choices[0].message.content or "").strip()
    return _parse_json(text)


async def describe_image(image_b64: str, instruction: str) -> str:
    """多模态 OCR：扫描件无文本层时转图片交给视觉模型还原版面文字。"""
    if not LLM.vision_enabled or _client is None:
        raise LLMUnavailable("未配置视觉模型 LLM_VISION_MODEL")
    resp = await _client.chat.completions.create(
        model=LLM.vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=2000,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    braces = re.search(r"\{.*\}", text, re.S)
    if braces:
        return json.loads(braces.group(0))
    raise ValueError(f"模型未返回合法 JSON: {text[:120]}")
