"""全局配置：所有密钥与模型参数统一从环境变量读取，禁止硬编码。

对应赛题要求「API 密钥等敏感信息不得出现在公开仓库，须通过环境变量方式配置」。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
# SCITRACE_DATA_DIR 用于把库、上传件、索引整体指向副本目录，验收时不污染正式库。
DATA_DIR = Path(os.getenv("SCITRACE_DATA_DIR") or (PROJECT_DIR / "data")).expanduser()
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# arXiv 导入：下载超时、下载地址与体积上限均可通过环境变量覆盖（验收脚本用本地桩服务注入故障）。
ARXIV_TIMEOUT_S = float(_env("ARXIV_TIMEOUT_S", "30"))
ARXIV_PDF_BASE_URL = _env("ARXIV_PDF_BASE_URL", "https://arxiv.org/pdf").rstrip("/")
ARXIV_MAX_BYTES = int(_env("ARXIV_MAX_BYTES", str(80 * 1024 * 1024)))


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI 兼容协议的大模型配置，可切换 DeepSeek / 硅基流动 / OpenAI 等。"""

    api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY") or _env("LLM_API_KEY"))
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-chat"))
    vision_model: str = field(default_factory=lambda: _env("LLM_VISION_MODEL", ""))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", ""))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.2")))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def vision_enabled(self) -> bool:
        return bool(self.api_key and self.vision_model)


@dataclass(frozen=True)
class RetrievalConfig:
    """混合检索参数：BM25(稀疏) + 向量(稠密) 用 RRF 融合。"""

    chunk_size: int = int(_env("CHUNK_SIZE", "820"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "160"))
    top_k: int = int(_env("TOP_K", "8"))
    rrf_k: int = int(_env("RRF_K", "60"))
    dedup_threshold: float = float(_env("DEDUP_THRESHOLD", "0.63"))


LLM = LLMConfig()
RETRIEVAL = RetrievalConfig()
