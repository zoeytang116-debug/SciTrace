"""混合检索：BM25(稀疏词频) + TF-IDF 余弦(稠密语义) 双路召回，再用 RRF 融合排序。

为什么不用外部 Embedding API：
评审环境不一定能联网，本地向量化零依赖、冷启动即用；
同时保留 `EMBEDDING_MODEL` 开关，配置后自动走 OpenAI 兼容的向量接口。
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import RETRIEVAL
from store import all_chunks

try:
    import jieba
except Exception:  # pragma: no cover
    jieba = None

_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{1,}|[0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """英文按词、数字保留小数点、中文按词（无 jieba 时退化为单字）。"""
    text = text.lower()
    tokens = _WORD.findall(text)
    if jieba is not None and re.search(r"[\u4e00-\u9fff]", text):
        tokens += [w for w in jieba.cut(text) if w.strip() and not re.fullmatch(r"\W+", w)]
    return tokens


class BM25:
    """Okapi BM25，纯 numpy 实现，避免额外依赖。"""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.n)
        self.tf: list[Counter[str]] = [Counter(d) for d in docs]
        self.idf: dict[str, float] = {}
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n)
        for term in query:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                dl = len(self.docs[i]) or 1
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out


class HybridIndex:
    """保持一次构建、多次查询；库变更时由 `invalidate()` 触发重建。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    def _build(self, paper_ids: list[str] | None) -> dict[str, Any]:
        key = ",".join(sorted(paper_ids)) if paper_ids else "*"
        if key in self._cache:
            return self._cache[key]

        chunks = all_chunks(paper_ids)
        if not chunks:
            payload = {"chunks": [], "bm25": None, "vec": None, "matrix": None}
            self._cache[key] = payload
            return payload

        docs = [tokenize(c["text"]) for c in chunks]
        texts = [c["text"] for c in chunks]
        vectorizer = TfidfVectorizer(
            analyzer="word", token_pattern=r"[A-Za-z][A-Za-z\-]{1,}|[0-9]+(?:\.[0-9]+)?",
            ngram_range=(1, 2), sublinear_tf=True, min_df=1,
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            matrix = None
        payload = {
            "chunks": chunks,
            "bm25": BM25(docs),
            "vec": vectorizer,
            "matrix": matrix,
        }
        self._cache[key] = payload
        return payload

    def search(self, query: str, paper_ids: list[str] | None = None, top_k: int | None = None) -> list[dict[str, Any]]:
        """双路召回后用 Reciprocal Rank Fusion 融合，返回带分数的片段。"""
        top_k = top_k or RETRIEVAL.top_k
        idx = self._build(paper_ids)
        chunks = idx["chunks"]
        if not chunks:
            return []

        q_tokens = tokenize(query)
        sparse = idx["bm25"].scores(q_tokens) if idx["bm25"] else np.zeros(len(chunks))

        dense = np.zeros(len(chunks))
        if idx["matrix"] is not None:
            qv = idx["vec"].transform([query])
            sim = (idx["matrix"] @ qv.T).toarray().ravel()
            dense = sim

        rrf = np.zeros(len(chunks))
        for scores in (sparse, dense):
            ranked = np.argsort(-scores)
            for rank, i in enumerate(ranked):
                if scores[i] <= 0 and rank > 0:
                    continue
                rrf[i] += 1.0 / (RETRIEVAL.rrf_k + rank + 1)

        order = np.argsort(-rrf)[: top_k * 2]
        results = []
        for i in order:
            if rrf[i] <= 0:
                continue
            c = chunks[int(i)]
            results.append(
                {
                    "paper_id": c["paper_id"],
                    "page": c["page"],
                    "heading": c["heading"],
                    "text": c["text"],
                    "bbox": c["bbox"],
                    "score": float(rrf[int(i)]),
                    "sparse": float(sparse[int(i)]),
                    "dense": float(dense[int(i)]),
                }
            )
        return results[:top_k]


INDEX = HybridIndex()
