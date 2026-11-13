"""语义级去重与版本演进比对（模块一核心亮点）。

判据：MinHash 签名估 Jaccard + SimHash 汉明距离双阈值，
只看正文内容，不看文件名 —— 因此 `1706.03762v1.pdf` 与 `v7.pdf` 会被正确识别为同一篇。
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Any

from parsing import fingerprint, jaccard_from_signatures, minhash_signature, simhash_distance
from store import all_chunks, get_paper, list_papers

# 2026-11-13 真实数据夹定：30 对负样本最高 0.4091，正样本 0.8490。
# 阈值取两者算术中点 0.6291，向百分位取 0.63；差值 0.4399。
DEDUP_THRESHOLD = 0.63
SIMHASH_SIGNAL_THRESHOLD = 0.72
MINHASH_SIGNAL_THRESHOLD = 0.45
TITLE_SIGNAL_THRESHOLD = 0.72

ARXIV_HEADER = re.compile(
    r"arXiv:\s*\d{4}\.\d{4,5}v\d+\s*\[[^\]]+\]\s*\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
    re.I,
)
CONFERENCE_FOOTER = re.compile(
    r"(?:Proceedings of [A-Z][^\n]{0,180}\d{4}[^\n]{0,100}|"
    r"\d{1,2}(?:st|nd|rd|th) Conference on [^\n]{0,180}\d{4}[^\n]{0,100}\.)",
    re.I,
)
TRAILING_PAGE_NUMBER = re.compile(r"\s+\d{1,4}\s*$")


def _normal_title(title: str, filename: str = "") -> str:
    value = re.sub(r"\s+", " ", title or "").strip()
    if not value or re.match(r"^arxiv:", value, re.I) or len(value) > 240:
        value = Path(filename).stem if filename else ""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _clean_body_text(chunks: list[dict[str, Any]], outline: list[dict[str, Any]]) -> str:
    reference_pages = [
        int(item["page"])
        for item in outline
        if re.match(r"^references\b", item.get("title", ""), re.I)
    ]
    reference_start = min(reference_pages) if reference_pages else 10**9
    parts: list[str] = []
    for chunk in sorted(chunks, key=lambda item: item.get("ord", 0)):
        if int(chunk.get("page", 0)) >= reference_start:
            continue
        text = ARXIV_HEADER.sub(" ", str(chunk.get("text", "")))
        text = CONFERENCE_FOOTER.sub(" ", text)
        text = TRAILING_PAGE_NUMBER.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def build_document_features(
    title: str,
    chunks: list[dict[str, Any]],
    outline: list[dict[str, Any]],
    filename: str = "",
    fallback_fingerprint: str = "",
    fallback_shingles: list[int] | None = None,
) -> dict[str, Any]:
    """从去除参考文献和页眉页脚后的正文生成三信号特征。"""
    body = _clean_body_text(chunks, outline)
    return {
        "simhash": fingerprint(re.sub(r"\s+", "", body)) if body else fallback_fingerprint,
        "minhash": minhash_signature(body) if body else (fallback_shingles or []),
        "title": _normal_title(title, filename),
    }


def compare_features(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    simhash_similarity = 1.0 - simhash_distance(left["simhash"], right["simhash"])
    minhash_similarity = jaccard_from_signatures(left["minhash"], right["minhash"])
    title_similarity = difflib.SequenceMatcher(
        None, left["title"], right["title"], autojunk=False
    ).ratio()
    votes = sum(
        (
            simhash_similarity >= SIMHASH_SIGNAL_THRESHOLD,
            minhash_similarity >= MINHASH_SIGNAL_THRESHOLD,
            title_similarity >= TITLE_SIGNAL_THRESHOLD,
        )
    )
    score = (simhash_similarity + minhash_similarity + title_similarity) / 3
    return {
        "score": round(score, 4),
        "simhash_similarity": round(simhash_similarity, 4),
        "minhash_similarity": round(minhash_similarity, 4),
        "title_similarity": round(title_similarity, 4),
        "votes": votes,
        "duplicate": votes >= 2 and score >= DEDUP_THRESHOLD,
    }


def _reference_title(entry: dict[str, Any]) -> str:
    raw = unicodedata.normalize("NFKC", str(entry.get("raw", "")))
    parts = [part.strip() for part in re.split(r"\.\s+", raw) if part.strip()]
    venue = re.compile(
        r"^(in\s|proceedings|arxiv|corr|journal|advances|conference|"
        r"empirical methods|technical report)",
        re.I,
    )
    title = ""
    for index, part in enumerate(parts):
        if index and venue.match(part):
            title = parts[index - 1]
            break
    if not title:
        title = parts[-1] if parts else raw[:160]
    title = re.sub(r"(\w)-\s+(\w)", r"\1\2", title)
    return re.sub(r"\s+", " ", title).strip(" .")


def _reference_key(entry: dict[str, Any]) -> str:
    title = unicodedata.normalize("NFKC", _reference_title(entry)).lower()
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def _reference_changes(
    new_refs: list[dict[str, Any]],
    old_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    new_by_key: dict[str, dict[str, Any]] = {}
    old_by_key: dict[str, dict[str, Any]] = {}
    for entry in new_refs:
        key = _reference_key(entry)
        if key:
            new_by_key.setdefault(key, entry)
    for entry in old_refs:
        key = _reference_key(entry)
        if key:
            old_by_key.setdefault(key, entry)
    added_keys = sorted(set(new_by_key) - set(old_by_key))
    removed_keys = sorted(set(old_by_key) - set(new_by_key))
    return {
        "new_count": len(new_refs),
        "old_count": len(old_refs),
        "added_count": len(added_keys),
        "removed_count": len(removed_keys),
        "added": [_reference_title(new_by_key[key]) for key in added_keys],
        "removed": [_reference_title(old_by_key[key]) for key in removed_keys],
    }


def _section_texts(chunks: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for chunk in sorted(chunks, key=lambda item: item.get("ord", 0)):
        heading = re.sub(r"\s+", " ", str(chunk.get("heading", ""))).strip()
        if not heading or re.match(r"^references\b", heading, re.I):
            continue
        grouped.setdefault(heading, []).append(str(chunk.get("text", "")))
    return {
        heading: re.sub(r"\s+", " ", " ".join(parts)).strip()
        for heading, parts in grouped.items()
    }


def _chapter_changes(
    new_paper: dict[str, Any],
    old_paper: dict[str, Any],
    new_chunks: list[dict[str, Any]],
    old_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    def outline_map(paper: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for item in paper.get("outline", []):
            title = re.sub(r"\s+", " ", str(item.get("title", ""))).strip()
            if title and not re.match(r"^references\b", title, re.I):
                result[re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()] = title
        return result

    new_outline = outline_map(new_paper)
    old_outline = outline_map(old_paper)
    added = [new_outline[key] for key in sorted(set(new_outline) - set(old_outline))]
    removed = [old_outline[key] for key in sorted(set(old_outline) - set(new_outline))]
    new_sections = _section_texts(new_chunks)
    old_sections = _section_texts(old_chunks)
    rewritten = []
    for key in sorted(set(new_outline) & set(old_outline)):
        title = new_outline[key]
        new_text = new_sections.get(title, "")
        old_text = old_sections.get(title, "")
        if not new_text or not old_text:
            continue
        ratio = difflib.SequenceMatcher(None, old_text, new_text, autojunk=False).ratio()
        if ratio < 0.90:
            rewritten.append({"title": title, "similarity": round(ratio, 4)})
    return {"added": added, "removed": removed, "rewritten": rewritten[:12]}


def find_duplicates(
    fingerprint_value: str,
    shingles: list[int],
    exclude_id: str = "",
    *,
    title: str = "",
    filename: str = "",
    chunks: list[dict[str, Any]] | None = None,
    outline: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """与库内已有文献比对，返回相似度高于阈值的候选。"""
    incoming = build_document_features(
        title,
        chunks or [],
        outline or [],
        filename,
        fingerprint_value,
        shingles,
    )
    candidates: list[dict[str, Any]] = []
    for paper in list_papers():
        if paper["id"] == exclude_id:
            continue
        existing = build_document_features(
            paper.get("title", ""),
            all_chunks([paper["id"]]),
            paper.get("outline", []),
            paper.get("filename", ""),
            paper.get("fingerprint", ""),
            paper.get("shingles") or [],
        )
        comparison = compare_features(incoming, existing)
        if comparison["duplicate"]:
            candidates.append(
                {
                    "existing_id": paper["id"],
                    "existing_title": paper["title"],
                    "existing_filename": paper["filename"],
                    "existing_year": paper.get("year", ""),
                    "similarity": comparison["score"],
                    "simhash_similarity": comparison["simhash_similarity"],
                    "minhash_similarity": comparison["minhash_similarity"],
                    "title_similarity": comparison["title_similarity"],
                    "signal_votes": comparison["votes"],
                    "duplicate": True,
                    "threshold": DEDUP_THRESHOLD,
                    "created_at": paper["created_at"],
                    "message": (
                        f"库中已存在内容高度重合的文献《{paper['title']}》"
                        f"（综合相似度 {comparison['score']:.0%}，"
                        f"{comparison['votes']}/3 信号命中），建议对比版本差异后再决定是否入库。"
                    ),
                }
            )
    candidates.sort(key=lambda x: -x["similarity"])
    return candidates


def compare_versions(id_new: str, id_old: str) -> dict[str, Any]:
    """逐段 diff 两个版本，输出新增/删除/修改段落与变更摘要。"""
    a, b = get_paper(id_new), get_paper(id_old)
    if not a or not b:
        raise ValueError("文献不存在")

    def body_chunks(paper_id: str, paper: dict[str, Any]) -> list[dict[str, Any]]:
        chunks = sorted(all_chunks([paper_id]), key=lambda c: c["ord"])
        reference_pages = [
            int(item["page"])
            for item in paper.get("outline", [])
            if re.match(r"^references\b", item.get("title", ""), re.I)
        ]
        if reference_pages:
            chunks = [chunk for chunk in chunks if chunk["page"] < min(reference_pages)]
        return chunks

    def paragraphs(chunks: list[dict[str, Any]]) -> list[str]:
        return [re.sub(r"\s+", " ", c["text"]).strip() for c in chunks]

    chunks_a = body_chunks(id_new, a)
    chunks_b = body_chunks(id_old, b)
    features_a = build_document_features(
        a.get("title", ""),
        chunks_a,
        a.get("outline", []),
        a.get("filename", ""),
        a.get("fingerprint", ""),
        a.get("shingles") or [],
    )
    features_b = build_document_features(
        b.get("title", ""),
        chunks_b,
        b.get("outline", []),
        b.get("filename", ""),
        b.get("fingerprint", ""),
        b.get("shingles") or [],
    )
    signals = compare_features(features_a, features_b)
    pa, pb = paragraphs(chunks_a), paragraphs(chunks_b)
    sm = difflib.SequenceMatcher(a=pb, b=pa, autojunk=False)
    added, removed = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert") and j2 > j1:
            added.append({"new_index": j1, "text": " ".join(pa[j1:j2])[:1200]})
        if tag in ("replace", "delete") and i2 > i1:
            removed.append({"old_index": i1, "text": " ".join(pb[i1:i2])[:1200]})

    # 段落索引难以体现真实重合度，补充字符级相似度作为整体修订幅度指标
    full_a = " ".join(pa)[:20000]
    full_b = " ".join(pb)[:20000]
    char_ratio = difflib.SequenceMatcher(a=full_b, b=full_a, autojunk=False).quick_ratio()

    chapters = _chapter_changes(a, b, chunks_a, chunks_b)
    references = _reference_changes(a.get("references", []), b.get("references", []))
    return {
        "new": {"id": id_new, "title": a["title"], "pages": a["n_pages"], "chunks": len(pa)},
        "old": {"id": id_old, "title": b["title"], "pages": b["n_pages"], "chunks": len(pb)},
        "summary": {
            "added_paragraphs": len(added),
            "removed_paragraphs": len(removed),
            "added_sections": chapters["added"],
            "removed_sections": chapters["removed"],
            "rewritten_sections": [item["title"] for item in chapters["rewritten"]],
            "references_old": references["old_count"],
            "references_new": references["new_count"],
            "references_added": references["added_count"],
            "references_removed": references["removed_count"],
            "ratio": round(char_ratio, 4),
        },
        "chapters": chapters,
        "references": references,
        "signals": {
            "similarity": signals["score"],
            "simhash_similarity": signals["simhash_similarity"],
            "minhash_similarity": signals["minhash_similarity"],
            "title_similarity": signals["title_similarity"],
            "votes": signals["votes"],
            "duplicate": signals["duplicate"],
            "threshold": DEDUP_THRESHOLD,
        },
        "added": added[:6],
        "removed": removed[:6],
    }
