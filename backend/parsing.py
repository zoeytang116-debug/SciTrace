"""PDF 细粒度解析：版面感知的标题树、摘要、引用、表格与公式抽取。

技术路线：PyMuPDF 直接读取文本层，保留每个文本块的 bbox 坐标，
使前端「点引用 → 跳到原 PDF 页面并高亮」成为可能（Source Tracing 的物理基础）。
无文本层的扫描件则渲染成图片交给视觉模型 OCR，保证「扫描件也可解析」。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf as fitz  # PyMuPDF 新命名空间，兼容原 fitz 用法

from config import RETRIEVAL, UPLOAD_DIR

HEADING_PATTERN = re.compile(
    r"^(abstract|introduction|related work|background|method|methods|approach|experiment|"
    r"experiments|result|results|discussion|conclusion|conclusions|references|appendix|"
    r"acknowledg|limitation|future work)\b",
    re.I,
)
NAMED_HEADING = re.compile(
    r"^(abstract|references|acknowledg(?:e)?ments?|attention visualizations|"
    r"appendix\b[^\n]{0,100})$",
    re.I,
)
NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-Z][^\n]{2,80})$")
REFERENCE_MARKER = re.compile(r"\[(\d{1,3})\]\s*")
ARXIV_ID = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5})(v\d+)?", re.I)
ARXIV_STEM = re.compile(r"(?<!\d)(\d{2})(\d{2})\.\d{4,5}(v\d+)?", re.I)
AUTHOR_NAME = re.compile(
    r"^(?:[^\W\d_][\w.'-]*)(?:\s+(?:[^\W\d_][\w.'-]*)){1,4}$",
    re.UNICODE,
)
INSTITUTION_HINT = re.compile(
    r"(google|university|institute|research|brain|laboratory|\blab\b|department|"
    r"school|college|\binc\b|corporation|openai|microsoft|deepmind|ai language)",
    re.I,
)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
MATH_FONT = ("math", "symbol", "cambria", "cmr", "cmsy")


@dataclass
class Segment:
    page: int
    text: str
    bbox: list[float]
    heading: str = ""
    font_size: float = 10.0


@dataclass
class ParseResult:
    meta: dict[str, Any] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    outline: list[dict[str, Any]] = field(default_factory=list)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())


def _body_font_size(doc: fitz.Document) -> float:
    """用字符数加权统计正文主字号，作为标题判定基准。"""
    counter: Counter[float] = Counter()
    for page in doc[: min(8, doc.page_count)]:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if len(span["text"].strip()) >= 3:
                        counter[round(span["size"], 1)] += len(span["text"])
    return counter.most_common(1)[0][0] if counter else 10.0


def _numbered_heading(text: str) -> tuple[str, str, int] | None:
    """识别章节编号，排除以年份开头的正文段落。"""
    match = NUMBERED_HEADING.match(text.strip())
    if not match:
        return None
    number, title = match.groups()
    if title.endswith(("。", ".")):
        return None
    return number, title.strip(), len(number.split("."))


def _collect_segments(doc: fitz.Document, body_size: float) -> tuple[list[Segment], list[dict]]:
    """按阅读顺序提取文本块，并按编号与字号建立章节树。"""
    segments: list[Segment] = []

    for pno, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            lines = block.get("lines", [])
            text = " ".join(
                "".join(span["text"] for span in line["spans"]).strip() for line in lines
            ).strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) < 2:
                continue
            max_size = max(
                (span["size"] for line in lines for span in line["spans"]), default=body_size
            )
            numbered = _numbered_heading(text)
            named = bool(NAMED_HEADING.match(text))
            visually_large = (
                max_size >= body_size * 1.14
                and len(text) <= 90
                and not text.endswith((".", "。"))
            )
            # 首页特大的非章节块是论文标题、arXiv 水印或版权行，不进入导航树。
            hero_noise = pno == 1 and max_size >= body_size * 1.30 and not (numbered or named)
            is_heading = not hero_noise and bool(numbered or named or visually_large)
            segments.append(
                Segment(page=pno, text=text, bbox=[round(v, 1) for v in block["bbox"]],
                        heading=text if is_heading else "", font_size=round(max_size, 1))
            )

    # 编号深度决定一级章节；其余导航项按视觉字号分二级和三级。
    top_level = {
        id(seg)
        for seg in segments
        if seg.heading and (numbered := _numbered_heading(seg.heading)) and numbered[2] == 1
    }
    secondary_sizes = sorted(
        {seg.font_size for seg in segments if seg.heading and id(seg) not in top_level},
        reverse=True,
    )
    secondary_size = secondary_sizes[0] if secondary_sizes else None

    outline: list[dict[str, Any]] = []
    for seg in segments:
        if seg.heading:
            if id(seg) in top_level:
                level = 1
            elif seg.font_size == secondary_size:
                level = 2
            else:
                level = 3
            outline.append(
                {
                    "level": level,
                    "title": seg.heading,
                    "page": seg.page,
                    "bbox": seg.bbox,
                }
            )
    return segments, outline


def _seg_font_size(seg: Segment, doc: fitz.Document) -> float:
    """兼容旧调用：返回该块内最大字号。"""
    return _block_font_size(seg)


_BLOCK_SIZE_CACHE: dict[tuple[int, tuple[float, ...]], float] = {}


def _block_font_size(seg: Segment) -> float:
    """从解析阶段缓存的字号信息读取块内最大字号（避免重复遍历页面）。"""
    return getattr(seg, "font_size", 10.0)


def _page_lines(page: fitz.Page) -> list[dict[str, Any]]:
    """保留首页行的文本、坐标与 span 字体信息，供作者区结构识别。"""
    lines: list[dict[str, Any]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            lines.append(
                {
                    "text": text,
                    "bbox": [round(v, 1) for v in line["bbox"]],
                    "spans": line["spans"],
                }
            )
    return lines


def _title_bottom(first_page: list[Segment], title_seg: Segment) -> float:
    """把跨多行的论文题名都排除在作者区之外。"""
    related = [
        seg
        for seg in first_page
        if abs(seg.font_size - title_seg.font_size) <= 0.2
        and title_seg.bbox[1] - 2 <= seg.bbox[1] <= title_seg.bbox[3] + 30
    ]
    return max((seg.bbox[3] for seg in related), default=title_seg.bbox[3])


def _title_text(first_page: list[Segment], title_seg: Segment) -> str:
    """合并同一字号且纵向相邻的多行论文标题。"""
    related = [
        seg
        for seg in first_page
        if abs(seg.font_size - title_seg.font_size) <= 0.3
        and abs(seg.bbox[1] - title_seg.bbox[1]) <= 45
        and not re.match(r"^arxiv:", seg.text, re.I)
    ]
    related.sort(key=lambda seg: (seg.bbox[1], seg.bbox[0]))
    joined = ""
    for seg in related:
        piece = seg.text.strip()
        if joined.endswith("-"):
            joined = joined[:-1] + piece if piece[:1].isupper() else joined + piece
        else:
            joined = f"{joined} {piece}" if joined else piece
    joined = re.sub(r"\b([A-Z]{2,})-\s+([A-Z]{2,})\b", r"\1\2", joined)
    return re.sub(r"\s+", " ", joined).strip()


def _marker_affiliations(page: fitz.Page) -> dict[str, str]:
    """从首页脚注中解析 †/‡ 等上标对应的机构。"""
    mapping: dict[str, str] = {}
    for line in page.get_text("text").splitlines():
        text = " ".join(line.split())
        match = re.match(r"^([*∗†‡]+)\s*(work performed.*)$", text, re.I)
        if not match:
            continue
        markers, note = match.groups()
        affiliation = re.search(r"(?:while at|at)\s+(.+?)(?:[.;]|$)", note, re.I)
        if not affiliation:
            continue
        cleaned = affiliation.group(1).strip()
        for marker in markers:
            mapping[marker] = cleaned
    return mapping


def _extract_author_details(
    page: fitz.Page,
    title_bottom: float,
    abstract_y: float,
    body_size: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """按首页作者区 span 提取姓名、上标和紧邻机构。"""
    lines = _page_lines(page)
    authors: list[dict[str, Any]] = []
    for line in lines:
        y0 = line["bbox"][1]
        if not (title_bottom - 2 <= y0 < abstract_y):
            continue
        for span in line["spans"]:
            name = " ".join(span["text"].split())
            font = span["font"].lower()
            is_bold = any(token in font for token in ("medi", "bold", "semi", "black", "heavy"))
            if (
                not is_bold
                or span["size"] < max(8.0, body_size * 0.75)
                or "@" in name
                or not AUTHOR_NAME.fullmatch(name)
            ):
                continue
            markers = "".join(
                other["text"].strip()
                for other in line["spans"]
                if other is not span
                and other["size"] < span["size"] * 0.8
                and other["bbox"][0] >= span["bbox"][2] - 1
                and other["text"].strip()
                and all(ch in "*∗†‡0123456789" for ch in other["text"].strip())
            )
            candidates = [
                item
                for item in authors
                if item["name"] == name
                and abs(item["y"] - y0) < 2
                and abs(item["x"] - span["bbox"][0]) < 2
            ]
            if not candidates:
                authors.append(
                    {
                        "name": name,
                        "markers": markers,
                        "y": y0,
                        "x": span["bbox"][0],
                        "x1": span["bbox"][2],
                    }
                )
    authors.sort(key=lambda item: (round(item["y"], 1), item["x"]))

    institution_lines: list[tuple[float, float, float, str]] = []
    for line in lines:
        y0 = line["bbox"][1]
        if not (title_bottom - 2 <= y0 < abstract_y):
            continue
        text = re.sub(r"\S+@\S+", "", line["text"])
        text = re.sub(r"[{}]", "", text)
        text = " ".join(text.split())
        if not text or not INSTITUTION_HINT.search(text):
            continue
        if any(author["name"] in text for author in authors):
            continue
        institution_lines.append((y0, line["bbox"][0], line["bbox"][2], text))

    marker_map = _marker_affiliations(page)
    details: list[dict[str, Any]] = []
    for author in authors:
        candidates = [
            item
            for item in institution_lines
            if item[0] >= author["y"] - 1 and item[0] - author["y"] <= 50
        ]
        nearest = min(
            candidates,
            key=lambda item: (
                item[0] - author["y"],
                abs((item[1] + item[2]) / 2 - (author["x"] + author["x1"]) / 2),
            ),
            default=None,
        )
        affiliations = [nearest[3]] if nearest else []
        if not affiliations:
            for marker in author["markers"]:
                if marker in marker_map:
                    affiliations = [marker_map[marker]]
                    break
        details.append(
            {
                "name": author["name"],
                "markers": author["markers"],
                "affiliations": affiliations,
            }
        )

    if not details:
        for line in lines:
            y0 = line["bbox"][1]
            text = " ".join(line["text"].split())
            if (
                title_bottom - 2 <= y0 < min(abstract_y, title_bottom + 100)
                and AUTHOR_NAME.fullmatch(text)
                and not INSTITUTION_HINT.search(text)
            ):
                details.append({"name": text, "markers": "", "affiliations": []})
                break

    unique_affiliations: list[str] = []
    for detail in details:
        for affiliation in detail["affiliations"]:
            if affiliation not in unique_affiliations:
                unique_affiliations.append(affiliation)
    return details, unique_affiliations


def _guess_title_authors(
    doc: fitz.Document,
    segments: list[Segment],
    body_size: float,
) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    """首页最大字号块视为标题，标题下方紧邻的小字块视为作者。

    关键点：按「字号最大 + 长度合理」双重条件筛选，避免把正文长句误判成标题。
    """
    if not segments:
        return "未命名文献", "", [], []
    first_page = [s for s in segments if s.page == 1] or segments[:5]
    scored: list[tuple[float, int, Segment]] = []
    for s in first_page[:15]:
        text = s.text.strip()
        if (
            not (6 <= len(text) <= 260)
            or HEADING_PATTERN.match(text)
            or re.match(r"^arxiv:", text, re.I)
        ):
            continue
        if re.fullmatch(r"[\d\W]+", text):
            continue
        scored.append((_block_font_size(s), len(text), s))
    if not scored:
        return first_page[0].text[:120], "", [], []
    title_seg = max(scored, key=lambda t: (t[0], t[1]))[2]
    title = re.sub(
        r"^(research|original article|article)\s*[:\-]?\s*",
        "",
        _title_text(first_page, title_seg),
        flags=re.I,
    )

    idx = first_page.index(title_seg) if title_seg in first_page else 0
    abstract_y = min(
        (
            seg.bbox[1]
            for seg in first_page
            if re.match(r"^abstract\b", seg.text, re.I) and seg.bbox[1] > title_seg.bbox[3]
        ),
        default=title_seg.bbox[3] + 240,
    )
    author_details, affiliations = _extract_author_details(
        doc[0],
        _title_bottom(first_page, title_seg),
        abstract_y,
        body_size,
    )
    authors = ", ".join(detail["name"] for detail in author_details)
    if not authors:
        for seg in first_page[idx + 1: idx + 4]:
            if len(seg.text) < 400 and re.search(r"[A-Z][a-z]+\s+[A-Za-z\.\-\s,&]{2,}", seg.text):
                authors = seg.text.strip()
                break
    return title.strip(), authors.strip(), author_details, affiliations


def _extract_abstract(segments: list[Segment]) -> str:
    joined = [s for s in segments if s.page <= 3]
    for i, seg in enumerate(joined):
        if re.match(r"^abstract", seg.text, re.I):
            body = re.sub(r"^abstract[\.:：\s]*", "", seg.text, flags=re.I)
            extra = " ".join(s.text for s in joined[i + 1: i + 3])
            if len(body) < 120:
                body = f"{body} {extra}"
            return body.strip()[:3000]
    return " ".join(s.text for s in joined[1:4])[:1200]


def _reference_entry(raw: str, number: int | None = None) -> dict[str, Any]:
    cleaned = re.sub(r"\s+", " ", raw).strip()
    arxiv = ARXIV_ID.search(cleaned)
    year = re.search(r"(19|20)\d{2}", cleaned)
    entry: dict[str, Any] = {
        "raw": cleaned[:600],
        "arxiv": arxiv.group(1) if arxiv else "",
        "year": year.group(0) if year else "",
        "title": _guess_ref_title(cleaned),
    }
    if number is not None:
        entry["number"] = number
    return entry


def _extract_numbered_references(text: str) -> list[dict[str, Any]]:
    """只按 [1]、[2]… 标记计数，避免把作者行误当参考文献。"""
    normalized = re.sub(r"\s+", " ", text)
    markers = list(REFERENCE_MARKER.finditer(normalized))
    entries: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(normalized)
        raw = normalized[marker.end():end].strip()
        if len(raw) < 20:
            continue
        entries.append(_reference_entry(raw, int(marker.group(1))))
    return entries


def _reference_columns(page: fitz.Page) -> list[list[tuple[float, float, str]]]:
    """按左/右栏收集文本行；第一列是行起点，续行会保留缩进坐标。"""
    width = page.rect.width
    grouped: list[dict[float, list[tuple[float, str]]]] = [{}, {}]
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            x0, y0, _, _ = line["bbox"]
            if y0 > page.rect.height - 28:
                continue
            column = 0 if x0 < width / 2 else 1
            grouped[column].setdefault(round(y0, 1), []).append((x0, text))

    columns: list[list[tuple[float, float, str]]] = []
    for rows_by_y in grouped:
        rows: list[tuple[float, float, str]] = []
        for y, items in sorted(rows_by_y.items()):
            items.sort(key=lambda item: item[0])
            rows.append((y, items[0][0], " ".join(text for _, text in items)))
        columns.append(rows)
    return columns


def _extract_layout_references(doc: fitz.Document, start_page: int) -> list[dict[str, Any]]:
    """解析无 [n] 编号的双栏作者-年份参考文献。"""
    entries: list[dict[str, Any]] = []
    current = ""

    def flush() -> None:
        nonlocal current
        cleaned = re.sub(r"\s+", " ", current).strip()
        if len(cleaned) >= 20:
            entries.append(_reference_entry(cleaned))
        current = ""

    for pno in range(max(0, start_page - 1), doc.page_count):
        columns = _reference_columns(doc[pno])
        appendix_col = None
        appendix_y = None
        for col_index, rows in enumerate(columns):
            for y, _, text in rows:
                if re.match(r"^appendix\s+for\b", text, re.I):
                    appendix_col = col_index
                    appendix_y = y
                    break
            if appendix_col is not None:
                break

        for col_index, rows in enumerate(columns):
            # BERT 的附录从右栏开始，而参考文献仍在同页左栏收尾。
            if appendix_col is not None and col_index != appendix_col:
                continue
            if appendix_y is not None:
                rows = [row for row in rows if row[0] < appendix_y]
            if not rows:
                continue

            margin = min(row[1] for row in rows)
            for _, x0, text in rows:
                if re.match(r"^references$", text, re.I):
                    continue
                if abs(x0 - margin) <= 2.0:
                    if current:
                        flush()
                    current = text
                elif current:
                    current = f"{current} {text}"

        if appendix_col is not None:
            break

    if current:
        flush()
    return entries[:120]


def _extract_references(doc: fitz.Document, segments: list[Segment]) -> list[dict[str, Any]]:
    """解析参考文献：优先按 [n] 编号，无编号时按双栏缩进切分。"""
    start_idx = next(
        (i for i, seg in enumerate(segments) if re.match(r"^references\b", seg.heading or "", re.I)),
        None,
    )
    if start_idx is not None:
        start_page = segments[start_idx].page
        end_idx = next(
            (
                i
                for i, seg in enumerate(segments[start_idx + 1:], start=start_idx + 1)
                if seg.heading and not re.match(r"^references\b", seg.heading, re.I)
            ),
            len(segments),
        )
        tail_text = "\n".join(seg.text for seg in segments[start_idx:end_idx])
    else:
        start_page = max(1, doc.page_count - 1)
        tail_text = "\n".join(seg.text for seg in segments if seg.page >= start_page)

    numbered = _extract_numbered_references(tail_text)
    if numbered:
        return numbered[:120]
    return _extract_layout_references(doc, start_page)


def _guess_ref_title(entry: str) -> str:
    """参考文献标题启发式：优先引号内文本，其次首个句点后的大写短语。"""
    quoted = re.search(r"“(.{8,120})”|\"(.{8,120})\"", entry)
    if quoted:
        return (quoted.group(1) or quoted.group(2)).strip()
    parts = [p.strip() for p in entry.split(".") if p.strip()]
    for part in parts:
        if len(part) >= 12 and not re.match(r"^(in |proceedings|arxiv|journal|conference)", part, re.I):
            return part[:160]
    return entry[:80]


def _detect_tables(page: fitz.Page) -> list[dict[str, Any]]:
    """表格抽取：利用 PyMuPDF 内置表格识别，返回可结构化的单元格。"""
    try:
        tables = page.find_tables()
    except Exception:
        return []
    out = []
    for tb in tables.tables:
        rows = tb.extract()
        if rows and len(rows) >= 2:
            out.append({"rows": [[(c or "")[:80] for c in r][:8] for r in rows[:8]]})
    return out


def _detect_formulas(page: fitz.Page) -> list[str]:
    """公式抽取：按数学字体族与符号密度识别行内/行间公式片段。"""
    found: list[str] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line["spans"]
            if not spans:
                continue
            math_chars = sum(len(re.findall(r"[=+\-*/^_{}\\]|\\frac|\\sum|\\mathcal", s["text"])) for s in spans)
            math_font = any(any(k in s["font"].lower() for k in MATH_FONT) for s in spans)
            if math_font or math_chars >= 4:
                text = "".join(s["text"] for s in spans).strip()
                if 3 <= len(text) <= 200:
                    found.append(text)
    return found[:12]


def _is_scanned(doc: fitz.Document) -> bool:
    total = sum(len(page.get_text().strip()) for page in doc[: min(6, doc.page_count)])
    return total < 120 * max(1, min(6, doc.page_count))


def _render_page_b64(page: fitz.Page, dpi: int = 150) -> str:
    pix = page.get_pixmap(dpi=dpi)
    return base64.b64encode(pix.tobytes("png")).decode()


def fingerprint(text: str) -> str:
    """SimHash 64 位，用于跨版本内容级去重（不依赖文件名）。"""
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())
    if not tokens:
        return "0" * 16
    weights = Counter(tokens)
    bits = [0] * 64
    for tok, w in weights.items():
        h = int(hashlib.blake2b(tok.encode(), digest_size=8).hexdigest(), 16)
        for i in range(64):
            bits[i] += w if (h >> i) & 1 else -w
    return f"{sum(1 << i for i, b in enumerate(bits) if b >= 0):016x}"


def minhash_signature(text: str, k: int = 64) -> list[int]:
    """MinHash 签名，快速估算两篇文献的 Jaccard 相似度。"""
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())
    shingles = {" ".join(words[i:i + 5]) for i in range(max(0, len(words) - 5))}
    if not shingles:
        return [0] * k
    sig = [2**63] * k
    for sh in shingles:
        h = int(hashlib.blake2b(sh.encode(), digest_size=8).hexdigest(), 16)
        for i in range(k):
            v = (h ^ (i * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
            if v < sig[i]:
                sig[i] = v
    return sig


def simhash_distance(a: str, b: str) -> float:
    if not a or not b:
        return 1.0
    x, y = int(a, 16), int(b, 16)
    return bin(x ^ y).count("1") / 64.0


def jaccard_from_signatures(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _split_long(text: str, limit: int) -> list[str]:
    """把超长文本块按句号切成不超过 limit 的片段，保证切片粒度可溯源。"""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    buf = ""
    for sent in re.split(r"(?<=[\.\!\?。；;])\s+", text):
        if len(buf) + len(sent) > limit and buf:
            parts.append(buf.strip())
            buf = sent
        else:
            buf += " " + sent if buf else sent
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _chunk_segments(segments: list[Segment]) -> list[dict[str, Any]]:
    """滑动窗口切块：保留所属章节路径与坐标，为引用溯源提供锚点。"""
    chunks: list[dict[str, Any]] = []
    buf: list[Segment] = []
    buf_len = 0
    current_heading = ""
    ordi = 0

    # 拆开超长块（整页压成一块时，切片会失去粒度，溯源只剩页码）
    expanded: list[Segment] = []
    for seg in segments:
        for piece in _split_long(seg.text, RETRIEVAL.chunk_size):
            expanded.append(Segment(page=seg.page, text=piece, bbox=seg.bbox,
                                   heading=seg.heading, font_size=seg.font_size))

    def flush() -> None:
        nonlocal buf, buf_len, ordi
        if not buf:
            return
        text = " ".join(s.text for s in buf).strip()
        if len(text) >= 40:
            xs = [s.bbox[0] for s in buf] + [s.bbox[2] for s in buf]
            ys = [s.bbox[1] for s in buf] + [s.bbox[3] for s in buf]
            chunks.append(
                {
                    "page": buf[0].page,
                    "heading": buf[0].heading or current_heading,
                    "text": text,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "ord": ordi,
                }
            )
            ordi += 1
        buf, buf_len = [], 0

    for seg in expanded:
        if seg.heading:
            current_heading = seg.heading
        buf.append(seg)
        buf_len += len(seg.text)
        if buf_len >= RETRIEVAL.chunk_size:
            flush()
    flush()
    return chunks


def _year_from_arxiv(path: Path, doc: fitz.Document) -> tuple[str, str]:
    """arXiv 文献统一取首次公开年份（YYMM 前缀）。"""
    match = ARXIV_STEM.search(path.stem)
    if not match:
        return "", ""
    base_year = 2000 + int(match.group(1))
    return str(base_year), "arxiv_filename"


def _year_from_front_matter(doc: fitz.Document) -> str:
    """从首页会议、版权、期刊或 arXiv 标注中取年份。"""
    if not doc.page_count:
        return ""
    keywords = re.compile(
        r"(proceedings|conference|copyright|published|journal|arxiv|naacl|acl|"
        r"neurips|nips|iclr|icml|©|c⃝|january|february|march|april|may|june|"
        r"july|august|september|october|november|december)",
        re.I,
    )
    years: list[str] = []
    for line in doc[0].get_text("text").splitlines():
        if keywords.search(line):
            years.extend(YEAR_PATTERN.findall(line))
    return Counter(years).most_common(1)[0][0] if years else ""


def _year_from_corroborated_metadata(doc: fitz.Document) -> str:
    metadata = doc.metadata or {}
    date_pattern = re.compile(r"D:((?:19|20)\d{2})")
    creation = date_pattern.search(str(metadata.get("creationDate", "") or ""))
    modified = date_pattern.search(str(metadata.get("modDate", "") or ""))
    if creation and modified and creation.group(1) == modified.group(1):
        return creation.group(1)
    return ""


def _year_from_references(references: list[dict[str, Any]]) -> str:
    years = [
        str(entry.get("year", ""))
        for entry in references
        if YEAR_PATTERN.fullmatch(str(entry.get("year", "")))
    ]
    return Counter(years).most_common(1)[0][0] if years else ""


def _extract_document_year(
    path: Path,
    doc: fitz.Document,
    references: list[dict[str, Any]],
) -> tuple[str, str]:
    """年份优先级：arXiv 文件名/版本首页 → 首页脚注 → 参考文献众数 → metadata。"""
    arxiv_year = _year_from_arxiv(path, doc)
    if arxiv_year[0]:
        return arxiv_year

    front_year = _year_from_front_matter(doc)
    if front_year:
        return front_year, "front_matter"

    metadata_dates_year = _year_from_corroborated_metadata(doc)
    if metadata_dates_year:
        return metadata_dates_year, "metadata:creation_mod"

    reference_year = _year_from_references(references)
    if reference_year:
        return reference_year, "references_mode"

    metadata = doc.metadata or {}
    for key, value in metadata.items():
        if key.lower() == "creationdate" or not value:
            continue
        years = YEAR_PATTERN.findall(str(value))
        if years:
            return years[-1], f"metadata:{key}"

    creation_years = YEAR_PATTERN.findall(str(metadata.get("creationDate", "") or ""))
    if creation_years and reference_year and creation_years[-1] == reference_year:
        return creation_years[-1], "metadata:creationDate_corroborated"
    return "", ""


def parse_pdf(path: Path, paper_id: str) -> ParseResult:
    """对外主入口：解析一篇 PDF 并返回元数据、切片与章节树。"""
    doc = fitz.open(path)
    body_size = _body_font_size(doc)
    scanned = _is_scanned(doc)

    if scanned:
        # 扫描件：逐页渲染为图片，交由视觉模型 OCR，重建可检索文本层
        segments = []
        ocr_note = ""
        from llm import LLM, available, describe_image

        use_ocr = LLM.vision_enabled
        for pno, page in enumerate(doc, start=1):
            text = ""
            if use_ocr:
                try:
                    text = describe_image(
                        _render_page_b64(page),
                        "请逐行转写这一页 PDF 图片中的全部文字，保留层级标题、公式与表格结构，"
                        "不要添加任何解释性内容。",
                    )
                except Exception as exc:  # OCR 失败不阻断入库
                    ocr_note = f"OCR 失败：{exc}"
            segments.append(Segment(page=pno, text=text or f"[第 {pno} 页待 OCR]", bbox=[0, 0, 595, 842]))
        outline = []
        title, authors = f"{path.stem}（扫描件）", ""
        author_details: list[dict[str, Any]] = []
        affiliations: list[str] = []
        abstract = ""
        references: list[dict[str, str]] = []
        tables: list[dict[str, Any]] = []
        formulas: list[str] = []
        if not use_ocr:
            ocr_note = "未配置视觉模型，扫描件仅完成图像级入库"
    else:
        segments, outline = _collect_segments(doc, body_size)
        title, authors, author_details, affiliations = _guess_title_authors(doc, segments, body_size)
        abstract = _extract_abstract(segments)
        references = _extract_references(doc, segments)
        tables = [t for pno in range(min(6, doc.page_count)) for t in _detect_tables(doc[pno])]
        formulas = [f for pno in range(min(4, doc.page_count)) for f in _detect_formulas(doc[pno])][:10]
        ocr_note = ""

    body_text = " ".join(s.text for s in segments)
    chunks = _chunk_segments(segments)
    year, year_source = _extract_document_year(path, doc, references)

    meta = {
        "id": paper_id,
        "filename": path.name,
        "title": title or path.stem,
        "authors": authors,
        "author_details": author_details,
        "affiliations": affiliations,
        "year": year,
        "year_source": year_source,
        "abstract": abstract,
        "n_pages": doc.page_count,
        "is_scanned": scanned,
        "status": "ready" if chunks else "empty",
        "fingerprint": fingerprint(re.sub(r"\s+", "", body_text)),
        "shingles": minhash_signature(body_text),
        "outline": outline,
        "references": references,
        "pdf_path": str(path),
        "n_tables": len(tables),
        "n_formulas": len(formulas),
        "tables": tables[:3],
        "formulas": formulas,
        "ocr_note": ocr_note,
        "doc_metadata": {k: v for k, v in (doc.metadata or {}).items() if v},
    }
    doc.close()
    return ParseResult(meta=meta, chunks=[{**c, "paper_id": paper_id} for c in chunks], outline=outline)


def save_upload(filename: str, data: bytes) -> Path:
    """落盘上传文件，文件名做安全化处理，避免路径穿越。"""
    safe = re.sub(r"[^\w.\-\u4e00-\u9fff]", "_", Path(filename).name)
    target = UPLOAD_DIR / safe
    if target.exists():
        stem, suffix = target.stem, target.suffix
        i = 1
        while (UPLOAD_DIR / f"{stem}_{i}{suffix}").exists():
            i += 1
        target = UPLOAD_DIR / f"{stem}_{i}{suffix}"
    target.write_bytes(data)
    return target
