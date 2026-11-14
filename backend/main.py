"""PaperLens 论文透镜 —— FastAPI 主服务。

路由分层：
/api/papers*   文献管理（上传、列表、详情、原文流、页面渲染）
/api/qa*       抗幻觉问答（普通 / SSE 流式）
/api/compare   多文献跨文对比
/api/survey    图谱 + 自动综述 + Future Work
/api/write     无幻觉论文框架
/api/translate 中英学术翻译
/api/viz       CSV → 学术图表 + Caption
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
import shutil
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agent
import dedup
import graph as graph_mod
import llm
import local_qa
import messages
import parsing
import store
from config import ARXIV_MAX_BYTES, ARXIV_PDF_BASE_URL, ARXIV_TIMEOUT_S, LLM, UPLOAD_DIR
from retrieval import INDEX


DEMO_FILENAME = "1706.03762v7.pdf"
DEMO_ID = "demo-v7"
DEMO_SOURCE = Path(__file__).resolve().parent.parent / "pdf" / DEMO_FILENAME
SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"

ARXIV_ID_PATTERN = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?$"
)
# arXiv 拒绝无 User-Agent 的下载请求。
ARXIV_USER_AGENT = "SciTrace/1.0 (arXiv importer; +https://github.com/zoeytang116-debug/SciTrace)"

logger = logging.getLogger("scitrace.arxiv")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _arxiv_handler = logging.StreamHandler()
    _arxiv_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_arxiv_handler)
    logger.propagate = False


def _restore_seed_data() -> None:
    """容器临时磁盘重置后，从镜像内 seed 恢复库和上传文件。"""
    seed_db = SEED_DIR / "library.db"
    if not store.DB_PATH.exists() and seed_db.exists():
        store.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_db, store.DB_PATH)

    seed_uploads = SEED_DIR / "uploads"
    if seed_uploads.exists():
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for source in seed_uploads.iterdir():
            target = UPLOAD_DIR / source.name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)


def _author_contract(authors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把解析器的作者详情收敛为前端与问答使用的稳定单机构契约。"""
    normalized: list[dict[str, Any]] = []
    for index, author in enumerate(authors):
        raw_affiliations = author.get("affiliations", [])
        if isinstance(raw_affiliations, str):
            raw_affiliations = [raw_affiliations]
        affiliation = "; ".join(
            str(item).strip() for item in raw_affiliations if str(item).strip()
        )
        normalized.append(
            {
                "order": index + 1,
                "name": str(author.get("name", "")).strip(),
                "affiliation": affiliation,
            }
        )
    return normalized


def _ensure_demo_paper() -> None:
    """确保首屏始终有一篇 v7 可展示；已有解析详情时不重复解析。"""
    papers = store.list_papers()
    demo_stem = Path(DEMO_FILENAME).stem
    existing = next(
        (
            paper
            for paper in papers
            if Path(paper.get("filename", "")).stem.startswith(demo_stem)
        ),
        None,
    )
    if existing and existing.get("author_details"):
        return

    source = Path(existing["pdf_path"]) if existing and existing.get("pdf_path") else DEMO_SOURCE
    if not source.exists():
        return
    paper_id = existing["id"] if existing else DEMO_ID
    parsed = parsing.parse_pdf(source, paper_id)
    store.insert_paper(parsed.meta, parsed.chunks)
    INDEX.invalidate()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _restore_seed_data()
    store.init_db()
    await asyncio.to_thread(_ensure_demo_paper)
    store.log_event("system", "startup", "PaperLens 服务启动")
    yield


app = FastAPI(title="PaperLens · AI4S 科研助手 Agent", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    paper_ids: list[str] = Field(default_factory=list)
    intent: str = ""


class CompareRequest(BaseModel):
    aspect: str
    paper_ids: list[str]


class FilesRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)


class WriteRequest(BaseModel):
    idea: str
    paper_ids: list[str] = Field(default_factory=list)


class TranslateRequest(BaseModel):
    text: str
    direction: str = "zh2en"
    style: str = "academic"


class EvidenceQARequest(BaseModel):
    question: str = Field(..., min_length=1)
    paper_id: str


class VizRequest(BaseModel):
    csv_text: str
    intent: str = ""


class ArxivImportRequest(BaseModel):
    arxiv_id: str = Field(..., min_length=1)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    papers = store.list_papers()
    return {
        "status": "ok",
        "llm_enabled": llm.available(),
        "model": LLM.model if llm.available() else "offline-extractive",
        "vision_enabled": LLM.vision_enabled,
        "papers": len(papers),
        "chunks": sum(p["n_chunks"] for p in papers),
        "dedup_threshold": dedup.DEDUP_THRESHOLD,
    }


@app.post("/api/papers/upload")
async def upload(files: list[UploadFile] = File(...), allow_duplicate: bool = False) -> dict[str, Any]:
    """批量上传：解析 → 语义去重检测 → 入库。命中重复时先返回预警，等用户确认。"""
    results = []
    for uf in files:
        if not uf.filename or not uf.filename.lower().endswith(".pdf"):
            results.append({"filename": uf.filename, "status": "rejected", "reason": "仅支持 PDF"})
            continue
        data = await uf.read()
        if len(data) > 80 * 1024 * 1024:
            results.append({"filename": uf.filename, "status": "rejected", "reason": "文件超过 80MB"})
            continue
        path = parsing.save_upload(uf.filename, data)
        paper_id = uuid.uuid4().hex[:12]
        try:
            parsed = await asyncio.to_thread(parsing.parse_pdf, path, paper_id)
        except Exception as exc:
            results.append({"filename": uf.filename, "status": "failed", "reason": f"解析失败：{exc}"})
            continue

        meta = parsed.meta
        duplicates = dedup.find_duplicates(
            meta["fingerprint"],
            meta["shingles"],
            exclude_id=paper_id,
            title=meta.get("title", ""),
            filename=meta.get("filename", ""),
            chunks=parsed.chunks,
            outline=parsed.outline,
        )
        if duplicates and not allow_duplicate:
            path.unlink(missing_ok=True)
            existing_year = str(duplicates[0].get("existing_year", ""))
            incoming_year = str(meta.get("year", ""))
            recommendation = "两版本年份相同，请选择保留版本"
            if existing_year.isdigit() and incoming_year.isdigit():
                if int(existing_year) > int(incoming_year):
                    recommendation = "建议保留库内版本"
                elif int(incoming_year) > int(existing_year):
                    recommendation = "建议保留新上传版本"
            results.append(
                {
                    "filename": uf.filename,
                    "status": "duplicate_warning",
                    "duplicate": True,
                    "parsed_title": meta["title"],
                    "similarity": duplicates[0]["similarity"],
                    "incoming_year": incoming_year,
                    "recommendation": recommendation,
                    "duplicates": duplicates[:3],
                    "message": duplicates[0]["message"],
                }
            )
            continue

        store.insert_paper(meta, parsed.chunks)
        store.log_event(paper_id, "upload", f"{uvf_name(uf.filename)} · {len(parsed.chunks)} 切片")
        INDEX.invalidate()
        results.append(
            {
                "filename": uf.filename,
                "status": "ok",
                "paper_id": paper_id,
                "title": meta["title"],
                "n_pages": meta["n_pages"],
                "n_chunks": len(parsed.chunks),
                "n_references": len(meta["references"]),
                "n_tables": meta.get("n_tables", 0),
                "is_scanned": meta["is_scanned"],
                "requires_ocr": bool(meta["is_scanned"]),
                "ocr_status": "需要 OCR" if meta["is_scanned"] else "",
                "ocr_note": meta.get("ocr_note", ""),
                "outline_count": len(parsed.outline),
                "authors": _author_contract(meta.get("author_details", [])),
                "affiliations": meta.get("affiliations", []),
                "duplicates": duplicates,
            }
        )
    return {"results": results}


def uvf_name(name: str) -> str:
    return Path(name).name


def _normalize_arxiv_id(raw: str) -> str:
    """接受 1706.03762 / v7 / arXiv: 前缀 / abs 链接，统一取出裸编号。"""
    value = (raw or "").strip()
    if not value:
        raise HTTPException(400, messages.EMPTY_ID)
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+)", value, re.IGNORECASE)
    if match:
        value = match.group(1)
    value = re.sub(r"^arxiv:\s*", "", value.strip(), flags=re.IGNORECASE)
    value = re.sub(r"\.pdf$", "", value.strip(), flags=re.IGNORECASE).strip().rstrip("/")
    if not ARXIV_ID_PATTERN.match(value):
        raise HTTPException(400, messages.INVALID_ID.format(raw=raw.strip()))
    return value


def _log_arxiv(
    level: int,
    *,
    arxiv_id: str,
    paper_id: str | None,
    elapsed_s: float,
    duplicate: bool | None,
    reason: str = "",
    **extra: Any,
) -> None:
    """结构化日志：来源、编号、paper id、耗时、是否重复、失败原因。只写日志，不碰 events 表。"""
    payload: dict[str, Any] = {
        "event": "arxiv_import",
        "source": "arxiv",
        "arxiv_id": arxiv_id,
        "paper_id": paper_id,
        "elapsed_s": round(elapsed_s, 3),
        "duplicate": duplicate,
    }
    if reason:
        payload["reason"] = reason
    payload.update(extra)
    logger.log(level, json.dumps(payload, ensure_ascii=False))


def _download_arxiv_pdf(arxiv_id: str, started: float) -> bytes:
    url = f"{ARXIV_PDF_BASE_URL}/{arxiv_id}"
    request = urllib.request.Request(url, headers={"User-Agent": ARXIV_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=ARXIV_TIMEOUT_S) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read(ARXIV_MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            detail = messages.NOT_FOUND.format(arxiv_id=arxiv_id)
        elif exc.code >= 500:
            detail = messages.SERVER_ERROR.format(status=exc.code)
        else:
            detail = messages.HTTP_FAILED.format(status=exc.code, reason=exc.reason)
        _log_arxiv(
            logging.WARNING,
            arxiv_id=arxiv_id,
            paper_id=None,
            elapsed_s=time.perf_counter() - started,
            duplicate=None,
            reason=detail,
        )
        raise HTTPException(400, detail) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        inner = getattr(exc, "reason", exc)
        is_timeout = (
            isinstance(exc, TimeoutError)
            or isinstance(inner, TimeoutError)
            or "timed out" in str(inner).lower()
        )
        detail = (
            messages.TIMEOUT.format(timeout=ARXIV_TIMEOUT_S)
            if is_timeout
            else messages.NETWORK_FAILED.format(reason=inner)
        )
        _log_arxiv(
            logging.WARNING,
            arxiv_id=arxiv_id,
            paper_id=None,
            elapsed_s=time.perf_counter() - started,
            duplicate=None,
            reason=detail,
        )
        raise HTTPException(400, detail) from exc

    if len(data) > ARXIV_MAX_BYTES:
        detail = messages.TOO_LARGE.format(max_mb=ARXIV_MAX_BYTES // (1024 * 1024))
        _log_arxiv(
            logging.WARNING,
            arxiv_id=arxiv_id,
            paper_id=None,
            elapsed_s=time.perf_counter() - started,
            duplicate=None,
            reason=detail,
        )
        raise HTTPException(400, detail)
    if not data.startswith(b"%PDF") and "pdf" not in content_type.lower():
        detail = messages.NOT_PDF.format(content_type=content_type or messages.UNKNOWN)
        _log_arxiv(
            logging.WARNING,
            arxiv_id=arxiv_id,
            paper_id=None,
            elapsed_s=time.perf_counter() - started,
            duplicate=None,
            reason=detail,
        )
        raise HTTPException(400, detail)
    return data


def _release_exception_chain(error: BaseException) -> None:
    """断开异常链对解析帧（可能持有打开的 PDF 句柄）的引用。"""
    seen: set[int] = set()
    stack: list[BaseException | None] = [error]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        stack.extend((current.__context__, current.__cause__))
        try:
            current.__traceback__ = None
        except (AttributeError, TypeError):
            continue


def _cleanup_upload_file(path: Path, error: BaseException | None = None) -> None:
    """删除临时文件；Windows 上解析器可能短暂占用句柄，先释放再重试。"""
    if error is not None:
        _release_exception_chain(error)
    for _ in range(60):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.05)
    logger.warning("临时文件清理失败：%s", path)


def _validate_pdf_bytes(data: bytes, arxiv_id: str, paper_id: str, started: float) -> None:
    """落盘前用内存流校验 PDF 是否可打开，损坏文件不产生任何临时文件。"""
    import pymupdf as fitz

    try:
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            if doc.page_count < 1:
                raise ValueError("PDF 无可解析页面")
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001 - 损坏/加密/空页统一收敛为 400
        detail = messages.PARSE_FAILED.format(error_type=type(exc).__name__)
        _log_arxiv(
            logging.WARNING,
            arxiv_id=arxiv_id,
            paper_id=paper_id,
            elapsed_s=time.perf_counter() - started,
            duplicate=None,
            reason=detail,
            error=str(exc),
        )
        raise HTTPException(400, detail) from exc


def _ingest_arxiv_pdf(arxiv_id: str, data: bytes, started: float) -> dict[str, Any]:
    """复用上传链路：校验 → 落盘 → 解析 → 去重判定 → 入库。"""
    paper_id = uuid.uuid4().hex[:12]
    _validate_pdf_bytes(data, arxiv_id, paper_id, started)
    path = parsing.save_upload(f"{arxiv_id}.pdf", data)
    parse_error: BaseException | None = None
    parsed = None
    try:
        parsed = parsing.parse_pdf(path, paper_id)
    except Exception as exc:  # noqa: BLE001 - 任何解析异常都收敛为 400
        parse_error = exc
    if parse_error is not None or parsed is None:
        error = parse_error or RuntimeError("解析器未返回结果")
        _cleanup_upload_file(path, error)
        detail = messages.PARSE_FAILED.format(error_type=type(error).__name__)
        _log_arxiv(
            logging.WARNING,
            arxiv_id=arxiv_id,
            paper_id=paper_id,
            elapsed_s=time.perf_counter() - started,
            duplicate=None,
            reason=detail,
            error=str(error),
        )
        raise HTTPException(400, detail) from error

    meta = parsed.meta
    duplicates = dedup.find_duplicates(
        meta["fingerprint"],
        meta["shingles"],
        exclude_id=paper_id,
        title=meta.get("title", ""),
        filename=meta.get("filename", ""),
        chunks=parsed.chunks,
        outline=parsed.outline,
    )
    if duplicates:
        _cleanup_upload_file(path)
        top = duplicates[0]
        _log_arxiv(
            logging.INFO,
            arxiv_id=arxiv_id,
            paper_id=top["existing_id"],
            elapsed_s=time.perf_counter() - started,
            duplicate=True,
            existing_id=top["existing_id"],
            dedup_score=round(float(top["similarity"]), 4),
        )
        return {
            "paper_id": top["existing_id"],
            "title": meta["title"],
            "n_pages": meta["n_pages"],
            "n_chunks": len(parsed.chunks),
            "duplicate": True,
            "dedup_score": round(float(top["similarity"]), 4),
            "existing_id": top["existing_id"],
            "existing_title": top.get("existing_title") or "",
            "elapsed_s": round(time.perf_counter() - started, 2),
        }

    store.insert_paper(meta, parsed.chunks)
    store.log_event(paper_id, "arxiv_import", f"{arxiv_id} · {len(parsed.chunks)} 切片")
    INDEX.invalidate()
    _log_arxiv(
        logging.INFO,
        arxiv_id=arxiv_id,
        paper_id=paper_id,
        elapsed_s=time.perf_counter() - started,
        duplicate=False,
        n_pages=meta["n_pages"],
        n_chunks=len(parsed.chunks),
    )
    return {
        "paper_id": paper_id,
        "title": meta["title"],
        "n_pages": meta["n_pages"],
        "n_chunks": len(parsed.chunks),
        "duplicate": False,
        "dedup_score": None,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


@app.post("/api/papers/import")
async def import_arxiv_paper(req: ArxivImportRequest) -> dict[str, Any]:
    """按 arXiv 编号下载论文 PDF，并复用上传链路完成解析、去重与入库。

    ### 支持的 arxiv_id 写法（大小写不敏感）
    - 裸编号：`1706.03762`
    - 带版本号：`1706.03762v7`
    - 带前缀：`arXiv:1706.03762`
    - abs / pdf 链接：`https://arxiv.org/abs/1706.03762`、`https://arxiv.org/pdf/1706.03762v7`
    - 旧式编号：`hep-th/9901001`（同时支持 `arXiv:` 前缀与 vN 后缀）

    ### 入参
    | 字段 | 类型 | 必填 | 说明 |
    |---|---|---|---|
    | `arxiv_id` | string | 是 | 上述任一写法，服务端统一归一化为裸编号 |

    ### 返回字段（HTTP 200）
    | 字段 | 类型 | 说明 |
    |---|---|---|
    | `paper_id` | string | 新入库论文 ID；命中重复时为库内已有论文 ID |
    | `title` | string | 解析出的标题 |
    | `n_pages` | int | 页数 |
    | `n_chunks` | int | 切片数 |
    | `duplicate` | bool | 是否被去重判定为同源（true 时不写库） |
    | `dedup_score` | number \\| null | 同源相似度；未重复时为 null |
    | `elapsed_s` | number | 从请求到返回的总耗时（秒） |
    | `existing_id` / `existing_title` | string | 仅 duplicate=true 时返回，指向库内同源论文 |

    ### 返回码与异常分支
    | 状态码 | 触发条件 | detail |
    |---|---|---|
    | 200 | 成功入库；或命中同源仅返回 `duplicate=true` | — |
    | 400 | 编号为空 / 格式非法（不发起下载） | 中文原因 |
    | 400 | arXiv 返回 404 | `arXiv 上找不到编号 …（HTTP 404）` |
    | 400 | arXiv 返回 5xx | `arXiv 服务暂时不可用（HTTP 5xx），请稍后重试` |
    | 400 | 连接失败 / 读取超时（默认 30 s，`ARXIV_TIMEOUT_S` 可配） | 中文原因 |
    | 400 | 返回内容非 PDF（Content-Type 与文件头均不是 PDF） | 中文原因 |
    | 400 | PDF 损坏、解析失败 | `PDF 解析失败：…`；临时文件已删除 |
    | 422 | 请求体缺少 `arxiv_id` | FastAPI 校验错误 |
    | 500 | 未预期异常 | 全局异常处理器 |

    所有 400 分支都不会写入论文、切片或图谱数据；已落盘的临时文件会删除。

    ### curl 示例
    ```bash
    # 裸编号
    curl -X POST http://127.0.0.1:7860/api/papers/import \\
         -H "Content-Type: application/json" \\
         -d '{"arxiv_id": "1706.03762"}'

    # 带版本号 / arXiv: 前缀 / abs 链接
    curl -X POST http://127.0.0.1:7860/api/papers/import \\
         -H "Content-Type: application/json" \\
         -d '{"arxiv_id": "1706.03762v7"}'
    curl -X POST http://127.0.0.1:7860/api/papers/import \\
         -H "Content-Type: application/json" \\
         -d '{"arxiv_id": "arXiv:1706.03762"}'
    curl -X POST http://127.0.0.1:7860/api/papers/import \\
         -H "Content-Type: application/json" \\
         -d '{"arxiv_id": "https://arxiv.org/abs/1706.03762"}'
    ```

    成功响应示例：
    ```json
    {"paper_id": "507990b1585e", "title": "Deep Residual Learning for Image Recognition",
     "n_pages": 12, "n_chunks": 58, "duplicate": false, "dedup_score": null, "elapsed_s": 12.7}
    ```
    重复响应示例（HTTP 200，不写库）：
    ```json
    {"paper_id": "5188570ee8b3", "title": "Attention Is All You Need", "n_pages": 15,
     "n_chunks": 41, "duplicate": true, "dedup_score": 1.0,
     "existing_id": "5188570ee8b3", "existing_title": "Attention Is All You Need", "elapsed_s": 9.79}
    ```
    """
    started = time.perf_counter()
    try:
        arxiv_id = _normalize_arxiv_id(req.arxiv_id)
    except HTTPException as exc:
        _log_arxiv(
            logging.WARNING,
            arxiv_id=(req.arxiv_id or "").strip(),
            paper_id=None,
            elapsed_s=time.perf_counter() - started,
            duplicate=None,
            reason=str(exc.detail),
        )
        raise
    data = await asyncio.to_thread(_download_arxiv_pdf, arxiv_id, started)
    return await asyncio.to_thread(_ingest_arxiv_pdf, arxiv_id, data, started)


@app.get("/api/papers")
async def get_papers() -> dict[str, Any]:
    papers = store.list_papers()
    for p in papers:
        p.pop("pdf_path", None)
    return {"papers": papers, "events": store.recent_events(10)}


@app.get("/api/papers/{paper_id}")
async def get_paper_detail(paper_id: str) -> dict[str, Any]:
    paper = store.get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "文献不存在")
    paper.pop("pdf_path", None)
    paper.pop("shingles", None)
    chunks = store.all_chunks([paper_id])
    paper["chunk_preview"] = [
        {"page": c["page"], "heading": c["heading"], "text": c["text"][:400], "bbox": c["bbox"]}
        for c in sorted(chunks, key=lambda c: c["ord"])[:6]
    ]
    return paper


@app.get("/api/papers/{paper_id}/parse")
async def get_parse_json(paper_id: str) -> dict[str, Any]:
    """首屏读取的稳定解析 JSON：作者保持数组，章节继续携带 page + bbox。"""
    paper = store.get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "文献不存在")
    authors = paper.get("author_details", [])
    if not authors and paper.get("authors"):
        authors = [
            {"name": name.strip(), "markers": "", "affiliations": []}
            for name in paper["authors"].split(",")
            if name.strip()
        ]
    return {
        "paper_id": paper_id,
        "filename": paper["filename"],
        "title": paper.get("title", ""),
        "authors_raw": paper.get("authors", ""),
        "authors": _author_contract(authors),
        "affiliations": paper.get("affiliations", []),
        "year": paper.get("year", ""),
        "year_source": paper.get("year_source", ""),
        "n_pages": paper.get("n_pages", 0),
        "n_references": len(paper.get("references", [])),
        "outline": paper.get("outline", []),
        "pdf_url": f"/api/papers/{paper_id}/file",
    }


@app.get("/api/papers/{paper_id}/file")
async def get_paper_file(paper_id: str) -> FileResponse:
    paper = store.get_paper(paper_id)
    if not paper or not Path(paper["pdf_path"]).exists():
        raise HTTPException(404, "PDF 文件缺失")
    return FileResponse(paper["pdf_path"], media_type="application/pdf")


@app.get("/api/papers/{paper_id}/page/{page_no}.png")
async def get_page_png(paper_id: str, page_no: int, dpi: int = 130) -> Response:
    """渲染单页为图片，供扫描件预览与非文本层场景使用。"""
    paper = store.get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "文献不存在")
    import pymupdf as fitz

    doc = fitz.open(paper["pdf_path"])
    if page_no < 1 or page_no > doc.page_count:
        doc.close()
        raise HTTPException(400, "页码越界")
    pix = doc[page_no - 1].get_pixmap(dpi=dpi)
    doc.close()
    return Response(content=pix.tobytes("png"), media_type="image/png")


@app.delete("/api/papers/{paper_id}")
async def delete_paper(paper_id: str) -> dict[str, str]:
    store.delete_paper(paper_id)
    INDEX.invalidate()
    return {"status": "deleted"}


@app.post("/api/qa")
async def qa(req: QARequest) -> dict[str, Any]:
    return await agent.answer(req.question, req.paper_ids, req.intent)


@app.post("/api/qa/evidence")
async def qa_evidence(req: EvidenceQARequest) -> dict[str, Any]:
    return await local_qa.answer_question(req.question, req.paper_id)


@app.post("/api/qa/stream")
async def qa_stream(req: QARequest) -> StreamingResponse:
    async def gen() -> Any:
        async for event in agent.stream_answer(req.question, req.paper_ids, req.intent):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/compare")
async def compare(req: CompareRequest) -> dict[str, Any]:
    if len(req.paper_ids) < 2:
        raise HTTPException(400, "跨文对比至少选择两篇文献")
    return await agent.compare(req.aspect, req.paper_ids)


@app.post("/api/survey")
async def survey(req: FilesRequest) -> dict[str, Any]:
    return await agent.survey(req.paper_ids)


@app.post("/api/graph")
async def citation_graph(req: FilesRequest) -> dict[str, Any]:
    return await asyncio.to_thread(
        graph_mod.build_citation_graph,
        req.paper_ids or None,
        False,
    )


@app.post("/api/write")
async def write(req: WriteRequest) -> dict[str, Any]:
    return await agent.write_framework(req.idea, req.paper_ids)


@app.post("/api/translate")
async def translate(req: TranslateRequest) -> dict[str, Any]:
    return await agent.translate(req.text, req.direction, req.style)


@app.post("/api/viz")
async def viz(req: VizRequest) -> dict[str, Any]:
    return await agent.render_figure(req.csv_text, req.intent)


@app.post("/api/dedup/compare")
async def dedup_compare(id_new: str, id_old: str) -> dict[str, Any]:
    return await asyncio.to_thread(dedup.compare_versions, id_new, id_old)


@app.exception_handler(Exception)
async def unhandled(request: Any, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


# ---- 前端静态资源（构建产物存在时自动挂载，实现单进程一键部署）----
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
