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
import json
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
import parsing
import store
from config import LLM, UPLOAD_DIR
from retrieval import INDEX


DEMO_FILENAME = "1706.03762v7.pdf"
DEMO_ID = "demo-v7"
DEMO_SOURCE = Path(__file__).resolve().parent.parent / "pdf" / DEMO_FILENAME
SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"

ARXIV_ID_PATTERN = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?$"
)
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
# arXiv 拒绝无 User-Agent 的下载请求。
ARXIV_USER_AGENT = "SciTrace/1.0 (arXiv importer; +https://github.com/zoeytang116-debug/SciTrace)"
ARXIV_TIMEOUT_S = 30
MAX_ARXIV_BYTES = 80 * 1024 * 1024


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
        raise HTTPException(400, "arXiv 编号不能为空")
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+)", value, re.IGNORECASE)
    if match:
        value = match.group(1)
    value = re.sub(r"^arxiv:\s*", "", value.strip(), flags=re.IGNORECASE)
    value = re.sub(r"\.pdf$", "", value.strip(), flags=re.IGNORECASE).strip().rstrip("/")
    if not ARXIV_ID_PATTERN.match(value):
        raise HTTPException(
            400,
            f"无法识别的 arXiv 编号「{raw.strip()}」，应形如 1706.03762、1706.03762v7 "
            "或 https://arxiv.org/abs/1706.03762",
        )
    return value


def _download_arxiv_pdf(arxiv_id: str) -> bytes:
    request = urllib.request.Request(
        ARXIV_PDF_URL.format(arxiv_id=arxiv_id),
        headers={"User-Agent": ARXIV_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=ARXIV_TIMEOUT_S) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read(MAX_ARXIV_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise HTTPException(400, f"arXiv 上找不到编号 {arxiv_id}（HTTP 404）") from exc
        raise HTTPException(400, f"下载 arXiv PDF 失败：HTTP {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise HTTPException(400, f"下载 arXiv PDF 失败：{reason}") from exc

    if len(data) > MAX_ARXIV_BYTES:
        raise HTTPException(400, "arXiv PDF 超过 80MB，已拒绝入库")
    if not data.startswith(b"%PDF") and "pdf" not in content_type.lower():
        raise HTTPException(
            400,
            f"下载内容不是 PDF（Content-Type: {content_type or '未知'}），"
            "可能编号不存在或 arXiv 返回了错误页",
        )
    return data


def _ingest_arxiv_pdf(arxiv_id: str, data: bytes, started: float) -> dict[str, Any]:
    """复用上传链路：落盘 → 解析 → 去重判定 → 入库。"""
    path = parsing.save_upload(f"{arxiv_id}.pdf", data)
    paper_id = uuid.uuid4().hex[:12]
    try:
        parsed = parsing.parse_pdf(path, paper_id)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"PDF 解析失败：{exc}") from exc

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
        path.unlink(missing_ok=True)
        top = duplicates[0]
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
    """按 arXiv 编号下载 PDF 并走既有解析、去重、入库链路。"""
    started = time.perf_counter()
    arxiv_id = _normalize_arxiv_id(req.arxiv_id)
    data = await asyncio.to_thread(_download_arxiv_pdf, arxiv_id)
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
