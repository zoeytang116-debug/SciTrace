#!/usr/bin/env python3
"""SciTrace · arXiv 导入验收脚本。

流程：
  1. 复制 data/library.db 到临时目录，用 SCITRACE_DATA_DIR 拉起隔离后端，
     顺序执行 a–m 用例（导入、去重、统计、图谱、问答、404、超时、非 PDF、损坏 PDF、5xx）。
  2. 关闭隔离后端，切回正式库（不设 SCITRACE_DATA_DIR），执行只读回归 r-e ~ r-h，
     并比对正式库 papers/chunks/edges 指纹，证明未被篡改。
  3. 输出 markdown 报告；失败时把服务端日志保留到 scripts/verify_logs/<时间戳>/。
  4. 无论成败，自动停止服务并清理临时目录。

用法：
  .venv/Scripts/python.exe scripts/arXiv_import_verify.py
  .venv/Scripts/python.exe scripts/arXiv_import_verify.py --report docs/arXiv_import_验收报告.md
  .venv/Scripts/python.exe scripts/arXiv_import_verify.py --keep-temp   # 调试时保留临时目录
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
OFFICIAL_DB = ROOT / "data" / "library.db"
DEFAULT_REPORT = ROOT / "docs" / "arXiv_import_验收报告.md"
LOG_DIR = ROOT / "scripts" / "verify_logs"
PYTHON = sys.executable

COPY_QUESTION_AUTHOR = "这篇论文的第六作者是谁，在哪个机构工作？"
COPY_QUESTION_QUBIT = "这篇论文的量子计算实验用了多少量子比特？"
AUTHOR_PAPER_ID = "5188570ee8b3"
REFUSAL_WORDING = "文献中未提及相关内容。"

HEALTH_TIMEOUT_S = 120
IMPORT_TIMEOUT_S = 180


@dataclass
class CaseResult:
    case_id: str
    name: str
    expected: str
    actual: str
    passed: bool
    elapsed_s: float


@dataclass
class VerifyState:
    results: list[CaseResult] = field(default_factory=list)
    imported_chunks: int = 0
    a_ok: bool = False
    failure_logs: list[Path] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(not item.passed for item in self.results)


# ---------- HTTP ----------

def http_json(
    base: str,
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: float = IMPORT_TIMEOUT_S,
) -> tuple[int, dict]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"detail": body}
        return exc.code, parsed


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------- 后端进程 ----------

def start_backend(data_dir: Path | None, env_extra: dict[str, str], log_path: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    env = os.environ.copy()
    env.pop("SCITRACE_DATA_DIR", None)
    env.pop("ARXIV_PDF_BASE_URL", None)
    env.pop("ARXIV_TIMEOUT_S", None)
    env.pop("ARXIV_MAX_BYTES", None)
    env["PYTHONUNBUFFERED"] = "1"
    if data_dir is not None:
        env["SCITRACE_DATA_DIR"] = str(data_dir)
    env.update(env_extra)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            PYTHON, "-m", "uvicorn", "main:app",
            "--app-dir", str(BACKEND_DIR),
            "--host", "127.0.0.1",
            "--port", str(port),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + HEALTH_TIMEOUT_S
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"后端进程提前退出，日志：{log_path}")
        try:
            status, _ = http_json(base, "GET", "/api/health", timeout=3)
            if status == 200:
                return process, base
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    stop_backend(process)
    raise RuntimeError(f"后端 120 秒内未就绪，日志：{log_path}")


def stop_backend(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


# ---------- 故障桩服务 ----------

class StubHandler(BaseHTTPRequestHandler):
    """按末段编号返回：超时 / 非 PDF / 损坏 PDF / 5xx。"""

    def log_message(self, *args) -> None:  # noqa: D401 - 静默访问日志
        return

    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # 超时用例里客户端先断开，桩继续写会抛 WinError 10053，属预期噪声
            return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        name = self.path.rstrip("/").split("/")[-1]
        if name == "0000.0001":
            time.sleep(3)
            self._respond(200, b"%PDF-1.4 slow", "application/pdf")
        elif name == "0000.0002":
            self._respond(200, b"<html>not a pdf</html>", "text/html; charset=utf-8")
        elif name == "0000.0003":
            self._respond(200, b"%PDF-1.4\nthis file is intentionally broken", "application/pdf")
        elif name == "0000.0004":
            self._respond(500, b"stub server error", "text/plain")
        else:
            self._respond(404, b"not found", "text/plain")


def start_stub() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


# ---------- 数据指纹 ----------

def db_digest(path: Path) -> str:
    """papers + chunks + edges 的稳定指纹；不含 events（启动会追加 startup 记录）。"""
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    digest = hashlib.sha256()
    try:
        for table, order in (
            ("papers", "id"),
            ("chunks", "paper_id, ord"),
            ("edges", "source, target, kind"),
        ):
            for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                digest.update(json.dumps(dict(row), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        return digest.hexdigest()
    finally:
        connection.close()


# ---------- 用例执行 ----------

def run_case(state: VerifyState, case_id: str, name: str, expected: str, func) -> None:
    started = time.perf_counter()
    try:
        passed, actual = func()
    except Exception as exc:  # noqa: BLE001 - 用例失败需记录而非中断
        passed, actual = False, f"异常：{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    state.results.append(CaseResult(case_id, name, expected, actual, passed, elapsed))
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {case_id} {name} ({elapsed:.1f}s) — {actual}")


def summarize(body: dict) -> str:
    keys = ("paper_id", "title", "n_pages", "n_chunks", "duplicate", "dedup_score", "elapsed_s", "detail")
    picked = {key: body.get(key) for key in keys if key in body}
    text = json.dumps(picked, ensure_ascii=False)
    return text if len(text) <= 220 else text[:217] + "..."


def run_copy_suite(state: VerifyState, base: str) -> None:
    def case_a():
        status, body = http_json(base, "POST", "/api/papers/import", {"arxiv_id": "1512.03385"})
        ok = (
            status == 200
            and body.get("duplicate") is False
            and int(body.get("n_pages") or 0) > 0
            and int(body.get("n_chunks") or 0) > 0
        )
        if ok:
            state.a_ok = True
            state.imported_chunks = int(body["n_chunks"])
        return ok, f"HTTP {status} {summarize(body)}"

    run_case(state, "a", "导入库内不存在的 1512.03385", "HTTP 200 / duplicate=false / n_pages>0 / n_chunks>0", case_a)

    def case_b():
        _, before = http_json(base, "GET", "/api/health", timeout=15)
        status, body = http_json(base, "POST", "/api/papers/import", {"arxiv_id": "1512.03385"})
        _, after = http_json(base, "GET", "/api/health", timeout=15)
        score = body.get("dedup_score")
        ok = (
            status == 200
            and body.get("duplicate") is True
            and isinstance(score, (int, float)) and not isinstance(score, bool)
            and before.get("papers") == after.get("papers")
        )
        return ok, f"HTTP {status} {summarize(body)}；papers {before.get('papers')}→{after.get('papers')}"

    run_case(state, "b", "重复提交同一编号", "HTTP 200 / duplicate=true / dedup_score 为数字 / 篇数不变", case_b)

    def case_c():
        status, body = http_json(base, "POST", "/api/papers/import", {"arxiv_id": "1706.03762"})
        score = body.get("dedup_score")
        ok = (
            status == 200
            and body.get("duplicate") is True
            and isinstance(score, (int, float)) and float(score) >= 0.95
        )
        return ok, f"HTTP {status} {summarize(body)}"

    run_case(state, "c", "导入库内已有的 1706.03762", "duplicate=true / dedup_score ≥ 0.95", case_c)

    def case_d():
        _, before = http_json(base, "GET", "/api/health", timeout=15)
        status, body = http_json(base, "POST", "/api/papers/import", {"arxiv_id": "abc123"}, timeout=30)
        _, after = http_json(base, "GET", "/api/health", timeout=15)
        detail = str(body.get("detail") or "")
        ok = status == 400 and "无法识别" in detail and before.get("papers") == after.get("papers")
        return ok, f"HTTP {status} detail={detail}；papers {before.get('papers')}→{after.get('papers')}"

    run_case(state, "d", "非法编号 abc123", "HTTP 400 中文原因 / 篇数不变", case_d)

    expected_papers = 22 if state.a_ok else 21
    expected_chunks = 2672 + state.imported_chunks

    def case_e():
        status, body = http_json(base, "GET", "/api/health", timeout=15)
        ok = (
            status == 200
            and body.get("papers") == expected_papers
            and body.get("chunks") == expected_chunks
            and body.get("dedup_threshold") == 0.63
        )
        return ok, (
            f"HTTP {status} papers={body.get('papers')} chunks={body.get('chunks')} "
            f"dedup_threshold={body.get('dedup_threshold')}"
        )

    run_case(
        state, "e", "基础统计接口校验",
        f"papers={expected_papers} / chunks={expected_chunks} / dedup_threshold=0.63", case_e,
    )

    def case_f():
        status, body = http_json(base, "POST", "/api/graph", {}, timeout=120)
        nodes = body.get("nodes") or []
        edges = body.get("edges") or []
        empty_labels = [node for node in nodes if not node.get("label")]
        ok = status == 200 and len(nodes) == expected_papers and len(edges) > 0 and not empty_labels
        return ok, f"HTTP {status} nodes={len(nodes)} edges={len(edges)} 空标签={len(empty_labels)}"

    run_case(state, "f", "图谱节点与边数量校验", f"nodes={expected_papers} / edges>0 / label 非空", case_f)

    def case_g():
        status, body = http_json(
            base, "POST", "/api/qa/evidence",
            {"question": COPY_QUESTION_AUTHOR, "paper_id": AUTHOR_PAPER_ID},
        )
        wording = str(body.get("wording") or "")
        evidence = body.get("evidence") or []
        ok = (
            status == 200
            and body.get("grounded") is True
            and len(evidence) > 0
            and "[P1]" in wording
            and "Aidan N. Gomez" in wording
            and "University of Toronto" in wording
        )
        return ok, f"HTTP {status} grounded={body.get('grounded')} evidence={len(evidence)} wording={wording[:80]}"

    run_case(state, "g", "问答正向命中（第六作者）", "grounded=true / evidence 非空 / 含 Aidan N. Gomez 与 University of Toronto", case_g)

    def case_h():
        status, body = http_json(
            base, "POST", "/api/qa/evidence",
            {"question": COPY_QUESTION_QUBIT, "paper_id": AUTHOR_PAPER_ID},
        )
        ok = (
            status == 200
            and body.get("grounded") is False
            and body.get("wording") == REFUSAL_WORDING
        )
        return ok, f"HTTP {status} grounded={body.get('grounded')} wording={body.get('wording')!r}"

    run_case(state, "h", "问答拒答分支（量子比特）", f"grounded=false / wording={REFUSAL_WORDING}", case_h)

    def case_i():
        _, before = http_json(base, "GET", "/api/health", timeout=15)
        status, body = http_json(base, "POST", "/api/papers/import", {"arxiv_id": "9999.99999"}, timeout=90)
        _, after = http_json(base, "GET", "/api/health", timeout=15)
        detail = str(body.get("detail") or "")
        ok = status == 400 and "404" in detail and before.get("papers") == after.get("papers")
        return ok, f"HTTP {status} detail={detail}；papers {before.get('papers')}→{after.get('papers')}"

    run_case(state, "i", "不存在的编号 9999.99999", "HTTP 400（arXiv 404）/ 篇数不变", case_i)


def run_fault_suite(state: VerifyState, base: str, uploads_dir: Path) -> None:
    expected_papers = 22 if state.a_ok else 21

    def failure_case(case_id: str, name: str, arxiv_id: str, keyword: str, expected: str) -> None:
        def case():
            _, before = http_json(base, "GET", "/api/health", timeout=15)
            status, body = http_json(base, "POST", "/api/papers/import", {"arxiv_id": arxiv_id}, timeout=60)
            _, after = http_json(base, "GET", "/api/health", timeout=15)
            detail = str(body.get("detail") or "")
            leftovers = sorted(path.name for path in uploads_dir.glob(f"{arxiv_id}*"))
            ok = (
                status == 400
                and keyword in detail
                and before.get("papers") == after.get("papers") == expected_papers
                and not leftovers
            )
            return ok, (
                f"HTTP {status} detail={detail}；papers {before.get('papers')}→{after.get('papers')}；"
                f"临时文件残留={leftovers or '无'}"
            )

        run_case(state, case_id, name, expected, case)

    failure_case("j", "下载超时分支", "0000.0001", "超时", "HTTP 400 超时中文原因 / 篇数不变")
    failure_case("k", "返回非 PDF", "0000.0002", "不是 PDF", "HTTP 400 非 PDF 中文原因 / 篇数不变")
    failure_case("l", "PDF 损坏解析失败", "0000.0003", "损坏", "HTTP 400 PDF 损坏中文原因 / 临时文件清理 / 篇数不变")
    failure_case("m", "arXiv 返回 5xx", "0000.0004", "服务暂时不可用", "HTTP 400 5xx 中文原因 / 篇数不变")


def run_official_suite(state: VerifyState, base: str, digest_before: str) -> str:
    def regression(case_id: str, name: str, expected: str, func) -> None:
        run_case(state, f"r-{case_id}", f"正式库回归 · {name}", expected, func)

    def case_e():
        status, body = http_json(base, "GET", "/api/health", timeout=15)
        ok = (
            status == 200
            and body.get("papers") == 21
            and body.get("chunks") == 2672
            and body.get("dedup_threshold") == 0.63
        )
        return ok, (
            f"HTTP {status} papers={body.get('papers')} chunks={body.get('chunks')} "
            f"dedup_threshold={body.get('dedup_threshold')}"
        )

    regression("e", "基础统计", "papers=21 / chunks=2672 / dedup_threshold=0.63", case_e)

    def case_f():
        status, body = http_json(base, "POST", "/api/graph", {}, timeout=120)
        nodes = body.get("nodes") or []
        edges = body.get("edges") or []
        ok = status == 200 and len(nodes) == 21 and len(edges) == 67
        return ok, f"HTTP {status} nodes={len(nodes)} edges={len(edges)}"

    regression("f", "引用图谱", "nodes=21 / edges=67", case_f)

    def case_g():
        status, body = http_json(
            base, "POST", "/api/qa/evidence",
            {"question": COPY_QUESTION_AUTHOR, "paper_id": AUTHOR_PAPER_ID},
        )
        wording = str(body.get("wording") or "")
        evidence = body.get("evidence") or []
        ok = (
            status == 200
            and body.get("grounded") is True
            and len(evidence) > 0
            and "[P1]" in wording
            and "Aidan N. Gomez" in wording
            and "University of Toronto" in wording
        )
        return ok, f"HTTP {status} grounded={body.get('grounded')} evidence={len(evidence)} wording={wording[:80]}"

    regression("g", "问答正向命中", "grounded=true / evidence 非空 / 含 Aidan N. Gomez 与 University of Toronto", case_g)

    def case_h():
        status, body = http_json(
            base, "POST", "/api/qa/evidence",
            {"question": COPY_QUESTION_QUBIT, "paper_id": AUTHOR_PAPER_ID},
        )
        ok = (
            status == 200
            and body.get("grounded") is False
            and body.get("wording") == REFUSAL_WORDING
        )
        return ok, f"HTTP {status} grounded={body.get('grounded')} wording={body.get('wording')!r}"

    regression("h", "问答拒答", f"grounded=false / wording={REFUSAL_WORDING}", case_h)

    def case_digest():
        digest_after = db_digest(OFFICIAL_DB)
        ok = digest_after == digest_before
        return ok, f"papers/chunks/edges 指纹 {digest_before[:12]}→{digest_after[:12]}（一致={ok}）"

    regression("db", "正式库数据指纹（只读）", "启动前后 papers/chunks/edges 指纹一致", case_digest)
    return db_digest(OFFICIAL_DB)


# ---------- 结构化日志检查 ----------

def check_structured_logs(log_path: Path) -> tuple[bool, str]:
    if not log_path.exists():
        return False, f"日志不存在：{log_path}"
    required = {"event", "source", "arxiv_id", "paper_id", "elapsed_s", "duplicate"}
    records = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "arxiv_import":
            records.append(payload)
    complete = [item for item in records if required.issubset(item.keys())]
    reasons = [item for item in records if "reason" in item]
    ok = len(records) > 0 and len(complete) == len(records) and len(reasons) > 0
    return ok, f"结构化日志 {len(records)} 条，字段完整 {len(complete)} 条，含失败原因 {len(reasons)} 条"


# ---------- 报告 ----------

def render_report(state: VerifyState, meta: dict) -> str:
    lines = [
        "# arXiv 文献导入 · 验收报告",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- 基线：SciTrace（arXiv 导入补全版）",
        f"- 隔离副本：`SCITRACE_DATA_DIR` 指向临时目录，正式库未被导入用例写入",
        f"- 后端启动命令：`python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port <临时端口>`",
        f"- 结论：**{'全部通过' if not state.failed else '存在失败用例'}**",
        "",
        "| # | 用例 | 预期 | 实际 | 耗时 | 结果 |",
        "|---|---|---|---|---|---|",
    ]
    for item in state.results:
        actual = item.actual.replace("|", "\\|")
        lines.append(
            f"| {item.case_id} | {item.name} | {item.expected} | {actual} | "
            f"{item.elapsed_s:.1f}s | {'PASS' if item.passed else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 副本库初始为正式库快照（21 篇 / 2672 块）；用例 a 入库后为 22 篇。",
            "- j/k/l/m 使用脚本内置桩服务注入超时、非 PDF、损坏 PDF、HTTP 500，"
            "通过 `ARXIV_PDF_BASE_URL` + `ARXIV_TIMEOUT_S=1` 指向桩服务，不访问外网。",
            "- 正式库回归只执行只读用例（health / graph / qa / 数据指纹），"
            "不向正式库导入论文；服务启动仍会按原有逻辑向 events 追加 startup 记录，"
            "papers / chunks / edges 指纹保持一致。",
            "- 结构化日志：`" + meta["log_check"] + "`。",
        ]
    )
    if state.failed:
        lines.extend(
            [
                "",
                "## 失败处理",
                "",
                f"- 服务端日志已保留到：`{meta['kept_logs']}`（如无则见临时目录提示）",
            ]
        )
    lines.append("")
    return "\n".join(lines)


# ---------- 主流程 ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="SciTrace arXiv 导入验收脚本")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="markdown 报告输出路径")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时目录（调试用）")
    parser.add_argument("--skip-official", action="store_true", help="跳过正式库只读回归")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if not OFFICIAL_DB.exists():
        print(f"正式库不存在：{OFFICIAL_DB}")
        return 2

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    temp_root = Path(tempfile.mkdtemp(prefix="scitrace_arxiv_verify_"))
    copy_data = temp_root / "data"
    copy_data.mkdir(parents=True)
    shutil.copy2(OFFICIAL_DB, copy_data / "library.db")

    state = VerifyState()
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "log_check": "未检查",
        "kept_logs": "",
    }
    stub_server = None
    processes: list[subprocess.Popen] = []
    official_digest = ""

    print(f"临时目录：{temp_root}")
    try:
        # 阶段一：副本库 + 真实 arXiv
        copy_log = temp_root / "server_copy.log"
        process, base = start_backend(copy_data, {}, copy_log)
        processes.append(process)
        print(f"副本库后端就绪：{base}（库：{copy_data / 'library.db'}）")
        run_copy_suite(state, base)
        stop_backend(process)
        processes.remove(process)
        log_ok, log_text = check_structured_logs(copy_log)
        meta["log_check"] = f"{'通过' if log_ok else '未通过'}，{log_text}"
        state.results.append(
            CaseResult("log", "结构化日志字段检查", "含来源/编号/paper id/耗时/是否重复/失败原因", log_text, log_ok, 0.0)
        )
        print(f"[{'PASS' if log_ok else 'FAIL'}] log {log_text}")

        # 阶段二：副本库 + 本地故障桩（超时/非 PDF/损坏 PDF/5xx）
        stub_server, stub_base = start_stub()
        fault_log = temp_root / "server_fault.log"
        process, base = start_backend(
            copy_data,
            {"ARXIV_PDF_BASE_URL": stub_base, "ARXIV_TIMEOUT_S": "1"},
            fault_log,
        )
        processes.append(process)
        print(f"故障桩后端就绪：{base} → 桩 {stub_base}")
        run_fault_suite(state, base, copy_data / "uploads")
        stop_backend(process)
        processes.remove(process)

        # 阶段三：正式库只读回归
        if not args.skip_official:
            official_digest = db_digest(OFFICIAL_DB)
            official_log = temp_root / "server_official.log"
            process, base = start_backend(None, {}, official_log)
            processes.append(process)
            print(f"正式库后端就绪：{base}（只读回归）")
            run_official_suite(state, base, official_digest)
            stop_backend(process)
            processes.remove(process)
    except Exception as exc:  # noqa: BLE001 - 启动或网络异常也要计入失败
        state.results.append(
            CaseResult("sys", "脚本执行异常", "无异常", f"{type(exc).__name__}: {exc}", False, 0.0)
        )
        print(f"[FAIL] sys {type(exc).__name__}: {exc}")
    finally:
        for process in list(processes):
            stop_backend(process)
        if stub_server is not None:
            stub_server.shutdown()
        if state.failed:
            keep_dir = LOG_DIR / stamp
            keep_dir.mkdir(parents=True, exist_ok=True)
            for log_file in temp_root.glob("server_*.log"):
                shutil.copy2(log_file, keep_dir / log_file.name)
            meta["kept_logs"] = str(keep_dir)
        try:
            report_path = Path(args.report)
            if not report_path.is_absolute():
                report_path = ROOT / report_path
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(render_report(state, meta), encoding="utf-8")
            print(f"报告已写入：{report_path}")
        except OSError as exc:
            print(f"报告写入失败：{exc}")
        if state.failed:
            print(f"失败日志已保留：{meta['kept_logs']}")
        if args.keep_temp:
            print(f"临时目录保留：{temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    return 0 if not state.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
