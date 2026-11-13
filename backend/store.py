"""SQLite 持久层：论文元数据、章节树、切片、引用边、图谱缓存。

使用标准库 sqlite3，零外部依赖，冷启动即可读写，满足评审环境的可复现性要求。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import DATA_DIR

DB_PATH = DATA_DIR / "library.db"
_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    title         TEXT,
    authors       TEXT,
    author_details_json TEXT,
    affiliations_json    TEXT,
    year          TEXT,
    abstract      TEXT,
    n_pages       INTEGER,
    n_chunks      INTEGER,
    is_scanned    INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'ready',
    fingerprint   TEXT,
    shingles      TEXT,
    outline       TEXT,
    refs_json     TEXT,
    pdf_path      TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id   TEXT NOT NULL,
    page       INTEGER,
    heading    TEXT,
    text       TEXT NOT NULL,
    bbox       TEXT,
    ord        INTEGER
);

CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    kind   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id   TEXT,
    action     TEXT,
    detail     TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    with _LOCK:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(papers)")}
        if "author_details_json" not in columns:
            conn.execute("ALTER TABLE papers ADD COLUMN author_details_json TEXT")
        if "affiliations_json" not in columns:
            conn.execute("ALTER TABLE papers ADD COLUMN affiliations_json TEXT")


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(r) for r in cur.fetchall()]


def insert_paper(meta: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO papers
               (id, filename, title, authors, author_details_json, affiliations_json,
                year, abstract, n_pages, n_chunks,
                is_scanned, status, fingerprint, shingles, outline, refs_json, pdf_path)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                meta["id"], meta["filename"], meta.get("title", ""), meta.get("authors", ""),
                json.dumps(meta.get("author_details", []), ensure_ascii=False),
                json.dumps(meta.get("affiliations", []), ensure_ascii=False),
                meta.get("year", ""), meta.get("abstract", ""), meta.get("n_pages", 0),
                len(chunks), int(meta.get("is_scanned", False)), meta.get("status", "ready"),
                meta.get("fingerprint", ""), json.dumps(meta.get("shingles", [])),
                json.dumps(meta.get("outline", []), ensure_ascii=False),
                json.dumps(meta.get("references", []), ensure_ascii=False),
                meta.get("pdf_path", ""),
            ),
        )
        conn.execute("DELETE FROM chunks WHERE paper_id = ?", (meta["id"],))
        conn.executemany(
            "INSERT INTO chunks (paper_id, page, heading, text, bbox, ord) VALUES (?,?,?,?,?,?)",
            [
                (c["paper_id"], c["page"], c["heading"], c["text"], json.dumps(c["bbox"]), c["ord"])
                for c in chunks
            ],
        )


def list_papers() -> list[dict[str, Any]]:
    with connect() as conn:
        papers = _rows(conn.execute("SELECT * FROM papers ORDER BY created_at DESC"))
    for p in papers:
        p["outline"] = json.loads(p["outline"] or "[]")
        p["references"] = json.loads(p.pop("refs_json", None) or "[]")
        p["author_details"] = json.loads(p.pop("author_details_json", None) or "[]")
        p["affiliations"] = json.loads(p.pop("affiliations_json", None) or "[]")
        p.pop("shingles", None)
        p["n_references"] = len(p["references"])
    return papers


def get_paper(paper_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    if not row:
        return None
    paper = dict(row)
    paper["outline"] = json.loads(paper["outline"] or "[]")
    paper["references"] = json.loads(paper.pop("refs_json", None) or "[]")
    paper["author_details"] = json.loads(paper.pop("author_details_json", None) or "[]")
    paper["affiliations"] = json.loads(paper.pop("affiliations_json", None) or "[]")
    paper["shingles"] = json.loads(paper["shingles"] or "[]")
    return paper


def all_chunks(paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM chunks"
    params: tuple = ()
    if paper_ids:
        placeholders = ",".join("?" for _ in paper_ids)
        sql += f" WHERE paper_id IN ({placeholders})"
        params = tuple(paper_ids)
    with connect() as conn:
        rows = _rows(conn.execute(sql, params))
    for r in rows:
        r["bbox"] = json.loads(r["bbox"] or "[]")
    return rows


def delete_paper(paper_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
        conn.execute("DELETE FROM edges WHERE source = ? OR target = ?", (paper_id, paper_id))


def replace_edges(edges: list[tuple[str, str, str]]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM edges")
        conn.executemany("INSERT INTO edges (source, target, kind) VALUES (?,?,?)", edges)


def load_edges() -> list[dict[str, str]]:
    with connect() as conn:
        return _rows(conn.execute("SELECT * FROM edges"))


def log_event(paper_id: str, action: str, detail: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO events (paper_id, action, detail) VALUES (?,?,?)",
            (paper_id, action, detail),
        )


def recent_events(limit: int = 20) -> list[dict[str, Any]]:
    with connect() as conn:
        return _rows(conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)))
