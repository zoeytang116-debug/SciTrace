"""局部引用图谱挖掘：引用边构建 → PageRank 基石论文识别 → 前沿综述与 Future Work。

图论依据：
- 基石论文（seminal）= 被引多且被高权重文献引用，用 PageRank 体现「引用含金量」；
- 边缘衍生论文 = 出度高、入度低，跟随基石论文派生的应用型工作。
"""

from __future__ import annotations

import re
import unicodedata
from csv import DictReader
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from parsing import _norm
from store import list_papers, replace_edges


def _inventory_titles() -> dict[str, str]:
    path = Path(__file__).resolve().parent.parent / "data" / "pdf_inventory.tsv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["file"]: row.get("title", "").strip()
            for row in DictReader(handle, delimiter="\t")
            if row.get("file")
        }


def _display_title(paper: dict[str, Any], inventory: dict[str, str]) -> str:
    filename = paper.get("filename", "")
    title = str(paper.get("title", "")).strip()
    if filename in inventory and inventory[filename]:
        return inventory[filename]
    stem = Path(filename).stem
    for known, known_title in inventory.items():
        if known_title and stem.startswith(Path(known).stem):
            return known_title
    if title and not re.match(r"^arxiv:", title, re.I):
        return title
    return stem or paper["id"]


def _title_index(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    inventory = _inventory_titles()
    for paper in papers:
        display = _display_title(paper, inventory)
        aliases = {
            _norm(paper.get("title", "")),
            _norm(display),
        }
        result.append(
            {
                "id": paper["id"],
                "label": display,
                "year": int(paper["year"]) if str(paper.get("year", "")).isdigit() else None,
                "aliases": {alias for alias in aliases if alias},
            }
        )
    return result


def build_citation_graph(paper_ids: list[str] | None = None, include_external: bool = True) -> dict[str, Any]:
    """构建局部引用网络，并对节点做 PageRank 与度中心性评估。"""
    papers = list_papers()
    if paper_ids:
        papers = [p for p in papers if p["id"] in paper_ids]
    all_papers = list_papers()
    title_candidates = _title_index(all_papers)
    inventory = _inventory_titles()

    graph = nx.DiGraph()
    for p in papers:
        graph.add_node(
            p["id"],
            label=_display_title(p, inventory),
            kind="library",
            year=p.get("year", ""),
            pages=p["n_pages"],
        )

    edges: set[tuple[str, str, str]] = set()
    for p in papers:
        source_year = int(p["year"]) if str(p.get("year", "")).isdigit() else None
        for ref in p.get("references", []):
            target = _match_reference(ref, source_year, title_candidates)
            if target and target != p["id"]:
                edges.add((p["id"], target, "library"))
                graph.add_edge(p["id"], target)
            elif include_external:
                ext_id = f"ext:{_ref_title(ref)[:80]}"
                if ext_id == "ext:":
                    continue
                if ext_id not in graph:
                    graph.add_node(
                        ext_id,
                        label=_ref_title(ref)[:70],
                        kind="external",
                        year=ref.get("year", ""),
                        arxiv=ref.get("arxiv", ""),
                    )
                    graph.add_edge(p["id"], ext_id)
                    edges.add((p["id"], ext_id, "external"))

    stats: dict[str, Any] = {"n_nodes": graph.number_of_nodes(), "n_edges": graph.number_of_edges()}
    ranks: dict[str, float] = {}
    if graph.number_of_nodes():
        ranks = nx.pagerank(graph, alpha=0.85) if graph.number_of_edges() else {}

    nodes = []
    for node, data in graph.nodes(data=True):
        indeg = graph.in_degree(node)
        outdeg = graph.out_degree(node)
        rank = ranks.get(node, 0.0)
        role = "library"
        if data["kind"] == "library":
            role = "seminal" if indeg >= 1 and rank > 0 else ("frontier" if outdeg > 0 else "isolated")
        nodes.append(
            {
                "id": node,
                "label": data.get("label", node),
                "kind": data["kind"],
                "role": role,
                "year": data.get("year", ""),
                "in_degree": indeg,
                "out_degree": outdeg,
                "pagerank": round(rank, 6),
                "arxiv": data.get("arxiv", ""),
            }
        )
    nodes.sort(key=lambda n: (-n["pagerank"], -n["in_degree"]))
    library_nodes = [node for node in nodes if node["kind"] == "library"]
    top_seminal = library_nodes[:2]
    edge_papers = sorted(
        library_nodes,
        key=lambda node: (node["in_degree"], node["pagerank"], node["out_degree"]),
    )[:3]
    top_ids = {node["id"] for node in top_seminal}
    edge_ids = {node["id"] for node in edge_papers}
    for node in nodes:
        if node["kind"] != "library":
            node["role"] = "external"
        elif node["id"] in top_ids:
            node["role"] = "seminal"
        elif node["id"] in edge_ids:
            node["role"] = "edge"
        elif node["out_degree"] > 0:
            node["role"] = "frontier"
        else:
            node["role"] = "isolated"

    replace_edges(edges)
    return {
        "nodes": nodes,
        "edges": [{"source": s, "target": t, "kind": k} for s, t, k in sorted(edges)],
        "stats": {
            **stats,
            "density": round(nx.density(graph), 4) if graph.number_of_nodes() > 1 else 0.0,
            "seminal": [n["label"] for n in top_seminal],
            "top_seminal": top_seminal,
            "edge_papers": edge_papers,
        },
    }


def _ref_title(ref: dict[str, str]) -> str:
    raw = unicodedata.normalize("NFKC", str(ref.get("raw", "")))
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
        title = parts[-1] if parts else str(ref.get("title", ""))
    title = re.sub(r"(\w)-\s+(\w)", r"\1\2", title)
    return re.sub(r"\s+", " ", title).strip(" .")


def _match_reference(
    ref: dict[str, str],
    source_year: int | None,
    candidates: list[dict[str, Any]],
) -> str | None:
    """只在参考文献列表中按标题词元匹配，并施加引用时间约束。"""
    title = _ref_title(ref)
    norm_ref = _norm(title)
    words_ref = set(re.findall(r"[a-z0-9]+", title.lower()))
    if not norm_ref or source_year is None:
        return None
    for candidate in candidates:
        target_year = candidate["year"]
        if target_year is None or target_year > source_year:
            continue
        if norm_ref in candidate["aliases"]:
            return candidate["id"]
        for alias in candidate["aliases"]:
            words_title = set(re.findall(r"[a-z0-9]+", alias))
            if len(words_ref) < 4 or len(words_title) < 4:
                continue
            overlap = len(words_ref & words_title) / max(1, len(words_ref | words_title))
            if overlap >= 0.60:
                return candidate["id"]
    return None


def future_work_candidates(paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """从多篇文献的 Future Work / Limitation 段落抽取仍待解决的开放问题。"""
    from store import all_chunks

    chunks = all_chunks(paper_ids)
    picks = [
        c for c in chunks
        if re.search(r"(future work|future research|limitation|open problem|we leave)", c["text"], re.I)
    ]
    return [
        {
            "paper_id": c["paper_id"],
            "page": c["page"],
            "heading": c["heading"],
            "text": c["text"][:900],
            "bbox": c["bbox"],
        }
        for c in picks[:8]
    ]


def dedupe_ideas(recent: list[dict[str, Any]], older: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """用时间线过滤「已被后续论文解决的旧 idea」：旧文献提出的问题若在新文献中被回应则标记 deprecated。"""
    deprecated = []
    recent_text = " ".join(r["text"].lower() for r in recent)
    for item in older:
        key_terms = set(re.findall(r"[a-z]{6,}", item["text"].lower())) & {
            "limitation", "future", "unsolved", "challenge", "lacking", "expensive", "bottleneck",
        }
        if key_terms and any(t in recent_text for t in ("solve", "address", "improve", "outperform")):
            deprecated.append({**item, "status": "已被后续工作覆盖", "checked_at": datetime.now(timezone.utc).isoformat()})
    return deprecated
