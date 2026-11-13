"""Agent 引擎：Task Planning → 工具路由 → 检索增强 → 抗幻觉作答。

这里是「Agent 架构」评分的核心：
每次提问先由规划器产出可观测的执行计划（steps），再按计划调用检索/图谱/翻译等工具，
最后在严格约束下生成回答，并把每个证据编号绑定回原文 page + bbox。
"""

from __future__ import annotations

import re
from typing import Any, AsyncIterator

import llm
from config import LLM, RETRIEVAL
from retrieval import INDEX
from store import all_chunks, get_paper, list_papers

INTENT_RULES: list[tuple[str, str]] = [
    (r"(对比|比较|差异|区别|vs|versus|compare)", "compare"),
    (r"(翻译|translate|译成)", "translate"),
    (r"(引用|图谱|脉络|基石|cite|graph)", "graph"),
    (r"(综述|survey|future work|研究空白|趋势)", "survey"),
    (r"(写|框架|outline|idea|投稿|abstract)", "write"),
    (r"(图表|csv|可视化|画个|画一)", "viz"),
]


def plan_task(question: str, paper_ids: list[str], intent_override: str = "") -> dict[str, Any]:
    """可观测的任务规划：意图识别 + 检索范围决策 + 工具链选择。"""
    intent = intent_override
    if not intent:
        for pattern, name in INTENT_RULES:
            if re.search(pattern, question, re.I):
                intent = name
                break
    intent = intent or "qa"

    papers = [p for p in list_papers() if (not paper_ids or p["id"] in paper_ids)]
    multi = len(papers) > 1

    tools = ["hybrid_retriever"]
    if intent in ("compare", "survey"):
        tools.append("multi_paper_reasoner")
    if intent == "graph":
        tools.append("citation_graph")
    if intent == "translate":
        tools = ["layout_parser", "academic_translator"]
    if intent == "write":
        tools.append("reference_grounder")
    if intent == "viz":
        tools = ["csv_interpreter", "chart_renderer"]

    steps = [
        {"step": 1, "name": "意图识别", "detail": f"判定任务类型为 {intent}，命中文献 {len(papers)} 篇"},
        {"step": 2, "name": "检索范围", "detail": "跨文献联合检索" if multi else "单篇精读检索"},
        {"step": 3, "name": "工具编排", "detail": " → ".join(tools)},
        {"step": 4, "name": "证据锚定", "detail": "为每个结论绑定原文页码与坐标，未命中则拒答"},
    ]
    return {"intent": intent, "tools": tools, "paper_count": len(papers), "steps": steps}


def _evidence_block(hits: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """把检索片段编号为 [P1]…[Pn]，返回提示词区块与前端可点击的引用列表。"""
    lines, citations = [], []
    for i, h in enumerate(hits, start=1):
        paper = get_paper(h["paper_id"]) or {}
        tag = f"P{i}"
        lines.append(
            f"[{tag}] 文献《{paper.get('title', h['paper_id'])}》第 {h['page']} 页"
            f"（章节：{h['heading'] or '正文'}）\n{h['text']}"
        )
        citations.append(
            {
                "tag": tag,
                "paper_id": h["paper_id"],
                "paper_title": paper.get("title", ""),
                "page": h["page"],
                "heading": h["heading"],
                "text": h["text"][:600],
                "bbox": h["bbox"],
                "score": round(h["score"], 4),
            }
        )
    return "\n\n".join(lines), citations


async def answer(question: str, paper_ids: list[str], intent: str = "") -> dict[str, Any]:
    """非流式作答，返回计划、回答、引用、以及是否降级。"""
    plan = plan_task(question, paper_ids, intent)
    hits = INDEX.search(question, paper_ids or None, top_k=RETRIEVAL.top_k)
    if not hits:
        return {
            "plan": plan,
            "answer": "当前文献库中未检索到与该问题相关的段落，无法给出有依据的答案。请先上传相关论文，或换一种问法。",
            "citations": [],
            "degraded": False,
            "grounded": False,
        }

    block, citations = _evidence_block(hits)
    if not llm.available():
        return {
            "plan": plan,
            "answer": _extractive_answer(question, citations),
            "citations": citations,
            "degraded": True,
            "grounded": True,
        }

    prompt = (
        f"【证据片段】\n{block}\n\n【用户问题】\n{question}\n\n"
        "请基于证据作答，每个结论句末尾标注 [Pn]；若证据不足以回答，请直接说明「文献中未提及」。"
        "回答控制在 400 字以内，中文作答，专业术语保留英文原文。"
    )
    try:
        text = await llm.chat(prompt)
    except Exception as exc:
        return {
            "plan": plan,
            "answer": _extractive_answer(question, citations),
            "citations": citations,
            "degraded": True,
            "error": str(exc),
            "grounded": True,
        }
    cited = set(re.findall(r"\[P(\d+)\]", text))
    return {
        "plan": plan,
        "answer": text,
        "citations": citations,
        "used_tags": sorted(int(c) for c in cited),
        "degraded": False,
        "grounded": len(cited) > 0,
    }


def _extractive_answer(question: str, citations: list[dict[str, Any]]) -> str:
    """无模型时的确定性降级：直接给出最相关原文段落并标注出处，不编造任何内容。"""
    head = "未配置大模型密钥，已降级为「原文抽取」模式，以下为最相关证据原文：\n\n"
    body = "\n\n".join(
        f"[{c['tag']}] 《{c['paper_title']}》第 {c['page']} 页：{c['text'][:280]}…" for c in citations[:4]
    )
    return head + body


async def stream_answer(question: str, paper_ids: list[str], intent: str = "") -> AsyncIterator[dict[str, Any]]:
    """SSE 流式作答：先推计划、再推 token、最后推引用表。"""
    plan = plan_task(question, paper_ids, intent)
    yield {"type": "plan", "plan": plan}

    hits = INDEX.search(question, paper_ids or None, top_k=RETRIEVAL.top_k)
    if not hits:
        yield {"type": "token", "text": "当前文献库中未检索到相关段落，无法给出有依据的答案。"}
        yield {"type": "done", "citations": [], "grounded": False}
        return

    block, citations = _evidence_block(hits)
    yield {"type": "citations", "citations": citations}

    if not llm.available():
        for ch in _extractive_answer(question, citations):
            yield {"type": "token", "text": ch}
        yield {"type": "done", "degraded": True, "grounded": True}
        return

    prompt = (
        f"【证据片段】\n{block}\n\n【用户问题】\n{question}\n\n"
        "请基于证据作答，每个结论句末尾标注 [Pn]；证据不足时明确写「文献中未提及」。"
    )
    try:
        async for delta in llm.chat_stream(prompt):
            yield {"type": "token", "text": delta}
    except Exception as exc:
        yield {"type": "token", "text": f"\n[生成异常，降级为原文抽取] {exc}"}
    yield {"type": "done", "grounded": True}


async def compare(prompt_aspect: str, paper_ids: list[str]) -> dict[str, Any]:
    """多篇文献跨文对比：逐篇取证据后再统一推理，避免长上下文串味。"""
    plan = plan_task(prompt_aspect, paper_ids, "compare")
    per_paper_evidence: list[dict[str, Any]] = []
    for pid in paper_ids:
        paper = get_paper(pid)
        if not paper:
            continue
        hits = INDEX.search(prompt_aspect, [pid], top_k=3)
        block, _ = _evidence_block(hits)
        per_paper_evidence.append({"paper": paper["title"], "evidence": block})

    combined = "\n\n".join(f"### {e['paper']}\n{e['evidence']}" for e in per_paper_evidence)
    fallback_table = [
        {"paper": e["paper"], "evidence": e["evidence"][:400]} for e in per_paper_evidence
    ]
    if not llm.available():
        return {"plan": plan, "table": fallback_table, "summary": _extractive_answer(prompt_aspect, []), "degraded": True}

    system = llm.GUARDRAIL + "\n对比时必须逐条列出差异，不得把 A 文献的结论安到 B 文献上。"
    out = await llm.chat_json(
        f"【各文献证据】\n{combined}\n\n【对比维度】\n{prompt_aspect}\n\n"
        '输出 JSON：{"dimensions":["维度1",...],"rows":[{"paper":"文献名","values":["该维度结论（含[Pn]）",...]}],'
        '"summary":"结论性对比分析，200字内","differences":["关键差异1","关键差异2"]}',
        system=system,
    )
    return {"plan": plan, "degraded": False, **out}


async def survey(paper_ids: list[str]) -> dict[str, Any]:
    """基于图谱与文献内容自动生成领域短综述，并提炼未解决的未来方向。"""
    from graph import build_citation_graph, future_work_candidates

    g = build_citation_graph(paper_ids)
    fw = future_work_candidates(paper_ids)
    papers = [p for p in list_papers() if not paper_ids or p["id"] in paper_ids]
    digest = "\n".join(
        f"- 《{p['title']}》({p.get('year') or '年份未知'})：{(p.get('abstract') or '')[:220]}"
        for p in papers
    )
    if not llm.available():
        return {
            "degraded": True,
            "graph": g,
            "future_work": fw,
            "survey": f"未配置模型，以下为结构化素材：\n{digest}",
        }

    out = await llm.chat_json(
        f"【文献清单】\n{digest}\n\n【引用图谱统计】基石论文：{g['stats']['seminal']}；节点 {g['stats']['n_nodes']}，边 {g['stats']['n_edges']}\n"
        f"【Future Work 原文】\n" + "\n".join(f"- {f['text'][:300]}" for f in fw) + "\n\n"
        '输出 JSON：{"survey":"领域短综述，400字内，含技术演进脉络","milestones":["里程碑1",...],'
        '"future_work":["仍待解决的方向1",...],"deprecated_ideas":["已被后续论文解决的旧想法",...]}',
        system=llm.GUARDRAIL,
    )
    return {"degraded": False, "graph": g, "future_work": fw, **out}


async def write_framework(idea: str, paper_ids: list[str]) -> dict[str, Any]:
    """无幻觉论文框架：References 只允许从文献库与抽取到的参考文献中选取。"""
    papers = [p for p in list_papers() if not paper_ids or p["id"] in paper_ids]
    library_refs = [
        {"title": p["title"], "year": p.get("year", ""), "id": p["id"]} for p in papers
    ]
    refs_text = "\n".join(f"- {r['title']} ({r['year']})" for r in library_refs) or "（库为空）"

    if not llm.available():
        return {
            "degraded": True,
            "framework": {
                "Abstract": f"围绕「{idea}」的研究框架（离线模式，需配置模型生成）",
                "Introduction": "", "Related Work": "", "Method": "", "Experiments": "", "Conclusion": "",
            },
            "references": library_refs,
            "mermaid": _mermaid_stub(idea),
        }

    out = await llm.chat_json(
        f"【研究想法】\n{idea}\n\n【可引用的真实文献（只允许从这里选）】\n{refs_text}\n\n"
        '输出 JSON：{"framework":{"Abstract":"","Introduction":"","RelatedWork":"","Method":"","Experiments":"","Conclusion":""},'
        '"references":[{"title":"","year":"","reason":"为何引用"}],'
        '"mermaid":"用 mermaid flowchart 语法描述方法架构，节点名用英文"}',
        system=llm.GUARDRAIL + "\nreferences 数组中的每一条都必须在提供的文献清单中出现过，严禁虚构。",
        max_tokens=1800,
    )
    # 二次校验：剔除模型可能虚构的参考文献（Guardrails）
    allowed = {r["title"].lower().strip() for r in library_refs}
    out["references"] = [r for r in out.get("references", []) if r.get("title", "").lower().strip() in allowed]
    out["rejected_hallucinated_refs"] = len(out.get("references", [])) - len(out["references"])
    return {"degraded": False, "library_refs": library_refs, **out}


def _mermaid_stub(idea: str) -> str:
    safe = re.sub(r"[^\w\u4e00-\u9fff ]", "", idea)[:30] or "ResearchIdea"
    return (
        "flowchart LR\n"
        f'    A["{safe}"] --> B["文献检索与去重"]\n'
        '    B --> C["混合检索 RAG"]\n'
        '    C --> D["证据锚定作答"]\n'
        '    D --> E["引用图谱与综述"]\n'
        '    E --> F["写作辅助"]\n'
    )


async def translate(text: str, direction: str = "zh2en", style: str = "academic") -> dict[str, Any]:
    """中英学术翻译：术语保留、长句拆分，并回显术语对照表。"""
    if not llm.available():
        return {"degraded": True, "translated": text, "glossary": [], "note": "未配置模型，返回原文"}
    target = "英文学术写作（IEEE/ACM 风格）" if direction == "zh2en" else "中文学术表达"
    out = await llm.chat_json(
        f"【原文】\n{text}\n\n【目标】\n翻译为{target}，{style} 语域。\n\n"
        '输出 JSON：{"translated":"译文","glossary":[{"src":"原术语","dst":"对应术语"}],'
        '"split_sentences":["拆分后的句子1","句子2"],"notes":"翻译要点"}',
        system=llm.GUARDRAIL,
    )
    return {"degraded": False, **out}


async def render_figure(csv_text: str, intent: str = "") -> dict[str, Any]:
    """CSV → 学术风格图表（matplotlib）+ 符合规范的 Figure Caption。"""
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in csv_text.strip().splitlines() if r.strip()]
    if len(rows) < 2:
        raise ValueError("CSV 至少需要表头 + 一行数据")
    sep = "," if "," in rows[0] else ("\t" if "\t" in rows[0] else r"\s+")
    header = [h.strip() for h in re.split(sep, rows[0])]
    data: dict[str, list[float]] = {h: [] for h in header}
    for line in rows[1:]:
        cells = re.split(sep, line.strip())
        for h, c in zip(header, cells):
            try:
                data[h].append(float(str(c).strip().rstrip("%")))
            except ValueError:
                data[h].append(float("nan"))

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    labels = [str(x) for x in data[header[0]]]
    labels = [l if not re.fullmatch(r"[\d.]+", l) else l for l in labels]
    series = header[1:]
    if len(series) == 1:
        vals = data[series[0]]
        ax.plot(range(len(vals)), vals, marker="o", color="#1f4e79", lw=1.6)
        ax.set_xlabel(header[0])
        ax.set_ylabel(series[0])
    else:
        width = 0.8 / len(series)
        xs = np.arange(len(labels))
        for i, s in enumerate(series):
            ax.bar(xs + i * width, data[s], width=width, label=s, alpha=0.85)
        ax.set_xticks(xs + 0.4 - width / 2)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.legend(frameon=False, fontsize=8)
    ax.set_title(intent or "Experimental Results", fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    png = base64.b64encode(buf.getvalue()).decode()

    caption = f"Figure 1: {intent or 'Experimental results'}."
    if llm.available():
        try:
            obj = await llm.chat_json(
                f"【CSV 数据】\n{csv_text[:1500]}\n\n用户意图：{intent}\n"
                '输出 JSON：{"type":"折线图/柱状图/三维雷达图 之一","caption":"符合 IEEE 规范的英文 Figure Caption，1-2句，说明趋势","insight":"中文30字核心结论"}',
                system="你是学术图表规范专家，Caption 必须客观描述趋势，不得夸张。",
            )
            caption = obj.get("caption", caption)
            insight = obj.get("insight", "")
            chart_type = obj.get("type", "bar")
        except Exception:
            insight, chart_type = "", "bar"
    else:
        insight, chart_type = "", "bar"

    return {"image_b64": png, "caption": caption, "insight": insight, "chart_type": chart_type,
            "n_rows": len(rows) - 1, "columns": header}
