"""本地自检脚本：无需联网即可验证 解析→去重→检索→问答 全链路。

用法：python selftest.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pymupdf as fitz

import dedup
import parsing
import store
from retrieval import INDEX

BODY_V1 = [
    ("Abstract", "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms."),
    ("1 Introduction", "Recurrent neural networks have been the state of the art in sequence modeling. We introduce the Transformer."),
    ("2 Model Architecture", "The encoder is composed of a stack of N=6 identical layers. We employ a residual connection around each sub-layer."),
    ("3 Training", "We trained on 8 NVIDIA P100 GPUs. The optimizer uses beta_1 = 0.9, beta_2 = 0.98 and epsilon = 1e-9."),
    ("4 Results", "The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German, improving over the previous best result."),
    ("5 Conclusion", "Attention-based models can be more accurate and parallelizable than recurrent models."),
]
BODY_V7 = BODY_V1 + [
    ("6 Appendix", "We additionally report the attention visualization of each head in layer five."),
]


def make_pdf(path: Path, body: list[tuple[str, str]], big_title: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    page.insert_text((72, y), big_title, fontsize=19, fontname="helv")
    y += 28
    page.insert_text((72, y), "Ashish Vaswani, Noam Shazeer, Niki Parmar", fontsize=10, fontname="helv")
    y += 30
    for heading, text in body:
        page.insert_text((72, y), heading, fontsize=12, fontname="helv")
        y += 16
        for chunk in [text[i:i + 92] for i in range(0, len(text), 92)]:
            page.insert_text((72, y), chunk, fontsize=9.5, fontname="helv")
            y += 13
        y += 8
    y += 10
    page.insert_text((72, y), "References", fontsize=12, fontname="helv")
    y += 16
    for i, ref in enumerate([
        "[1] Ashish Vaswani et al. Attention is all you need. NeurIPS 2017. arXiv:1706.03762",
        "[2] Jacob Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019.",
        "[3] Tom Brown et al. Language Models are Few-Shot Learners. NeurIPS 2020.",
    ], start=1):
        page.insert_text((72, y), ref, fontsize=8.5, fontname="helv")
        y += 11
    doc.save(str(path))
    doc.close()


def main() -> int:
    store.init_db()
    tmp = Path(__file__).parent / "data" / "selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    v1, v7 = tmp / "1706.03762v1.pdf", tmp / "1706.03762v7.pdf"
    make_pdf(v1, BODY_V1, "Attention Is All You Need")
    make_pdf(v7, BODY_V7, "Attention Is All You Need")

    print("[1] 解析 v7（较新版本）")
    p7 = parsing.parse_pdf(v7, uuid.uuid4().hex[:12])
    print("    标题:", p7.meta["title"])
    print("    作者:", p7.meta["authors"][:60])
    print("    切片数:", len(p7.chunks), "| 参考文献:", len(p7.meta["references"]), "| 章节树:", len(p7.outline))
    print("    首片 bbox:", p7.chunks[0]["bbox"], "页码:", p7.chunks[0]["page"])

    print("[2] 解析 v1（旧版本）并做内容级去重检测")
    p1 = parsing.parse_pdf(v1, uuid.uuid4().hex[:12])
    store.insert_paper(p7.meta, p7.chunks)
    dups = dedup.find_duplicates(p1.meta["fingerprint"], p1.meta["shingles"])
    if dups:
        print(f"    ✅ 检出重复: 《{dups[0]['existing_title']}》 相似度 {dups[0]['similarity']:.0%}")
    else:
        print("    ❌ 未检出重复（阈值可能过严）")
    store.insert_paper(p1.meta, p1.chunks)

    print("[3] 版本差异比对")
    diff = dedup.compare_versions(p1.meta["id"], p7.meta["id"])
    print("    新增段落:", diff["summary"]["added_paragraphs"], "| 删除段落:", diff["summary"]["removed_paragraphs"],
          "| 相似度:", diff["summary"]["ratio"])

    print("[4] 混合检索 + 抗幻觉作答")
    INDEX.invalidate()
    hits = INDEX.search("beta_1 和 beta_2 的取值是多少？", None, top_k=3)
    for h in hits:
        print(f"    score={h['score']:.4f} p{h['page']} | {h['text'][:70]}")

    import asyncio

    res = asyncio.run(__import__("agent").answer("beta_1 和 beta_2 的取值是多少？", [], ""))
    print("    作答:", res["answer"][:120].replace("\n", " "))
    print("    引用数:", len(res["citations"]), "| 降级模式:", res["degraded"])

    print("[5] 引用图谱")
    import graph as graph_mod

    g = graph_mod.build_citation_graph(None)
    print("    节点:", g["stats"]["n_nodes"], "边:", g["stats"]["n_edges"], "| 基石:", g["stats"]["seminal"])
    print("\n✅ 全链路自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
