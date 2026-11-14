# Codex 任务卡 · 问答措辞与多篇检索修复

> 立卡人：豆包（规格/验收）。立卡时间 2026-11-14。
> 前提：GPT-2 论文已由豆包入库（`paper_id = a547f6a9fdde`，24 页 / 92 块），库现为 **21 篇 / 2672 块**。
> 回滚点：前端与后端当前 HEAD（`f065165` 之后的最新提交）。

## 一、三个必须修的缺陷（按严重度排序）

### 缺陷 A（严重，会毁掉演示）：`/api/qa/evidence` 有证据却输出"文献中未提及"

复现（本地 `http://127.0.0.1:7860`，两问法都复现）：

```
POST /api/qa/evidence
{"question":"这篇论文的第六作者是谁，在哪个机构工作？","paper_id":"5188570ee8b3"}
→ HTTP 200, grounded=True, refusal=False, evidence=1 条（tag=P1, page=1, bbox=[244.6,283.5,330.8,306.0], text="Aidan N. Gomez / University of Toronto"）
→ 但 wording = "文献中未提及 [P1]。"        ← 错

{"question":"第 6 作者是谁？","paper_id":"5188570ee8b3"}
→ wording = "文献中未提及 [P1]。"            ← 错

{"question":"第六作者 Aidan N. Gomez 所属机构是什么？","paper_id":"5188570ee8b3"}
→ wording = "Aidan N. Gomez 所属机构是 University of Toronto [P1]。"  ← 对
```

对照：同一个问题走 `POST /api/qa`（`paper_ids":["5188570ee8b3"]`）时答案是**正确**的：
`根据文献作者列表，第六作者为 Aidan N. Gomez [P1]。其所属机构为 University of Toronto [P1]。`

**要求**：定位 `backend/local_qa.py`（或 `agent.py`）里"证据是否足以回答"的判定分支，找出为什么"谁/哪位"类问句在 evidence 非空时被判成未提及。**不许把"文献中未提及"整段硬编码兜底**，也不许靠调大阈值糊过去。

**验收**：上面两种错问法都返回含 `Aidan N. Gomez` 与 `University of Toronto` 的措辞，且带 `[P1]`；`grounded=True`。

### 缺陷 B（中）：拒答时措辞为空，前端漏出内部文案

```
{"question":"这篇论文的量子计算实验用了多少量子比特？","paper_id":"5188570ee8b3"}
→ refusal=True, grounded=False, evidence=0, wording=""（空）
→ 前端 `result.wording || "未配置模型，仅展示证据"` 于是把"未配置模型"显示给用户 ← 错
```

**要求**：拒答时后端必须返回自己的拒答措辞（例如"文献中未提及相关内容"），前端兜底文案改成中性说法（如"没有找到证据"），不得把"未配置模型"这类内部状态漏给评委。

**验收**：上述拒答题的 `wording` 非空且不含"未配置模型"。

### 缺陷 C（中，影响官方 3 道多篇题）：多篇问答证据配额不均，漏召回关键片段

现象（`POST /api/qa`，`paper_ids = [GPT-1 `7db1bbeeb4a6`, GPT-2 `a547f6a9fdde`, GPT-3 `8d0ecd72aff1`]`）：

- M2「对比 GPT-1/2/3 参数变化」→ 只答出 GPT-3 的 8 档（125M–175B），说 **GPT-2 参数未提及**。但 GPT-2 论文 p1 明写 `a 1.5B parameter Transformer`，就在库里。
- M4「GPT-1 到 GPT-3 上下文长度」→ 只说 GPT-3 = 2048，说 GPT-1/GPT-2 未提及。实际：GPT-2 p4 `increase the context size from 512 to 1024 tokens`；GPT-1 p5 `sequences of 512 tokens`。

根因判断：多篇问答时 `TOP_K=8` 的席位被 GPT-3（篇幅最长）占满，另两篇各只进一两条边缘片段。

**要求**：多篇问答的检索改为**按论文配额**（每篇至少给 N 条，再按分数补满总席位），并保证"数值型片段"（含数字/单位/billion/tokens 等）在配额内优先。不得改单篇检索的行为与去重/图谱逻辑。

**验收（必须贴真实输出）**：
- M2 的答案同时出现三篇的参数信息，且至少引到 GPT-2 的 `1.5B`（p1 附近）。
- M4 的答案给出 `512`（GPT-1）、`1024`（GPT-2）、`2048`（GPT-3），三者都有 `[Pn]` 引用。
- 若某篇确实没有该信息，必须明确说明"该篇未报告"，不得含糊。

## 二、不许碰的（回归红线）

| 项 | 现值 |
|---|---|
| `GET /api/health` | papers=21, chunks=2672, llm_enabled=true, dedup_threshold=0.63 |
| `GET /api/papers` | 21 篇 |
| `POST /api/graph {}` | 20 节点 64 边（注：图谱只统计有引用关系的 20 篇，加 GPT-2 后若变 21 节点需说明原因） |
| `POST /api/dedup/compare` | v7 vs v1 → 0.8490，votes=3 |
| 官方单篇 Q1–Q5 | 全部 grounded=true 且带 page+bbox |

## 三、提交方式

一条提交，message 写清改了什么 + 怎么验证：

```
问答: 修复有证据却判未提及 / 拒答措辞为空 / 多篇按论文配额召回
| qa/evidence Q1 正确, refusal wording 非空, M2 含 1.5B, M4 含 512/1024/2048, graph 20/64 不变
```

做完把 6 条验收输出贴回来，我复核。
