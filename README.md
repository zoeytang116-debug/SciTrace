# SciTrace · 文溯 —— 可溯源的科研助手 Agent

> **每一句回答都能点回原文那一页那一行。**
>
> 2026 传智杯 AI 大模型创新应用挑战赛 · 科研助手 Agent · B 组 · 个人参赛

**在线体验**：https://www.modelscope.cn/studios/zoey11677/SciTrace
**直接访问**：https://zoey11677-scitrace.ms.show

---

## 这是什么

大模型读论文最严重的问题不是"答不准"，而是**答错了你无法察觉**。一句编造的引文就足以毁掉一段综述的可信度。

SciTrace 把"可验证"做成硬约束：

- 答案里每个结论都挂 `[Pn]` 锚点，点击跳到原 PDF 那一页，并按 bbox 画出原文矩形高亮
- 参考文献只能从库内已有条目里挑，未命中即剔除，界面显示"已拒绝 N 条"
- 检索不到证据就明确输出"文献中未提及"，同时列出已检索到的候选片段

## 五个能跑通的能力

| 能力 | 做法 | 实测 |
|---|---|---|
| PDF 细粒度解析 | 字号 + 编号正则双判据建标题树；作者按上标切分、脚注回填机构；参考文献按编号切段 | 39 份官方 PDF，平均 5.94 s/份；v7 一级 7 / 三级 15 / 参考文献 40；八作者三机构全对 |
| 抗幻觉问答 | 本地证据检索 + LLM 只负责措辞；证据不足拒答 | 官方 9 题：9/9 有证据、6/9 完整答对带页码、3/9 正确拒答、0 崩溃，1.1–2.2 s |
| 语义去重 | SimHash + MinHash + 标题相似三信号投票 | 正样本 0.8490 / 负样本最高 0.4091 / 阈值 0.63 / 负样本误判 0-30 |
| 引用图谱 | 引用关系抽取 → PageRank → 基石/边缘论文 → 自动综述 | 21 节点 67 边，时间倒错 0；Layer Norm PageRank 0.179677 居首；Transformer 被引 9 排第 2 |
| 阅读 / 翻译 / 写作 / 可视化 | 三栏阅读器、章节树跳转、引用双向跳转、学术体翻译、CSV 出图 | 翻译 1.4 s、写作 2.7 s、综述 4.0 s |

## 架构

```
浏览器（单文件前端，无构建）
   │ HTTP/JSON
   ▼
FastAPI（backend/main.py）
   ├─ parsing.py    PyMuPDF → 页码 + bbox + 标题树 + 作者机构 + 参考文献
   ├─ store.py      SQLite（papers / chunks / edges）
   ├─ retrieval.py  jieba 分词 + 关键词召回 + RRF 融合
   ├─ local_qa.py   证据裁剪；不足则拒答
   ├─ llm.py        DeepSeek（OpenAI 兼容）—— 只把证据写成自然语言
   ├─ graph.py      networkx + PageRank
   └─ dedup.py      SimHash / MinHash / 标题相似，三信号投票
   │
   ▼
DeepSeek API（仅措辞，不参与事实判定）
```

**关键设计：裁决与表达分离。** 页码、bbox、参考文献全部由本地代码给出，模型拿到的输入里没有第二个页码可选——从结构上消灭"幻觉页码"。

## 本地安装

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip.exe
cp .env.example .env                           # 填入 DEEPSEEK_API_KEY
.venv/bin/python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 7860
```

打开 http://127.0.0.1:7860

### 环境变量

| 变量 | 含义 |
|---|---|
| `DEEPSEEK_API_KEY` | 措辞层密钥（线上走平台 Secrets，不写进仓库） |
| `LLM_BASE_URL` | OpenAI 兼容端点，默认 `https://api.deepseek.com/v1` |
| `LLM_MODEL` | 模型名，默认 `deepseek-chat` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 切块粒度，默认 820 / 160 |
| `TOP_K` / `RRF_K` | 召回条数与 RRF 常数，默认 8 / 60 |
| `DEDUP_THRESHOLD` | 去重阈值，实测夹定为 0.63 |

不填 key 也能跑：`llm_enabled=false`，问答退回本地模板措辞（`degraded=true`），功能不中断。

## Docker 部署

```bash
docker build -t scitrace .
docker run -p 7860:7860 -e DEEPSEEK_API_KEY=sk-xxx scitrace
```

镜像基于 `python:3.13-slim`，非 root 用户运行，监听 `0.0.0.0:7860`；`.env` 不在镜像内（`.dockerignore` 已排除）。

**线上（魔搭创空间 / Docker 类型）的三个硬约束**：端口必须 `7860`；单文件超 10 MiB 必须走 Git LFS；`data/seed/` 在启动时自愈，绕过免费实例"重启丢数据"。

## 主要接口

| 端点 | 用途 |
|---|---|
| `GET /api/health` | 状态：`llm_enabled / papers / chunks / dedup_threshold` |
| `POST /api/papers/upload` | 上传 PDF（含去重判定） |
| `GET /api/papers` · `GET /api/papers/{id}` | 文献列表与详情（作者机构、章节树、参考文献） |
| `GET /api/papers/{id}/page/{n}.png` | 渲染原页（bbox 高亮的底图） |
| `POST /api/qa` | 多文问答，返回 `answer + citations(page, bbox, tag)` |
| `POST /api/qa/evidence` | 单题完整证据链（`grounded / refusal / candidates / tried_queries`） |
| `POST /api/compare` | 同源版本对比（章节级 + 参考文献条目级） |
| `POST /api/dedup/compare` | 去重相似度明细（三信号分值） |
| `POST /api/graph` · `POST /api/survey` | 引用图谱 PageRank / 自动综述 |
| `POST /api/translate` | 学术体中英互译 + 术语对照 |
| `POST /api/write` | idea → 写作框架（引用只取库内） |
| `POST /api/viz` | CSV → 图表 + 英文 Caption |

## 实测数据

- 测试语料：`pdf/` 39 份官方 PDF（含 1 份扫描件、1 份 541 页长文档），**不使用任何自造 PDF 作为验收证据**
- 逐份指标：`docs/parse_metrics.tsv`
- 修前/修后记录与官方 9 题明细：`docs/03-测试记录.md`

## 目录

```
backend/          11 个 Python 模块（解析/存储/检索/问答/措辞/图谱/去重/接口）
frontend/dist/    单文件前端（index.html，63.9 KB，无构建）
data/seed/        启动自愈快照（library.db + 21 篇 PDF）
docs/             测试记录与逐份指标
pdf/              39 份官方测试数据
Dockerfile · requirements.txt · .env.example
```

## 团队信息

- **唐丽晴**（单人参赛）· 山西科技学院 · 计算机科学与技术 · B 组
- 项目角色：选题、规格与验收标准制定、文档与演示脚本、部署验收；代码实现由 AI 编程助手（Codex + DeepSeek）完成，AI 协同贡献说明见技术文档第 7 章

## 数据与合规

- 测试语料为公开论文（arXiv / ACL / ICLR 等），仅用于比赛评测
- 密钥仅通过环境变量与平台 Secrets 注入，不进入仓库、镜像、截图
- 界面所有指标均可由仓库脚本复跑

## License

Apache-2.0
