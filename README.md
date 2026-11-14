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
| `SCITRACE_DATA_DIR` | 数据目录覆盖（库 / 上传件 / 索引），验收副本库时使用 |
| `ARXIV_TIMEOUT_S` | arXiv 下载超时秒数，默认 30 |
| `ARXIV_PDF_BASE_URL` | arXiv PDF 下载地址，默认 `https://arxiv.org/pdf`（验收桩服务可覆盖） |
| `ARXIV_MAX_BYTES` | 单篇 arXiv PDF 体积上限，默认 80 MiB |

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
| `POST /api/papers/import` | 按 arXiv 编号下载并入库（支持 4 种写法，去重命中不重复入库） |
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

## arXiv 文献导入模块

按 arXiv 编号一键下载 PDF，复用上传链路的**解析 → 去重 → 建图**，不另写解析器。

```bash
curl -X POST http://127.0.0.1:7860/api/papers/import \
     -H "Content-Type: application/json" \
     -d '{"arxiv_id": "1706.03762"}'
```

支持 `1706.03762`、`1706.03762v7`、`arXiv:1706.03762`、`https://arxiv.org/abs/1706.03762`
以及旧式编号 `hep-th/9901001`。前端在「文献库」标题下提供输入框与「导入」按钮：
请求期间按钮置灰并提示「正在拉取并解析 arXiv 文献，请稍候。」，成功后刷新列表、
选中新论文并滚动定位到新增条目，后端返回的中文报错原样展示。

返回 `paper_id / title / n_pages / n_chunks / duplicate / dedup_score / elapsed_s`；
与库内文献同源时 `duplicate=true`，只提示相似度、不重复入库。
完整状态码与异常分支见接口 docstring（`backend/main.py`）。

### 从零启动演示

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip.exe
cp .env.example .env                             # 填入 DEEPSEEK_API_KEY（可留空，退回本地措辞）
.venv/bin/python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 7860
```

打开 http://127.0.0.1:7860 → 左侧「文献库」输入 `1706.03762` → 点「导入」。
库为空时首次启动会按原有逻辑从 `data/seed/` 自愈。

### `SCITRACE_DATA_DIR`（隔离数据目录）

设置后，库文件、上传件与检索索引整体指向该目录，用于副本库验收、避免污染正式库：

```bash
SCITRACE_DATA_DIR=/tmp/scitrace-copy \
  .venv/bin/python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 7861
```

目录内需先放一份 `library.db`（可复制正式库），`uploads/` 与 `index/` 会自动创建；
未设置时仍使用项目内 `data/`。

### 验收脚本

```bash
.venv/Scripts/python.exe scripts/arXiv_import_verify.py   # Windows
.venv/bin/python scripts/arXiv_import_verify.py            # Linux / macOS
```

脚本自动完成：复制正式库到临时目录 → 用 `SCITRACE_DATA_DIR` 拉起隔离后端 →
顺序执行 a–m 用例（导入 / 重复 / 已有论文 / 非法编号 / 统计 / 图谱 / 问答正向 /
问答拒答 / 404 / 超时 / 非 PDF / 损坏 PDF / 5xx）→ 切回正式库跑只读回归 →
输出 `docs/arXiv_import_验收报告.md`。任一用例失败即标记失败并把服务端日志保留到
`scripts/verify_logs/<时间戳>/`；无论成败都会停止服务、清理临时目录。
超时、非 PDF、损坏 PDF 与 5xx 由脚本内置的本地桩服务注入
（`ARXIV_PDF_BASE_URL` + `ARXIV_TIMEOUT_S=1`），不需要外网。

### 实测性能与已知限制

| 场景 | 实测 |
|---|---|
| 本机导入 `1512.03385`（12 页 / 58 块） | 11.2 s，200，`duplicate=false` |
| 本机重复导入库内已有 `1706.03762` | 200，`duplicate=true`，相似度 1.0，库位数不变 |
| 本机非法编号 `abc123` | 400，中文原因，不写库 |
| 公网（魔搭免费 CPU）导入同一编号 | **超过 120 s，被网关 504**——容器出网带宽受限，2 MB 级 PDF 下不完；同一容器查「不存在的编号」仅 1.5 s，说明 DNS 与连通性正常，瓶颈是下载带宽 |

因此**公网演示时不要现场导入整篇论文**：本机与离线上传路径正常，公网入口保留但受带宽限制。这是平台侧限制，不是接口缺陷。

### 离线备选方案

网络不通时，arXiv 导入会返回明确的中文失败原因（超时 / 无法连接），其它功能不受影响。
离线场景改用**预置本地 PDF**：`pdf/` 内的论文通过「拖入 PDF 或点击选择」上传入口导入，
复用同一套 `/api/papers/upload` 解析、去重、建图链路，行为与 arXiv 导入一致。
为避免与在线导入混淆，离线预置入口默认隐藏，不额外暴露旁路。

### 备注

服务启动会按原有逻辑向 `events` 表追加一条 `startup` 记录（既有行为）；
本模块不会修改 `papers`、`chunks`、`edges` 的已有数据。验收脚本在正式库回归阶段
会对这三张表做启动前后的指纹比对。

## 实测数据

- 测试语料：`pdf/` 39 份官方 PDF（含 1 份扫描件、1 份 541 页长文档），**不使用任何自造 PDF 作为验收证据**
- 逐份指标：`docs/parse_metrics.tsv`
- **降级说明**：模型额度耗尽或接口不可用时自动降级为**纯证据模式**——仍返回原文证据、页码与 bbox，仅缺少自然语言措辞，溯源与去重、图谱、解析等功能不受影响。
- 修前/修后记录与官方 9 题明细：`docs/03-测试记录.md`

## 命题书功能对照表（逐条自查）

对照官方命题书 1.1–1.6 六个模块逐条自查，**做没做、做到什么程度都写清楚**，不把未实现的部分写成已实现。

| 命题书条目 | 状态 | 说明与证据 |
|---|---|---|
| 1.1.1 PDF 内容去重 | ✅ | 文件名不同也认得出同源：v7 与 v1 相似度 0.8490、三信号 3/3、阈值 0.63（由实测夹定）；重传时弹窗并建议保留库内版本，可列出参考文献 40→36 的差异 |
| 1.1.2 细粒度解析 | ✅ / 🔶 | 标题树（一级 7 / 三级 15）、作者 8 位 3 机构、参考文献 40 条、表格与公式计数、541 页书籍、扫描件标记 OCR；**图表 caption 与附录未单独结构化** |
| 1.2.1 单篇问答 | ✅ | 官方 5 题全部命中并带 page + bbox，1.1–1.6 s |
| 1.2.2 多篇问答 | ✅ | 官方 4 题：BERS/GPT 系列对比 3 题完整、1 题（GPT-1 总参数量）如实标注"该篇未报告" |
| 1.3.1 基本阅读 | ✅ | 三栏阅读器，按原页图像渲染，保留双栏版式 |
| 1.3.2 章节树与文献跳转 | ✅ | 章节树点击跳页（如 3.2.2 → p4，bbox [108.0, 630.8, 230.6, 640.7]）；参考文献条目可点 |
| 1.3.3 翻译 | ✅ / 🔶 | 选段翻译 + 术语对照表 + 学术体说明；**中英整篇对照联动滚动未做** |
| 1.4 论文自动下载与推荐 | ✅ / 🔶 | 按 arXiv 编号一键下载入库：支持 4 种写法、超时可配，404 / 5xx / 超时 / 非 PDF / 损坏 PDF 全部分支返回 400 + 中文原因；去重命中只提示相似度、不重复入库；**论文追踪与推荐未做** |
| 1.5 图谱·核心节点·综述 | ✅ / 🔶 | 21 节点 67 边、时间倒错 0、PageRank 排基石/边缘、自动综述 454 字；**Future Work 探索为简版** |
| 1.6 写作与可视化 | ✅ / 🔶 | 论文框架（References 只从库内挑）、CSV→图表+自动图注、中译英；**TikZ / Mermaid 拓扑图未做** |

## 目录

```
backend/          12 个 Python 模块（解析/存储/检索/问答/措辞/图谱/去重/接口/报错文案）
frontend/dist/    单文件前端（index.html，91.4 KB，无构建）
scripts/          arXiv_import_verify.py（隔离副本库验收，a–m + 正式库只读回归）
data/seed/        启动自愈快照（library.db + 21 篇 PDF）
docs/             测试记录、逐份指标、arXiv 导入验收报告
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
