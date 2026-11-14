# 任务卡 · 补齐命题书 1.4「论文自动下载与导入」

> 给执行方（Codex / Trae / Qoder / 其他 AI 编程工具）：**只读本文件，按本节执行，不要通读其它 md。**
> 项目根 `D:\学习\传智杯`，启动方式与现有端点见 `README.md` 的「安装」与「接口」两节。

## 一、为什么做这一条

命题书 1.1–1.6 六模块逐条自查后，**只有 1.4「论文自动下载 / 追踪 / 推荐」整节空白**（其余五节已实现，1.3.3 / 1.6 有两处半做）。
"产品功能与工程质量"占评分 30%，六节里空一节是最扎眼的缺口。本次只补这一节里**最能演示的一小块：按 arXiv 编号一键下载并入库**。

## 二、范围（只允许动这些）

允许修改：
- `backend/main.py`：新增一个端点
- `backend/ingest.py` 或现有的上传入库函数：如需抽出一个可复用的「解析 + 入库」函数
- `frontend/dist/index.html`：文献库区域新增一个输入框 + 一个按钮

**不允许修改**：`backend/agent.py`、`backend/local_qa.py`、`backend/dedup.py`、`backend/graph.py`、`backend/parsing.py` 的任何现有逻辑；不允许改 `data/library.db` 的现有 21 条记录；不允许改 `data/seed/`。

## 三、接口契约

```
POST /api/papers/import
Content-Type: application/json
{ "arxiv_id": "1706.03762" }
```

行为：
1. 归一化输入：接受 `1706.03762`、`1706.03762v7`、`arXiv:1706.03762`、`https://arxiv.org/abs/1706.03762` 四种写法，统一取出裸编号。
2. 下载 `https://arxiv.org/pdf/{arxiv_id}`，**必须带 `User-Agent`**（arXiv 会拒绝无 UA 请求），超时 30 s。
3. 存到 `data/uploads/{文件名}.pdf`，复用现有上传入库链路（同一套解析 + 去重 + 建图），**不要另写一套解析**。
4. 若下载失败或返回非 PDF（Content-Type 不是 `application/pdf`），返回 400 + 明确中文原因，不写库。
5. 若该论文与库内已有论文被去重判定为同源（复用 `dedup.py`），返回 `duplicate: true` 与相似度分数，**不重复入库**。

返回：
```json
{ "paper_id": "…", "title": "…", "n_pages": 15, "n_chunks": 96,
  "duplicate": false, "dedup_score": null, "elapsed_s": 3.4 }
```

## 四、前端

文献库标题右侧加一行：

```
[ 输入 arXiv 编号，如 1706.03762 ]  [ 导入 ]
```

- 成功后：刷新文献列表、选中新论文、提示「已导入《标题》，15 页 / 96 块」
- 重复时：提示「疑似与库内《标题》同源，相似度 0.8x，未重复入库」
- 失败时：显示后端返回的中文原因，不要只弹 "请求失败"

## 五、验收（必须贴真实输出，缺一条算没做）

**关键约束：验收必须在副本库上做，不许污染正式库。**

```powershell
# 1) 复制一份库到临时目录，用副本跑一个临时服务
Copy-Item "D:\学习\传智杯\data\library.db" "$env:TEMP\lib_test.db"
# 启动时通过环境变量指向副本（若无该变量，先在 config 里支持 DB_PATH 覆盖，这属于允许修改范围）
```

| # | 验收项 | 期望 |
|---|---|---|
| a | 导入一篇库内没有的论文（建议 `1512.03385`） | 200，`duplicate=false`，`n_pages>0`，`n_chunks>0` |
| b | 重复导入同一编号 | 200，`duplicate=true`，`dedup_score` 为数字，库内篇数不变 |
| c | 导入库内已有的 v7（`1706.03762`） | `duplicate=true` 且分数接近 1.0 |
| d | 非法编号 `abc123` | 400，中文原因，库内篇数不变 |
| e | **正式库回归**：`GET /api/health` | `papers=21`、`chunks=2672`、`dedup_threshold=0.63` |
| f | **正式库回归**：`POST /api/graph {}` | 21 节点 / 67 边 |
| g | **正式库回归**：`POST /api/qa/evidence`，问「这篇论文的第六作者是谁，在哪个机构工作？」+ `paper_id=5188570ee8b3` | `grounded=true`、`evidence` 非空、措辞含 `Aidan N. Gomez`、`University of Toronto` |
| h | **正式库回归**：同一端点问「这篇论文的量子计算实验用了多少量子比特？」 | `grounded=false`、`wording=文献中未提及相关内容。` |

## 六、回归红线（碰了就算失败，必须回滚）

1. 正式库 `data/library.db` 必须仍是 **21 篇 / 2672 块**，图谱仍是 **21 节点 67 边**。任何一行现有记录的字段都不许变。
2. 问答两条口径不许变：Q1 必须带 `[P1]` 且措辞含 Aidan N. Gomez；量子比特问题必须拒答且 wording 为「文献中未提及相关内容。」。
3. 去重常量不许动：`DEDUP_THRESHOLD=0.63`、SIMHASH 0.72、MINHASH 0.45、TITLE 0.72。
4. 不许在代码里硬编码任何 API key；不许把 `.env` 打进镜像；`Dockerfile` 不许 `COPY . .`。
5. 端口仍为 `0.0.0.0:7860`。

## 七、提交

一条 commit，message 格式：

```
1.4: 按 arXiv 编号导入论文 | 副本库导入 200 / 重复判定 true / 正式库 21 篇 2672 块与图谱 21-67 回归不变
```

**做完就停，不要顺手改别的。**
