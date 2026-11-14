# arXiv 文献导入 · 验收报告

- 生成时间：2026-11-14 13:26:48
- 基线：SciTrace（arXiv 导入补全版）
- 隔离副本：`SCITRACE_DATA_DIR` 指向临时目录，正式库未被导入用例写入
- 后端启动命令：`python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port <临时端口>`
- 结论：**全部通过**

| # | 用例 | 预期 | 实际 | 耗时 | 结果 |
|---|---|---|---|---|---|
| a | 导入库内不存在的 1512.03385 | HTTP 200 / duplicate=false / n_pages>0 / n_chunks>0 | HTTP 200 {"paper_id": "10bb7728ace0", "title": "Deep Residual Learning for Image Recognition", "n_pages": 12, "n_chunks": 58, "duplicate": false, "dedup_score": null, "elapsed_s": 12.2} | 12.2s | PASS |
| b | 重复提交同一编号 | HTTP 200 / duplicate=true / dedup_score 为数字 / 篇数不变 | HTTP 200 {"paper_id": "10bb7728ace0", "title": "Deep Residual Learning for Image Recognition", "n_pages": 12, "n_chunks": 58, "duplicate": true, "dedup_score": 1.0, "elapsed_s": 10.65}；papers 22→22 | 10.8s | PASS |
| c | 导入库内已有的 1706.03762 | duplicate=true / dedup_score ≥ 0.95 | HTTP 200 {"paper_id": "5188570ee8b3", "title": "Attention Is All You Need", "n_pages": 15, "n_chunks": 41, "duplicate": true, "dedup_score": 1.0, "elapsed_s": 9.91} | 9.9s | PASS |
| d | 非法编号 abc123 | HTTP 400 中文原因 / 篇数不变 | HTTP 400 detail=无法识别的 arXiv 编号「abc123」，应形如 1706.03762、1706.03762v7 或 https://arxiv.org/abs/1706.03762；papers 22→22 | 0.1s | PASS |
| e | 基础统计接口校验 | papers=22 / chunks=2730 / dedup_threshold=0.63 | HTTP 200 papers=22 chunks=2730 dedup_threshold=0.63 | 0.0s | PASS |
| f | 图谱节点与边数量校验 | nodes=22 / edges>0 / label 非空 | HTTP 200 nodes=22 edges=69 空标签=0 | 0.2s | PASS |
| g | 问答正向命中（第六作者） | grounded=true / evidence 非空 / 含 Aidan N. Gomez 与 University of Toronto | HTTP 200 grounded=True evidence=1 wording=第 6 作者是 Aidan N. Gomez，所属机构为 University of Toronto [P1]。 | 0.5s | PASS |
| h | 问答拒答分支（量子比特） | grounded=false / wording=文献中未提及相关内容。 | HTTP 200 grounded=False wording='文献中未提及相关内容。' | 1.6s | PASS |
| i | 不存在的编号 9999.99999 | HTTP 400（arXiv 404）/ 篇数不变 | HTTP 400 detail=arXiv 上找不到编号 9999.99999（HTTP 404）；papers 22→22 | 0.4s | PASS |
| log | 结构化日志字段检查 | 含来源/编号/paper id/耗时/是否重复/失败原因 | 结构化日志 5 条，字段完整 5 条，含失败原因 2 条 | 0.0s | PASS |
| j | 下载超时分支 | HTTP 400 超时中文原因 / 篇数不变 | HTTP 400 detail=下载 arXiv PDF 超时（1 秒），请稍后重试；papers 22→22；临时文件残留=无 | 1.2s | PASS |
| k | 返回非 PDF | HTTP 400 非 PDF 中文原因 / 篇数不变 | HTTP 400 detail=下载内容不是 PDF（Content-Type: text/html; charset=utf-8），可能编号不存在或 arXiv 返回了错误页；papers 22→22；临时文件残留=无 | 0.1s | PASS |
| l | PDF 损坏解析失败 | HTTP 400 PDF 损坏中文原因 / 临时文件清理 / 篇数不变 | HTTP 400 detail=PDF 文件损坏或无法解析（FileDataError），已拒绝入库；papers 22→22；临时文件残留=无 | 0.1s | PASS |
| m | arXiv 返回 5xx | HTTP 400 5xx 中文原因 / 篇数不变 | HTTP 400 detail=arXiv 服务暂时不可用（HTTP 500），请稍后重试；papers 22→22；临时文件残留=无 | 0.1s | PASS |
| r-e | 正式库回归 · 基础统计 | papers=21 / chunks=2672 / dedup_threshold=0.63 | HTTP 200 papers=21 chunks=2672 dedup_threshold=0.63 | 0.0s | PASS |
| r-f | 正式库回归 · 引用图谱 | nodes=21 / edges=67 | HTTP 200 nodes=21 edges=67 | 0.2s | PASS |
| r-g | 正式库回归 · 问答正向命中 | grounded=true / evidence 非空 / 含 Aidan N. Gomez 与 University of Toronto | HTTP 200 grounded=True evidence=1 wording=第 6 作者是 Aidan N. Gomez，所属机构为 University of Toronto [P1]。 | 0.5s | PASS |
| r-h | 正式库回归 · 问答拒答 | grounded=false / wording=文献中未提及相关内容。 | HTTP 200 grounded=False wording='文献中未提及相关内容。' | 1.6s | PASS |
| r-db | 正式库回归 · 正式库数据指纹（只读） | 启动前后 papers/chunks/edges 指纹一致 | papers/chunks/edges 指纹 3f5bf54bcccb→3f5bf54bcccb（一致=True） | 0.1s | PASS |

## 说明

- 副本库初始为正式库快照（21 篇 / 2672 块）；用例 a 入库后为 22 篇。
- j/k/l/m 使用脚本内置桩服务注入超时、非 PDF、损坏 PDF、HTTP 500，通过 `ARXIV_PDF_BASE_URL` + `ARXIV_TIMEOUT_S=1` 指向桩服务，不访问外网。
- 正式库回归只执行只读用例（health / graph / qa / 数据指纹），不向正式库导入论文；服务启动仍会按原有逻辑向 events 追加 startup 记录，papers / chunks / edges 指纹保持一致。
- 结构化日志：`通过，结构化日志 5 条，字段完整 5 条，含失败原因 2 条`。
