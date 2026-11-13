# 科研助手 Agent —— SciTrace · 文溯

每一句回答都能点回原文那一页、那一行，证据不足时明确拒答的科研助手。

## 安装

1. 创建并激活虚拟环境：

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. 安装后端依赖：

   ```powershell
   pip install -r requirements.txt
   ```

3. 复制环境变量模板并填入自己的密钥：

   ```powershell
   Copy-Item .env.example .env
   ```

## 启动

```powershell
$env:PORT = "7860"
uvicorn main:app --app-dir backend --host 0.0.0.0 --port $env:PORT
```

## 部署

目标平台：魔搭创空间（ModelScope Studios），Docker 类型，免费 CPU 基础实例（16GB 内存）。

### 构建镜像

```bash
docker build -t scitrace .
docker run --rm -p 7860:7860 -e PORT=7860 scitrace
```

Dockerfile 基于 `python:3.13-slim`，安装 `requirements.txt` 中的固定版本，并复制：

- `backend/`
- `frontend/dist/`
- `pdf/`
- `data/`（含运行时数据库、上传文件和 seed 副本）

工作目录为 `/home/user/app`，使用非 root 用户运行。容器固定监听 `0.0.0.0:7860`，不能改为平台占用的 `8080`。

### 魔搭创空间

1. 创建 Docker 类型的公开创空间 `SciTrace`。
2. 在平台 Secrets 中配置所需模型环境变量。
3. 在本地设置推送所需环境变量后运行部署脚本。
4. 创空间会按根目录 `Dockerfile` 构建并映射 `7860` 端口。

创空间地址占位：

```text
https://www.modelscope.cn/studios/<用户名>/SciTrace
```

推送前先做 dry-run：

```powershell
$env:MODELSCOPE_USERNAME = "<用户名>"
$env:MODELSCOPE_STUDIO = "SciTrace"
.\deploy_modelscope.ps1
```

确认文件数和体积后显式推送：

```powershell
$env:MODELSCOPE_API_TOKEN = "<模型空间访问令牌>"
.\deploy_modelscope.ps1 -Push
```

脚本从环境变量读取 token，不会把 token 写入仓库或打印；但 dry-run 不读取 token。脚本排除 `.venv`、`.git`、`.archive`、`__pycache__`、`*.pyc`、`*.log`、`.env`，默认分支为 `master`，不执行 force push。

可选环境变量：

```text
DEEPSEEK_API_KEY=
LLM_API_KEY=
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_VISION_MODEL=
EMBEDDING_MODEL=
```

使用 DeepSeek 时，把 DeepSeek key 写入创空间 Secret `DEEPSEEK_API_KEY` 或 `LLM_API_KEY`（前者优先）。未配置时系统仍可运行本地解析、检索、证据问答、去重和图谱，但不会调用模型。

### 免费层限制

免费 CPU 创空间长期无访问时会自动休眠，再次访问需要冷启动。具体休眠时长、构建额度和存储额度以魔搭创空间当前政策为准。

### 数据烘焙与启动自愈

- `data/library.db` 与 `data/uploads/` 同时烘焙进镜像。
- `data/seed/library.db` 与 `data/seed/uploads/` 是只读种子副本。
- 服务启动时若运行目录中的 `data/library.db` 不存在，会从 `data/seed/library.db` 复制恢复，并补齐缺失的上传文件。
- 创空间重启可能清空运行时磁盘；上述 seed 自愈会在启动时恢复 20 篇内置文献。
