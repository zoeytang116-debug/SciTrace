"""arXiv 导入模块的错误提示文案。

所有对外可见的中文报错集中在此，业务代码只做格式化，不内联文案。
"""

from __future__ import annotations

EMPTY_ID = "arXiv 编号不能为空"
INVALID_ID = (
    "无法识别的 arXiv 编号「{raw}」，应形如 1706.03762、1706.03762v7 "
    "或 https://arxiv.org/abs/1706.03762"
)
NOT_FOUND = "arXiv 上找不到编号 {arxiv_id}（HTTP 404）"
SERVER_ERROR = "arXiv 服务暂时不可用（HTTP {status}），请稍后重试"
HTTP_FAILED = "下载 arXiv PDF 失败：HTTP {status} {reason}"
TIMEOUT = "下载 arXiv PDF 超时（{timeout:g} 秒），请稍后重试"
NETWORK_FAILED = "下载 arXiv PDF 失败：{reason}"
TOO_LARGE = "arXiv PDF 超过 {max_mb}MB，已拒绝入库"
NOT_PDF = (
    "下载内容不是 PDF（Content-Type: {content_type}），"
    "可能编号不存在或 arXiv 返回了错误页"
)
UNKNOWN = "未知"
PARSE_FAILED = "PDF 文件损坏或无法解析（{error_type}），已拒绝入库"
