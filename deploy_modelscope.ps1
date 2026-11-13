[CmdletBinding()]
param(
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$excludedDirectories = @(".venv", ".git", ".archive", "__pycache__")
$excludedExtensions = @(".pyc", ".log", ".out", ".err")
$excludedFiles = @(".env")

function Test-ExcludedPath {
    param([string]$RelativePath)

    $parts = $RelativePath -split "[\\/]"
    foreach ($part in $parts) {
        if ($excludedDirectories -contains $part) {
            return $true
        }
    }
    $leaf = $parts[-1]
    if ($excludedFiles -contains $leaf) {
        return $true
    }
    return $excludedExtensions -contains [IO.Path]::GetExtension($leaf).ToLowerInvariant()
}

$files = Get-ChildItem -LiteralPath $projectRoot -File -Recurse -Force |
    Where-Object {
        $relative = [IO.Path]::GetRelativePath($projectRoot, $_.FullName)
        -not (Test-ExcludedPath $relative)
    }
$totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
$totalMiB = [math]::Round($totalBytes / 1MB, 2)
$totalMB = [math]::Round($totalBytes / 1e6, 2)

Write-Output "将上传的文件数：$($files.Count)"
Write-Output "总体积：$totalMiB MiB（$totalMB MB）"

if (-not $Push) {
    Write-Output "DRY_RUN=1；未读取 token，未连接远程仓库。"
    exit 0
}

$token = $env:MODELSCOPE_API_TOKEN
$username = $env:MODELSCOPE_USERNAME
$studio = if ($env:MODELSCOPE_STUDIO) { $env:MODELSCOPE_STUDIO } else { "SciTrace" }
if ([string]::IsNullOrWhiteSpace($token)) {
    throw "缺少环境变量 MODELSCOPE_API_TOKEN"
}
if ([string]::IsNullOrWhiteSpace($username)) {
    throw "缺少环境变量 MODELSCOPE_USERNAME"
}

$staging = Join-Path $env:TEMP ("scitrace-modelscope-" + [guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Path $staging | Out-Null
    foreach ($file in $files) {
        $relative = [IO.Path]::GetRelativePath($projectRoot, $file.FullName)
        $destination = Join-Path $staging $relative
        $parent = Split-Path -Parent $destination
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $destination
    }

    & git -C $staging init -b master | Out-Null
    & git -C $staging add --all
    & git -C $staging -c user.name="SciTrace Deploy" -c user.email="deploy@local.invalid" `
        commit -m "Deploy SciTrace ModelScope Space" | Out-Null

    $escapedToken = [uri]::EscapeDataString($token)
    $remoteUrl = "https://oauth2:$escapedToken@www.modelscope.cn/studios/$username/$studio.git"
    $rawOutput = & git -C $staging push --porcelain $remoteUrl master:master 2>&1
    $exitCode = $LASTEXITCODE
    $safeOutput = ($rawOutput | Out-String).Replace($token, "***").Replace($escapedToken, "***")
    if ($safeOutput.Trim()) {
        Write-Output $safeOutput.Trim()
    }
    if ($exitCode -ne 0) {
        throw "git push 失败，退出码 $exitCode；未执行 force push"
    }
    Write-Output "推送完成：master"
} finally {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}
