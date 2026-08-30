[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ModelPath,

    [ValidateRange(1, 65529)]
    [int]$Port = 2020,

    [ValidateRange(64, 262144)]
    [int]$ContextTokens = 262144,

    [ValidateRange(1, 16)]
    [int]$MaxRunningRequests = 1,

    [string]$DesktopPython = (Join-Path $env:LOCALAPPDATA 'FreeToken\venv\Scripts\python.exe'),

    [string]$VisionPackagesPath,

    [switch]$EnableVision,

    [ValidateSet('layer-stream', 'gpu')]
    [string]$VisionExecution = 'layer-stream',

    [switch]$EnableCacheReport
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceDir = Join-Path $repoRoot 'python'
$shimDir = Join-Path $PSScriptRoot 'windows-ple-mmap'
if (-not $VisionPackagesPath) {
    $VisionPackagesPath = Join-Path $repoRoot '.local\vision-packages'
}

if (-not (Test-Path -LiteralPath $ModelPath -PathType Container)) {
    throw "Model directory does not exist: $ModelPath"
}
if (-not (Test-Path -LiteralPath $DesktopPython -PathType Leaf)) {
    throw "FreeToken Desktop Python does not exist: $DesktopPython. Install FreeToken Desktop or pass -DesktopPython."
}
if (-not (Test-Path -LiteralPath $sourceDir -PathType Container)) {
    throw "FreeToken source directory does not exist: $sourceDir"
}
if (-not (Test-Path -LiteralPath (Join-Path $shimDir 'sitecustomize.py') -PathType Leaf)) {
    throw "Windows compatibility shim does not exist: $shimDir"
}

$venvRoot = Split-Path (Split-Path $DesktopPython -Parent) -Parent
$installedKernelDir = Join-Path $venvRoot 'Lib\site-packages\freetoken\kernel'
if (-not (Test-Path -LiteralPath $installedKernelDir -PathType Container)) {
    throw "FreeToken Desktop Windows kernels do not exist: $installedKernelDir"
}

$cudaRoot = $env:CUDA_PATH
if (-not $cudaRoot) {
    $cudaRoot = (& $DesktopPython -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or '')").Trim()
}
if (-not $cudaRoot -or -not (Test-Path -LiteralPath (Join-Path $cudaRoot 'bin\nvcc.exe') -PathType Leaf)) {
    throw 'CUDA Toolkit with nvcc.exe was not found. Install CUDA 13 and set CUDA_PATH.'
}

$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
$resolvedPython = (Resolve-Path -LiteralPath $DesktopPython).Path
$env:CUDA_PATH = (Resolve-Path -LiteralPath $cudaRoot).Path

$pathParts = @($shimDir)
if ($EnableVision) {
    if (-not (Test-Path -LiteralPath $VisionPackagesPath -PathType Container)) {
        throw "Local picture packages do not exist: $VisionPackagesPath. Run install-qwen38-vision-deps-windows.ps1 first."
    }
    $resolvedVisionPackages = (Resolve-Path -LiteralPath $VisionPackagesPath).Path
    $priorPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $resolvedVisionPackages + $(if ($priorPythonPath) { ";$priorPythonPath" } else { '' })
        & $resolvedPython -c "import PIL, torch, torchvision; assert torch.__version__.startswith('2.11.'); assert torch.version.cuda and torch.version.cuda.startswith('13.'); assert torchvision.__version__.startswith('0.26.'); assert torchvision.extension._has_ops()"
        if ($LASTEXITCODE -ne 0) {
            throw 'Local picture packages are missing or incompatible with Desktop Torch.'
        }
    }
    finally {
        $env:PYTHONPATH = $priorPythonPath
    }
    $pathParts += $resolvedVisionPackages
    $env:FREETOKEN_LOAD_VISION = '1'
    $env:FREETOKEN_VISION_EXECUTION = $VisionExecution
}
else {
    # Make the fallback deterministic even if the parent shell previously ran picture mode.
    $env:FREETOKEN_LOAD_VISION = '0'
    $env:FREETOKEN_VISION_EXECUTION = 'gpu'
}
$pathParts += $sourceDir
$env:PYTHONPATH = ($pathParts -join ';') + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { '' })

Write-Host "Starting the unofficial Desktop-assisted Windows server"
Write-Host "  Model:  $resolvedModel"
Write-Host "  API:    http://127.0.0.1:$Port/v1"
Write-Host "  Context tokens: $ContextTokens"
Write-Host "  Active requests: $MaxRunningRequests"
Write-Host "  Picture input: $($EnableVision.IsPresent)"
Write-Host "  Picture execution: $(if ($EnableVision) { $VisionExecution } else { 'disabled' })"
Write-Host 'FreeToken Desktop supplies the Windows runtime but does not need to be open.'

$serveArgs = @(
    '-m', 'freetoken.cli', 'serve',
    '--model', $resolvedModel,
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--ple-backend', 'mmap',
    '--moe-backend', 'offload',
    '--moe-cache-auto',
    '--max-running-requests', "$MaxRunningRequests",
    '--kv-reserve-tokens', "$ContextTokens",
    '--expert-load', 'serial'
)
if ($EnableCacheReport) {
    $serveArgs += '--enable-cache-report'
}

& $resolvedPython @serveArgs
exit $LASTEXITCODE
