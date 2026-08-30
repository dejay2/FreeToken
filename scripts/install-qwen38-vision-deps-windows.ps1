[CmdletBinding()]
param(
    [string]$DesktopPython = (Join-Path $env:LOCALAPPDATA 'FreeToken\venv\Scripts\python.exe'),
    [string]$TargetPath,
    [switch]$IncludeTestTools
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $TargetPath) {
    $TargetPath = Join-Path $repoRoot '.local\vision-packages'
}
if (-not (Test-Path -LiteralPath $DesktopPython -PathType Leaf)) {
    throw "FreeToken Desktop Python does not exist: $DesktopPython"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv was not found. Install uv, then run this script again.'
}

$resolvedPython = (Resolve-Path -LiteralPath $DesktopPython).Path
New-Item -ItemType Directory -Force -Path $TargetPath | Out-Null
$resolvedTarget = (Resolve-Path -LiteralPath $TargetPath).Path

# Keep Torch in the untouched Desktop environment. Only the matching picture package
# and Pillow are placed beside this checkout.
& uv pip install --python $resolvedPython --target $resolvedTarget --upgrade --no-deps 'pillow==12.3.0'
if ($LASTEXITCODE -ne 0) { throw 'Pillow installation failed.' }
& uv pip install --python $resolvedPython --target $resolvedTarget --upgrade --no-deps `
    --index 'https://download.pytorch.org/whl/cu130' 'torchvision==0.26.0+cu130'
if ($LASTEXITCODE -ne 0) { throw 'TorchVision installation failed.' }

if ($IncludeTestTools) {
    & uv pip install --python $resolvedPython --target $resolvedTarget --upgrade 'pytest>=9,<10'
    if ($LASTEXITCODE -ne 0) { throw 'pytest installation failed.' }
}

if (Test-Path -LiteralPath (Join-Path $resolvedTarget 'torch')) {
    throw "The local picture folder unexpectedly contains Torch: $resolvedTarget\torch"
}

$priorPythonPath = $env:PYTHONPATH
$checkPath = Join-Path $resolvedTarget '.freetoken-vision-check.py'
try {
    $env:PYTHONPATH = $resolvedTarget + $(if ($priorPythonPath) { ";$priorPythonPath" } else { '' })
    @'
import PIL
import torch
import torchvision
assert torch.__version__.startswith("2.11."), torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("13."), torch.version.cuda
assert torchvision.__version__.startswith("0.26."), torchvision.__version__
assert torchvision.extension._has_ops(), "TorchVision compiled operations did not load"
print(f"Pillow {PIL.__version__}")
print(f"Torch {torch.__version__} / CUDA {torch.version.cuda}")
print(f"TorchVision {torchvision.__version__}")
'@ | Set-Content -LiteralPath $checkPath -Encoding UTF8
    & $resolvedPython $checkPath
    if ($LASTEXITCODE -ne 0) { throw 'The local picture packages are incompatible with Desktop Torch.' }
}
finally {
    Remove-Item -LiteralPath $checkPath -Force -ErrorAction SilentlyContinue
    $env:PYTHONPATH = $priorPythonPath
}

Write-Host "Picture packages are ready at $resolvedTarget"
