#Requires -Version 5.1
<##
.SYNOPSIS
Start the local FreeToken settings helper without opening a window.

.DESCRIPTION
This is the exact command used by the per-user Task Scheduler entry. It keeps the
helper on loopback, uses the FreeToken Desktop Python installation, puts the
checkout source on PYTHONPATH, and records all output in the user's local
FreeToken log directory. The helper stays torch-free; the Windows compatibility
shim belongs to the GPU server launcher, not this control page.
#>
[CmdletBinding()]
param(
    # 2031 avoids engine worker ports 2020-2029; measured by the stop script's port sweep.
    [int]$Port = 2031
)
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonExe = Join-Path $env:LOCALAPPDATA 'FreeToken\venv\Scripts\python.exe'
$sourceDir = Join-Path $repoRoot 'python'
$logDir = Join-Path $env:LOCALAPPDATA 'FreeToken'
$logPath = Join-Path $logDir 'settings-helper.log'

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "FreeToken Desktop Python does not exist: $pythonExe"
}
if (-not (Test-Path -LiteralPath $sourceDir -PathType Container)) {
    throw "FreeToken source directory does not exist: $sourceDir"
}

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$started = Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'
Add-Content -LiteralPath $logPath -Value "`r`n=== settings helper start $started ==="

$priorPythonPath = $env:PYTHONPATH
$pathParts = @($sourceDir)
if ($priorPythonPath) {
    $pathParts += $priorPythonPath
}
$env:PYTHONPATH = $pathParts -join ';'

$exitCode = 0
Push-Location $repoRoot
try {
    # The helper is deliberately a foreground child: Task Scheduler owns this wrapper,
    # while Tee-Object-style line writes keep failures visible after a hidden launch.
    # Uvicorn writes normal startup messages to stderr; Continue prevents PowerShell 5.1
    # from turning those messages into terminating NativeCommandError records.
    $childErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $pythonExe -m freetoken.daemon.settings --port $Port --job-id settings-helper *>&1 |
            ForEach-Object {
                $line = [string]$_
                Add-Content -LiteralPath $logPath -Value $line
                Write-Output $line
            }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $childErrorAction
    }
}
catch {
    $message = "settings helper failed: $($_.Exception.Message)"
    Add-Content -LiteralPath $logPath -Value $message
    throw
}
finally {
    Pop-Location
}

$finished = Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'
Add-Content -LiteralPath $logPath -Value "=== settings helper exit $exitCode at $finished ==="
exit $exitCode
