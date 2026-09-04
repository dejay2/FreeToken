#Requires -Version 5.1
<##
.SYNOPSIS
Register the FreeToken settings helper for the current user's logon.

.DESCRIPTION
Task Scheduler stores one per-user logon task. /F makes repeated registration
replace the same task instead of creating duplicates; no Administrator rights are
needed because the task belongs to the current user. If this Windows installation
rejects a non-administrator Task Scheduler registration, the script uses the
per-user Startup folder instead.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$taskName = 'FreeTokenSettingsHelper'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$startScript = Join-Path $repoRoot 'scripts\start-settings-helper.ps1'
$startupDir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup'
$shortcutPath = Join-Path $startupDir ($taskName + '.lnk')

if (-not (Test-Path -LiteralPath $startScript -PathType Leaf)) {
    throw "Settings helper start script does not exist: $startScript"
}

# Keep this command aligned with P0-spec.md section 6: hidden, unrestricted PowerShell
# invokes the checked-in wrapper, which supplies the Desktop Python and PYTHONPATH.
$taskRun = 'powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $startScript
& schtasks.exe /Create /TN $taskName /TR $taskRun /SC ONLOGON /F
if ($LASTEXITCODE -eq 0) {
    # Remove a stale fallback so exactly one startup mechanism remains after a later
    # successful Task Scheduler registration.
    if (Test-Path -LiteralPath $shortcutPath) {
        Remove-Item -LiteralPath $shortcutPath -Force
    }
    Write-Output "Registered $taskName for the current user's logon with Task Scheduler."
    Write-Output "Command: $taskRun"
    exit 0
}

Write-Warning "Task Scheduler registration was rejected (exit code $LASTEXITCODE); using the per-user Startup folder."
New-Item -ItemType Directory -Path $startupDir -Force | Out-Null
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$shortcut.Arguments = '-WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $startScript
$shortcut.WorkingDirectory = $repoRoot
$shortcut.WindowStyle = 7
$shortcut.Description = 'FreeToken local settings helper'
$shortcut.Save()

Write-Output "Registered $taskName for the current user's logon in the Startup folder."
Write-Output "Shortcut: $shortcutPath"
