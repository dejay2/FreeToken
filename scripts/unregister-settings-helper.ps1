#Requires -Version 5.1
<##
.SYNOPSIS
Remove the current user's FreeToken settings-helper task or Startup shortcut.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$taskName = 'FreeTokenSettingsHelper'
$startupDir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup'
$shortcutPath = Join-Path $startupDir ($taskName + '.lnk')

$taskOutput = @(& schtasks.exe /Delete /TN $taskName /F 2>&1)
$taskExitCode = $LASTEXITCODE
$removedShortcut = $false
if (Test-Path -LiteralPath $shortcutPath) {
    Remove-Item -LiteralPath $shortcutPath -Force
    $removedShortcut = $true
}

if ($taskExitCode -eq 0) {
    Write-Output "Unregistered $taskName from Task Scheduler."
}
elseif ($removedShortcut) {
    Write-Output "Unregistered $taskName from the Startup folder."
}
elseif ($taskOutput -match 'cannot find|does not exist|not found') {
    Write-Output "$taskName was not registered."
}
else {
    $detail = ($taskOutput -join ' ').Trim()
    throw "Could not remove $taskName from Task Scheduler: $detail"
}
