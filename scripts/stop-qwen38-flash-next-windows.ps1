#Requires -Version 5.1
<#
.SYNOPSIS
Stop a FreeToken server on this box and wait until the card and the port are actually free.

.DESCRIPTION
A live run leaves more than one process behind. The launcher starts
`python.exe -m freetoken.cli serve`, which spawns scheduler/tokenizer children through
`multiprocessing.spawn`; when the launcher is killed from the console its children can
survive as orphans (`python.exe ... spawn_main`, parent gone) still holding the ZMQ ports,
and the next boot dies with

    ZMQError: Address in use (tcp://127.0.0.1:2033)

This script selects, prints, kills and then WAITS:

  * `python.exe` / `pythonw.exe` whose command line runs `freetoken.cli serve` (for -Port,
    or every port when -Port is 0 or omitted);
  * every descendant python process of those, to any depth;
  * any orphaned `python.exe ... spawn_main` whose parent no longer exists -- these are the
    ones that hold the ZMQ ports after the parent is gone.

It never touches a non-python process. In particular `ft.exe`, the FreeToken Desktop daemon
on port 1900, is excluded by name even though its own command line mentions
`freetoken.cli serve` -- killing it takes the Windows runtime down with it.

After killing it waits, up to -TimeoutSeconds, for BOTH:

  * `nvidia-smi` to report less than -VramFreeThresholdMB in use (the card is only really
    free once the driver has torn the context down; booting into a half-released card is
    how `cudaHostRegister failed ... out of memory` happens), and
  * no listener left on -Port or the nine ports above it (the ZMQ side ports).

.PARAMETER Port
The server's HTTP port, e.g. 2020. 0 (the default) means every FreeToken server found.

.PARAMETER DryRun
Print the selection and exit without killing anything.

.PARAMETER DotSourceOnly
Define the functions and return. For the test suite; does not enumerate a single process.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop-qwen38-flash-next-windows.ps1 -Port 2020

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop-qwen38-flash-next-windows.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [ValidateRange(0, 65529)]
    [int]$Port = 0,

    [ValidateRange(1, 900)]
    [int]$TimeoutSeconds = 120,

    # The card counts as free below this. A settled RTX 5090 with no server sits near 0.5 GB
    # (the desktop compositor); 3 GB leaves room for that without accepting a live server.
    [ValidateRange(0, 131072)]
    [int]$VramFreeThresholdMB = 3072,

    [switch]$DryRun,

    # Leave orphaned `spawn_main` children alone. The orphan rule is deliberately blind to
    # WHICH tool spawned them -- a multiprocessing child's command line says nothing about
    # its parent's module -- so a stale child of some other python tool on the box matches
    # too. That is the point (an orphan of the last server is exactly what holds the ZMQ
    # port), but -DryRun first, and this switch is the way out.
    [switch]$SkipOrphans,

    [switch]$DotSourceOnly
)

# PowerShell 5.1 only: no `&&`, no `||`, no ternary, no null-coalescing anywhere below.

# Process names this script is ever allowed to kill. `ft.exe` -- the Desktop daemon on port
# 1900, whose command line also mentions freetoken.cli -- is not on it, and neither is
# anything else: the selection is by command line, so the name check is the backstop that
# keeps a command-line coincidence from taking down the runtime.
$script:FreeTokenKillableNames = @('python.exe', 'pythonw.exe')

function Test-FreeTokenKillableName {
    param([string]$Name)
    if (-not $Name) { return $false }
    return $script:FreeTokenKillableNames -contains $Name.ToLowerInvariant()
}

function Test-FreeTokenServeProcess {
    <#
    .SYNOPSIS
    Is this a FreeToken server process (optionally: on this port)?
    #>
    param(
        [string]$Name,
        [string]$CommandLine,
        [int]$Port = 0
    )
    if (-not (Test-FreeTokenKillableName -Name $Name)) { return $false }
    if (-not $CommandLine) { return $false }
    if ($CommandLine -notmatch 'freetoken\.cli') { return $false }
    # `serve` as its own token: `freetoken.cli bench` must not match, and neither must a
    # model path that happens to contain the word.
    if ($CommandLine -notmatch '(?i)freetoken\.cli\s+serve(\s|$)') { return $false }
    if ($Port -le 0) { return $true }
    # --port 2020 and --port=2020, but not --port 20200
    return $CommandLine -match ("(?i)--port[=\s]+" + [regex]::Escape("$Port") + "(\s|$)")
}

function Test-FreeTokenSpawnMainProcess {
    <#
    .SYNOPSIS
    Is this one of multiprocessing's spawned children?
    #>
    param([string]$Name, [string]$CommandLine)
    if (-not (Test-FreeTokenKillableName -Name $Name)) { return $false }
    if (-not $CommandLine) { return $false }
    return $CommandLine -match 'spawn_main'
}

function Get-FreeTokenDescendantIds {
    <#
    .SYNOPSIS
    Every killable descendant of -Id in -Processes, to any depth.

    .DESCRIPTION
    Pure over the snapshot it is handed, so the walk is testable without a process. Windows
    recycles pids, so a snapshot can contain a parent cycle; the visited set makes the walk
    terminate anyway.
    #>
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][array]$Processes,
        [Parameter(Mandatory = $true)][int]$Id
    )
    $found = New-Object 'System.Collections.Generic.HashSet[int]'
    $frontier = @($Id)
    $visited = New-Object 'System.Collections.Generic.HashSet[int]'
    $null = $visited.Add($Id)
    while ($frontier.Count -gt 0) {
        $next = @()
        foreach ($parentId in $frontier) {
            foreach ($proc in $Processes) {
                if ([int]$proc.ParentProcessId -ne [int]$parentId) { continue }
                $childId = [int]$proc.ProcessId
                if (-not (Test-FreeTokenKillableName -Name $proc.Name)) { continue }
                if (-not $visited.Add($childId)) { continue }
                $null = $found.Add($childId)
                $next += $childId
            }
        }
        $frontier = $next
    }
    return @($found)
}

function Get-FreeTokenOrphanIds {
    <#
    .SYNOPSIS
    Killable `spawn_main` processes whose parent is no longer in the snapshot.

    .DESCRIPTION
    These are what breaks the next boot: the launcher is gone, so nothing will ever reap
    them, and they still hold the ZMQ ports. A spawn_main whose parent IS alive belongs to
    that parent's tree and is only killed through it.
    #>
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][array]$Processes
    )
    $alive = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($proc in $Processes) { $null = $alive.Add([int]$proc.ProcessId) }
    $orphans = @()
    foreach ($proc in $Processes) {
        if (-not (Test-FreeTokenSpawnMainProcess -Name $proc.Name -CommandLine $proc.CommandLine)) {
            continue
        }
        if ($alive.Contains([int]$proc.ParentProcessId)) { continue }
        $orphans += [int]$proc.ProcessId
    }
    return @($orphans)
}

function Select-FreeTokenKillSet {
    <#
    .SYNOPSIS
    The process ids this script would kill, from a snapshot.
    #>
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][array]$Processes,
        [int]$Port = 0
    )
    $selected = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($proc in $Processes) {
        if (-not (Test-FreeTokenServeProcess -Name $proc.Name -CommandLine $proc.CommandLine -Port $Port)) {
            continue
        }
        $null = $selected.Add([int]$proc.ProcessId)
        foreach ($childId in (Get-FreeTokenDescendantIds -Processes $Processes -Id ([int]$proc.ProcessId))) {
            $null = $selected.Add([int]$childId)
        }
    }
    foreach ($orphanId in (Get-FreeTokenOrphanIds -Processes $Processes)) {
        $null = $selected.Add([int]$orphanId)
    }
    return @($selected)
}

function Get-FreeTokenWatchPorts {
    <#
    .SYNOPSIS
    The HTTP port plus the nine ZMQ side ports above it. Empty when no port was named.
    #>
    param([int]$Port = 0)
    if ($Port -le 0) { return @() }
    return @($Port..($Port + 9))
}

function Get-FreeTokenProcessSnapshot {
    <#
    .SYNOPSIS
    Every process, in the shape the helpers above expect. The one CIM call in the script.
    #>
    return @(Get-CimInstance -ClassName Win32_Process |
        Select-Object ProcessId, ParentProcessId, Name, CommandLine)
}

function Get-FreeTokenListeningPorts {
    <#
    .SYNOPSIS
    Which of -Ports still have a listener. Falls back to netstat where Get-NetTCPConnection
    is missing (Server Core, older images).
    #>
    param([AllowEmptyCollection()][array]$Ports)
    if (-not $Ports -or $Ports.Count -eq 0) { return @() }
    $listening = @()
    $cmdlet = Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue
    if ($cmdlet) {
        foreach ($p in $Ports) {
            $conn = $null
            try {
                $conn = Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction Stop
            }
            catch {
                $conn = $null
            }
            if ($conn) { $listening += [int]$p }
        }
        return @($listening)
    }
    $netstat = @(netstat -ano -p tcp)
    foreach ($p in $Ports) {
        foreach ($line in $netstat) {
            if ($line -match ("[:\.]" + [regex]::Escape("$p") + "\s") -and $line -match 'LISTENING') {
                $listening += [int]$p
                break
            }
        }
    }
    return @($listening)
}

function Get-FreeTokenVramUsedMB {
    <#
    .SYNOPSIS
    Total VRAM in use across the visible GPUs, or $null when nvidia-smi is unavailable.

    .DESCRIPTION
    A query, not a CUDA context: it does not touch the card the way opening a device would.
    #>
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return $null }
    $out = $null
    try {
        $out = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    }
    catch {
        return $null
    }
    if (-not $out) { return $null }
    $total = 0
    foreach ($line in @($out)) {
        $text = "$line".Trim()
        if ($text -match '^\d+$') { $total += [int]$text }
    }
    return $total
}

if ($DotSourceOnly) { return }

$ErrorActionPreference = 'Stop'

$snapshot = Get-FreeTokenProcessSnapshot
$byId = @{}
foreach ($proc in $snapshot) { $byId[[int]$proc.ProcessId] = $proc }

$orphanIds = @(Get-FreeTokenOrphanIds -Processes $snapshot)
$targetIds = @(Select-FreeTokenKillSet -Processes $snapshot -Port $Port)
if ($SkipOrphans) {
    $targetIds = @($targetIds | Where-Object { $orphanIds -notcontains $_ })
}
# Never this shell, and never the tree it is running inside.
$selfTree = @($PID) + @(Get-FreeTokenDescendantIds -Processes $snapshot -Id $PID)
$targetIds = @($targetIds | Where-Object { $selfTree -notcontains $_ })
$serverIds = @($targetIds | Where-Object { $orphanIds -notcontains $_ })
$killableOrphanIds = @($targetIds | Where-Object { $orphanIds -contains $_ })

function Write-FreeTokenProcessLine {
    param([int]$Id)
    $proc = $byId[[int]$Id]
    $cmd = "$($proc.CommandLine)"
    if ($cmd.Length -gt 120) { $cmd = $cmd.Substring(0, 117) + '...' }
    Write-Host "  pid $Id  $($proc.Name)  $cmd"
}

if ($Port -gt 0) {
    Write-Host "FreeToken server processes on port ${Port}:"
}
else {
    Write-Host 'FreeToken server processes on every port:'
}
if ($serverIds.Count -eq 0) {
    Write-Host '  (none running)'
}
foreach ($id in ($serverIds | Sort-Object)) { Write-FreeTokenProcessLine -Id $id }

Write-Host 'Orphaned multiprocessing children (parent gone; these hold the ZMQ ports):'
if ($killableOrphanIds.Count -eq 0) {
    Write-Host '  (none)'
}
foreach ($id in ($killableOrphanIds | Sort-Object)) { Write-FreeTokenProcessLine -Id $id }

if ($DryRun) {
    Write-Host '-DryRun: nothing was killed.'
    return
}

foreach ($id in $targetIds) {
    $proc = $byId[[int]$id]
    # Re-check the name at the kill site: the snapshot is a moment old, and a pid can be
    # recycled between reading it and acting on it.
    if (-not (Test-FreeTokenKillableName -Name $proc.Name)) { continue }
    try {
        Stop-Process -Id ([int]$id) -Force -ErrorAction Stop
        Write-Host "  killed pid $id"
    }
    catch {
        Write-Host "  pid ${id}: $($_.Exception.Message)"
    }
}

$watchPorts = Get-FreeTokenWatchPorts -Port $Port
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$vramUsed = $null
$listening = @()
$settled = $false
while ((Get-Date) -lt $deadline) {
    $listening = @(Get-FreeTokenListeningPorts -Ports $watchPorts)
    $vramUsed = Get-FreeTokenVramUsedMB
    $vramOk = $true
    if ($null -ne $vramUsed) { $vramOk = $vramUsed -lt $VramFreeThresholdMB }
    if ($listening.Count -eq 0 -and $vramOk) {
        $settled = $true
        break
    }
    Start-Sleep -Seconds 2
}

if ($null -eq $vramUsed) {
    Write-Host 'VRAM in use: unknown (nvidia-smi not found)'
}
else {
    Write-Host "VRAM in use: $vramUsed MB (threshold $VramFreeThresholdMB MB)"
}
if ($watchPorts.Count -eq 0) {
    Write-Host 'Ports watched: none (-Port was not given)'
}
elseif ($listening.Count -eq 0) {
    Write-Host "Ports $($watchPorts[0])-$($watchPorts[-1]): free"
}
else {
    Write-Host "Ports still listening: $($listening -join ', ')"
}

if ($settled) {
    Write-Host 'Settled: the card and the ports are free.'
    exit 0
}
Write-Host "NOT settled within $TimeoutSeconds s. Re-run, or check the processes by hand."
exit 1
