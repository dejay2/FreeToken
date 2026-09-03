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
    [int]$MaxRunningRequests = 4,

    [string]$DesktopPython = (Join-Path $env:LOCALAPPDATA 'FreeToken\venv\Scripts\python.exe'),

    [string]$VisionPackagesPath,

    [switch]$EnableVision,

    [ValidateSet('layer-stream', 'gpu')]
    [string]$VisionExecution = 'layer-stream',

    # Where the 856 MiB of picture weights live. 'ram' reads them into process memory at
    # boot and holds them for the life of the server, even for text-only sessions. 'mmap'
    # maps their extent of the bf16 shard copy-on-write and faults it in only when a
    # picture arrives, handing that 856 MiB back to the standby list the 47.7 GiB PLE table
    # is served from. Layer-stream picture execution only.
    [ValidateSet('ram', 'mmap')]
    [string]$VisionWeights = 'ram',

    [switch]$EnableCacheReport,

    # Weight-only int8 for every dense projection the checkpoint ships bf16 (GDN, QSA,
    # hyper-connections, shared experts, lm_head): ~2x the decode GEMV speed and 3.9 GiB of
    # VRAM back for -MoECacheSize. Measured 2026-09-02 on the RTX 5090: 8k decode 53 -> 72
    # tok/s with the freed VRAM spent on slots. Quantization is per-output-row symmetric.
    [ValidateSet('', 'int8')]
    [string]$DenseQuant = '',

    # Keep the 1.27 GB token embedding in pinned host RAM (rows gathered over PCIe); frees
    # that VRAM for expert slots. Untied lm_head only.
    [switch]$EmbedHost,

    [ValidateRange(-1, 1024)]
    [int]$CudaGraphMaxBS = -1,

    [ValidateRange(0, 1048576)]
    [int]$MoECacheSize = 0,

    # Hard KV-pool capacity in tokens (--num-tokens). The default reserves the model's
    # full 262,144-token context; pass 0 to return to automatic sizing.
    [ValidateRange(0, 4194304)]
    [int]$KVCacheTokens = 262144,

    # Main QSA K/V storage. BF16 is the unchanged default and passes no new engine argument;
    # FP8 uses E4M3 plus per-token/head scales while the compressed QSA index stays BF16.
    [ValidateSet('bf16', 'fp8')]
    [string]$KVDtype = 'bf16',

    # Move completed QSA KV plus its GDN/PLE snapshot outside VRAM between turns. The launcher
    # passes even 'off' explicitly so an inherited FREETOKEN_KV_PARK cannot override this choice.
    [ValidateSet('off', 'ram', 'ssd')]
    [string]$KVPark = 'off',

    [ValidateRange(0, 86400000)]
    [int]$KVParkIdleMs = 0,

    [ValidateRange(64, 4194304)]
    [int]$KVParkMinTokens = 8192,

    [ValidateRange(0.125, 128)]
    [double]$KVParkRAMGiB = 2.0,

    [string]$KVParkSSDDir = '~/.cache/freetoken/kv-park',

    [ValidateRange(0.125, 8192)]
    [double]$KVParkSSDGiB = 32.0,

    # SSD keeps files as truth; these are two bounded pinned staging windows, not full copies.
    [ValidateRange(1, 4096)]
    [int]$KVParkWindowMiB = 256,

    # How the MoE expert banks are read into host RAM (--expert-load). 'parallel' is the
    # cache-bypassing multi-threaded reader (FILE_FLAG_NO_BUFFERING on Windows); 'serial' is
    # the older one-shard-at-a-time read; 'auto' lets the loader pick (it resolves to
    # parallel here, since this checkpoint's expert tensors are scattered).
    # Measured 2026-09-02 on the RTX 5090, same flags, back to back: the expert-load phase
    # is 43 s serial -> 37 s parallel (-14 %), standby stays flat across the load either way
    # (+0.0 / -0.4 GiB while 64 GiB of banks are pinned), and boot-to-serving is a wash
    # (73.1 s -> 73.2 s). Parallel costs ~1.4 GiB more peak RAM for its whole-shard buffers
    # (peak physical 85.4 -> 86.8 GiB, min available 10.3 -> 8.8 GiB) and the loader's
    # low-RAM fallback to serial is inert on Windows (it reads /proc/meminfo), so on a
    # tighter box pass -ExpertLoad serial explicitly.
    [ValidateSet('auto', 'serial', 'parallel')]
    [string]$ExpertLoad = 'parallel',

    # Accumulate the per-(MoE layer, expert) decode routing histogram and serve it at
    # GET /v1/cache/routing (--moe-collect-decode-freq). Boot-time only: the counters are
    # device-side ops that have to exist before CUDA graph capture. Research knob -- it
    # adds one scatter_add_ per MoE layer per decode step.
    [switch]$CollectRoutingStats,

    # MoE layers that keep every expert permanently resident in VRAM and allocate NO host
    # bank at all (--moe-gpu-owned-layers). Each owned layer hands back 1.32 GiB of host RAM
    # and costs 1.32 GiB of VRAM (about 512 LRU slots). Do NOT lower -MoECacheSize yourself:
    # it is the TOTAL expert-slot budget and the engine charges the owned layers to it, so
    # the card holds the same MoE bytes either way. 'auto' (six layers) therefore needs
    # -MoECacheSize >= 4096 = 6 x 512 charged + the 1024-slot prefill-overlap floor.
    # 'auto' is the six hungriest layers measured on this box
    # (docs/research/routing-skew-2026-09-02); 'auto:N' takes the first N; an explicit id
    # list, a count or a fraction also work. Empty (the default) leaves the feature off.
    # FREETOKEN_MOE_GPU_OWNED_LAYERS is the env fallback, read only here.
    [string]$GpuOwnedLayers = '',

    # VRAM the expert cache must NOT spend because it is allocated AFTER the cache is sized
    # (--moe-vram-reserve-bytes): the resident MTP draft head, the CUDA-graph pools, the
    # picture layer-stream workspace. -1 (the default) leaves the engine's auto composition
    # alone -- 0.75 GiB of graph pools plus 2.25 GiB of draft head when speculation is on.
    # Pass 0 to reserve nothing (the pre-2026-09 sizing), or a byte count to override.
    [ValidateRange(-1, 34359738368)]
    [long]$MoEVramReserveBytes = -1,

    # Free VRAM the expert cache must leave after every known reservation
    # (--moe-cache-headroom-bytes). -1 keeps the engine default (1.5 GiB, the floor every
    # healthy boot measured; the 569 MiB-free run halved decode throughput). An explicit
    # -MoECacheSize that leaves less refuses to boot, naming the largest size that fits.
    [ValidateRange(-1, 34359738368)]
    [long]$MoECacheHeadroomBytes = -1
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
    # Cheap argument check before the package probe below, so a bad combination reports
    # itself instead of a torchvision traceback.
    if ($VisionWeights -eq 'mmap' -and $VisionExecution -ne 'layer-stream') {
        throw "-VisionWeights mmap needs -VisionExecution layer-stream (got '$VisionExecution'). In gpu mode every picture tensor is copied to CUDA at load, so the mapping would be read once and then be dead address space."
    }
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
    $env:FREETOKEN_VISION_WEIGHTS = $VisionWeights
}
else {
    # Make the fallback deterministic even if the parent shell previously ran picture mode.
    $env:FREETOKEN_LOAD_VISION = '0'
    $env:FREETOKEN_VISION_EXECUTION = 'gpu'
    $env:FREETOKEN_VISION_WEIGHTS = 'ram'
}
# Switches only ever SET these; an operator who exported the env vars keeps them.
if ($DenseQuant -ne '') { $env:FREETOKEN_DENSE_QUANT = $DenseQuant }
if ($EmbedHost) { $env:FREETOKEN_EMBED_HOST = '1' }
# the env var is a fallback for the parameter, never an override of it
if (-not $GpuOwnedLayers -and $env:FREETOKEN_MOE_GPU_OWNED_LAYERS) {
    $GpuOwnedLayers = $env:FREETOKEN_MOE_GPU_OWNED_LAYERS
}
$pathParts += $sourceDir
$env:PYTHONPATH = ($pathParts -join ';') + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { '' })

Write-Host "Starting the unofficial Desktop-assisted Windows server"
Write-Host "  Model:  $resolvedModel"
Write-Host "  API:    http://127.0.0.1:$Port/v1"
Write-Host "  Context tokens: $ContextTokens"
Write-Host "  Active requests: $MaxRunningRequests"
Write-Host "  KV dtype: $KVDtype"
Write-Host "  MoE cache slots: $(if ($MoECacheSize -gt 0) { $MoECacheSize } else { 'auto' })"
Write-Host "  GPU-owned MoE layers: $(if ($GpuOwnedLayers) { $GpuOwnedLayers } else { 'off' })"
Write-Host "  Picture input: $($EnableVision.IsPresent)"
Write-Host "  Picture execution: $(if ($EnableVision) { $VisionExecution } else { 'disabled' })"
Write-Host "  Picture weights: $(if ($EnableVision) { $VisionWeights } else { 'disabled' })"
if ($KVPark -ne 'off') {
    Write-Host "  KV parking: $KVPark (idle $KVParkIdleMs ms, minimum $KVParkMinTokens tokens)"
}
Write-Host 'FreeToken Desktop supplies the Windows runtime but does not need to be open.'

# --moe-cache-size and --moe-cache-auto are mutually exclusive; an explicit size opts out
# of the auto sizing entirely.
$moeCacheArgs = if ($MoECacheSize -gt 0) {
    @('--moe-cache-size', "$MoECacheSize")
}
else {
    @('--moe-cache-auto')
}

$serveArgs = @(
    '-m', 'freetoken.cli', 'serve',
    '--model', $resolvedModel,
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--ple-backend', 'mmap',
    '--moe-backend', 'offload'
) + $moeCacheArgs + @(
    '--max-running-requests', "$MaxRunningRequests",
    '--kv-reserve-tokens', "$ContextTokens",
    '--expert-load', $ExpertLoad
)
if ($KVDtype -eq 'fp8') {
    $serveArgs += @('--kv-dtype', 'fp8')
}
$serveArgs += @('--kv-park', $KVPark)
if ($KVPark -ne 'off') {
    $serveArgs += @(
        '--kv-park-idle-ms', "$KVParkIdleMs",
        '--kv-park-min-tokens', "$KVParkMinTokens",
        '--kv-park-ram-gib', "$KVParkRAMGiB",
        '--kv-park-ssd-dir', $KVParkSSDDir,
        '--kv-park-ssd-gib', "$KVParkSSDGiB",
        '--kv-park-window-mib', "$KVParkWindowMiB"
    )
}
if ($EnableCacheReport) {
    $serveArgs += '--enable-cache-report'
}
if ($CollectRoutingStats) {
    $serveArgs += '--moe-collect-decode-freq'
}
if ($GpuOwnedLayers) {
    $serveArgs += @('--moe-gpu-owned-layers', $GpuOwnedLayers)
}
if ($MoEVramReserveBytes -ge 0) {
    $serveArgs += @('--moe-vram-reserve-bytes', "$MoEVramReserveBytes")
}
if ($MoECacheHeadroomBytes -ge 0) {
    $serveArgs += @('--moe-cache-headroom-bytes', "$MoECacheHeadroomBytes")
}
if ($CudaGraphMaxBS -ge 0) {
    $serveArgs += @('--cuda-graph-max-bs', "$CudaGraphMaxBS")
}
if ($KVCacheTokens -gt 0) {
    $serveArgs += @('--num-tokens', "$KVCacheTokens")
}

& $resolvedPython @serveArgs
exit $LASTEXITCODE
