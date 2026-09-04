[CmdletBinding()]
param(
    # General Windows launcher: the parameters of start-qwen38-flash-next-mmap-windows.ps1
    # with the model-specific choices made from the model's own config.json instead of
    # assumed. Read config.json (never the weights) and:
    #   - pass --ple-backend mmap only when the model has a PLE table (Qwen3.8-Flash-Next);
    #   - allow KV parking only on the QSA+GDN hybrid it was built for (qwen4_exp), else
    #     force it off with a note (the engine refuses it: kvcache/park_store.py);
    #   - load the picture tower only when the model has a vision_config;
    #   - keep the MTP switches off for a model that ships no MTP head.
    # -DryRun prints the resolved engine command and exits without starting anything, so a
    # profile can be checked without touching the card.
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ModelPath,

    [ValidateRange(1, 65529)]
    [int]$Port = 2020,

    # Technical ceiling only; the real limit is the model's max_position_embeddings, which
    # the settings page enforces and this script re-checks from config.json.
    [ValidateRange(64, 4194304)]
    [int]$ContextTokens = 0,

    [ValidateRange(1, 16)]
    [int]$MaxRunningRequests = 4,

    [string]$DesktopPython = (Join-Path $env:LOCALAPPDATA 'FreeToken\venv\Scripts\python.exe'),

    [string]$VisionPackagesPath,

    [switch]$EnableVision,

    [ValidateSet('layer-stream', 'gpu')]
    [string]$VisionExecution = 'layer-stream',

    [ValidateSet('ram', 'mmap')]
    [string]$VisionWeights = 'ram',

    # 'auto' = mmap when the model has a PLE table, nothing otherwise.
    [ValidateSet('auto', 'mmap', 'pinned', 'off')]
    [string]$PleBackend = 'auto',

    [switch]$EnableCacheReport,

    [ValidateSet('', 'int8')]
    [string]$DenseQuant = '',

    [switch]$EmbedHost,

    [ValidateRange(-1, 1024)]
    [int]$CudaGraphMaxBS = -1,

    [ValidateRange(0, 1048576)]
    [int]$MoECacheSize = 0,

    # 0 = automatic sizing (--moe-cache-auto / no --num-tokens).
    [ValidateRange(0, 4194304)]
    [int]$KVCacheTokens = 0,

    [ValidateSet('bf16', 'fp8')]
    [string]$KVDtype = 'bf16',

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

    [ValidateRange(1, 4096)]
    [int]$KVParkWindowMiB = 256,

    [ValidateSet('auto', 'serial', 'parallel')]
    [string]$ExpertLoad = 'auto',

    [switch]$CollectRoutingStats,

    # "auto" / "auto:N" use the ranking measured for Qwen3.8 (engine GPU_OWNED_LAYER_RANK);
    # a plain count spreads the layers evenly, which is what the settings page writes for
    # any other model.
    [string]$GpuOwnedLayers = '',

    [ValidateRange(-1, 34359738368)]
    [long]$MoEVramReserveBytes = -1,

    [ValidateRange(-1, 34359738368)]
    [long]$MoECacheHeadroomBytes = -1,

    # Print the engine command and exit 0 without starting Python.
    [switch]$DryRun
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
$configPath = Join-Path $ModelPath 'config.json'
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "No config.json in the model directory: $ModelPath"
}

# ---- what the model is, from config.json alone -------------------------------------
$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$text = if ($config.PSObject.Properties['text_config'] -and $config.text_config) { $config.text_config } else { $config }
function Get-Field($obj, $name) {
    if ($null -ne $obj -and $obj.PSObject.Properties[$name]) { return $obj.$name }
    return $null
}
$architecture = if ($config.architectures) { [string]$config.architectures[0] } else { '' }
$modelType = [string](Get-Field $text 'model_type')
if (-not $modelType) { $modelType = [string](Get-Field $config 'model_type') }
$pleIds = Get-Field $text 'ple_layer_ids'
if ($null -eq $pleIds) { $pleIds = Get-Field $config 'ple_layer_ids' }
$hasPle = ($null -ne $pleIds) -and (@($pleIds).Count -gt 0)
$hasVision = $null -ne (Get-Field $config 'vision_config')
$mtpLayers = Get-Field $text 'mtp_num_hidden_layers'
if ($null -eq $mtpLayers) { $mtpLayers = Get-Field $text 'num_nextn_predict_layers' }
if ($null -eq $mtpLayers) { $mtpLayers = Get-Field $config 'mtp_num_hidden_layers' }
if ($null -eq $mtpLayers) { $mtpLayers = Get-Field $config 'num_nextn_predict_layers' }
$hasMtp = ($null -ne $mtpLayers) -and ([int]$mtpLayers -gt 0)
$maxContext = Get-Field $text 'max_position_embeddings'
if ($null -eq $maxContext) { $maxContext = Get-Field $config 'max_position_embeddings' }
$numLayers = Get-Field $text 'num_hidden_layers'
if ($null -eq $numLayers) { $numLayers = Get-Field $text 'n_layer' }
if ($null -eq $numLayers) { $numLayers = Get-Field $config 'num_hidden_layers' }
if ($null -eq $numLayers) { $numLayers = Get-Field $config 'n_layer' }
$firstDense = Get-Field $text 'first_k_dense_replace'
if ($null -eq $firstDense) { $firstDense = Get-Field $config 'first_k_dense_replace' }
$moeLayerCount = if ($null -ne $numLayers) { [math]::Max(0, [int]$numLayers - [int]($firstDense -as [int])) } else { 0 }
$experts = Get-Field $text 'num_experts'
if ($null -eq $experts) { $experts = Get-Field $text 'n_routed_experts' }
if ($null -eq $experts) { $experts = Get-Field $text 'num_local_experts' }
if ($null -eq $experts) { $experts = Get-Field $config 'num_experts' }
if ($null -eq $experts) { $experts = Get-Field $config 'n_routed_experts' }
if ($null -eq $experts) { $experts = Get-Field $config 'num_local_experts' }
$isMoe = ($null -ne $experts) -and ([int]$experts -gt 0)
$modelExpertCount = if ($isMoe -and $moeLayerCount -gt 0) { $moeLayerCount * [int]$experts } else { 0 }
# KV parking is implemented for the QSA + GDN hybrid cache only (kvcache/park_store.py).
$parkingSupported = ($modelType -eq 'qwen4_exp_text') -or ($modelType -eq 'qwen4_exp')

$notes = @()
if ($ContextTokens -le 0) {
    $ContextTokens = if ($maxContext) { [int]$maxContext } else { 32768 }
    $notes += "Context tokens not given; using the model's own limit $ContextTokens"
}
elseif ($maxContext -and $ContextTokens -gt [int]$maxContext) {
    throw "-ContextTokens $ContextTokens is longer than this model can read ($maxContext, config.json max_position_embeddings)"
}
$resolvedPle = switch ($PleBackend) {
    'auto' { if ($hasPle) { 'mmap' } else { 'off' } }
    'off' { 'off' }
    default {
        if (-not $hasPle) { $notes += "-PleBackend $PleBackend ignored: this model has no PLE table"; 'off' } else { $PleBackend }
    }
}
if ($KVPark -ne 'off' -and -not $parkingSupported) {
    $notes += "KV parking '$KVPark' forced off: the engine parks chats only for the Qwen3.8-Flash-Next (qwen4_exp) memory layout"
    $KVPark = 'off'
}
if ($EnableVision -and -not $hasVision) {
    $notes += 'Picture input switched off: this model has no picture tower (no vision_config)'
    $EnableVision = $false
}
if (-not $isMoe -and ($MoECacheSize -gt 0 -or $GpuOwnedLayers)) {
    $notes += 'Expert-slot settings ignored: this model has no routed experts'
    $MoECacheSize = 0
    $GpuOwnedLayers = ''
}
if ($isMoe -and $modelExpertCount -gt 0 -and $MoECacheSize -gt $modelExpertCount) {
    $notes += "MoE cache slots clamped from $MoECacheSize to ${modelExpertCount}: this model has only $modelExpertCount expert pieces"
    $MoECacheSize = $modelExpertCount
}
if (-not $hasMtp) {
    # The MTP head is a Qwen3.8 private file; keep every entry switch off for other models.
    foreach ($name in 'FREETOKEN_MTP_SPECULATE', 'FREETOKEN_MTP_RESIDENT', 'FREETOKEN_MTP_SHADOW', 'FREETOKEN_MTP_SPEC_GRAPH') {
        if ((Get-Item -Path "env:$name" -ErrorAction SilentlyContinue).Value -eq '1') {
            $notes += "$name forced off: this model ships no MTP head"
        }
        Set-Item -Path "env:$name" -Value '0'
    }
}

# ---- runtime checks (same as the Qwen launcher) ---------------------------------------
if (-not $DryRun) {
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
    $env:CUDA_PATH = (Resolve-Path -LiteralPath $cudaRoot).Path
}

$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
$resolvedPython = if (Test-Path -LiteralPath $DesktopPython -PathType Leaf) { (Resolve-Path -LiteralPath $DesktopPython).Path } else { $DesktopPython }

$pathParts = @($shimDir)
if ($EnableVision) {
    if ($VisionWeights -eq 'mmap' -and $VisionExecution -ne 'layer-stream') {
        throw "-VisionWeights mmap needs -VisionExecution layer-stream (got '$VisionExecution')."
    }
    if (-not $DryRun) {
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
    }
    $env:FREETOKEN_LOAD_VISION = '1'
    $env:FREETOKEN_VISION_EXECUTION = $VisionExecution
    $env:FREETOKEN_VISION_WEIGHTS = $VisionWeights
}
else {
    $env:FREETOKEN_LOAD_VISION = '0'
    $env:FREETOKEN_VISION_EXECUTION = 'gpu'
    $env:FREETOKEN_VISION_WEIGHTS = 'ram'
}
# Resolve these switches for every run so a previous profile cannot leak into this one.
$env:FREETOKEN_DENSE_QUANT = if ($DenseQuant -ne '') { $DenseQuant } else { 'none' }
$env:FREETOKEN_EMBED_HOST = if ($EmbedHost) { '1' } else { '0' }
if (-not $PSBoundParameters.ContainsKey('GpuOwnedLayers') -and $env:FREETOKEN_MOE_GPU_OWNED_LAYERS) {
    $GpuOwnedLayers = $env:FREETOKEN_MOE_GPU_OWNED_LAYERS
}
$pathParts += $sourceDir
$env:PYTHONPATH = ($pathParts -join ';') + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { '' })

Write-Host "Starting the unofficial Desktop-assisted Windows server"
Write-Host "  Model:  $resolvedModel ($architecture)"
Write-Host "  API:    http://127.0.0.1:$Port/v1"
Write-Host "  Context tokens: $ContextTokens"
Write-Host "  Active requests: $MaxRunningRequests"
Write-Host "  KV dtype: $KVDtype"
Write-Host "  PLE table: $resolvedPle"
Write-Host "  MoE cache slots: $(if ($MoECacheSize -gt 0) { $MoECacheSize } else { 'auto' })"
Write-Host "  GPU-owned MoE layers: $(if ($GpuOwnedLayers) { $GpuOwnedLayers } else { 'off' })"
Write-Host "  Picture input: $([bool]$EnableVision)"
Write-Host "  KV parking: $KVPark"
foreach ($note in $notes) { Write-Host "  Note: $note" }

# [string[]] so a one-element result stays an array (PowerShell unrolls single-element
# arrays, and '--moe-cache-auto' + @(...) would otherwise concatenate into one string).
[string[]]$moeCacheArgs = if (-not $isMoe) { @() } elseif ($MoECacheSize -gt 0) { @('--moe-cache-size', "$MoECacheSize") } else { @('--moe-cache-auto') }

$serveArgs = @(
    '-m', 'freetoken.cli', 'serve',
    '--model', $resolvedModel,
    '--host', '127.0.0.1',
    '--port', "$Port"
)
if ($resolvedPle -ne 'off') { $serveArgs += @('--ple-backend', $resolvedPle) }
if ($isMoe) { $serveArgs += @('--moe-backend', 'offload') }
$serveArgs += $moeCacheArgs + @(
    '--max-running-requests', "$MaxRunningRequests",
    '--kv-reserve-tokens', "$ContextTokens"
)
if ($isMoe) { $serveArgs += @('--expert-load', $ExpertLoad) }
if ($KVDtype -eq 'fp8') { $serveArgs += @('--kv-dtype', 'fp8') }
if ($parkingSupported) {
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
}
if ($EnableCacheReport) { $serveArgs += '--enable-cache-report' }
if ($CollectRoutingStats -and $isMoe) { $serveArgs += '--moe-collect-decode-freq' }
if ($isMoe -and $GpuOwnedLayers) { $serveArgs += @('--moe-gpu-owned-layers', $GpuOwnedLayers) }
if ($isMoe -and $MoEVramReserveBytes -ge 0) { $serveArgs += @('--moe-vram-reserve-bytes', "$MoEVramReserveBytes") }
if ($isMoe -and $MoECacheHeadroomBytes -ge 0) { $serveArgs += @('--moe-cache-headroom-bytes', "$MoECacheHeadroomBytes") }
if ($CudaGraphMaxBS -ge 0) { $serveArgs += @('--cuda-graph-max-bs', "$CudaGraphMaxBS") }
if ($KVCacheTokens -gt 0) { $serveArgs += @('--num-tokens', "$KVCacheTokens") }

if ($DryRun) {
    Write-Host 'DRY RUN: the engine command would be'
    Write-Host ("  " + $resolvedPython + ' ' + ($serveArgs -join ' '))
    exit 0
}

& $resolvedPython @serveArgs
exit $LASTEXITCODE
