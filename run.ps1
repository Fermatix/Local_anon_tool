# LocalAnonPipeline in one command (Windows).
#
#   powershell -ExecutionPolicy Bypass -File run.ps1 <export folder or archive> [output folder]
#
# Output: out\<export name> in this folder. The models archive (localanon-models-<date>.zip, sent
# separately) must lie in this folder before the first start: it is unpacked into models\. The first start
# also installs everything into .venv next to this
# script: the package with all its dependencies, onnxruntime-gpu with CUDA libraries when an NVIDIA card is
# present, and the speech recognition model (whisper) sized for this machine. This is the only step that
# uses the network. Later starts reuse .venv (reinstalling only when pyproject.toml changes) and run offline;
# if the hardware changed and the run plan needs another whisper size, it is downloaded first.
#
#   $env:LOCALANON_GPU = "no"    # do not install onnxruntime-gpu (yes: always; default: auto)
#
# This file is ASCII only: Windows PowerShell 5.1 reads scripts in the ANSI code page.
$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Cli = Join-Path $Venv "Scripts\localanon.exe"
$Mark = Join-Path $Venv ".localanon-installed"
$Config = Join-Path $Root "localanon.toml"
$Gpu = if ($env:LOCALANON_GPU) { $env:LOCALANON_GPU } else { "auto" }

function Say([string]$Text) { Write-Host "== $Text" }
function Die([string]$Text) { [Console]::Error.WriteLine("run.ps1: $Text"); exit 2 }

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "run.ps1: failed (exit code $LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

if ($args.Count -lt 1 -or $args[0] -in @("-h", "--help", "/?")) {
    [Console]::Error.WriteLine("usage: run.ps1 <export folder or archive> [output folder]")
    exit 2
}
$In = [string]$args[0]
if (-not (Test-Path -LiteralPath $In)) { Die "input not found: $In" }
$In = (Resolve-Path -LiteralPath $In).Path
$Name = Split-Path -Leaf ($In.TrimEnd([char[]]"\/"))
if (Test-Path -LiteralPath $In -PathType Leaf) {
    if ($Name -like "*.tar.gz") { $Name = $Name.Substring(0, $Name.Length - 7) }
    elseif ($Name -match "\.(zip|tar|tgz|7z|rar)$") { $Name = [IO.Path]::GetFileNameWithoutExtension($Name) }
}
$Out = if ($args.Count -ge 2) { [string]$args[1] } else { Join-Path (Join-Path $Root "out") $Name }

function Find-Python {    # first interpreter >= 3.11: the py launcher, then python on PATH
    $check = "import sys; sys.exit(sys.version_info < (3, 11))"
    foreach ($c in @(@("py", "-3.13"), @("py", "-3.12"), @("py", "-3.11"), @("python"), @("python3"))) {
        if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue)) { continue }
        $rest = @($c | Select-Object -Skip 1)
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $c[0] @rest -c $check *> $null
            if ($LASTEXITCODE -eq 0) { return , $c }
        } catch {
        } finally {
            $ErrorActionPreference = $prevEap
        }
    }
    return $null
}

if (Test-Path -LiteralPath $Py) {
    $BaseExe = $Py; $BaseArgs = @()
} else {
    $found = Find-Python
    if (-not $found) { Die "Python 3.11 or newer is required (https://www.python.org/downloads/)" }
    $BaseExe = $found[0]; $BaseArgs = @($found | Select-Object -Skip 1)
}

# The models come as a separate archive: the first start unpacks it into models\ (checked by models.lock.json)
$env:PYTHONPATH = $Root
& $BaseExe @BaseArgs @("-m", "localanon.fetch", "--ensure-models", $Root)
if ($LASTEXITCODE -ne 0) { exit 1 }
Remove-Item Env:PYTHONPATH

if ($Gpu -eq "auto") {
    $Gpu = "no"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & nvidia-smi -L *> $null
            if ($LASTEXITCODE -eq 0) { $Gpu = "yes" }
        } catch {
        } finally {
            $ErrorActionPreference = $prevEap
        }
    }
}
$Stamp = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Root "pyproject.toml")).Hash.ToLower() + " gpu=$Gpu"

function Get-PkgVersion([string]$Dist) {    # installed version of a distribution or ""
    $code = "import sys, importlib.metadata as m; print(next((d.version for d in m.distributions(name=sys.argv[1])), ''))"
    return ([string](& $Py -c $code $Dist)).Trim()
}

function Install-Venv {
    Say "installing into $Venv (first start or pyproject.toml changed)"
    if (-not (Test-Path -LiteralPath $Py)) { Invoke-Checked $BaseExe ($BaseArgs + @("-m", "venv", $Venv)) }
    Invoke-Checked $Py @("-m", "pip", "install", "--upgrade", "pip")
    # editable: the package stays in this folder and finds models\ next to it
    Invoke-Checked $Py @("-m", "pip", "install", "-e", $Root)
    # the engine's C accelerator (Aho-Corasick): optional, a fresh Python may have no wheel for it
    $acOk = $false
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Py -m pip install -q pyahocorasick *> $null
        $acOk = ($LASTEXITCODE -eq 0)
    } catch {
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if (-not $acOk) {
        [Console]::Error.WriteLine("run.ps1: pyahocorasick is not installed - the engine works without it, only slower")
    }
    if ($Gpu -eq "yes") {
        Say "NVIDIA card: onnxruntime-gpu instead of the CPU onnxruntime"
        # GPU build of the same version as the CPU build; if PyPI has none, the latest one
        $ver = Get-PkgVersion "onnxruntime"
        $ok = $false
        if ($ver) {
            & $Py -m pip install "onnxruntime-gpu[cuda,cudnn]==$ver"
            $ok = ($LASTEXITCODE -eq 0)
        }
        if (-not $ok) { Invoke-Checked $Py @("-m", "pip", "install", "onnxruntime-gpu[cuda,cudnn]") }
        # Both builds write the same onnxruntime module: removing the CPU build deletes files of the GPU
        # build too, so the GPU build is reinstalled on top without deps (otherwise pip brings the CPU one back).
        Invoke-Checked $Py @("-m", "pip", "uninstall", "-y", "onnxruntime")
        $GpuVer = Get-PkgVersion "onnxruntime-gpu"
        Invoke-Checked $Py @("-m", "pip", "install", "--force-reinstall", "--no-deps", "onnxruntime-gpu==$GpuVer")
        # faster-whisper (ctranslate2) needs cuBLAS 12 and cuDNN 9 for CUDA 12, onnxruntime-gpu[cuda,cudnn]
        # brings CUDA 13 libraries; both cuDNN packages write the same files: cu12 is always the last pip step.
        Say "speech recognition on the card: cuBLAS 12 and cuDNN 9 (last)"
        Invoke-Checked $Py @("-m", "pip", "install", "nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*")
        Say "check: localanon hw --probe-cuda"
        & $Cli @("hw", "--probe-cuda")
        if ($LASTEXITCODE -ne 0) {
            [Console]::Error.WriteLine("run.ps1: the card did not pass the check (reason above) - processing runs on CPU")
        }
    }
    Say "speech recognition model (whisper) for this machine"
    Invoke-Checked $Py @("-m", "localanon.fetch", "--asr", "auto")
    Set-Content -LiteralPath $Mark -Value $Stamp -Encoding ascii
}

$installed = ""
if (Test-Path -LiteralPath $Mark) { $installed = (Get-Content -LiteralPath $Mark -Raw).Trim() }
if (-not (Test-Path -LiteralPath $Py) -or $installed -ne $Stamp) {
    Install-Venv
} else {
    # the hardware changed after the installation (more memory, another card): the run plan picks another
    # whisper size; without it the run falls back to the size in the cache or removes media with a mark
    $env:HF_HUB_OFFLINE = "1"
    & $Py @("-m", "localanon.fetch", "--asr", "auto", "--check")
    $asrRc = $LASTEXITCODE
    Remove-Item Env:HF_HUB_OFFLINE
    if ($asrRc -ne 0) {
        Say "speech recognition model (whisper) for this machine"
        & $Py @("-m", "localanon.fetch", "--asr", "auto")
        if ($LASTEXITCODE -ne 0) {
            [Console]::Error.WriteLine("run.ps1: whisper download failed - media are transcribed with the model in the cache, if any")
        }
    }
}

if (-not (Test-Path -LiteralPath $Config)) {
    # fresh salt: tokens are derived from it, so the file stays on this machine
    Invoke-Checked $Cli @("init", $Config)
}

Say "processing $In"
$env:HF_HUB_OFFLINE = "1"
& $Cli @("run", "--in", $In, "--out", $Out, "--config", $Config)
$rc = $LASTEXITCODE

Write-Host ""
Write-Host "Output folder: $Out"
Write-Host "  report (stays with you): $(Join-Path $Out 'report.html')"
if ($rc -eq 0) {
    foreach ($f in @(Get-ChildItem -LiteralPath $Out -Filter "*.zip" -File -ErrorAction SilentlyContinue)) {
        Write-Host "  to hand over: $($f.FullName)"
    }
    $manifest = Join-Path $Out "local_manifest.json"
    if (Test-Path -LiteralPath $manifest) { Write-Host "  to hand over: $manifest" }
} else {
    Write-Host "  no archive (exit code $rc): see the report and the message above"
}
Write-Host "Settings (executor names, chats to drop, files to restore): $Config - edit and run again."
exit $rc
