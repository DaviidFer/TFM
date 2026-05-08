# Shared helper for EC2/local: finds the TFM repo root (folder with .git + requirements.txt).
# Usage (dot-source): . "$PSScriptRoot\Resolve-TfmProjectDir.ps1"

function Test-TfmProjectDirCandidate {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $false
    }
    return (Test-Path $Path) -and (Test-Path (Join-Path $Path "requirements.txt")) -and (Test-Path (Join-Path $Path ".git"))
}

function Resolve-TfmProjectCandidate {
    param([string]$Path)
    if (Test-TfmProjectDirCandidate -Path $Path) {
        return (Resolve-Path $Path).Path
    }
    return $null
}

function Get-TfmProjectDir {
    [CmdletBinding()]
    param()

    $ErrorActionPreference = "Stop"

    $candidates = @()

    if ($env:TFM_PROJECT_DIR) {
        $candidates += $env:TFM_PROJECT_DIR.Trim()
    }

    try {
        $cwd = (Get-Location).Path
        $candidates += $cwd
        $parent = Split-Path -Path $cwd -Parent
        while (-not [string]::IsNullOrWhiteSpace($parent) -and $parent -ne $cwd) {
            $candidates += $parent
            $cwd = $parent
            $parent = Split-Path -Path $cwd -Parent
        }
    }
    catch {
    }

    # Prefer the current public repo name if present.
    $candidates += @(
        "C:\tfm\tfm-project-gitpublic",
        "C:\tfm\tfm-project",
        "C:\tfm-trading",
        "C:\TFM",
        "C:\tfm\TFM"
    )

    foreach ($cand in $candidates) {
        $resolved = Resolve-TfmProjectCandidate -Path $cand
        if ($resolved) {
            return $resolved
        }
    }

    # Search one level under C:\tfm (bootstrap root).
    $tfmRoot = "C:\tfm"
    if (Test-Path $tfmRoot) {
        $repos = Get-ChildItem -Path $tfmRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { Test-TfmProjectDirCandidate -Path $_.FullName } |
            Sort-Object @{ Expression = { if ($_.Name -eq "tfm-project-gitpublic") { 0 } elseif ($_.Name -eq "tfm-project") { 1 } else { 2 } } }, LastWriteTime -Descending
        if ($repos) {
            return $repos[0].FullName
        }
    }

    throw @'
No se encontro el proyecto TFM (requirements.txt + .git).

Rutas prioritarias:
- C:\tfm\tfm-project-gitpublic
- C:\tfm\tfm-project

Solucion:
1) Clonar o bootstrap:
   cd C:\tfm
   git clone https://github.com/DaviidFer/TFM.git tfm-project-gitpublic

2) O en sesion PowerShell define la ruta real del repo:
   $env:TFM_PROJECT_DIR = "C:\tfm\tfm-project-gitpublic"
'@
}

function Set-TfmProjectEnvironment {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectDir
    )

    $resolved = (Resolve-Path $ProjectDir).Path
    $env:TFM_PROJECT_DIR = $resolved
    $env:TFM_ARTIFACTS_ROOT = Join-Path $resolved "app\.tmp"
    $env:TFM_DB_PATH = Join-Path $resolved "app\.tmp\supervisor\supervisor.sqlite"
}
