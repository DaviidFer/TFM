# Shared helper for EC2/local: finds the TFM repo root (folder with .git + requirements.txt).
# Usage (dot-source): . "$PSScriptRoot\Resolve-TfmProjectDir.ps1"

function Get-TfmProjectDir {
    [CmdletBinding()]
    param()

    $ErrorActionPreference = "Stop"

    if ($env:TFM_PROJECT_DIR) {
        $cand = $env:TFM_PROJECT_DIR.Trim()
        if ((Test-Path $cand) -and (Test-Path (Join-Path $cand "requirements.txt"))) {
            return (Resolve-Path $cand).Path
        }
    }

    # Default path written by bootstrap_windows_ec2.ps1 (Machine env TFM_PROJECT_DIR).
    $bootstrapDefault = "C:\tfm\tfm-project"
    if ((Test-Path $bootstrapDefault) -and (Test-Path (Join-Path $bootstrapDefault "requirements.txt"))) {
        return (Resolve-Path $bootstrapDefault).Path
    }

    # Legacy/alternate names users tried manually.
    $extra = @(
        "C:\tfm-trading",
        "C:\TFM",
        "C:\tfm\TFM"
    )
    foreach ($p in $extra) {
        if ((Test-Path $p) -and (Test-Path (Join-Path $p "requirements.txt"))) {
            return (Resolve-Path $p).Path
        }
    }

    # Search one level under C:\tfm (bootstrap root).
    $tfmRoot = "C:\tfm"
    if (Test-Path $tfmRoot) {
        Get-ChildItem -Path $tfmRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $req = Join-Path $_.FullName "requirements.txt"
            $git = Join-Path $_.FullName ".git"
            if ((Test-Path $req) -and (Test-Path $git)) {
                return $_.FullName
            }
        }
    }

    throw @'
No se encontro el proyecto TFM (requirements.txt + .git).

Esperado tras bootstrap: C:\tfm\tfm-project

Solucion:
1) Clonar o bootstrap:
   cd C:\tfm
   git clone https://github.com/DaviidFer/TFM.git tfm-project
   cd .\tfm-project
   .\scripts\cloud\bootstrap_windows_ec2.ps1 -RepoUrl https://github.com/DaviidFer/TFM.git -Branch main

2) O en sesion PowerShell define la ruta real del repo:
   $env:TFM_PROJECT_DIR = "C:\tfm\tfm-project"
'@
}
