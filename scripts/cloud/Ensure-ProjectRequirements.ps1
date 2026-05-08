. "$PSScriptRoot\Ensure-NumpyNumbaAbi.ps1"

function Get-RequirementsFingerprint([string]$RequirementsFile) {
    if (-not (Test-Path $RequirementsFile)) {
        throw "No existe requirements.txt en $RequirementsFile"
    }
    return (Get-FileHash -Path $RequirementsFile -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-ProjectDependencyProbe([string]$PythonExe) {
    & $PythonExe -c "import sys, streamlit, pandas, numpy, numba, boto3; np=tuple(int(x) for x in numpy.__version__.split('.')[:2]); nb=tuple(int(x) for x in numba.__version__.split('.')[:2]); py=sys.version_info[:2]; ok=np < (2, 4); ok = ok and (py != (3, 11) or ((0, 62) <= nb < (0, 63))); sys.exit(0 if ok else 1)"
    return ($LASTEXITCODE -eq 0)
}

function Ensure-ProjectRequirements([string]$PythonExe, [string]$ProjectDir, [switch]$Force) {
    $requirementsFile = Join-Path $ProjectDir "requirements.txt"
    if (-not (Test-Path $requirementsFile)) {
        throw "No existe requirements.txt en $ProjectDir"
    }

    $pyVersion = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo leer la version de Python"
    }
    $pyVersion = ($pyVersion | Select-Object -First 1).Trim()
    $fullSyncSupported = ($pyVersion -eq "3.11")

    $stateDir = Join-Path $ProjectDir "app\.tmp\bootstrap"
    $fingerprintFile = Join-Path $stateDir "requirements.sha256"
    New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

    $currentFingerprint = Get-RequirementsFingerprint -RequirementsFile $requirementsFile
    $storedFingerprint = ""
    if (Test-Path $fingerprintFile) {
        $storedFingerprint = (Get-Content $fingerprintFile -Raw | Out-String).Trim().ToLowerInvariant()
    }

    $needsInstall = $Force.IsPresent -or ($storedFingerprint -ne $currentFingerprint)
    if (-not $needsInstall) {
        $needsInstall = -not (Test-ProjectDependencyProbe -PythonExe $PythonExe)
    }

    if ($needsInstall -and -not $fullSyncSupported) {
        Write-Host "Saltando sincronizacion completa de requirements en Python $pyVersion; usar Python 3.11/.venv para despliegue cloud."
        Ensure-NumpyNumbaAbi -PythonExe $PythonExe
        return
    }

    if ($needsInstall) {
        Write-Host "Sincronizando dependencias del proyecto desde requirements.txt ..."
        & $PythonExe -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) {
            throw "Fallo actualizando pip"
        }

        & $PythonExe -m pip install -r $requirementsFile
        if ($LASTEXITCODE -ne 0) {
            throw "Fallo instalando requirements.txt"
        }

        if ($pyVersion -eq "3.11") {
            & $PythonExe -m pip install --no-deps --ignore-requires-python "pyeventbt==0.0.9"
            if ($LASTEXITCODE -ne 0) {
                throw "Fallo instalando pyeventbt==0.0.9 para Python 3.11"
            }
        }

        Set-Content -Path $fingerprintFile -Value $currentFingerprint -NoNewline
    }

    Ensure-NumpyNumbaAbi -PythonExe $PythonExe
}
