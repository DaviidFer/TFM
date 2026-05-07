[CmdletBinding()]
param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\Resolve-TfmProjectDir.ps1"

function Get-PythonExe([string]$ProjectDir) {
    $venvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return "python"
}

function Ensure-StreamlitDependencies([string]$PythonExe, [string]$ProjectDir) {
    if ($SkipInstall) {
        return
    }
    $requirementsFile = Join-Path $ProjectDir "requirements.txt"
    & $PythonExe -c "import streamlit" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "streamlit no encontrado; instalando desde requirements.txt ..."
        & $PythonExe -m pip install --upgrade pip
        & $PythonExe -m pip install -r $requirementsFile
        if ($LASTEXITCODE -ne 0) {
            throw "Fallo pip install -r requirements.txt"
        }
        $pyVersion = (& $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" | Select-Object -First 1).Trim()
        if ($pyVersion -eq "3.11") {
            & $PythonExe -m pip install --no-deps --ignore-requires-python "pyeventbt==0.0.9"
        }
    }
}

$projectDir = Get-TfmProjectDir
$logDir = Join-Path $projectDir "app\.tmp\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "streamlit.log"
$streamlitPort = if ($env:STREAMLIT_PORT) { $env:STREAMLIT_PORT } else { "8501" }

Start-Transcript -Path $logFile -Append | Out-Null
try {
    Set-Location $projectDir
    $pythonExe = Get-PythonExe -ProjectDir $projectDir
    Ensure-StreamlitDependencies -PythonExe $pythonExe -ProjectDir $projectDir

    & $pythonExe -m streamlit run "app\ui\dashboard.py" --server.port $streamlitPort --server.address "0.0.0.0" --server.headless "true" --browser.gatherUsageStats "false"
}
finally {
    Stop-Transcript | Out-Null
}
