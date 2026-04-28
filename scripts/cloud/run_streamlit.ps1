[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Get-ProjectDir {
    if ($env:TFM_PROJECT_DIR) {
        return $env:TFM_PROJECT_DIR
    }
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Get-PythonExe([string]$ProjectDir) {
    $venvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return "python"
}

$projectDir = Get-ProjectDir
$logDir = Join-Path $projectDir "app\.tmp\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "streamlit.log"
$streamlitPort = if ($env:STREAMLIT_PORT) { $env:STREAMLIT_PORT } else { "8501" }

Start-Transcript -Path $logFile -Append | Out-Null
try {
    Set-Location $projectDir
    $pythonExe = Get-PythonExe -ProjectDir $projectDir
    & $pythonExe -m streamlit run "app\ui\dashboard.py" --server.port $streamlitPort --server.address "0.0.0.0"
}
finally {
    Stop-Transcript | Out-Null
}

