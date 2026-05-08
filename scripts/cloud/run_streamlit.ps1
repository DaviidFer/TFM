[CmdletBinding()]
param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\Resolve-TfmProjectDir.ps1"
. "$PSScriptRoot\Ensure-ProjectRequirements.ps1"

function Get-PythonExe([string]$ProjectDir) {
    $venvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return "python"
}

$projectDir = Get-TfmProjectDir
Set-TfmProjectEnvironment -ProjectDir $projectDir
$logDir = Join-Path $projectDir "app\.tmp\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "streamlit.log"
$streamlitPort = if ($env:STREAMLIT_PORT) { $env:STREAMLIT_PORT } else { "8501" }

Start-Transcript -Path $logFile -Append | Out-Null
try {
    Set-Location $projectDir
    $pythonExe = Get-PythonExe -ProjectDir $projectDir
    if (-not $SkipInstall) {
        Ensure-ProjectRequirements -PythonExe $pythonExe -ProjectDir $projectDir
    }

    & $pythonExe -m streamlit run "app\ui\dashboard.py" --server.port $streamlitPort --server.address "0.0.0.0" --server.headless "true" --browser.gatherUsageStats "false"
}
finally {
    Stop-Transcript | Out-Null
}
