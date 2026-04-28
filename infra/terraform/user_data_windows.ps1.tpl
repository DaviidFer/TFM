<powershell>
$ErrorActionPreference = "Stop"
$bootstrapRoot = "C:\tfm"
$projectDir = "C:\tfm\tfm-project"
$bootstrapLogDir = Join-Path $bootstrapRoot "logs"
$bootstrapLogFile = Join-Path $bootstrapLogDir "bootstrap.log"

function Install-ChocolateyIfNeeded {
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        return
    }
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString("https://community.chocolatey.org/install.ps1"))
}

function Install-ChocoPackageIfNeeded([string]$CommandName, [string]$PackageName, [string]$ExtraArgs = "") {
    if (Get-Command $CommandName -ErrorAction SilentlyContinue) {
        return
    }
    if ([string]::IsNullOrWhiteSpace($ExtraArgs)) {
        choco install $PackageName -y
    }
    else {
        choco install $PackageName -y --params $ExtraArgs
    }
}

New-Item -ItemType Directory -Force -Path $bootstrapRoot | Out-Null
New-Item -ItemType Directory -Force -Path $bootstrapLogDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $bootstrapRoot "bootstrap") | Out-Null

Start-Transcript -Path $bootstrapLogFile -Append | Out-Null
try {
    Install-ChocolateyIfNeeded
    Install-ChocoPackageIfNeeded -CommandName "git" -PackageName "git"
    Install-ChocoPackageIfNeeded -CommandName "python" -PackageName "python" -ExtraArgs "'/InstallDir:C:\Python311 /PrependPath:1'"
    Install-ChocoPackageIfNeeded -CommandName "aws" -PackageName "awscli"

    [Environment]::SetEnvironmentVariable("AWS_REGION", "${aws_region}", "Machine")
    [Environment]::SetEnvironmentVariable("TFM_S3_BUCKET", "${s3_bucket_name}", "Machine")
    [Environment]::SetEnvironmentVariable("TFM_S3_PREFIX", "${s3_prefix}", "Machine")
    [Environment]::SetEnvironmentVariable("TFM_PROJECT_DIR", $projectDir, "Machine")
    [Environment]::SetEnvironmentVariable("STREAMLIT_PORT", "${streamlit_port}", "Machine")

    if (-not (Test-Path $projectDir)) {
        git clone --branch "${github_branch}" "${github_repo_url}" $projectDir
    }
    else {
        Set-Location $projectDir
        git fetch --all
        git checkout "${github_branch}"
        git pull origin "${github_branch}"
    }

    Set-Location $projectDir
    $envExample = Join-Path $projectDir ".env.example"
    $envFile = Join-Path $projectDir ".env"
    if ((Test-Path $envExample) -and (-not (Test-Path $envFile))) {
        Copy-Item $envExample $envFile
    }

    $venvPython = Join-Path $projectDir ".venv\Scripts\python.exe"
    $bootstrapPython = if (Test-Path "C:\Python311\python.exe") { "C:\Python311\python.exe" } else { "python" }
    if (-not (Test-Path $venvPython)) {
        & $bootstrapPython -m venv ".venv"
    }

    & $venvPython -m pip install --upgrade pip
    if (Test-Path (Join-Path $projectDir "requirements.txt")) {
        & $venvPython -m pip install -r "requirements.txt"
    }
}
finally {
    Stop-Transcript | Out-Null
}
</powershell>

