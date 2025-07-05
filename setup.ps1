# setup_pyldpc.ps1

$pythonVersion = "3.10.13"
$venvName = "venv310"
$projectDir = "$PSScriptRoot"
$requirementsFile = Join-Path $projectDir "requirements.txt"

Write-Host "Starting Python $pythonVersion environment setup..."

# Step 1: Install Python if needed
$pythonInstalled = & py -3.10 --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python $pythonVersion not found. Downloading..."
    $installer = "python-$pythonVersion-amd64.exe"
    $url = "https://www.python.org/ftp/python/$pythonVersion/$installer"
    Invoke-WebRequest -Uri $url -OutFile $installer
    Start-Process -Wait -FilePath .\$installer -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0"
    Remove-Item .\$installer
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
} else {
    Write-Host "Python $pythonVersion is already installed."
}

# Step 2: Create virtual environment
$venvPath = Join-Path $projectDir $venvName
if (!(Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    & py -3.10 -m venv $venvName
} else {
    Write-Host "Virtual environment already exists. Skipping."
}

# Step 3: Create a default requirements.txt if missing
if (!(Test-Path $requirementsFile)) {
    Write-Host "Creating default requirements.txt..."
    @"
matplotlib
scipy
scikit-commpy
"@ | Set-Content -Encoding UTF8 $requirementsFile
}

# Step 4: Activate environment, install numpy + pyldpc + rest, run test python file
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
Write-Host "Installing dependencies (numpy, pyldpc, others)..."

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command",
    @"
& '$activateScript';
python -m pip install --upgrade pip;
pip install wheel;
pip install numpy;
pip install --no-build-isolation pyldpc;
pip install -r '$requirementsFile';
Write-Host '===========================' -ForegroundColor Cyan
Write-Host ' Running test.py...' -ForegroundColor Cyan
Write-Host '===========================' -ForegroundColor Cyan
python src\main.py;
"@
)