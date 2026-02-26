# setup.ps1 - Automated environment setup for SpatialTranscriptFormer

$ErrorActionPreference = 'Stop'

Write-Host "--- SpatialTranscriptFormer Setup ---" -ForegroundColor Cyan

$EnvName = "SpatialTranscriptFormer"

# Check if conda exists
try {
      conda --version | Out-Null
}
catch {
      Write-Error "Conda was not found. Please ensure Conda is installed and added to your PATH."
      exit 1
}

# Check if conda environment exists
$CondaEnv = conda env list | Select-String $EnvName
if ($null -eq $CondaEnv) {
      Write-Host "Creating conda environment '$EnvName' with Python 3.9..." -ForegroundColor Yellow
      conda create -n $EnvName python=3.9 -y
}
else {
      Write-Host "Conda environment '$EnvName' already exists." -ForegroundColor Green
}

Write-Host "Installing PyTorch (CUDA 11.8)..." -ForegroundColor Yellow
conda run -n $EnvName pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) {
      Write-Error "Failed to install PyTorch."
      exit $LASTEXITCODE
}

Write-Host "Installing/Updating package in editable mode..." -ForegroundColor Yellow
conda run -n $EnvName pip install -e .[dev]
if ($LASTEXITCODE -ne 0) {
      Write-Error "Failed to install SpatialTranscriptFormer."
      exit $LASTEXITCODE
}

Write-Host "Checking Hugging Face authentication..." -ForegroundColor Yellow
$HFLoginStatus = conda run -n $EnvName huggingface-cli whoami 2>&1
if ($LASTEXITCODE -ne 0 -or $HFLoginStatus -match "Not logged in") {
      $HFNeedLogin = $true
}
else {
      $HFNeedLogin = $false
      Write-Host "Hugging Face authentication found: $HFLoginStatus" -ForegroundColor Green
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "             SETUP COMPLETE!             " -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: You must activate the environment before using the tools:" -ForegroundColor Yellow
Write-Host "  conda activate $EnvName" -ForegroundColor Cyan
Write-Host ""

if ($HFNeedLogin) {
      Write-Host "------------------------------------------------------------" -ForegroundColor DarkYellow
      Write-Host "DATASET ACCESS REQUIRES AUTHENTICATION" -ForegroundColor Red
      Write-Host "The HEST-1k dataset on Hugging Face is gated. You must provide an access token." -ForegroundColor DarkYellow
      Write-Host "Please do ONE of the following before downloading data:"
      Write-Host "  Option A (Persistent): Run 'conda run -n $EnvName huggingface-cli login' and paste your token."
      Write-Host "  Option B (Temporary): Set the 'HF_TOKEN' environment variable."
      Write-Host "Get your token from: https://huggingface.co/settings/tokens" -ForegroundColor DarkCyan
      Write-Host "------------------------------------------------------------" -ForegroundColor DarkYellow
      Write-Host ""
}

Write-Host "You can then use the following commands:"
Write-Host "  stf-download --help"
Write-Host "  stf-split --help"
Write-Host "  stf-build-vocab --help"
Write-Host ""
Write-Host "To run tests, use: .\test.ps1"
