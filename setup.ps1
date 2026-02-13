# setup.ps1 - Automated environment setup for SpatialTranscriptFormer

Write-Host "--- SpatialTranscriptFormer Setup ---" -ForegroundColor Cyan

$EnvName = "SpatialTranscriptFormer"

# Check if conda environment exists
$CondaEnv = conda env list | Select-String $EnvName
if ($null -eq $CondaEnv) {
      Write-Host "Creating conda environment '$EnvName' with Python 3.10..." -ForegroundColor Yellow
      conda create -n $EnvName python=3.10 -y
}
else {
      Write-Host "Conda environment '$EnvName' already exists." -ForegroundColor Green
}

Write-Host "Installing/Updating package in editable mode..." -ForegroundColor Yellow
conda run -n $EnvName pip install -e .[dev]

Write-Host ""
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "You can now use the following commands:"
Write-Host "  stf-download --help"
Write-Host "  stf-split --help"
Write-Host ""
Write-Host "To run tests, use: .\test.ps1"
