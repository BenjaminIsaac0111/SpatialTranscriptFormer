# test.ps1 - Run project tests

Write-Host "--- Running SpatialTranscriptFormer Tests ---" -ForegroundColor Cyan

$EnvName = "SpatialTranscriptFormer"

conda run -n $EnvName pytest tests/
