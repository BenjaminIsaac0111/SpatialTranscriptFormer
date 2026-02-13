# PowerShell Script to run and track HEST Bowel Cancer download progress

$TargetDir = "A:\hest_data"
$TargetSizeGB = 95
$TargetSizeFactor = 1GB * $TargetSizeGB
$PythonExe = "C:\Users\wispy\miniconda3\envs\SpatialTranscriptFormer\python.exe"

Write-Host "--- HEST Bowel Cancer Download Tracker ---" -ForegroundColor Cyan
Write-Host "Target Directory: $TargetDir"
Write-Host "Estimated Total Size: $TargetSizeGB GB"
Write-Host ""

# Check if Python script is running using CIM for CommandLine access
function Get-DownloadProcess {
    return Get-CimInstance Win32_Process -Filter "Name = 'python.exe' and CommandLine like '%download_bowel_cancer.py%'"
}

$Process = Get-DownloadProcess

if ($null -eq $Process) {
    Write-Host "Download script is not running. Starting it now..." -ForegroundColor Yellow
    if (-not (Test-Path $PythonExe)) {
        Write-Host "Error: Python executable not found at $PythonExe" -ForegroundColor Red
        exit
    }
    # Start the process in the background
    Start-Process $PythonExe -ArgumentList "scripts/download_bowel_cancer.py" -NoNewWindow
    Start-Sleep -Seconds 5 # Give it a bit more time to initialize
    $Process = Get-DownloadProcess
    
    if ($null -eq $Process) {
        Write-Host "Error: Failed to start the download script or it crashed immediately." -ForegroundColor Red
        Write-Host "Please check for error messages above." -ForegroundColor Yellow
        exit
    }
    Write-Host "Started download script (PID: $($Process.ProcessId))." -ForegroundColor Green
}
else {
    Write-Host "Download script is already running (PID: $($Process.ProcessId))." -ForegroundColor Green
}

$StartTime = Get-Date
$DownloadPID = $Process.ProcessId

while ($true) {
    if (Test-Path $TargetDir) {
        # Using .NET for much faster file enumeration than Get-ChildItem -Recurse
        [long]$CurrentSize = 0
        try {
            $files = [System.IO.Directory]::EnumerateFiles($TargetDir, "*", [System.IO.SearchOption]::AllDirectories)
            foreach ($file in $files) {
                $CurrentSize += (New-Object System.IO.FileInfo($file)).Length
            }
        }
        catch {
            # Directory might be busy or locked
        }
        
        $Percent = [Math]::Min(100, ($CurrentSize / $TargetSizeFactor) * 100)
        $CurrentGB = [Math]::Round($CurrentSize / 1GB, 2)
        
        $TimeElapsed = (Get-Date) - $StartTime
        if ($CurrentSize -gt 0 -and $TimeElapsed.TotalSeconds -gt 0) {
            $TotalEstimatedTime = New-TimeSpan -Seconds ($TimeElapsed.TotalSeconds * ($TargetSizeFactor / $CurrentSize))
            $RemainingTime = $TotalEstimatedTime - $TimeElapsed
            $Speed = [Math]::Round(($CurrentSize / 1MB) / $TimeElapsed.TotalSeconds, 2)
        }
        else {
            $RemainingTime = New-TimeSpan -Seconds 0
            $Speed = 0
        }

        $ProgressBar = "[" + ("#" * [Math]::Floor($Percent / 5)) + ("." * (20 - [Math]::Floor($Percent / 5))) + "]"
        
        # Format ETA as HH:MM:SS
        $etaStr = if ($RemainingTime.TotalSeconds -gt 0) { 
            "$([int]$RemainingTime.TotalHours):$($RemainingTime.Minutes.ToString('00')):$($RemainingTime.Seconds.ToString('00'))" 
        }
        else { "Calculating..." }

        $Status = "$ProgressBar $($Percent.ToString("F2"))% | $CurrentGB / $TargetSizeGB GB | Speed: $Speed MB/s | ETA: $etaStr"
        Write-Progress -Activity "Downloading HEST Bowel Cancer Subset" -Status $Status -PercentComplete $Percent
        
        # Also print to console
        Write-Host "`r$Status" -NoNewline
    }
    else {
        Write-Host "`rWaiting for target directory to be created..." -NoNewline
    }
    
    # Check if process is still running
    $CurrentProc = Get-CimInstance Win32_Process -Filter "ProcessId = $DownloadPID"
    if ($null -eq $CurrentProc -and $Percent -lt 99) {
        Write-Host "`nWARNING: Download process (PID $DownloadPID) seems to have stopped unexpectedly!" -ForegroundColor Red
        Write-Host "Check the terminal output above for any Python errors." -ForegroundColor Yellow
        break
    }

    if ($Percent -ge 100) {
        Write-Host "`nDownload complete!" -ForegroundColor Green
        break
    }
    
    Start-Sleep -Seconds 5
}
