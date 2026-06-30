<#
.SYNOPSIS
    Stop the TabVision Studio servers (backend :5000 and frontend :5173).
#>
[CmdletBinding()]
param()

function Stop-Port {
    param([int]$Port, [string]$Label)
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Host "  $Label (:$Port) not running." -ForegroundColor DarkGray
        return
    }
    $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $pids) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host "  Stopped $Label (:$Port) pid $procId." -ForegroundColor Green
        } catch {
            Write-Host "  Could not stop pid $procId for $Label : $_" -ForegroundColor Yellow
        }
    }
}

Write-Host 'Stopping TabVision Studio...' -ForegroundColor Cyan
Stop-Port -Port 5000 -Label 'Backend'
Stop-Port -Port 5173 -Label 'Frontend'
