$repo = "C:\Users\adamp\maddis-4DGS-Video-Generator"
$peakVram = 0; $peakShared = 0; $baseShared = $null
function SharedMB { try { ((Get-Counter '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction Stop).CounterSamples | Measure-Object CookedValue -Sum).Sum / 1MB } catch { 0 } }
while ($true) {
  $g = (nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits) -split ',\s*'
  $used = [int]$g[0]; $total = [int]$g[1]
  if ($used -gt $peakVram) { $peakVram = $used }
  $shared = [math]::Round((SharedMB), 0)
  if ($baseShared -eq $null) { $baseShared = $shared }
  $spill = $shared - $baseShared
  if ($spill -gt $peakShared) { $peakShared = $spill }
  $py = Get-Process python -ErrorAction SilentlyContinue
  $latest = Get-ChildItem (Join-Path $repo "output_3dgs") -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $frames = @(); if ($latest) { $frames = @(Get-ChildItem $latest.FullName -Filter "*_standard.ply" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime) }
  $zero = @($frames | Where-Object Length -eq 0).Count
  $spf = "-"
  if ($frames.Count -ge 2) {
    $last = $frames | Select-Object -Last ([math]::Min(5, $frames.Count))
    $secs = ($last[-1].LastWriteTime - $last[0].LastWriteTime).TotalSeconds / ($last.Count - 1)
    $spf = "{0:N1} s per frame" -f $secs
  }
  Clear-Host
  Write-Host "=== PC LIVE STATS ===" -ForegroundColor Cyan
  if ($py) { Write-Host "Job:          RUNNING" -ForegroundColor Green } else { Write-Host "Job:          not running" -ForegroundColor Yellow }
  Write-Host ("GPU memory:   {0} / {1} MB   (peak {2} MB)" -f $used, $total, $peakVram)
  $c = "Green"; if ($spill -gt 300) { $c = "Red" }
  Write-Host ("RAM spill:    {0} MB   (peak {1} MB)   <- above 0 = slow" -f $spill, $peakShared) -ForegroundColor $c
  Write-Host ("GPU busy:     {0}%" -f $g[2])
  Write-Host ("Temp / power: {0} C / {1} W" -f $g[3], $g[4])
  if ($latest) { Write-Host ("Output:       {0}" -f $latest.Name) }
  Write-Host ("Frames done:  {0}" -f $frames.Count)
  Write-Host ("Speed:        {0}" -f $spf)
  if ($zero -gt 0) { Write-Host ("0-byte files: {0}" -f $zero) -ForegroundColor Red } else { Write-Host "0-byte files: none" -ForegroundColor Green }
  Write-Host ""
  Write-Host "Updates every 2s. Ctrl + C exits this view only." -ForegroundColor DarkGray
  Start-Sleep 2
}