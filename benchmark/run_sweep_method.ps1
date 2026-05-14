# Launch a long-running sweep for one method (PPO or MPC) as a detached
# background process. Resumable: per-cell checkpoints land in sweep_state.json.
#
# Usage:
#     powershell -NoProfile -ExecutionPolicy Bypass -File run_sweep_method.ps1 PPO
#     powershell -NoProfile -ExecutionPolicy Bypass -File run_sweep_method.ps1 MPC
#
# Or detached so the shell can close:
#     $p = Start-Process powershell.exe -ArgumentList @(
#       '-NoProfile','-ExecutionPolicy','Bypass',
#       '-File','./run_sweep_method.ps1','PPO'
#     ) -PassThru -WindowStyle Hidden
#     "PID=$($p.Id)" | Out-File sweep_${args[0].ToLower()}.pid

param(
  [Parameter(Mandatory=$true)]
  [ValidateSet('Random','Greedy','GA','PPO','MPC')]
  [string]$Method
)

Set-Location -Path $PSScriptRoot
$lower = $Method.ToLower()
$logfile = "sweep_$lower.log"
$t0 = Get-Date
"=== $Method sweep start $($t0.ToString('yyyy-MM-dd HH:mm:ss')) ===" | Out-File -FilePath $logfile -Encoding utf8
python -m sweep.runner `
  --manifest sweep_manifest.csv `
  --state sweep_state.json `
  --data-root data_dev `
  --results-root results `
  --only-method $Method `
  --resume *>> $logfile
$t1 = Get-Date
"=== $Method sweep end   $($t1.ToString('yyyy-MM-dd HH:mm:ss'))  wall=$([int](($t1-$t0).TotalSeconds))s ===" | Out-File -FilePath $logfile -Encoding utf8 -Append
