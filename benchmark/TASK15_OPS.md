# Task 15 ops cheatsheet

Two parallel tracks because PPO is GPU-friendly but MPC is CPU-only:

| Track | Method | Where | Wall (est.) | Launch |
|-------|--------|-------|-------------|--------|
| A | PPO | Colab × 5 shards | 14-22h per shard, parallel | `ppo_sweep_bundle.zip` + ipynb |
| B | MPC | Local 12-thread | overnight × 1-2 | `run_sweep_method.ps1 MPC` |

The two tracks don't share state files (PPO writes shard-specific JSON; MPC writes the canonical `sweep_state.json`). After both finish, results merge cleanly under `results/{preset}/{method}/seed{N}.json`.

## Track A: PPO via Colab (5 parallel shards)

### Setup once
1. Build bundle: `python build_decision_gate_bundle.py --ppo`
2. Upload `ppo_sweep_bundle.zip` to `Drive/MyDrive/arcgis-farmland-mpc/`

### Each Colab session (×5)
1. New Colab notebook → Runtime → A100 GPU (T4 if A100 unavailable)
2. Open `sweep/ppo_sweep_colab.ipynb` from the bundle (or upload directly)
3. Edit Cell 0: set `SHARD_ID = 0` (then 1, 2, 3, 4 in the other 4 sessions)
4. Run all
5. Result auto-saves to `Drive/MyDrive/arcgis-farmland-mpc/ppo_results_shard{N}.zip`

### Merge back locally
```powershell
cd D:\test\_publish\arcgis-farmland-mpc\benchmark
foreach ($i in 0..4) {
  unzip -o "ppo_results_shard$i.zip" -d .
}
(Get-ChildItem results/ -Recurse -Filter *.json | Where-Object { $_.FullName -match '\\PPO\\' }).Count
# expect 35
```

## Track B: MPC local

### Launch detached
```powershell
cd D:\test\_publish\arcgis-farmland-mpc\benchmark
$p = Start-Process powershell.exe -ArgumentList @(
  '-NoProfile','-ExecutionPolicy','Bypass',
  '-File','./run_sweep_method.ps1','MPC'
) -PassThru -WindowStyle Hidden
"PID=$($p.Id)" | Out-File -FilePath sweep_mpc.pid -Encoding utf8
```

### Check progress
```powershell
python -c "import json; from collections import Counter; s=json.load(open('sweep_state.json')); mpc=[c for c in s['cells'] if c['method']=='MPC']; print(Counter(c['status'] for c in mpc)); print('running:', [c['cell_id'] for c in mpc if c['status']=='running'])"
```

### Stop (resumable)
```powershell
$pid = (Get-Content sweep_mpc.pid -Raw).Split('=')[1].Trim()
Stop-Process -Id $pid -Force
Get-WmiObject Win32_Process -Filter "ParentProcessId=$pid" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Restart: same `Start-Process` command — `--resume` is built into the launcher.

## After both tracks finish

Validate: 175 result JSONs total (35 cells × 5 methods).
```powershell
(Get-ChildItem results/ -Recurse -Filter *.json).Count   # expect 175
```

Then commit and proceed to Task 16 (`docs/BENCHMARK_RESULTS.md` aggregation).
