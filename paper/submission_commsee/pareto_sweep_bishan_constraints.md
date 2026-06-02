# Bishan execution-constraint frontier

Source date: 2026-06-02

This is an execution-constraint frontier. Runtime reward-weight overrides are not treated as a full policy Pareto sweep unless Tool 2 sampling and Tool 3 training are re-run under those weights.

| Mode | Slope delta (%) | Contiguity delta | Steep-tail delta (ha) | Baimu count delta | Baimu area delta (ha) | Cultivated-area delta (ha) | Reward | Swaps | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unconstrained | -1.753 | +0.0124 | -691.9 | +5 | -483.9 | -486.0 | +71.78 | 427 | 302.1 |
| Cultivated floor -500 ha | -1.746 | +0.0118 | -681.5 | +6 | -503.9 | -498.5 | +74.73 | 438 | 299.4 |
| Cultivated floor -250 ha | -1.261 | +0.0180 | -468.6 | +3 | -244.9 | -249.7 | +59.37 | 420 | 299.2 |
| Cultivated floor -100 ha | -0.918 | +0.0197 | -329.2 | +3 | -78.5 | -100.0 | +56.87 | 415 | 303.8 |
| No net cultivated loss | -0.675 | +0.0204 | -218.5 | +1 | +29.8 | +3.0 | +45.05 | 401 | 305.0 |
| No net cultivated + baimu area | -0.711 | +0.0207 | -234.0 | +2 | +29.2 | +0.2 | +51.27 | 414 | 334.2 |
| Connectivity conservative | -0.608 | +0.0259 | -210.5 | +1 | +43.4 | +4.8 | +43.72 | 408 | 302.3 |
