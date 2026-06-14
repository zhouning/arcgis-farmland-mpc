# Neijiang Dongxing execution-constraint frontier

This is an execution-constraint frontier. It changes Tool 4 hard constraints and local pair-selection settings, not Tool 2 sampling or Tool 3 training.

| Mode | Slope delta (%) | Contiguity delta | Steep-tail delta (ha) | Baimu count delta | Baimu area delta (ha) | Cultivated-area delta (ha) | Reward | Swaps | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unconstrained | -0.500 | +0.0327 | -448.5 | -7 | +246.5 | +58.5 | +77.44 | 463 | 3847.2 |
| Cultivated floor -500 ha | -0.527 | +0.0318 | -448.5 | -6 | +183.7 | -0.8 | +80.66 | 473 | 2357.0 |
| Cultivated floor -250 ha | -0.527 | +0.0318 | -448.5 | -6 | +183.7 | -0.8 | +80.66 | 473 | 4479.3 |
| Cultivated floor -100 ha | -0.527 | +0.0318 | -448.5 | -6 | +183.7 | -0.8 | +81.05 | 473 | 3746.4 |
| No net cultivated loss | -0.457 | +0.0322 | -344.8 | -8 | +287.5 | +101.9 | +82.31 | 471 | 2577.9 |
| No net cultivated + baimu area | -0.454 | +0.0322 | -343.6 | -8 | +286.5 | +100.9 | +76.87 | 471 | 4270.3 |
| Connectivity conservative | -0.440 | +0.0404 | -367.5 | -12 | +344.0 | +126.6 | +58.10 | 475 | 4080.7 |
