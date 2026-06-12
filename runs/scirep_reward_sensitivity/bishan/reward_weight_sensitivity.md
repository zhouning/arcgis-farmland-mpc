# Retrained Bishan reward-weight sensitivity

Each profile re-runs Tool 2 sampling and Tool 3 ensemble training before no-net-loss Tool 4 planning.

| Profile | Slope delta (%) | Contiguity delta | Baimu count delta | Baimu area delta (ha) | Cultivated area delta (ha) | Rank acc mean | Reward std median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default reward | -0.7247 | +0.02700 | +1 | +27.49 | +0.44 | 0.9033 | 0.6855 |
| Baimu low | -0.6522 | +0.02090 | +0 | +17.34 | +0.18 | 0.9273 | 0.5928 |
| Baimu high | -0.8379 | +0.03433 | +1 | +56.21 | +0.64 | 0.8733 | 0.6843 |
