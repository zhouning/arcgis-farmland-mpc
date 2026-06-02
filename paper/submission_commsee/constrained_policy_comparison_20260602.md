# Cultivated-area constraint comparison

Source date: 2026-06-02

All deltas below are GIS-only recomputations from the optimized shapefile, except the Bishan unconstrained row, which uses the existing Bishan audit anchors in `policy_translation_bishan.json` and the manuscript verification table.

| County | Planning mode | Slope delta (%) | Contiguity delta | Steep-tail delta (ha) | Baimu count delta | Baimu area delta (ha) | Cultivated-area delta (ha) | Paired swaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bishan | Unconstrained candidate | -2.001 | +0.0128 | -687.3 | +8 | -473.3 | -505.2 | 428 |
| Bishan | Hard no-net-loss floor | -0.675 | +0.0204 | -218.5 | +1 | +29.8 | +3.0 | 401 |
| Neijiang Dongxing | Unconstrained candidate | -0.664 | +0.0324 | -218.5 | +7 | -103.6 | -261.0 | 454 |
| Neijiang Dongxing | Hard no-net-loss floor | -0.492 | +0.0373 | -330.4 | 0 | +252.6 | +62.2 | 466 |

Interpretation for manuscript/rebuttal:

- The original unconstrained mode is a useful audit scenario but not a policy-compliant plan in either county because it loses cultivated area.
- The hard cumulative cultivated-area floor directly addresses the central red-line failure: Bishan changes from -505.2 ha to +3.0 ha, and Neijiang changes from -261.0 ha to +62.2 ha.
- The constraint trades off part of Bishan's slope reduction, but the constrained output still improves slope, contiguity, and qualifying large-patch area.
- Neijiang is less costly under the constraint: steep-tail reduction is larger in the constrained audit (-330.4 ha), while cultivated area and baimu area both become positive.
