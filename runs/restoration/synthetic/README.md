# Synthetic Natural Resources Optimization Case

Case: abandoned-mine and degraded-slope ecological restoration priority planning.

This synthetic dataset is designed as a non-farmland companion case for geospatial optimization papers. It demonstrates that a model-based planning pipeline can be framed around general natural-resource decisions: selecting, restoring, protecting, or prioritising spatial units under hard constraints and multi-objective rewards.

## Files

- `restoration_units.geojson`: 420 candidate restoration polygons with attributes.
- `restoration_units_attributes.csv`: same attributes without geometry.
- `adjacency_edges.csv`: queen-contiguity graph between restoration units.
- `water_network.geojson`: synthetic river and tributary.
- `ecological_sources.geojson`: protected ecological source patches used for connectivity rewards.
- `settlements.geojson`: exposed settlements for fairness/risk-reduction objectives.
- `synthetic_dem.npy`, `synthetic_slope_deg.npy`, `raster_grid_meta.json`: portable synthetic terrain rasters.
- `scenario_config.json`: suggested action space, constraints, and reward terms.
- `preview.png`: quick visual overview.

## Optimization framing

Spatial unit: restoration polygon.

Action: choose one candidate unit to restore at each step. The action mask excludes already-restored units and units that would violate remaining budget or scenario-specific rules.

Objectives:

1. reduce mine/disturbance and erosion risk;
2. increase habitat connectivity to ecological sources and already restored units;
3. improve riparian/water-quality protection;
4. reduce exposure for nearby settlements;
5. respect restoration-budget constraints.

This maps to the same algorithmic abstraction as the farmland case: a high-branching discrete spatial planning problem with action masking, multi-objective rewards, and spatial adjacency effects.

## Suggested paper wording

"To test whether the approach is specific to farmland consolidation, we constructed a second, non-agricultural natural-resources benchmark: ecological restoration prioritisation for abandoned mines and degraded slopes. The decision unit is a restoration polygon, the action is to select a unit for restoration under a fixed budget, and rewards combine risk reduction, riparian benefit, human-exposure reduction and habitat-connectivity gains. This preserves the same optimisation structure as county-scale consolidation while changing the semantics of the spatial units, actions and reward terms."

## Caveat

The data are synthetic. Use them to demonstrate algorithmic portability and reproducibility, not to claim empirical restoration benefits for a real county.
