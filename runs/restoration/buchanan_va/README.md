# Real Natural Resources Case: Buchanan County, Virginia

This directory contains a real public-data companion case for the farmland optimization paper.

Business scenario: abandoned mine land reclamation / ecological restoration prioritization.

## Downloaded public data

- `eamlis_buchanan_va.geojson` and `.csv`: 1249 abandoned mine land records from OSMRE e-AMLIS public ArcGIS service.
- `buchanan_county_boundary.geojson`: Census TIGER/Cartographic Boundary county boundary for Buchanan County, Virginia (GEOID 51027).
- `nhd_flowline_buchanan.geojson`: 3157 NHD large-scale flowlines clipped to the county boundary from the USGS Hydrography MapServer.
- `usgs_3dep_dem_buchanan_900x700.tif`: USGS 3DEP elevation image exported for the county bounding box.
- `usgs_3dep_slope_deg_buchanan.tif`: slope raster derived from the DEM.

## Model-ready derived data

- `planning_units_2km.geojson`: 562 2-km planning units clipped to the county boundary.
- `planning_units_2km_candidates.geojson`: 522 candidate units with AML points or high riparian/slope risk.
- `planning_units_2km_attributes.csv`: aggregated features per planning unit.
- `planning_units_2km_adjacency.csv`: adjacency graph among planning units.
- `scenario_config.json`: optimization framing and suggested reward terms.
- `preview.png`: overview map.

## Optimization framing

Spatial unit: 2-km planning cell.

Action: select one candidate unit for reclamation/restoration priority at each planning step.

Hard constraints: already selected units are masked; a budget proxy can mask units whose restoration cost exceeds remaining budget.

Objectives: reduce abandoned-mine risk, prioritize units near streams, prefer connected restoration clusters, and control restoration cost.

## Source notes

The e-AMLIS web map metadata describes the inventory as OSMRE/state mining agency data based on the inventory as of September 2019. Use this real case to test algorithmic portability and data plumbing; avoid making policy claims about current remediation status without checking the latest state AML records.
