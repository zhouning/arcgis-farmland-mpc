# Zenodo Metadata Draft: Software Release

Use this metadata for the GitHub-Zenodo software archive.

## Basic fields

- Upload type: Software
- Title: ArcGIS Farmland MPC: reproducible model-based planning for county-scale farmland consolidation
- Version: v1.0-scirep
- Creators:
  - Ning Zhou, ORCID: 0009-0002-5647-7388
  - Xiang Jing
- License: MIT
- Access right: Open access
- Publication date: use the release date shown by Zenodo

## Description

This archive contains the software, trained ensembles, deterministic benchmark generators, verification scripts and Scientific Reports submission package supporting the manuscript "Reproducible model-based planning for county-scale farmland consolidation in fragmented mountain landscapes." The repository implements a learned transition-model and model-predictive-control workflow for auditable county-scale cadastral scenario generation, plus the ArcGIS Pro toolbox and command-line reproduction workflow used in the manuscript.

The release includes open synthetic benchmark data, public-data restoration boundary-check artefacts, trained ensembles, de-identified logs and the Scientific Reports manuscript package. Raw Bishan District and Neijiang Dongxing District cadastral records derive from the Third National Land Survey of China and are not redistributed because of data-governance restrictions. Derived and anonymised artefacts are included where public redistribution is permitted.

## Keywords

- farmland consolidation
- model predictive control
- learned surrogate model
- GIS
- ArcGIS Pro
- land-use planning
- Scientific Reports
- reproducible research

## Related identifiers

- GitHub repository: https://github.com/zhouning/arcgis-farmland-mpc
- Manuscript: add the Scientific Reports article DOI after acceptance
- Optional data record: add the dataset DOI if a separate dataset deposit is created

## Notes for Zenodo

Do not upload restricted cadastral shapefiles, GeoPackages, DEM rasters or local ignored run caches as manual files. Let Zenodo archive the GitHub release, which contains only tracked repository content.
