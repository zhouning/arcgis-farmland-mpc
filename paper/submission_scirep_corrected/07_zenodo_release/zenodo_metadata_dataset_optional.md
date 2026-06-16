# Zenodo Metadata Draft: Optional Dataset Record

Use this only if you decide to create a separate Dataset DOI in addition to the GitHub-Zenodo software DOI.

## Basic fields

- Upload type: Dataset
- Title: Open reproduction data for model-based county-scale farmland consolidation planning
- Creators:
  - Ning Zhou, ORCID: 0009-0002-5647-7388
  - Xiang Jing
- Access right: Open access
- Version: v1.0-scirep
- License: CC-BY 4.0 for generated open benchmark and derived artefacts where rights permit. Third-party public inputs retain their original licences. Code remains MIT licensed in the software release.

## Description

This dataset record, if created, should contain only public or redistributable reproduction artefacts supporting the Scientific Reports manuscript "Reproducible model-based planning for county-scale farmland consolidation in fragmented mountain landscapes." Suitable contents include deterministic synthetic farmland benchmark files, public-data Buchanan County restoration inputs and outputs, derived aggregate tables, anonymised pairwise data where redistribution is permitted, trained ensemble metadata, de-identified logs, random seeds, hyperparameters and figure source summaries.

Raw Bishan District and Neijiang Dongxing District cadastral records from the Third National Land Survey of China must not be uploaded. Local shapefiles, GeoPackages, DEM rasters and ignored run caches should also be excluded unless their redistribution rights have been checked.

## File manifest requirements

Before publishing a separate dataset record, include a README or manifest with:

- filename
- file format
- short description
- related manuscript figure, table or supplementary section
- units and variables for tabular files
- provenance or generating script
- licence or access condition

## Suggested Data Availability wording after a dataset DOI exists

Open reproduction data and derived artefacts that can be publicly redistributed are archived on Zenodo at `[DATA_DOI]`. The record contains the synthetic farmland benchmark, public-data Buchanan restoration case, trained-ensemble metadata, de-identified logs, random seeds, hyperparameters and figure source summaries. Raw Bishan and Neijiang cadastral records are not redistributed because of Third National Land Survey data-governance restrictions.
