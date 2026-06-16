# Zenodo Release Record for Paper 9

This folder documents the GitHub-Zenodo release supporting the Scientific Reports submission:

`Reproducible model-based planning for county-scale farmland consolidation in fragmented mountain landscapes`

## Recommended route

Use the GitHub-Zenodo integration to archive a public GitHub release. This is the cleanest route because the repository already contains the code, manuscript package, trained ensembles, benchmark generators, verification scripts and tracked reproduction artefacts.

Current cleaned submission release tag:

`v1.0.1-scirep`

Current GitHub release title:

`Paper 9 Scientific Reports cleaned submission release`

Release notes file:

`github_release_notes_v1.0.1-scirep.md`

Zenodo version DOI for this cleaned submission release:

https://doi.org/10.5281/zenodo.20713695

The earlier `v1.0-scirep` release exists as a historical pre-cleanup archive. Use `v1.0.1-scirep` and `10.5281/zenodo.20713695` for reproducible Scientific Reports review.

## What the Zenodo software DOI covers

The GitHub release archive includes the tracked repository contents. Tracked-file summary at release preparation:

| Path pattern | Files | Approx. size |
|---|---:|---:|
| `paper/checkpoints/**` | 741 | 331.2 MB |
| `paper/submission_scirep_corrected/**` | 32 | 4.51 MB |
| `benchmark/**` | 230 | 0.30 MB |
| `runs/restoration/**` | 254 | 3.24 MB |
| `verification/**` | 7 | 0.07 MB |
| `docs/**` | 8 | 0.07 MB |
| `scripts/**` | 16 | 0.15 MB |
| `farmland_mpc/**` | 38 | 0.39 MB |
| `notebooks/**` | 2 | 0.13 MB |
| `neijiang_cross_region/**` | 41 | 1.43 MB |

Total tracked repository size is about 351 MB, which is suitable for a Zenodo GitHub-release archive.

## Do not upload separately

Do not upload or manually add any of the following to a public Zenodo record:

- raw Bishan or Neijiang cadastral records from the Third National Land Survey
- local shapefiles, GeoPackages, rasters, DEM tiles or geodatabases
- ignored local caches under `runs/neijiang/pareto/`
- ignored local sensitivity caches under `runs/scirep_extra*/` and `runs/scirep_reward_sensitivity*/`, except files that are already tracked in Git
- temporary pytest directories, local logs, credentials or machine-specific paths

The public release should disclose that real-county raw cadastral records are restricted, while derived and anonymised artefacts needed for verification are available through the release.

## Completed actions

1. GitHub-Zenodo integration was enabled for `zhouning/arcgis-farmland-mpc`.
2. A cleaned GitHub release was published from `main` with tag `v1.0.1-scirep`.
3. Zenodo archived the release and minted version DOI `10.5281/zenodo.20713695`.
4. The manuscript, README files, declarations checklist and `CITATION.cff` were updated to cite the cleaned release DOI.

## Official help pages

- GitHub: Referencing and citing content, including Zenodo archiving: https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
- GitHub: About `CITATION.cff` files: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files
- Zenodo quick start: https://help.zenodo.org/docs/get-started/quickstart/

## Optional dataset DOI

A separate Zenodo Dataset record is optional. Use it only if you want a DOI that separates "data artefacts" from "software release". The current GitHub release DOI already captures the tracked reproduction artefacts. If a separate Dataset record is created, use `zenodo_metadata_dataset_optional.md` as the metadata draft and do not include restricted raw cadastral records.
