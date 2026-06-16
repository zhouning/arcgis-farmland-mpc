# Zenodo Release Preparation for Paper 9

This folder prepares the repository for a Zenodo DOI supporting the Scientific Reports submission:

`Reproducible model-based planning for county-scale farmland consolidation in fragmented mountain landscapes`

## Recommended route

Use the GitHub-Zenodo integration to archive a public GitHub release. This is the cleanest route because the repository already contains the code, manuscript package, trained ensembles, benchmark generators, verification scripts and tracked reproduction artefacts.

Recommended release tag:

`v1.0-scirep`

Recommended GitHub release title:

`Paper 9 Scientific Reports submission release`

Do not create the GitHub release before all release-preparation changes have been pushed to `main` and the repository has been enabled in Zenodo's GitHub settings. The GitHub release event is what Zenodo uses to archive the repository and mint the software DOI.

## What the Zenodo software DOI will cover

The GitHub release archive should include the tracked repository contents. Current tracked-file summary:

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

## User actions required

The local preparation, commit and push can be done from this workstation. The following browser-side steps require the repository owner to log in:

1. Log in to Zenodo with GitHub: https://zenodo.org/login
2. Open the Zenodo GitHub settings page: https://zenodo.org/account/settings/github/
3. Enable the repository `zhouning/arcgis-farmland-mpc`.
4. On GitHub, create a release from `main` with tag `v1.0-scirep`.
5. Use `Paper 9 Scientific Reports submission release` as the release title.
6. Paste `github_release_notes_v1.0-scirep.md` into the GitHub release body.
7. Publish the GitHub release.
8. Wait for Zenodo to archive the release and mint a DOI.
9. Send the DOI back so the manuscript, README and `CITATION.cff` can be updated before final journal submission.

If the GitHub release is published on a date other than 2026-06-16, update `date-released` in `CITATION.cff` to the actual release date before publishing the release.

## Official help pages

- GitHub: Referencing and citing content, including Zenodo archiving: https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
- GitHub: About `CITATION.cff` files: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files
- Zenodo quick start: https://help.zenodo.org/docs/get-started/quickstart/

## Optional dataset DOI

A separate Zenodo Dataset record is optional. Use it only if you want a DOI that separates "data artefacts" from "software release". The current GitHub release DOI already captures the tracked reproduction artefacts. If a separate Dataset record is created, use `zenodo_metadata_dataset_optional.md` as the metadata draft and do not include restricted raw cadastral records.
