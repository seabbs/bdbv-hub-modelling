# Data

Extraction of the hub data, its vintages, and the synthetic vintages used in testing.
{ .page-subtitle }

What the data show is on the [Data results page](../data.md).

## Target and notation

The target is $y_{p}(w)$, the final count of confirmed cases in province $p$ with symptom onset in epi week $w$ (Sunday to Saturday, labelled by its Saturday).
It includes onsets the hub imputes for cases reported without one.
A vintage $v$ is the data of one hub release.
It gives, for each $p$ and $w$, known onsets, imputed onsets and their total $Y_{p,v}(w)$.
A forecast made from vintage $v$ serves the round with reference date $r_v$, the first Saturday whose submission window (Sunday to Tuesday, Kinshasa time) closes on or after the vintage date.
Horizon $h$ is the number of weeks from $r_v$ to $w$.
Backtest forecasts cover $h = -4, \dots, 3$.

## Extraction

| Hub file | What it holds | Vintages |
|---|---|---|
| `target-data/time-series.csv` | Weekly confirmed cases by onset week, by province and nationally. Also health zone totals. | All, in one file |
| `auxiliary-data/daily-confirmed-case.csv` | Daily cases by onset day, observed and imputed, nationally and by province and health zone. The observed and imputed split of the weekly target comes from here. | All, in one file |
| `auxiliary-data/weekly-reporting-delays-by-province.csv` | One row per known-onset case: province, onset week and onset-to-report delay. | Latest only; earlier ones from the archive folder and git history |
| `auxiliary-data/locations.csv` | Location codes, names, types and parents. | Static |
| `auxiliary-data/health-zone-code-mapping.csv` | Health zone codes mapped to P-codes and DHIS2 codes. | Static |
| `hub-config/*.json` | Rounds, tasks, output types, hub settings and the metadata schema. | Copied at download |

The hub does not publish deaths, suspected cases, counts by report date or populations.
Province populations are the Institut National de la Statistique's 2019 figures (*Annuaire statistique RDC 2020*), and province centres the means of health zone centroids weighted by WorldPop 2025 counts, as in BVDOutbreakSize.

## Vintages

Each hub release is a vintage, labelled by its `as_of` date and of kind `real` or `simulated`.
From the 2026-11-01 release on, `as_of` is the extraction date, a day after the last reported case.
The hub states that earlier releases were labelled by the last reported case ([hub issue 90](https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/90)).
We keep each release's published label and map a rewritten label back to it.
The data cut-off of a real release is its latest onset day over all places, one date per release.
This gives 2026-10-07 for the release labelled 2026-10-09, 2026-10-16 for 2026-10-17 and 2026-10-31 for 2026-11-01.
The last day or two before extraction are thin, as new reports take time to reach the line list.
The latest onset day is a cut-off taken from the data that every model can share.
The renewal models learn the publication lag from consecutive releases relative to this cut-off, so the lag adjusts to it and is not counted twice.
A simulated release states its cut-off.
The models, the scoring and the zone stage all use this cut-off.
Recent weeks are revised in later releases, so each forecast is made from one vintage and scored on the counts later releases report.

## Release snapshots

`auxiliary-data/daily-confirmed-case.csv` holds, for each release, the daily onsets known at that release.
The count of a day is its known plus imputed onsets, and the counts of a week sum to the hub's weekly target.
A release has an `as_of` date and a data cut-off ([Vintages](#vintages)).
A release observes the days up to its cut-off, and a day in range with no row counts 0.
The baseline models compare the counts of a day across releases by its age at each cut-off ([Supporting Information, *Baselines*](../paper/si.html#baselines)).

## Delay and missingness data

The hub imputes an onset for each case reported without one by subtracting a delay drawn from a fit to its published delays ([hub issue 61](https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/61), [Supporting Information, *Renewal model*](../paper/si.html#renewal-model)).
So the share of cases without an onset is measured at every vintage.

## BVDOutbreakSize estimates

`renewalbvd` uses the Rt estimates in the public `results-<run>` releases of [BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) (MIT licence, [Zenodo 10.5281/zenodo.20312758](https://doi.org/10.5281/zenodo.20312758)).
These are the national $R_T$ draws, the patch summaries with the national $R_T$ they are relative to where the release has them, and the release's data cut-off.
Which release a forecast uses, how growth rates are turned into $R_T$ and how the estimates enter the model are in the paper's [Supporting Information, *Renewal model*](../paper/si.html#renewal-model).
The release's joint-fit R-hat is read from the fit diagnostics table in its rendered site, and a release without a readable table is passed over for the next older one.
Each fit records the release it used, its R-hat and why newer releases were skipped.
