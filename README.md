# Cacao soil cadmium depth composition

Analysis code for "Site composition at depth can reverse soil cadmium depth comparisons in compiled cacao data". The code describes how the sites represented at different soil depths affect cadmium depth comparisons, using the public cacao compilation of Wade et al. (2022), transcribed means from Ghana and Ecuador, and a separate five-farm example from Peru. In the omission analysis, all concentrations are held fixed; only the set of retained deep observations changes. The pooled-curve diagnostics separately use constructed flattened and nonincreasing profiles.

The unmatched difference between deep and shallow means, `U`, equals the paired difference among retained sites, `P`, plus the difference in shallow means between retained and all sites, `C` (`U = P + C`). The code enumerates all one- and two-site omissions of deep values and computes exact bounds under rounding of the published values.

## Requirements

Python 3.10 or later with NumPy and pandas. R is needed only for the optional LOESS replay.

```sh
conda env create -f environment.yml
conda activate cacao-depth
```

## Run

Download the pinned source file and run the analysis:

```sh
python run_analysis.py --download-source --output outputs/reproduction
```

or use an existing, unmodified copy of the source CSV:

```sh
python run_analysis.py --source /path/to/Extracted_and_finalized_data.csv --output outputs/reproduction
```

The output directory must not already exist. The source file is checked against its SHA-256 before any calculation.

## Input

The Wade et al. (2022) dataset is downloaded from its original repository and is not redistributed here.

- Repository: https://github.com/jordon-wade/Cacao-cadmium-meta-analysis
- Commit: `b16cd1def36732550f0c25182e82a71260cc3596`
- File: `Extracted and finalized data.csv` (Windows-1252, 1,253 rows)
- SHA-256: `09f328a87063373ccf0d15f5159a7b12775dbf8623233b35a74d734ab51a365a`

The analysis retains 973 rows with numeric soil Cd and valid depth intervals, including 43 zeros. The article reports 1,122 final data points; the row-level mapping between the two counts could not be recovered.

`data/` contains the attributed transcriptions of published means and the R reference fits. `SOURCES.md` lists all sources, and `THIRD_PARTY_NOTICES.md` describes redistribution limits.

## Output

A run writes 44 CSV tables to `OUTPUT/tables`, together with input and environment records, code hashes and a completion status. The source CSV is cached under `OUTPUT/inputs`; do not commit that directory. Each run is compared with the reference values in `reference/` and stops if they differ beyond floating-point tolerance.

Table file prefixes identify the analysis stage that produces them:

| Prefix | Contents |
|---|---|
| `input_*`, `primary_means_*` | Eligible source records and checked means |
| `code3_*` | Depth coverage, constructed profiles and pooled-curve comparisons |
| `code4_*` | All omission sets at the published values |
| `code5_*` | Rounding bounds and the Peru example |
| `code6_*` | Summary tables and figure source data |

Column names in these tables use `naive`, `paired_visible`, `selection`, `visible`, `hidden` and `mask` for the manuscript terms unmatched, paired, composition, retained, omitted and omission.

## Tests

```sh
python -m unittest discover -s tests -v
```

The 16 tests cover source parsing, overlapping intervals, site identifiers, rounding bounds checked by corner enumeration, transcription checks and the Peru contrasts checked against an independent decimal calculation. They do not require the source download.

All 44 tables from this code matched the original Windows run within `rtol=1e-11`, `atol=1e-10` (largest absolute difference about 1e-13); see `reference/reproduction_check.json`.

## Optional R replay

The default run checks the stored R 4.5.0 LOESS results without running R. To refit them:

```sh
Rscript reproduction/verify_loess.R outputs/reproduction/inputs/data/r_reference/r_input.csv outputs/r_replay
```

Other R versions may give slightly different values in sparse regions.

## Citation and license

Please cite this software using `CITATION.cff` and acknowledge the original data sources in `SOURCES.md`.

Repository: https://github.com/EJSAHN/cacao-cadmium-depth-composition

The original analysis code, project documentation and newly computed reference outputs are dedicated under CC0 1.0 Universal; see `LICENSE`. This dedication does not cover the original observations, source publications, transcribed third-party measurements or dependencies. Their attribution and terms remain as stated in `THIRD_PARTY_NOTICES.md`. No ownership of the original observations is claimed.
