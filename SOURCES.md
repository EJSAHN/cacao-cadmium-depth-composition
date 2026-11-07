# Data sources and numerical provenance

## Public compilation

Wade et al. (2022), *Drivers of cadmium accumulation in Theobroma cacao L. beans: A quantitative synthesis of soil–plant relationships across the Cacao Belt*, PLOS ONE, https://doi.org/10.1371/journal.pone.0261989.

The source repository is https://github.com/jordon-wade/Cacao-cadmium-meta-analysis. The pinned file and checksum are in `README.md` and `run_analysis.py`. The publication is CC BY. A separate license for every repository asset was not established. The complete CSV and original R code are not redistributed here. The source's 1,253 rows are not equated with the publication's 1,122 final points.

## Ghana numerical transcription

Ackah, J. E. (2012), *Distribution of heavy metals in cocoa farm soils in the Western Region of Ghana*, MPhil thesis, University of Ghana, Legon, Ghana.

Institutional record: https://ugspace.ug.edu.gh/items/b68e00a8-9f13-4e7f-b04b-5698e256361e

Full-text locator: https://ugspace.ug.edu.gh/bitstreams/ee74a93b-c24a-4d90-84db-370b75e5c92d/download

`data/code2_ghana_primary_cd_check.csv` contains 37 means transcribed from Tables 4.4a–d and checked against the compilation by site and interval. There are eight site series; seven have complete 0–30 and 30–50 cm support. Asankragwa lacks required deeper observations. Reported field sampling limits do not establish random missingness. Analytical triplicates are not independent farms. The thesis, table artwork and narrative text are not included.

## Ecuador numerical transcription

Chavez, E., He, Z. L., Stoffella, P. J., Mylavarapu, R. S., Li, Y. C., Moyano, B., and Baligar, V. C. (2015), *Concentration of cadmium in cacao beans and its relationship with soil cadmium in southern Ecuador*, Science of the Total Environment, 533, 205–214, https://doi.org/10.1016/j.scitotenv.2015.06.106.

`data/code2_ecuador_primary_cd_check.csv` contains 76 total-recoverable Cd means from Tables 1–2. All 76 values and depth intervals were checked against the publisher PDF of the article. The source comprises 19 farm composites (four cores per site and depth), not individual soil cores. Assay: EPA 3050B HNO3/H2O2 digestion with ICP-OES determination in triplicate. The paper and table artwork are not included.

Public bibliographic record: https://pubmed.ncbi.nlm.nih.gov/26172587/

## Separate Peru example

Guarín, D., Rodríguez Hamamura, N., Blancas Cortez, J., Benavides, J., Spargo, J., Guiltinan, M., Maximova, S., Acha de la Cruz, O., and Drohan, P. J. (2023), *Cadmium contamination in cacao farms of Piura, North Peru: A comprehensive assessment of geogenic and anthropogenic sources and implications for future production*, Environmental Challenges, 13, 100765, https://doi.org/10.1016/j.envc.2023.100765.

`data/external/guarin2023_published_means.csv` contains 20 numerical mean tokens and their displayed standard errors from Table 1, page 4: five farms, nominal 5/20 cm depths, and two assays. The transcription was visually checked against the published PDF. The original publication specifies CC BY-NC-ND 4.0; no paper, table artwork or narrative text is redistributed or relicensed. The two assays share farms and are not independent studies. The displayed token precision is inconsistent, and its sensitivity calculation is not a verified measurement-precision model. The newly calculated masks are not a result claimed by the source authors.

## Derived numerical references

`data/r_reference` stores newly calculated base-R reference fits, fit counts, warnings and execution/session records. The independently written `reproduction/verify_loess.R` generates these calculations. The original authors' R code was inspected for the fitting method but is not included. The R input is regenerated from the original CSV, rather than redistributed. The reference manifest records its expected SHA-256.

`data/external/guarin2023_all_masks_decimal_reference.csv` is an independent Decimal calculation for testing the Peru contrasts. Files in `reference/` are derived comparison summaries, not additional independent data.

CSV study identifiers are original repository identifiers and are not necessarily the numbered studies in the source publication's supplement. None are renumbered for presentation.
