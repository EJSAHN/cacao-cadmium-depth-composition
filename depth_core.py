"""Auditable descriptive depth analysis for the cacao cadmium source table.

The grouping key identifies candidate site-depth series, not verified individual
soil pits. These functions do not infer causal depth effects or pool contrasts
across different depth spans. Original CSV strings and duplicate headers are
preserved separately from analysis-ready numeric fields.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd


DEPTH_BINS = (0, 10, 20, 30, 40, 50, 60, 80, 100, 150)
_MISSING = {"", "na", "n/a", "nan"}
_REQUIRED = (
    "Sample.ID", "Study.ID", "SiteName", "Block.Rep", "Country",
    "SampleDepth.min", "SampleDepth.max", "Soil.TotalCd",
)


def _text(values: pd.Series) -> pd.Series:
    """Normalize analysis identifiers only; original strings stay in raw_df."""
    out = values.astype(str).str.strip()
    return out.mask(out.str.lower().isin(_MISSING), pd.NA)


def _numbers(values: pd.Series, name: str) -> pd.Series:
    text = _text(values)
    parsed = pd.to_numeric(text, errors="coerce")
    bad = text.notna() & (parsed.isna() | ~np.isfinite(parsed))
    if bad.any():
        examples = [f"CSV row {int(i) + 2}: {text.loc[i]!r}" for i in text.index[bad][:5]]
        raise ValueError(f"Invalid numeric value in {name}: " + "; ".join(examples))
    return parsed.astype(float)


def _key(study: object, site: object, block: object) -> object:
    if pd.isna(study) or pd.isna(site):
        return pd.NA
    # Percent-encoding prevents literal separators or a literal missing marker
    # from colliding with genuine missing block identifiers.
    parts = [quote(str(study), safe=""), quote(str(site), safe="")]
    parts.append("@MISSING" if pd.isna(block) else quote(str(block), safe=""))
    return "|".join(parts)


def read_source(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read CP1252 CSV and return (unaltered string frame, clean_total, audit).

    Missing Cd or depth bounds exclude a row from clean_total. Zero Cd remains.
    Invalid nonmissing numbers, negative concentrations, or invalid intervals
    fail explicitly. SHA-256/version verification belongs to the caller.
    """
    with Path(path).open("r", encoding="cp1252", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError("Source CSV is empty")
    header, values = rows[0], rows[1:]
    widths = Counter(len(row) for row in values)
    if any(width != len(header) for width in widths):
        raise ValueError(f"Inconsistent CSV row widths: header={len(header)}, rows={dict(widths)}")
    counts = Counter(header)
    for name in _REQUIRED:
        if counts[name] != 1:
            raise ValueError(f"Required source field must occur exactly once: {name} ({counts[name]})")
    raw = pd.DataFrame(values, columns=header, dtype=str)
    clean = pd.DataFrame(index=raw.index)
    clean["source_row"] = raw.index + 2  # Header is CSV row 1.
    for source, target in (
        ("Sample.ID", "sample_id"), ("Study.ID", "study_id"),
        ("SiteName", "site_name"), ("Block.Rep", "block_rep"),
        ("Country", "country"), ("TotalCd.method", "total_cd_method"),
        ("DOI.primary", "doi_primary"), ("Study.context", "study_context"),
    ):
        if counts[source] > 1:
            raise ValueError(f"Ambiguous analysis metadata field: {source}")
        clean[target] = _text(raw[source]) if counts[source] == 1 else pd.Series(pd.NA, index=raw.index)
    for source, target in (
        ("SampleDepth.min", "depth_min_cm"),
        ("SampleDepth.max", "depth_max_cm"),
        ("Soil.TotalCd", "total_cd_mg_kg"),
    ):
        clean[target] = _numbers(raw[source], source)
    bad_cd = clean["total_cd_mg_kg"].notna() & (clean["total_cd_mg_kg"] < 0)
    if bad_cd.any():
        raise ValueError(f"Negative total Cd at CSV rows {clean.loc[bad_cd, 'source_row'].tolist()}")
    have_bounds = clean[["depth_min_cm", "depth_max_cm"]].notna().all(axis=1)
    bad_interval = have_bounds & (
        (clean["depth_min_cm"] < 0) | (clean["depth_max_cm"] <= clean["depth_min_cm"])
    )
    if bad_interval.any():
        raise ValueError(f"Invalid depth intervals at CSV rows {clean.loc[bad_interval, 'source_row'].tolist()}")
    complete = have_bounds & clean["total_cd_mg_kg"].notna()
    missing_counts = {col: int(clean[col].isna().sum()) for col in
                      ("total_cd_mg_kg", "depth_min_cm", "depth_max_cm")}
    clean = clean.loc[complete].copy()
    clean["depth_mid_cm"] = (clean["depth_min_cm"] + clean["depth_max_cm"]) / 2.0
    clean["site_depth_series"] = [
        _key(study, site, block) for study, site, block in
        clean[["study_id", "site_name", "block_rep"]].itertuples(index=False, name=None)
    ]
    clean = clean.reset_index(drop=True)
    duplicate_headers = []
    for name, number in counts.items():
        if number > 1:
            positions = [i for i, value in enumerate(header) if value == name]
            duplicate_headers.append({
                "name": name, "occurrences": number,
                "positions_1based": [i + 1 for i in positions],
                "nonmissing_counts_by_position": [int(_text(raw.iloc[:, i]).notna().sum()) for i in positions],
            })
    profile = profile_table(clean)
    eligible = profile["eligible_primary"] if not profile.empty else pd.Series(dtype=bool)
    audit = {
        "source_encoding": "cp1252",
        "source_rows": len(raw), "source_header_count": len(header),
        "source_unique_header_count": len(counts),
        "duplicate_headers": duplicate_headers,
        "raw_strings_and_duplicate_headers_preserved": True,
        "clean_total_rows": len(clean),
        "excluded_missing_cd_or_depth_rows": int((~complete).sum()),
        "missing_numeric_fields": missing_counts,
        "zero_total_cd_rows_retained": int((clean["total_cd_mg_kg"] == 0).sum()),
        "positive_total_cd_rows": int((clean["total_cd_mg_kg"] > 0).sum()),
        "rows_missing_series_identity": int(clean["site_depth_series"].isna().sum()),
        "distinct_studies_with_cd_and_depth": int(clean["study_id"].nunique()),
        "distinct_countries_with_cd_and_depth": int(clean["country"].nunique()),
        "candidate_multidepth_series": int((profile["interval_count"] >= 2).sum()) if not profile.empty else 0,
        "eligible_multidepth_series": int(eligible.sum()),
        "eligible_multidepth_rows": int(profile.loc[eligible, "row_count"].sum()) if not profile.empty else 0,
        "series_with_duplicate_intervals": int((profile["duplicate_interval_count"] > 0).sum()) if not profile.empty else 0,
        "series_with_overlapping_intervals": int((profile["overlapping_interval_pairs"] > 0).sum()) if not profile.empty else 0,
        "identity_definition": "Study.ID + SiteName + Block.Rep; candidate site-depth series, not verified pits",
    }
    return raw, clean, audit


def profile_table(clean: pd.DataFrame) -> pd.DataFrame:
    """Inventory candidate series and flag ambiguous repeated/overlapping layers.

    Primary contrasts require at least two distinct intervals, no repeated
    interval, no overlapping intervals, and no contradictory country labels.
    A missing Block.Rep is retained as a missing-key marker, not a replicate.
    """
    columns = ["site_depth_series", "study_id", "site_name", "block_rep", "country",
               "row_count", "interval_count", "duplicate_interval_count",
               "overlapping_interval_pairs", "country_count", "min_depth_cm",
               "max_depth_cm", "eligible_primary", "exclusion_reason"]
    output = []
    linked = clean.loc[clean["site_depth_series"].notna()]
    for series_id, group in linked.groupby("site_depth_series", sort=True, dropna=False):
        intervals = sorted(set(zip(group["depth_min_cm"], group["depth_max_cm"])))
        duplicates = len(group) - len(intervals)
        overlaps = sum(max(a[0], b[0]) < min(a[1], b[1])
                       for i, a in enumerate(intervals) for b in intervals[i + 1:])
        country_count = group["country"].nunique(dropna=True)
        reasons = []
        if len(intervals) < 2:
            reasons.append("single_depth_interval")
        if duplicates:
            reasons.append("duplicate_depth_interval")
        if overlaps:
            reasons.append("overlapping_depth_intervals")
        if country_count > 1:
            reasons.append("conflicting_country_labels")
        first = group.iloc[0]
        output.append({
            "site_depth_series": series_id,
            **{name: first[name] for name in ("study_id", "site_name", "block_rep", "country")},
            "row_count": len(group), "interval_count": len(intervals),
            "duplicate_interval_count": duplicates,
            "overlapping_interval_pairs": int(overlaps), "country_count": int(country_count),
            "min_depth_cm": float(group["depth_min_cm"].min()),
            "max_depth_cm": float(group["depth_max_cm"].max()),
            "eligible_primary": not reasons, "exclusion_reason": ";".join(reasons),
        })
    return pd.DataFrame(output, columns=columns)


def support_table(clean: pd.DataFrame, axes=("upper", "midpoint", "lower"),
                  bins=DEPTH_BINS) -> pd.DataFrame:
    """Count source support in right-closed depth bins, including the first edge.

    A concentration measured over a layer is not relabeled as a point
    measurement: axes are explicitly alternative depth coordinates. Effective
    study number is 1/sum(p_study**2) using row fractions within each bin.
    """
    edges = np.asarray(bins, dtype=float)
    if len(edges) < 2 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError("bins must contain increasing finite depth edges")
    mapping = {"upper": "depth_max_cm", "midpoint": "depth_mid_cm", "lower": "depth_min_cm"}
    output = []
    for axis in axes:
        if axis not in mapping:
            raise ValueError(f"Unknown depth axis: {axis}")
        coordinates = clean[mapping[axis]]
        if coordinates.isna().any() or ((coordinates < edges[0]) | (coordinates > edges[-1])).any():
            raise ValueError(f"Depth coordinates outside supplied bins or missing for axis {axis}")
        codes = pd.cut(coordinates, bins=edges, right=True, include_lowest=True, labels=False)
        for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            group = clean.loc[codes == idx]
            study_counts = group["study_id"].fillna("@MISSING_STUDY").value_counts()
            fractions = study_counts / len(group) if len(group) else pd.Series(dtype=float)
            output.append({
                "axis": axis, "bin_index": idx, "bin_lower_cm": float(left),
                "bin_upper_cm": float(right),
                "bin_label": f"{'[' if idx == 0 else '('}{left:g}, {right:g}]",
                "n_rows": len(group), "n_studies": int(group["study_id"].nunique()),
                "n_countries": int(group["country"].nunique()),
                "n_series": int(group["site_depth_series"].nunique()),
                "max_study_fraction": float(fractions.max()) if len(group) else np.nan,
                "effective_studies": float(1 / np.square(fractions).sum()) if len(group) else np.nan,
                "study_ids": ";".join(sorted(group["study_id"].dropna().unique(), key=str)),
                "countries": ";".join(sorted(group["country"].dropna().unique(), key=str)),
                "dominant_study_id": str(study_counts.index[0]) if len(group) else "",
            })
    return pd.DataFrame(output)


def contrasts(clean: pd.DataFrame, profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Describe surface/deep and prespecified near-30-cm interior contrasts.

    Surface/deep refer to smallest/largest upper depth coordinate in each
    eligible series. The interior anchor is selected by depth alone, nearest
    30 cm among interior observations (ties choose shallower). It is never
    selected as the minimum Cd value. Raw differences and log ratios cover
    varying spans and must not be pooled as identical-depth effects.
    """
    profiles = profile_table(clean) if profiles is None else profiles
    allowed = set(profiles.loc[profiles["eligible_primary"], "site_depth_series"])
    columns = ["site_depth_series", "study_id", "site_name", "block_rep", "country", "n_layers",
               "surface_min_cm", "surface_upper_cm", "surface_mid_cm", "surface_cd_mg_kg",
               "deep_min_cm", "deep_upper_cm", "deep_mid_cm", "deep_cd_mg_kg",
               "upper_depth_span_cm", "raw_delta_mg_kg", "log_deep_surface_ratio",
               "delta_per_10cm", "log_ratio_per_10cm", "positive_endpoints",
               "has_interior_anchor", "interior_anchor_upper_cm", "interior_anchor_cd_mg_kg",
               "early_delta_mg_kg", "late_delta_mg_kg", "early_log_ratio", "late_log_ratio",
               "u_pattern_at_depth_anchor"]
    output = []
    for series_id, group in clean.loc[clean["site_depth_series"].isin(allowed)].groupby("site_depth_series", sort=True):
        group = group.sort_values(["depth_max_cm", "depth_min_cm", "source_row"])
        surface, deep = group.iloc[0], group.iloc[-1]
        span = float(deep["depth_max_cm"] - surface["depth_max_cm"])
        if span <= 0:
            raise ValueError(f"Nonpositive upper-depth span in eligible series {series_id}")
        low, high = float(surface["total_cd_mg_kg"]), float(deep["total_cd_mg_kg"])
        positive = low > 0 and high > 0
        log_ratio = float(np.log(high / low)) if positive else np.nan
        row = {
            "site_depth_series": series_id,
            **{name: surface[name] for name in ("study_id", "site_name", "block_rep", "country")},
            "n_layers": len(group),
            "surface_min_cm": float(surface["depth_min_cm"]),
            "surface_upper_cm": float(surface["depth_max_cm"]),
            "surface_mid_cm": float(surface["depth_mid_cm"]), "surface_cd_mg_kg": low,
            "deep_min_cm": float(deep["depth_min_cm"]), "deep_upper_cm": float(deep["depth_max_cm"]),
            "deep_mid_cm": float(deep["depth_mid_cm"]), "deep_cd_mg_kg": high,
            "upper_depth_span_cm": span, "raw_delta_mg_kg": high - low,
            "log_deep_surface_ratio": log_ratio, "delta_per_10cm": 10 * (high - low) / span,
            "log_ratio_per_10cm": 10 * log_ratio / span, "positive_endpoints": positive,
            "has_interior_anchor": len(group) >= 3,
            "interior_anchor_upper_cm": np.nan, "interior_anchor_cd_mg_kg": np.nan,
            "early_delta_mg_kg": np.nan, "late_delta_mg_kg": np.nan,
            "early_log_ratio": np.nan, "late_log_ratio": np.nan,
            "u_pattern_at_depth_anchor": pd.NA,
        }
        if len(group) >= 3:
            interior = group.iloc[1:-1].copy()
            interior["anchor_distance"] = abs(interior["depth_max_cm"] - 30.0)
            anchor = interior.sort_values(["anchor_distance", "depth_max_cm", "source_row"]).iloc[0]
            middle = float(anchor["total_cd_mg_kg"])
            early, late = middle - low, high - middle
            row.update({
                "interior_anchor_upper_cm": float(anchor["depth_max_cm"]),
                "interior_anchor_cd_mg_kg": middle,
                "early_delta_mg_kg": early, "late_delta_mg_kg": late,
                "early_log_ratio": float(np.log(middle / low)) if middle > 0 and low > 0 else np.nan,
                "late_log_ratio": float(np.log(high / middle)) if high > 0 and middle > 0 else np.nan,
                "u_pattern_at_depth_anchor": bool(early < 0 and late > 0),
            })
        output.append(row)
    return pd.DataFrame(output, columns=columns)
