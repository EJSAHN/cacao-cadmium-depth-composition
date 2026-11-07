"""Descriptive, exactly additive depth-curve diagnostics.

This module implements its own fixed-neighbourhood, local quadratic tricube
smoother.  It has no robust iterations, inference, confidence intervals, or
claim of numerical equivalence to R's ``loess``.  At every grid point, the SAME
linear smoothing operator is applied to observed Cd, its series mean, and the
within-series residual.  Thus the three curves obey an additive identity.

The context curve replaces each reconstructed site's depth observations with
their arithmetic mean while retaining the observed depths and row weights.
It is a diagnostic of the observed sampling composition, not a causal effect.
Reconstructed series must already have been checked by the calling pipeline.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


_AXES = {
    "upper": "depth_max_cm",
    "max": "depth_max_cm",
    "mid": "depth_mid_cm",
    "midpoint": "depth_mid_cm",
    "lower": "depth_min_cm",
    "min": "depth_min_cm",
    "depth_max_cm": "depth_max_cm",
    "depth_mid_cm": "depth_mid_cm",
    "depth_min_cm": "depth_min_cm",
}
_COLUMNS = [
    "grid_depth", "observed", "context", "within", "n_local", "rank",
    "method", "status",
]


def _prepare(df: pd.DataFrame, axis: str, weighting: str) -> tuple:
    if axis not in _AXES:
        raise ValueError("axis must be upper, midpoint, lower, or a depth column name")
    if weighting not in {"rows", "equal_series", "equal_study"}:
        raise ValueError("weighting must be rows, equal_series, or equal_study")
    depth_col = _AXES[axis]
    required = {"study_id", "site_depth_series", "total_cd_mg_kg", depth_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    work = df.copy()
    # Missing concentrations/depths cannot contribute to any of the curves.
    work[depth_col] = pd.to_numeric(work[depth_col], errors="coerce")
    work["total_cd_mg_kg"] = pd.to_numeric(work["total_cd_mg_kg"], errors="coerce")
    keep = np.isfinite(work[depth_col]) & np.isfinite(work["total_cd_mg_kg"])
    work = work.loc[keep].copy()
    if work[["study_id", "site_depth_series"]].isna().any().any():
        raise ValueError("Finite observations need non-missing study and series IDs")
    work["study_id"] = work["study_id"].astype(str)
    work["site_depth_series"] = work["site_depth_series"].astype(str)
    if (work["study_id"].str.strip().eq("") |
            work["site_depth_series"].str.strip().eq("")).any():
        raise ValueError("Study and series IDs cannot be empty")
    if work.groupby("site_depth_series")["study_id"].nunique().gt(1).any():
        raise ValueError("Each site_depth_series must belong to exactly one study")
    if "country" in work and work.groupby("site_depth_series")["country"].nunique().gt(1).any():
        raise ValueError("A site_depth_series cannot span countries")

    # Stable numerical order also makes ties independent of incoming row order.
    work = work.sort_values(
        ["site_depth_series", depth_col, "total_cd_mg_kg"], kind="mergesort"
    ).reset_index(drop=True)
    context = work.groupby("site_depth_series", sort=False)["total_cd_mg_kg"].transform("mean")
    y = work["total_cd_mg_kg"].to_numpy(dtype=float)
    c = context.to_numpy(dtype=float)
    responses = np.column_stack((y, c, y - c))
    if weighting == "rows":
        base_weights = np.ones(len(work), dtype=float)
    else:
        group = "site_depth_series" if weighting == "equal_series" else "study_id"
        counts = work.groupby(group, sort=False)[group].transform("size")
        base_weights = 1.0 / counts.to_numpy(dtype=float)
    metadata = {
        "axis": axis,
        "depth_column": depth_col,
        "weighting": weighting,
        "n_input_rows": int(len(df)),
        "n_rows": int(len(work)),
        "n_excluded_nonfinite": int(len(df) - len(work)),
        "n_series": int(work["site_depth_series"].nunique()),
        "n_studies": int(work["study_id"].nunique()),
        "description": "Descriptive fixed linear smoother; no confidence intervals or causal interpretation",
        "context_definition": "Arithmetic mean of Cd within each retained site depth series",
    }
    return work, work[depth_col].to_numpy(dtype=float), responses, base_weights, metadata


def _operator(x: np.ndarray, center: float, q: int, base_weights: np.ndarray) -> tuple:
    """Return intercept operator, nonzero local count, rank, and fitted degree.

    All observations tied at the q-th nearest distance enter the neighbourhood.
    A tiny bandwidth expansion makes that boundary deterministic; negligible
    kernel weights are dropped relative to the largest local kernel weight.
    A deficient quadratic falls back to a linear fit, then a weighted mean.
    """
    distances = np.abs(x - center)
    radius = float(np.partition(distances, q - 1)[q - 1])
    if radius == 0.0:
        local = distances == 0.0
        weights = base_weights * local
        return weights / weights.sum(), int(local.sum()), 1, 0

    boundary = np.nextafter(radius, np.inf)
    local = distances <= boundary
    bandwidth = radius * (1.0 + 1e-8)
    z = (x[local] - center) / bandwidth
    kernel = np.maximum(0.0, 1.0 - np.abs(z) ** 3) ** 3
    weights = base_weights[local] * kernel
    weights /= weights.max()
    effective = weights > 1e-14
    weights = weights[effective]
    z = z[effective]
    indices = np.flatnonzero(local)[effective]
    root_w = np.sqrt(weights)

    for degree in (2, 1, 0):
        design = np.column_stack([z ** power for power in range(degree + 1)])
        weighted_design = design * root_w[:, None]
        # An explicit SVD threshold avoids treating near-zero boundary weights
        # or effectively coincident depths as an estimable quadratic.
        singular = np.linalg.svd(weighted_design, compute_uv=False)
        rank = int(np.sum(singular > singular[0] * 1e-10))
        if rank == degree + 1:
            local_operator = np.linalg.pinv(weighted_design, rcond=1e-10)[0] * root_w
            operator = np.zeros(len(x), dtype=float)
            operator[indices] = local_operator
            return operator, int(len(indices)), rank, degree
    raise RuntimeError("Internal error: even the local weighted mean was undefined")


def smooth_decomposition(
    df: pd.DataFrame,
    axis: str = "upper",
    span: float = 0.75,
    weighting: str = "rows",
    grid: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Return observed = context + within descriptive depth curves.

    Parameters
    ----------
    df : DataFrame
        Columns: study_id, site_depth_series, total_cd_mg_kg, and the requested
        depth column. The calling pipeline supplies checked, globally unique
        series IDs. Singleton series contribute zero within-series residual.
    axis : str
        'upper' uses the recorded interval's maximum depth; 'midpoint' and
        'lower' use its midpoint and minimum. No interval is treated as a core
        measured at a true point: these are explicitly alternative plot axes.
    span : float
        q = ceil(span * retained row count) nearest neighbours, including ties.
        This is a row-based neighbourhood even with alternative base weights.
    weighting : str
        'rows': every row weight 1; 'equal_series': each series total weight 1;
        'equal_study': each study total weight 1. These are descriptive choices.
    grid : iterable, optional
        Default is 201 points across observed support. A supplied grid keeps
        its order; outside-support points receive NaNs and status outside_support.

    Returns
    -------
    DataFrame
        grid_depth, observed, context, within, n_local, rank, method, status.
        Nonfinite depth/Cd rows are excluded and counted in result.attrs. The
        status identifies quadratic fits, fallbacks, and unsupported points.
        No extrapolation, inference, or robust reweighting is performed.
    """
    if not np.isfinite(span) or not 0.0 < span <= 1.0:
        raise ValueError("span must lie in (0, 1]")
    _, x, responses, base_weights, metadata = _prepare(df, axis, weighting)
    if grid is None:
        grid_array = (np.unique(x) if len(x) and x.min() == x.max()
                      else np.linspace(x.min(), x.max(), 201) if len(x)
                      else np.array([], dtype=float))
    else:
        grid_array = np.asarray(list(grid), dtype=float)
        if grid_array.ndim != 1 or not np.isfinite(grid_array).all():
            raise ValueError("grid must be a one-dimensional sequence of finite numbers")
    q = max(1, int(math.ceil(float(span) * len(x))))
    records = []
    for center in grid_array:
        if not len(x) or center < x.min() or center > x.max():
            status = "no_valid_data" if not len(x) else "outside_support"
            records.append((center, np.nan, np.nan, np.nan, 0, 0, "none", status))
            continue
        operator, n_local, rank, degree = _operator(x, float(center), q, base_weights)
        fitted = operator @ responses
        method = {2: "local_quadratic", 1: "local_linear", 0: "weighted_mean"}[degree]
        status = {2: "ok", 1: "fallback_linear", 0: "fallback_mean"}[degree]
        records.append((center, *fitted.tolist(), n_local, rank, method, status))
    result = pd.DataFrame.from_records(records, columns=_COLUMNS)
    result["n_local"] = result["n_local"].astype(int)
    result["rank"] = result["rank"].astype(int)
    metadata.update({"span": float(span), "q": q if len(x) else 0})
    metadata["support_min"] = float(x.min()) if len(x) else None
    metadata["support_max"] = float(x.max()) if len(x) else None
    valid = result[["observed", "context", "within"]].dropna()
    metadata["max_additivity_error"] = float(
        (valid["observed"] - valid["context"] - valid["within"]).abs().max()
    ) if len(valid) else None
    result.attrs.update(metadata)
    return result


def covariance_decomposition(
    df: pd.DataFrame, axis: str = "upper", weighting: str = "rows"
) -> dict:
    """Weighted descriptive identity cov(depth, Cd) = context + within.

    Uses population (not sample-corrected) weighted covariances. No p-value,
    confidence interval, or biological/causal attribution is implied.
    """
    _, x, responses, weights, metadata = _prepare(df, axis, weighting)
    if len(x):
        weights = weights / weights.sum()
        x_centered = x - weights @ x
        covariances = (weights * x_centered) @ responses
        total, context, within = map(float, covariances)
        error = total - context - within
    else:
        total = context = within = error = float("nan")
    return {
        **metadata,
        "total_covariance": total,
        "context_covariance": context,
        "within_covariance": within,
        "additivity_error": error,
        "status": "ok" if len(x) else "no_valid_data",
    }
