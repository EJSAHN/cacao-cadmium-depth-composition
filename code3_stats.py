"""Small, deterministic CODE3 diagnostics; no causal effect or hypothesis test.

Grid settings are fixed for sensitivity analysis, not selected for a desired
result. Constructed profiles are explicitly counterexamples, never corrected
observations. Common-support contrasts remain study-specific.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from depth_core import profile_table
from depth_models import _prepare, _operator

SPANS = (.35, .5, .75, 1.)
METHODS = ('local_quadratic', 'local_linear', 'positive_kernel_mean')
STATES = ('observed', 'flattened', 'descending', 'ascending')


def eligible_ids(data):
    p = profile_table(data)
    return set(p.loc[p.eligible_primary, 'site_depth_series'])


def fixed_operator(x, center, span=.75, method='local_quadratic', weights=None):
    """Linear operator with a shared neighbourhood; local mean has no negatives.

    The quadratic is exactly CODE1's operator, not R loess. The linear/mean
    use the same tricube kernel and tie rules. Rank deficiency is reported.
    Caller must enforce global support before use.
    """
    x = np.asarray(x, dtype=float)
    if method not in METHODS or not np.isfinite(span) or not 0 < span <= 1:
        raise ValueError('Unknown method or invalid span')
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(center):
        raise ValueError('Finite nonempty coordinates required')
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != x.shape or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Weights must be finite and positive')
    q = max(1, int(math.ceil(span*len(x))))
    if method == 'local_quadratic':
        return _operator(x, float(center), q, w)
    distances = np.abs(x-center)
    radius = float(np.partition(distances, q-1)[q-1])
    if radius == 0:
        a = w*(distances == 0)
        return a/a.sum(), int(np.sum(a > 0)), 1, 0
    loc = distances <= np.nextafter(radius, np.inf)
    z = (x[loc]-center)/(radius*(1+1e-8))
    kernel_w = w[loc]*np.maximum(0, 1-np.abs(z)**3)**3
    kernel_w /= kernel_w.max()
    keep = kernel_w > 1e-14
    ix, z, kernel_w = np.flatnonzero(loc)[keep], z[keep], kernel_w[keep]
    degree = 1 if method == 'local_linear' else 0
    if degree:
        design = np.column_stack((np.ones(len(z)), z))
        weighted = design*np.sqrt(kernel_w)[:, None]
        singular = np.linalg.svd(weighted, compute_uv=False)
        rank = int(np.sum(singular > singular[0]*1e-10))
        if rank == 2:
            a = np.zeros(len(x))
            a[ix] = np.linalg.pinv(weighted, rcond=1e-10)[0]*np.sqrt(kernel_w)
            return a, len(ix), 2, 1
    a = np.zeros(len(x))
    a[ix] = kernel_w/kernel_w.sum()
    return a, len(ix), 1, 0


def fitted_contrast(data, axis='upper', method='local_quadratic', span=.75,
                    left=50., right=80., weighting='rows'):
    """Observed = series-mean context + residual, using one linear operator."""
    work, x, responses, weights, _ = _prepare(data, axis, weighting)
    result = dict(axis=axis, method=method, span=float(span), weighting=weighting,
                  left_coordinate_cm=float(left), right_coordinate_cm=float(right),
                  n_rows=len(work), n_series=int(work.site_depth_series.nunique()),
                  n_studies=int(work.study_id.nunique()),
                  support_min_cm=float(x.min()) if len(x) else np.nan,
                  support_max_cm=float(x.max()) if len(x) else np.nan,
                  observed_change=np.nan, flattened_change=np.nan, within_change=np.nan,
                  left_fitted=np.nan, right_fitted=np.nan, additivity_error=np.nan,
                  left_degree=-1, right_degree=-1, left_n_local=0, right_n_local=0,
                  left_negative_weight_mass=np.nan, right_negative_weight_mass=np.nan,
                  support_status='outside_observed_range')
    if not len(x) or min(left, right)<x.min() or max(left, right)>x.max():
        return result
    oa, na, _, da = fixed_operator(x, left, span, method, weights)
    ob, nb, _, db = fixed_operator(x, right, span, method, weights)
    f = np.vstack((oa, ob))@responses
    change = f[1]-f[0]
    result.update(observed_change=float(change[0]), flattened_change=float(change[1]),
                  within_change=float(change[2]), left_fitted=float(f[0, 0]),
                  right_fitted=float(f[1, 0]), additivity_error=float(change[0]-change[1]-change[2]),
                  left_degree=da, right_degree=db, left_n_local=na, right_n_local=nb,
                  left_negative_weight_mass=float(-oa[oa<0].sum()),
                  right_negative_weight_mass=float(-ob[ob<0].sum()),
                  support_status='in_observed_range')
    return result


def smoother_sensitivity(data):
    """Seven source/axis scenarios x three methods x four fixed spans."""
    data = data.copy()
    data['study_id'] = data.study_id.astype(str)
    allowed = eligible_ids(data)
    multi = data.loc[data.site_depth_series.isin(allowed)]
    conservative = data.loc[~data.study_id.isin(['17', '27'])]
    audited = data.loc[data.study_id.isin(['20', '31'])]
    scenarios = [('all_rows_upper', data, 'upper'),
                 ('multidepth_upper', multi, 'upper'),
                 ('conservative_upper', conservative, 'upper'),
                 ('audited_upper', audited, 'upper'),
                 ('multidepth_midpoint', multi.loc[~multi.study_id.isin(['17', '27'])], 'midpoint'),
                 ('conservative_midpoint', conservative, 'midpoint'),
                 ('audited_midpoint', audited, 'midpoint')]
    rows=[]
    for scenario, subset, axis in scenarios:
        for method in METHODS:
            for span in SPANS:
                row=fitted_contrast(subset, axis, method, span)
                row.update(scenario=scenario, source_interpretation=(
                    'as_extracted_aggregates_present' if scenario in ('all_rows_upper','multidepth_upper')
                    else 'primary_means_checked_20_31' if scenario.startswith('audited')
                    else '17_27_excluded_22_not_fully_checked'))
                rows.append(row)
    return pd.DataFrame(rows)


def construct_profiles(data, state):
    """Within eligible series only; retain depths, means and frozen rows.

    Ascending/descending also preserve the exact multiset of concentrations.
    Flattening replaces the multiset with its arithmetic mean. Every construct
    is hypothetical and must never be presented as a measured soil profile.
    """
    if state not in STATES:
        raise ValueError('Unknown constructed profile state')
    out=data.copy().reset_index(drop=True)
    allow=eligible_ids(out)
    out['observed_cd_mg_kg']=out.total_cd_mg_kg
    out['eligible_constructed_series']=out.site_depth_series.isin(allow)
    for sid, group in out.groupby('site_depth_series', sort=True):
        if sid not in allow or state == 'observed':
            continue
        ix=group.sort_values(['depth_min_cm','depth_max_cm']).index
        y=out.loc[ix,'total_cd_mg_kg'].to_numpy()
        if state == 'flattened': values=np.full(len(y),y.mean())
        else:
            values=np.sort(y)
            if state == 'descending': values=values[::-1]
        if not np.isclose(values.sum(),y.sum(),rtol=1e-13,atol=1e-12):
            raise AssertionError('Constructed series changed its total')
        out.loc[ix,'total_cd_mg_kg']=values
    out['profile_state']=state
    return out


def counterexample_diagnostics(data):
    """Audited two-study subset only, with every deterministic construction."""
    audited=data.loc[data.study_id.astype(str).isin(['20','31'])].copy()
    contrasts, records, curves=[],[],[]
    for state in STATES:
        constructed=construct_profiles(audited,state)
        records.append(constructed)
        for axis in ('upper','midpoint'):
            for method in METHODS:
                for span in SPANS:
                    row=fitted_contrast(constructed,axis,method,span)
                    row.update(profile_state=state,scenario='audited_20_31',
                               n_frozen_rows=int((~constructed.eligible_constructed_series).sum()),
                               hypothetical=state!='observed')
                    contrasts.append(row)
                # Full diagnostic curves at the original fixed span only.
                work,x,responses,weights,_=_prepare(constructed,axis,'rows')
                if not len(x):continue
                for center in np.linspace(x.min(),x.max(),101):
                    operator,n,rank,degree=fixed_operator(x,float(center),.75,method,weights)
                    curves.append(dict(scenario='audited_20_31',profile_state=state,
                        hypothetical=state!='observed',axis=axis,method=method,span=.75,
                        coordinate_cm=float(center),fitted_cd_mg_kg=float(operator@responses[:,0]),
                        n_local=n,rank=rank,degree=degree))
    return pd.DataFrame(contrasts),pd.concat(records,ignore_index=True),pd.DataFrame(curves)


def _whole_interval_index(group, lower, upper):
    """Require complete exact tiling, never split a layer or interpolate."""
    subset=group.loc[(group.depth_min_cm>=lower)&(group.depth_max_cm<=upper)].sort_values('depth_min_cm')
    if not len(subset):return None
    cursor=float(lower)
    for row in subset.itertuples(index=False):
        if float(row.depth_min_cm)!=cursor:return None
        cursor=float(row.depth_max_cm)
    if cursor!=float(upper):return None
    widths=subset.depth_max_cm-subset.depth_min_cm
    value=float(np.dot(widths,subset.total_cd_mg_kg)/(upper-lower))
    layers=';'.join(f'{a:g}-{b:g}' for a,b in zip(subset.depth_min_cm,subset.depth_max_cm))
    return value,layers


def common_support(data):
    """Depth-weighted 0-30 index vs intact 30-50 layer, Ghana/Ecuador only.

    This is a concentration index over depth, NOT a cadmium stock. Interpreting
    it as a mass-weighted concentration additionally assumes equal bulk density
    (and compatible fine-earth fractions), which these data do not establish.
    Studies are not pooled; different analytical methods remain a limitation.
    """
    data=data.loc[data.study_id.astype(str).isin(['20','31'])].copy()
    allow=eligible_ids(data)
    rows=[]
    for sid,group in data.groupby('site_depth_series',sort=True):
        first=group.iloc[0]
        record=dict(study_id=str(first.study_id),country=first.country,
                    site_name=first.site_name,site_depth_series=sid,
                    shallow_index_mg_kg=np.nan,deep_cd_mg_kg=np.nan,delta_mg_kg=np.nan,
                    shallow_layers='',deep_layer='30-50',status='ineligible_series')
        if sid in allow:
            shallow=_whole_interval_index(group,0,30)
            deep=group.loc[group.depth_min_cm.eq(30)&group.depth_max_cm.eq(50)]
            if shallow is not None and len(deep)==1:
                d=float(deep.total_cd_mg_kg.iloc[0])
                record.update(shallow_index_mg_kg=shallow[0],shallow_layers=shallow[1],
                              deep_cd_mg_kg=d,delta_mg_kg=d-shallow[0],status='complete_exact_union')
            else:record['status']='missing_exact_0_30_union_or_30_50_layer'
        rows.append(record)
    cols=['study_id','country','site_name','site_depth_series','shallow_index_mg_kg',
          'deep_cd_mg_kg','delta_mg_kg','shallow_layers','deep_layer','status']
    sites=pd.DataFrame(rows,columns=cols)
    summary=[]
    for (study,country),group in sites.groupby(['study_id','country'],sort=True):
        valid=group.loc[group.status.eq('complete_exact_union')]
        summary.append(dict(study_id=str(study),country=country,n_candidate_series=len(group),
            n_complete_series=len(valid),n_excluded_series=len(group)-len(valid),
            shallow_index_mean_mg_kg=float(valid.shallow_index_mg_kg.mean()),
            deep_mean_mg_kg=float(valid.deep_cd_mg_kg.mean()),
            mean_delta_mg_kg=float(valid.delta_mg_kg.mean()),median_delta_mg_kg=float(valid.delta_mg_kg.median()),
            n_increasing=int(valid.delta_mg_kg.gt(1e-12).sum()),
            n_decreasing=int(valid.delta_mg_kg.lt(-1e-12).sum()),
            n_equal=int(valid.delta_mg_kg.abs().le(1e-12).sum()),
            estimand='depth_weighted_0_30_concentration_index_vs_intact_30_50',
            bulk_density_assumption='not_needed_for_index; constant_density_needed_for_mass_weighted_interpretation',
            inferential_intervals=False))
    return sites,pd.DataFrame(summary)


def ghana_leave_one_site(data):
    """Influence in audited subset; cannot supply independent replication."""
    data=data.loc[data.study_id.astype(str).isin(['20','31'])].copy()
    ghana=data.loc[data.study_id.astype(str).eq('20')]
    removals=[('none',set()),('all_Ghana',set(ghana.site_depth_series))]
    removals += [(str(sid),{sid}) for sid in sorted(ghana.site_depth_series.unique())]
    rows=[]
    for label,excluded in removals:
        subset=data.loc[~data.site_depth_series.isin(excluded)]
        flat=construct_profiles(subset,'flattened')
        descending=construct_profiles(subset,'descending')
        for method in METHODS:
            result=fitted_contrast(subset,'upper',method,.75)
            # Explicit constructions have no ambiguity for these two studies.
            result.update(omitted_ghana_series=label,n_omitted_series=len(excluded),
                flattened_constructed_change=fitted_contrast(flat,'upper',method,.75)['observed_change'],
                descending_constructed_change=fitted_contrast(descending,'upper',method,.75)['observed_change'],
                remaining_ghana_series=int(subset.loc[subset.study_id.astype(str).eq('20'),'site_depth_series'].nunique()))
            rows.append(result)
    return pd.DataFrame(rows)


def _json_records(frame):
    # pandas maps NaN to null, making missing support unambiguous in JSON.
    return json.loads(frame.to_json(orient='records',double_precision=15))


def run_code3_analysis(data, output_dir, permutations=2000, seed=20261106):
    """Write seven small tables. Compatibility seed/draw arguments are unused."""
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    sensitivity=smoother_sensitivity(data)
    counter,records,curves=counterexample_diagnostics(data)
    sites,studies=common_support(data)
    influence=ghana_leave_one_site(data)
    tables={'code3_smoother_sensitivity.csv':sensitivity,
            'code3_counterexample_contrasts.csv':counter,
            'code3_counterexample_records.csv':records,
            'code3_counterexample_curves.csv':curves,
            'code3_common_support_sites.csv':sites,
            'code3_common_support_studies.csv':studies,
            'code3_ghana_leave_one_site.csv':influence}
    for name,frame in tables.items():frame.to_csv(output_dir/name,index=False)
    focal=counter.loc[counter.axis.eq('upper')&counter.span.eq(.75)]
    valid=sensitivity.loc[sensitivity.support_status.eq('in_observed_range')]
    return dict(deterministic=True,monte_carlo_performed=False,compatibility_seed_ignored=int(seed),
        compatibility_permutations_ignored=int(permutations),n_sensitivity_settings=len(sensitivity),
        n_supported_sensitivity_settings=len(valid),n_counterexample_settings=len(counter),
        positive_flattened_sensitivity_settings=int(valid.flattened_change.gt(1e-12).sum()),
        negative_flattened_sensitivity_settings=int(valid.flattened_change.lt(-1e-12).sum()),
        zero_flattened_sensitivity_settings=int(valid.flattened_change.abs().le(1e-12).sum()),
        fixed_spans=list(SPANS),methods=list(METHODS),
        max_additivity_error=float(valid.additivity_error.abs().max()) if len(valid) else None,
        audited_counterexamples_upper_span075=_json_records(focal),
        common_support_study_results=_json_records(studies),
        common_support_complete_series=int(sites.status.eq('complete_exact_union').sum()),
        common_support_excluded_series=int(sites.status.ne('complete_exact_union').sum()),
        outputs=list(tables),
        interpretation='Descriptive robustness and constructed counterexamples, not p-values, causal fractions or population CIs.',
        original_R_loess_equivalence_claimed=False,
        caveats=['Settings fixed after CODE2 inspection; exploratory, not preregistered.',
                 'Two-study counterexamples retain assay, site-selection and depth-coverage differences.',
                 'Ordered profiles are hypothetical; monotonicity is constructed, not a measured finding.',
                 'Common support uses thickness-weighted index, not mass-weighted Cd or stock.',
                 'Leave-one-site checks Ghana influence, not independent study replication.'])
