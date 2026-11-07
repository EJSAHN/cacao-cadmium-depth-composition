"""Observation masks on fixed empirical site summaries; no ML or inference.

Every included value is held fixed. Only whether the 30-50 cm layer is visible
changes. The mixed-study dataset is a deliberate stress-test population, not a
worldwide estimand. Mask fractions are exhaustive descriptions, not probabilities
of real missingness, p-values or population confidence intervals.
"""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from code3_stats import common_support

TOL = 1e-12
VALUE_COLUMNS = ('shallow_index_mg_kg', 'deep_cd_mg_kg', 'delta_mg_kg')


def _validated_sites(sites):
    cols={'site_depth_series','study_id','country','site_name',*VALUE_COLUMNS}
    if not cols.issubset(sites.columns):
        raise ValueError('Missing fixed-site value/identity columns')
    s=sites.copy().sort_values('site_depth_series',kind='mergesort').reset_index(drop=True)
    if s.site_depth_series.isna().any() or s.site_depth_series.duplicated().any():
        raise ValueError('Site identities must be nonmissing and unique')
    if not len(s) or not np.isfinite(s[list(VALUE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError('Need nonempty finite fixed-site values')
    if not np.allclose(s.deep_cd_mg_kg-s.shallow_index_mg_kg,s.delta_mg_kg,rtol=0,atol=1e-10):
        raise ValueError('Site delta does not equal deep minus shallow')
    return s


def sign(value):
    return 'positive' if value>TOL else 'negative' if value < -TOL else 'zero'


def mask_contrast(sites, observed_ids, scope='mixed_20_31', mask='unnamed'):
    """An exact bookkeeping identity using the same deep-observed sites.

    naive = mean(D_visible) - mean(S_all)
          = mean(D-S)_visible + mean(S_visible) - mean(S_all).

    Selection term is an algebraic difference of retained shallow means, not a
    causal estimate. No outcomes are reweighted, imputed or rearranged.
    """
    s=_validated_sites(sites)
    wanted=set(observed_ids)
    known=set(s.site_depth_series)
    if not wanted.issubset(known):raise ValueError('Unknown observed site identity')
    visible=s.loc[s.site_depth_series.isin(wanted)]
    n=len(s);k=len(visible)
    shallow_all=float(s.shallow_index_mg_kg.mean())
    baseline=float(s.delta_mg_kg.mean())
    result=dict(scope=scope,mask=mask,hypothetical_mask=True,n_shallow=n,n_deep_observed=k,
                n_deep_hidden=n-k,shallow_all_mean_mg_kg=shallow_all,
                deep_visible_mean_mg_kg=np.nan,shallow_visible_mean_mg_kg=np.nan,
                naive_delta_mg_kg=np.nan,paired_visible_delta_mg_kg=np.nan,
                selection_term_mg_kg=np.nan,identity_error_mg_kg=np.nan,
                full_pairs_delta_mg_kg=baseline,baseline_sign=sign(baseline),
                naive_sign='unsupported',paired_sign='unsupported',naive_flip_from_full=False,
                paired_flip_from_full=False,naive_paired_disagree=False,
                naive_flip_while_paired_retains_baseline=False,
                n_ghana_deep=int(visible.study_id.astype(str).eq('20').sum()),
                n_ecuador_deep=int(visible.study_id.astype(str).eq('31').sum()),
                status='no_deep_observations')
    if not k:return result
    dm=float(visible.deep_cd_mg_kg.mean());sm=float(visible.shallow_index_mg_kg.mean())
    naive=dm-shallow_all;paired=float(visible.delta_mg_kg.mean());selection=sm-shallow_all
    error=naive-paired-selection
    if abs(error)>1e-10:raise AssertionError('Observation-mask identity failed')
    ns,ps,bs=sign(naive),sign(paired),sign(baseline)
    opposite=lambda a,b: {a,b}=={'positive','negative'}
    result.update(deep_visible_mean_mg_kg=dm,shallow_visible_mean_mg_kg=sm,
                  naive_delta_mg_kg=naive,paired_visible_delta_mg_kg=paired,
                  selection_term_mg_kg=selection,identity_error_mg_kg=error,
                  naive_sign=ns,paired_sign=ps,naive_flip_from_full=opposite(ns,bs),
                  paired_flip_from_full=opposite(ps,bs),naive_paired_disagree=opposite(ns,ps),
                  naive_flip_while_paired_retains_baseline=opposite(ns,bs) and ps==bs,
                  status='supported')
    return result


def exhaustive_omissions(sites, n_hidden, scope='mixed_20_31'):
    """Enumerate every labelled mask at one or two omitted deep layers."""
    s=_validated_sites(sites)
    if n_hidden not in (1,2):raise ValueError('Only one/two omissions are enumerated')
    identities=list(s.site_depth_series)
    out=[]
    if n_hidden>=len(s):return pd.DataFrame()
    lookup=s.set_index('site_depth_series')
    for hidden in combinations(identities,n_hidden):
        kept=set(identities).difference(hidden)
        row=mask_contrast(s,kept,scope,mask='hide_'+str(n_hidden))
        row.update(hidden_site_ids=';'.join(hidden),
                   hidden_site_names=';'.join(str(lookup.loc[i,'site_name']) for i in hidden),
                   hidden_deep_values_mg_kg=';'.join(format(float(lookup.loc[i,'deep_cd_mg_kg']),'.12g') for i in hidden))
        out.append(row)
    return pd.DataFrame(out)


def retention_bounds(sites,scope='mixed_20_31'):
    """Exact ranges over ALL subsets of size k, found by order statistics.

    These are extremal ranges, not quantiles. Naive and matched extrema generally
    use different masks. Their source IDs are included to make that explicit.
    Only the naive extremizing masks receive the exact joint decomposition.
    """
    s=_validated_sites(sites)
    ds=s.sort_values(['deep_cd_mg_kg','site_depth_series'])
    ps=s.sort_values(['delta_mg_kg','site_depth_series'])
    rows=[]
    for k in range(1,len(s)+1):
        lo=ds.iloc[:k];hi=ds.iloc[-k:]
        low=mask_contrast(s,set(lo.site_depth_series),scope,'naive_min_at_k')
        high=mask_contrast(s,set(hi.site_depth_series),scope,'naive_max_at_k')
        pl=ps.iloc[:k];ph=ps.iloc[-k:]
        rows.append(dict(scope=scope,n_sites=len(s),n_deep_observed=k,n_deep_hidden=len(s)-k,
            naive_min_mg_kg=low['naive_delta_mg_kg'],naive_max_mg_kg=high['naive_delta_mg_kg'],
            paired_min_mg_kg=float(pl.delta_mg_kg.mean()),paired_max_mg_kg=float(ph.delta_mg_kg.mean()),
            paired_at_naive_min_mg_kg=low['paired_visible_delta_mg_kg'],
            paired_at_naive_max_mg_kg=high['paired_visible_delta_mg_kg'],
            selection_at_naive_min_mg_kg=low['selection_term_mg_kg'],
            selection_at_naive_max_mg_kg=high['selection_term_mg_kg'],
            naive_min_site_ids=';'.join(lo.site_depth_series),naive_max_site_ids=';'.join(hi.site_depth_series),
            paired_min_site_ids=';'.join(pl.site_depth_series),paired_max_site_ids=';'.join(ph.site_depth_series),
            baseline_delta_mg_kg=float(s.delta_mg_kg.mean()),
            both_naive_signs_possible=low['naive_delta_mg_kg']<-TOL and high['naive_delta_mg_kg']>TOL,
            bounds_type='exact_subset_extrema_not_confidence_interval',
            paired_extrema_use_separate_masks=True))
    return pd.DataFrame(rows)


def _json_records(df):
    return json.loads(df.to_json(orient='records',double_precision=15))


def run_code4_analysis(data,output_dir):
    """Write small fixed-value masking diagnostics; do not alter input data."""
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    inventory,study_summary=common_support(data)
    sites=inventory.loc[inventory.status.eq('complete_exact_union')].copy()
    sites=_validated_sites(sites)
    # Source rows specify the original intervals that made each fixed index.
    shallow_sources=[];deep_sources=[]
    for sid in sites.site_depth_series:
        g=data.loc[data.site_depth_series.eq(sid)]
        shallow_sources.append(';'.join(str(int(v)) for v in g.loc[g.depth_min_cm.ge(0)&g.depth_max_cm.le(30)].sort_values('depth_min_cm').source_row))
        deep_sources.append(';'.join(str(int(v)) for v in g.loc[g.depth_min_cm.eq(30)&g.depth_max_cm.eq(50)].source_row))
    sites['shallow_source_rows']=shallow_sources;sites['deep_source_rows']=deep_sources
    scopes={'mixed_20_31':sites,'Ghana_20':sites.loc[sites.study_id.astype(str).eq('20')],
            'Ecuador_31':sites.loc[sites.study_id.astype(str).eq('31')]}
    scenarios=[];membership=[];singles=[];doubles=[];bounds=[]
    for scope,frame in scopes.items():
        base=mask_contrast(frame,set(frame.site_depth_series),scope,'all_deep_visible')
        base['hypothetical_mask']=False;scenarios.append(base)
        singles.append(exhaustive_omissions(frame,1,scope))
        doubles.append(exhaustive_omissions(frame,2,scope))
        bounds.append(retention_bounds(frame,scope))
    for study,country in [('20','Ghana'),('31','Ecuador')]:
        kept=set(sites.loc[sites.study_id.astype(str).eq(study),'site_depth_series'])
        label=country+'_deep_only'
        scenarios.append(mask_contrast(sites,kept,'mixed_20_31',label))
        for row in sites.itertuples():
            membership.append(dict(mask=label,site_depth_series=row.site_depth_series,country=row.country,
                shallow_visible=True,deep_visible=row.site_depth_series in kept,
                shallow_index_mg_kg=row.shallow_index_mg_kg,deep_cd_mg_kg=row.deep_cd_mg_kg,
                values_unchanged=True,hypothetical_mask=True))
    scenarios=pd.DataFrame(scenarios)
    singles=pd.concat(singles,ignore_index=True);doubles=pd.concat(doubles,ignore_index=True)
    bounds=pd.concat(bounds,ignore_index=True)
    outputs={'code4_common_support_inventory.csv':inventory,'code4_site_values.csv':sites,
             'code4_mask_scenarios.csv':scenarios,'code4_mask_membership.csv':pd.DataFrame(membership),
             'code4_single_omissions.csv':singles,'code4_double_omissions.csv':doubles,
             'code4_retention_bounds.csv':bounds}
    for name,frame in outputs.items():frame.to_csv(output_dir/name,index=False)
    mixed_single=singles.loc[singles.scope.eq('mixed_20_31')]
    mixed_double=doubles.loc[doubles.scope.eq('mixed_20_31')]
    witnesses=mixed_single.loc[mixed_single.naive_flip_while_paired_retains_baseline]
    mb=bounds.loc[bounds.scope.eq('mixed_20_31')]
    baseline=float(sites.delta_mg_kg.mean())
    can_flip=mb.naive_min_mg_kg.lt(-TOL) if baseline>TOL else mb.naive_max_mg_kg.gt(TOL)
    min_any=int(mb.loc[can_flip,'n_deep_hidden'].min()) if abs(baseline)>TOL and can_flip.any() else None
    exact_terms=pd.concat([scenarios,singles,doubles],ignore_index=True)
    scope_results=[]
    for scope,frame in scopes.items():
        one=singles.loc[singles.scope.eq(scope)]
        two=doubles.loc[doubles.scope.eq(scope)]
        sb=bounds.loc[bounds.scope.eq(scope)]
        b=float(frame.delta_mg_kg.mean())
        f=sb.naive_min_mg_kg.lt(-TOL) if b>TOL else sb.naive_max_mg_kg.gt(TOL)
        w1=one.loc[one.naive_flip_while_paired_retains_baseline]
        w2=two.loc[two.naive_flip_while_paired_retains_baseline]
        scope_results.append(dict(scope=scope,n_sites=len(frame),baseline_delta_mg_kg=b,
            single_masks=len(one),single_naive_flips=int(one.naive_flip_from_full.sum()),
            single_naive_flips_with_paired_baseline_sign=len(w1),double_masks=len(two),
            double_naive_flips=int(two.naive_flip_from_full.sum()),
            double_naive_flips_with_paired_baseline_sign=len(w2),
            minimum_hidden_for_naive_flip_exact=int(sb.loc[f,'n_deep_hidden'].min()) if abs(b)>TOL and f.any() else None,
            minimum_hidden_for_flip_with_paired_baseline_sign=1 if len(w1) else 2 if len(w2) else None,
            joint_minimum_search_limit_omissions=2))
    within_study_double_witnesses=doubles.loc[doubles.scope.ne('mixed_20_31')&doubles.naive_flip_while_paired_retains_baseline]
    report={
        'analysis':'fixed_values_changed_observation_mask','deterministic':True,'monte_carlo_performed':False,
        'no_outcomes_rearranged_or_imputed':True,'n_complete_sites':len(sites),
        'n_candidate_sites':len(inventory),'n_excluded_incomplete_sites':len(inventory)-len(sites),
        'mixed_table_baseline_delta_mg_kg':baseline,
        'named_scenarios':_json_records(scenarios),
        'mixed_single_masks':len(mixed_single),
        'mixed_single_naive_sign_flips':int(mixed_single.naive_flip_from_full.sum()),
        'mixed_single_naive_flip_with_paired_baseline_sign':len(witnesses),
        'mixed_double_masks':len(mixed_double),
        'mixed_double_naive_sign_flips':int(mixed_double.naive_flip_from_full.sum()),
        'mixed_double_naive_flip_with_paired_baseline_sign':int(mixed_double.naive_flip_while_paired_retains_baseline.sum()),
        'single_omission_witnesses':_json_records(witnesses),
        'scope_results':scope_results,
        'within_study_double_omission_witnesses':_json_records(within_study_double_witnesses),
        'minimum_hidden_deep_for_any_naive_sign_flip_exact':min_any,
        'minimum_hidden_deep_for_naive_flip_with_paired_sign_retained':1 if len(witnesses) else 2 if mixed_double.naive_flip_while_paired_retains_baseline.any() else None,
        'joint_minimum_search_limit_omissions':2,
        'max_decomposition_error_mg_kg':float(exact_terms.identity_error_mg_kg.abs().max()),
        'outputs':list(outputs),
        'interpretation':'Finite-table selection stress test, not an estimate of worldwide depth effects or real missingness probabilities.',
        'caveats':[
            'Designed after CODE3; exploratory, not preregistered.',
            '0-30 concentration index is depth weighted, not Cd stock or established mass-weighted concentration.',
            'Mixed-study baseline is an arithmetic reference for this table; study, method and site selection are confounded.',
            'Masks are hypothetical; all observed source values remain fixed and unchanged.',
            'All shallow summaries remain visible, but only selected30-50 layers are visible; deeper layers are outside this common-support exercise.',
            'Exact extrema are adversarial attainable limits, not confidence intervals; no assumption makes masks equiprobable.',
            'Paired estimates target retained sites and can themselves change under selection; they do not recover missing population effects.',
            'Single/double mask counts are exhaustive descriptions, not independent experiments or p-values.'
        ]}
    json.dumps(report,allow_nan=False)
    return report
