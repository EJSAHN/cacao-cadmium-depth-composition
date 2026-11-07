"""Conditional rounding-box audit for fixed-cohort observation masks.

Each already source-checked reported layer mean receives half of its assumed
last reporting unit on either side. These closed boxes are conservative under
nearest rounding. They are not confidence intervals, sampling uncertainty, or
an allowance for laboratory error. Original values are never replaced.

Expressions use rational coefficients attached to original CSV row identities.
Repeated occurrences of a cell are combined BEFORE propagation, so cancellation
in the shallow-composition term is retained exactly.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from code3_stats import common_support
from code4_stats import mask_contrast, _validated_sites
from source_checks import verify_primary_means

TOL = 1e-12
PRECISION = {
    '20': {'digits': 1, 'half_step': 0.05, 'reference': 'code2_ghana_primary_cd_check.csv'},
    '31': {'digits': 2, 'half_step': 0.005, 'reference': 'code2_ecuador_primary_cd_check.csv'},
}
METRICS = {
    'naive': 'naive_delta_mg_kg',
    'paired': 'paired_visible_delta_mg_kg',
    'selection': 'selection_term_mg_kg',
    'baseline': 'full_pairs_delta_mg_kg',
}


def combine(*terms):
    """Combine (rational multiplier, row-coefficient dict) pairs exactly."""
    out = {}
    for multiplier, coefficients in terms:
        multiplier = Fraction(multiplier)
        for source_row, coefficient in coefficients.items():
            row = int(source_row)
            out[row] = out.get(row, Fraction(0)) + multiplier * Fraction(coefficient)
    return {row: coefficient for row, coefficient in sorted(out.items()) if coefficient}


def average(expressions):
    expressions = list(expressions)
    if not expressions:
        raise ValueError('Cannot average an empty set of cell expressions')
    weight = Fraction(1, len(expressions))
    return combine(*[(weight, expression) for expression in expressions])


def interval_sign(lower, upper):
    """Universal strict sign over a closed box; zero contact is not robust."""
    if lower > TOL:
        return 'positive'
    if upper < -TOL:
        return 'negative'
    if lower == 0 and upper == 0:
        return 'zero'
    return 'includes_zero'


def linear_box(coefficients, cell_values, half_steps):
    """Exact affine-box formula, evaluated in floating point (mg/kg).

    The coefficient dictionary must already combine shared raw cells. Returning
    a radius for each expression does not imply independent expression errors.
    """
    keys = sorted(coefficients)
    if any(row not in cell_values or row not in half_steps for row in keys):
        raise ValueError('Coefficient refers to an unknown original source cell')
    if any(not np.isfinite(float(half_steps[row])) or float(half_steps[row]) < 0 for row in keys):
        raise ValueError('Cell half-steps must be finite and nonnegative')
    center = sum(float(coefficients[row]) * float(cell_values[row]) for row in keys)
    radius = sum(abs(float(coefficients[row])) * float(half_steps[row]) for row in keys)
    if not np.isfinite([center, radius]).all() or radius < 0:
        raise ValueError('Nonfinite or negative rounding-box radius')
    lower, upper = center - radius, center + radius
    return dict(center=center, radius=radius, lower=lower, upper=upper,
                robust_sign=interval_sign(lower, upper))


def endpoint_values(coefficients, cell_values, half_steps, endpoint):
    """One attainable rounding-box corner for this expression's min or max."""
    if endpoint not in ('lower', 'upper'):
        raise ValueError('Endpoint must be lower or upper')
    direction = -1 if endpoint == 'lower' else 1
    values = dict(cell_values)
    for row, coefficient in coefficients.items():
        values[row] = float(cell_values[row]) + direction * (1 if coefficient > 0 else -1) * float(half_steps[row])
    return values


def _source_tokens(path):
    with Path(path).open(encoding='cp1252', newline='') as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    if header.count('Soil.TotalCd') != 1:
        raise ValueError('Ambiguous original Cd source column')
    column = header.index('Soil.TotalCd')
    return {row_number: row[column] for row_number, row in enumerate(rows[1:], 2)}


def rounding_cells(data, package_dir):
    """Preserve text tokens and the external, study-specific precision assumption.

    Precision is NOT inferred from Python floats or the CSV token's trailing
    zeros. It is explicitly assigned from the source-table reporting convention.
    Tokens are checked against that convention and the 113 verified means.
    """
    package_dir = Path(package_dir)
    checked = verify_primary_means(data, package_dir / 'data')
    source_tokens = _source_tokens(package_dir / 'data' / 'wade2022_source.csv')
    refs = []
    keys = ['study_id', 'site_name', 'depth_min_cm', 'depth_max_cm']
    for study, settings in PRECISION.items():
        ref = pd.read_csv(package_dir / 'data' / settings['reference'], dtype=str, keep_default_na=False)
        ref = ref.rename(columns={'primary_cd_mg_kg': 'primary_transcription_text'})
        ref['depth_min_cm'] = pd.to_numeric(ref.depth_min_cm)
        ref['depth_max_cm'] = pd.to_numeric(ref.depth_max_cm)
        ref['assumed_reported_decimal_places'] = settings['digits']
        ref['rounding_half_step_mg_kg'] = settings['half_step']
        refs.append(ref[keys + ['primary_transcription_text', 'assumed_reported_decimal_places', 'rounding_half_step_mg_kg']])
    out = checked.merge(pd.concat(refs, ignore_index=True), on=keys, validate='one_to_one')
    series = data[['source_row', 'site_depth_series']]
    out = out.merge(series, on='source_row', validate='one_to_one')
    out = out.sort_values('source_row', kind='mergesort').reset_index(drop=True)
    out['original_source_cd_text'] = out.source_row.map(source_tokens)
    for row in out.itertuples():
        scale = Decimal(10) ** int(row.assumed_reported_decimal_places)
        try:
            original = Decimal(row.original_source_cd_text.strip())
            reference = Decimal(row.primary_transcription_text.strip())
        except (InvalidOperation, AttributeError) as exc:
            raise ValueError('Malformed original or primary concentration token') from exc
        if original != reference or original * scale != (original * scale).to_integral_value():
            raise ValueError('Value/token incompatible with source precision assumption')
    out['rounding_lower_mg_kg'] = out.total_cd_mg_kg - out.rounding_half_step_mg_kg
    out['rounding_upper_mg_kg'] = out.total_cd_mg_kg + out.rounding_half_step_mg_kg
    if out.rounding_lower_mg_kg.lt(0).any():
        raise ValueError('Unexpected negative rounding bound; symmetric-box audit requires review')
    out['precision_assumption'] = 'nearest_rounding_to_study_reported_decimal_place'
    out['precision_inferred_from_float'] = False
    out['closed_bounds_conservative_for_rounding_ties'] = True
    out['source_value_modified'] = False
    return out


def site_expressions(data, sites):
    """Reconstruct 0-30 weighted index and intact 30-50 layer from source cells."""
    result = {}
    for site in sites.itertuples():
        group = data.loc[data.site_depth_series.eq(site.site_depth_series)]
        shallow = group.loc[group.depth_min_cm.ge(0) & group.depth_max_cm.le(30)].sort_values('depth_min_cm')
        deep = group.loc[group.depth_min_cm.eq(30) & group.depth_max_cm.eq(50)]
        cursor = Fraction(0)
        weights = {}
        for row in shallow.itertuples():
            lo, hi = Fraction(str(float(row.depth_min_cm))), Fraction(str(float(row.depth_max_cm)))
            if lo != cursor or hi <= lo:
                raise ValueError('Shallow original intervals do not tile 0-30')
            weights[int(row.source_row)] = (hi - lo) / 30
            cursor = hi
        if cursor != 30 or sum(weights.values()) != 1 or len(deep) != 1:
            raise ValueError('Incomplete exact original-cell support')
        result[site.site_depth_series] = {
            'shallow': weights, 'deep': {int(deep.source_row.iloc[0]): Fraction(1)}}
    return result


def contrast_expressions(sites, observed_ids, expressions):
    all_ids = sorted(sites.site_depth_series)
    observed = sorted(set(observed_ids))
    if not observed or not set(observed).issubset(all_ids):
        raise ValueError('Need at least one known deep-observed site')
    all_shallow = average(expressions[sid]['shallow'] for sid in all_ids)
    visible_shallow = average(expressions[sid]['shallow'] for sid in observed)
    visible_deep = average(expressions[sid]['deep'] for sid in observed)
    all_deep = average(expressions[sid]['deep'] for sid in all_ids)
    terms = {
        'naive': combine((1, visible_deep), (-1, all_shallow)),
        'paired': combine((1, visible_deep), (-1, visible_shallow)),
        'selection': combine((1, visible_shallow), (-1, all_shallow)),
        'baseline': combine((1, all_deep), (-1, all_shallow)),
    }
    if combine((1, terms['naive']), (-1, terms['paired']), (-1, terms['selection'])):
        raise AssertionError('Raw-cell decomposition coefficient identity failed')
    return terms


def rounding_contrast(sites, observed_ids, expressions, cell_values, half_steps,
                      scope='mixed_20_31', mask='unnamed'):
    record = mask_contrast(sites, observed_ids, scope, mask)
    terms = contrast_expressions(sites, observed_ids, expressions)
    boxes = {}
    for metric, original_key in METRICS.items():
        box = linear_box(terms[metric], cell_values, half_steps)
        boxes[metric] = box
        if abs(box['center'] - record[original_key]) > 1e-10:
            raise AssertionError('Original-cell linear contrast disagrees with fixed site means')
        record.update({f'{metric}_lower_mg_kg': box['lower'],
                       f'{metric}_upper_mg_kg': box['upper'],
                       f'{metric}_rounding_radius_mg_kg': box['radius'],
                       f'{metric}_robust_sign': box['robust_sign']})
    ns, ps, bs = [boxes[metric]['robust_sign'] for metric in ('naive', 'paired', 'baseline')]
    opposite = lambda a, b: {a, b} == {'positive', 'negative'}
    record['robust_naive_flip_from_full'] = opposite(ns, bs)
    record['robust_naive_paired_disagree'] = opposite(ns, ps)
    record['robust_flip_while_paired_retains_baseline'] = opposite(ns, bs) and ps == bs
    # The common scale multiplies every original cell's half-reporting-unit.
    # It is a deterministic distance to possible zero contact, not an error SD.
    if record['naive_flip_while_paired_retains_baseline']:
        thresholds = {metric: abs(boxes[metric]['center']) / boxes[metric]['radius']
                      for metric in ('naive', 'paired', 'baseline') if boxes[metric]['radius'] > 0}
        limiter = min(thresholds, key=thresholds.get)
        record['critical_rounding_multiplier'] = thresholds[limiter]
        record['critical_rounding_limiter'] = limiter
    else:
        record['critical_rounding_multiplier'] = None
        record['critical_rounding_limiter'] = 'not_a_point_estimate_reversal'
    record['interval_type'] = 'conditional_rounding_box_not_confidence_interval'
    record['shared_cell_coefficients_combined'] = True
    record['coefficient_identity_exact'] = True
    return record, terms


def _records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def run_code5_analysis(data, output_dir, package_dir=None):
    """Audit every CODE4 one/two-omission mask under reported-value rounding."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    package_dir = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    cells = rounding_cells(data, package_dir)
    inventory, _ = common_support(data)
    sites = _validated_sites(inventory.loc[inventory.status.eq('complete_exact_union')])
    expressions = site_expressions(data, sites)
    cell_values = dict(zip(cells.source_row.astype(int), cells.total_cd_mg_kg.astype(float)))
    half_steps = dict(zip(cells.source_row.astype(int), cells.rounding_half_step_mg_kg.astype(float)))
    used = {row for expression in expressions.values() for coefficients in expression.values() for row in coefficients}
    cells['used_in_complete_common_support'] = cells.source_row.isin(used)
    sites['shallow_source_rows'] = [';'.join(map(str, expressions[sid]['shallow'])) for sid in sites.site_depth_series]
    sites['deep_source_rows'] = [';'.join(map(str, expressions[sid]['deep'])) for sid in sites.site_depth_series]
    scopes = {'mixed_20_31': sites,
              'Ghana_20': sites.loc[sites.study_id.astype(str).eq('20')],
              'Ecuador_31': sites.loc[sites.study_id.astype(str).eq('31')]}
    baselines, singles, doubles, scenarios, coefficients, endpoints = [], [], [], [], [], []

    def calculate(frame, observed, scope, label):
        return rounding_contrast(frame, observed, expressions, cell_values, half_steps, scope, label)

    for scope, frame in scopes.items():
        ids = sorted(frame.site_depth_series)
        base, _ = calculate(frame, ids, scope, 'all_deep_visible')
        base['hypothetical_mask'] = False
        baselines.append(base)
        lookup = frame.set_index('site_depth_series')
        for n_hidden, target in ((1, singles), (2, doubles)):
            for hidden in combinations(ids, n_hidden):
                record, _ = calculate(frame, set(ids).difference(hidden), scope, f'hide_{n_hidden}')
                record['hidden_site_ids'] = ';'.join(hidden)
                record['hidden_site_names'] = ';'.join(str(lookup.loc[sid, 'site_name']) for sid in hidden)
                record['hidden_deep_values_mg_kg'] = ';'.join(format(float(lookup.loc[sid, 'deep_cd_mg_kg']), '.12g') for sid in hidden)
                target.append(record)

    # These witnesses were selected in CODE4, before the CODE5 rounding audit.
    specifications = [('mixed_20_31', 'hide_Buako', {'Buako'}),
                      ('mixed_20_31', 'hide_Enchi', {'Enchi'}),
                      ('Ghana_20', 'hide_Buako_and_Enchi', {'Buako', 'Enchi'})]
    for scope, label, hidden_names in specifications:
        frame = scopes[scope]
        hidden = frame.loc[frame.site_name.isin(hidden_names)]
        if set(hidden.site_name) != hidden_names:
            raise ValueError('Predefined CODE4 witness site missing')
        observed = set(frame.site_depth_series).difference(hidden.site_depth_series)
        record, terms = calculate(frame, observed, scope, label)
        record['hidden_site_ids'] = ';'.join(sorted(hidden.site_depth_series))
        record['hidden_site_names'] = ';'.join(sorted(hidden_names))
        scenarios.append(record)
        for metric, term in terms.items():
            for source_row in sorted(cell_values):
                coefficient = term.get(source_row, Fraction(0))
                coefficients.append(dict(scope=scope, mask=label, metric=metric, source_row=source_row,
                    coefficient_numerator=coefficient.numerator, coefficient_denominator=coefficient.denominator,
                    coefficient=float(coefficient), original_cd_mg_kg=cell_values[source_row],
                    half_step_mg_kg=half_steps[source_row],
                    lower_endpoint_cell_mg_kg=cell_values[source_row] - (1 if coefficient > 0 else -1 if coefficient < 0 else 0) * half_steps[source_row],
                    upper_endpoint_cell_mg_kg=cell_values[source_row] + (1 if coefficient > 0 else -1 if coefficient < 0 else 0) * half_steps[source_row]))
            for endpoint in ('lower', 'upper'):
                corner = endpoint_values(term, cell_values, half_steps, endpoint)
                evaluated = {name: sum(float(coef) * corner[row] for row, coef in expr.items())
                             for name, expr in terms.items()}
                target = record[f'{metric}_{endpoint}_mg_kg']
                error = evaluated[metric] - target
                identity = evaluated['naive'] - evaluated['paired'] - evaluated['selection']
                if abs(error) > 1e-10 or abs(identity) > 1e-10:
                    raise AssertionError('Constructive rounding endpoint check failed')
                endpoints.append(dict(scope=scope, mask=label, optimized_metric=metric, endpoint=endpoint,
                    target_bound_mg_kg=target, attained_value_mg_kg=evaluated[metric],
                    endpoint_error_mg_kg=error, identity_error_mg_kg=identity,
                    naive_at_corner_mg_kg=evaluated['naive'], paired_at_corner_mg_kg=evaluated['paired'],
                    selection_at_corner_mg_kg=evaluated['selection'], baseline_at_corner_mg_kg=evaluated['baseline'],
                    separate_metric_extrema_use_separate_corners=True))
    for study, country in [('20', 'Ghana'), ('31', 'Ecuador')]:
        observed = sites.loc[sites.study_id.astype(str).eq(study), 'site_depth_series']
        record, _ = calculate(sites, observed, 'mixed_20_31', country + '_deep_only')
        record['hidden_site_ids'] = ';'.join(sorted(set(sites.site_depth_series).difference(observed)))
        record['hidden_site_names'] = ';'.join(sorted(sites.loc[~sites.site_depth_series.isin(observed), 'site_name']))
        scenarios.append(record)

    baselines = pd.DataFrame(baselines)
    singles, doubles = pd.DataFrame(singles), pd.DataFrame(doubles)
    scenarios = pd.DataFrame(scenarios)
    coefficients, endpoints = pd.DataFrame(coefficients), pd.DataFrame(endpoints)
    scope_rows = []
    for scope, frame in scopes.items():
        base = baselines.loc[baselines.scope.eq(scope)].iloc[0]
        row = dict(scope=scope, n_sites=len(frame), baseline_delta_mg_kg=float(base.full_pairs_delta_mg_kg),
                   baseline_lower_mg_kg=float(base.baseline_lower_mg_kg),
                   baseline_upper_mg_kg=float(base.baseline_upper_mg_kg),
                   baseline_robust_sign=base.baseline_robust_sign)
        for label, all_masks in [('single', singles), ('double', doubles)]:
            masks = all_masks.loc[all_masks.scope.eq(scope)]
            row.update({f'{label}_masks': len(masks),
                        f'{label}_point_naive_flips': int(masks.naive_flip_from_full.sum()),
                        f'{label}_rounding_robust_naive_flips': int(masks.robust_naive_flip_from_full.sum()),
                        f'{label}_point_flip_with_paired_baseline': int(masks.naive_flip_while_paired_retains_baseline.sum()),
                        f'{label}_rounding_robust_flip_with_paired_baseline': int(masks.robust_flip_while_paired_retains_baseline.sum())})
        scope_rows.append(row)
    scope_summary = pd.DataFrame(scope_rows)
    outputs = {'code5_primary_rounding_cells.csv': cells,
               'code5_common_support_inventory.csv': inventory,
               'code5_site_values.csv': sites,
               'code5_rounding_baselines.csv': baselines,
               'code5_rounding_scenarios.csv': scenarios,
               'code5_rounding_single_omissions.csv': singles,
               'code5_rounding_double_omissions.csv': doubles,
               'code5_rounding_scope_summary.csv': scope_summary,
               'code5_witness_rounding_coefficients.csv': coefficients,
               'code5_witness_rounding_endpoints.csv': endpoints}
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    report = dict(analysis='conditional_reported_precision_audit_of_CODE4_masks', deterministic=True,
        monte_carlo_performed=False, n_primary_means_checked=len(cells),
        n_original_cells_in_common_support=int(cells.used_in_complete_common_support.sum()),
        n_candidate_sites=len(inventory), n_complete_sites=len(sites), n_excluded_incomplete_sites=len(inventory)-len(sites),
        single_masks=len(singles), double_masks=len(doubles), total_enumerated_masks=len(singles)+len(doubles),
        rounding_assumptions={'20': {'reported_decimal_places': 1, 'half_step_mg_kg': 0.05},
                              '31': {'reported_decimal_places': 2, 'half_step_mg_kg': 0.005}},
        precision_assigned_from_study_reporting_convention_not_float=True,
        coefficients_rational_and_combined_by_original_source_row=True,
        scope_results=_records(scope_summary), baselines=_records(baselines), named_scenarios=_records(scenarios),
        mixed_single_robust_witnesses=_records(singles.loc[singles.scope.eq('mixed_20_31') & singles.robust_flip_while_paired_retains_baseline]),
        all_witness_endpoints_verified=True,
        max_endpoint_error_mg_kg=float(endpoints.endpoint_error_mg_kg.abs().max()),
        max_endpoint_identity_error_mg_kg=float(endpoints.identity_error_mg_kg.abs().max()),
        outputs=list(outputs),
        interpretation='Conditional rounding robustness for this finite table, not independent replication or population inference.',
        caveats=[
            'Nearest-rounding boxes are conditional on the specified source-table reporting precision; raw pre-rounded means are unavailable.',
            'These bounds include only reporting-rounding uncertainty, not laboratory error, sampling error, transcription error, or missing sites.',
            'A robust sign is a universal statement over this rounding box, not a confidence level or probability.',
            'Rounding multipliers scale reporting half-units together; they are neither measurement-error SDs nor estimated experimental tolerances.',
            'Extrema of different expressions may require different original-cell corners; never add independently minimized expression bounds.',
            'All one/two-omission masks are exhaustive finite-table descriptions; their counts are not independent replications or missingness probabilities.',
            'Mixed study, country, site selection and assay are confounded; the mixed baseline is only an arithmetic reference.',
            'The 0-30 index is depth weighted, not cadmium stock or a verified mass-weighted concentration.',
            'Paired-visible contrasts target the retained sites and do not recover a population effect or missing outcomes.',
            'CODE4 witnesses predate this audit, but the project and CODE5 are exploratory rather than preregistered.',
        ])
    json.dumps(report, allow_nan=False)
    return report
