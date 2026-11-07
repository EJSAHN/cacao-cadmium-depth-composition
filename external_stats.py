"""Standalone published-value observation-mask audit of Guarin et al. (2023).

Nominal 5 and 20 cm farm means are not the 0-30/30-50 cm estimand in the main
analysis. No data are pooled. Literal-token rounding is an explicit sensitivity
scenario, NOT verified reporting precision: trailing zeros may be suppressed.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path

SOURCE_FILE = 'data/external/guarin2023_published_means.csv'
SOURCE_SHA256 = 'de17b3ff7cf89e10d255d7f629b765a3f3c62ecd5a1879a0c8835bec3afcf82e'
ASSAYS = ('EPA3050B', 'Mehlich3')
FARMS = ('F1', 'F2', 'F3', 'F4', 'F5')
ROUNDING_MODEL = 'conditional_literal_token_half_unit_not_verified_reporting_precision'


def combine(*terms):
    result = {}
    for factor, expression in terms:
        for key, value in expression.items():
            result[key] = result.get(key, Fraction(0)) + Fraction(factor) * value
    return {key: value for key, value in sorted(result.items()) if value}


def _mean(expressions):
    expressions = list(expressions)
    if not expressions:
        raise ValueError('Cannot average an empty selection')
    return combine(*[(Fraction(1, len(expressions)), x) for x in expressions])


def expressions_for(farms, observed):
    """Combine shared original cells before taking each interval radius."""
    farms = tuple(sorted(farms)); observed = tuple(sorted(observed))
    if len(set(farms)) != len(farms) or len(set(observed)) != len(observed):
        raise ValueError('Farm identities must be unique')
    if not observed or not set(observed).issubset(farms):
        raise ValueError('Need one or more known retained farms')
    s_all = _mean({(farm, 5): Fraction(1)} for farm in farms)
    s_obs = _mean({(farm, 5): Fraction(1)} for farm in observed)
    d_all = _mean({(farm, 20): Fraction(1)} for farm in farms)
    d_obs = _mean({(farm, 20): Fraction(1)} for farm in observed)
    expressions = {
        'naive': combine((1, d_obs), (-1, s_all)),
        'paired': combine((1, d_obs), (-1, s_obs)),
        'selection': combine((1, s_obs), (-1, s_all)),
        'baseline': combine((1, d_all), (-1, s_all)),
    }
    if combine((1, expressions['naive']), (-1, expressions['paired']), (-1, expressions['selection'])):
        raise AssertionError('Shared-cell coefficient identity failed')
    return expressions


def literal_half_step(token):
    """Stress scenario based literally on text, not a claim about lab precision."""
    try:
        value = Decimal(str(token))
    except InvalidOperation as exc:
        raise ValueError('Invalid printed numeric token') from exc
    if not value.is_finite():
        raise ValueError('Printed numeric token must be finite')
    return Fraction(Decimal(10) ** value.as_tuple().exponent) / 2


def box(expression, values, half_steps):
    center = sum((coefficient * values[key] for key, coefficient in expression.items()), Fraction(0))
    radius = sum((abs(coefficient) * half_steps[key] for key, coefficient in expression.items()), Fraction(0))
    if radius < 0:
        raise ValueError('Negative interval radius')
    lower, upper = center-radius, center+radius
    return {'center': center, 'radius': radius, 'lower': lower, 'upper': upper,
            'sign': 'positive' if lower > 0 else 'negative' if upper < 0 else 'includes_zero'}


def _sign(value):
    return 'positive' if value > 0 else 'negative' if value < 0 else 'zero'


def load_source(package_dir):
    source = Path(package_dir) / SOURCE_FILE
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('External source transcription checksum mismatch')
    with source.open(encoding='utf-8', newline='') as handle:
        records = list(csv.DictReader(handle))
    required = {'study_key', 'country', 'region', 'farm_id', 'depth_cm', 'depth_support',
                'assay', 'cd_mean_printed', 'se_printed', 'cd_unit', 'source_table',
                'source_page', 'source_doi', 'verification', 'rounding_precision_status'}
    if len(records) != 20 or not required.issubset(records[0]):
        raise ValueError('Expected 20 transcribed assay/farm/depth records')
    expected = {(assay, farm, depth) for assay in ASSAYS for farm in FARMS for depth in (5, 20)}
    seen = set()
    for row in records:
        key = (row['assay'], row['farm_id'], int(row['depth_cm']))
        if key in seen or key not in expected:
            raise ValueError('Unexpected or duplicate external assay/farm/depth key')
        seen.add(key)
        if (row['study_key'] != 'guarin2023' or row['cd_unit'] != 'mg/kg'
                or row['source_doi'] != '10.1016/j.envc.2023.100765'
                or row['depth_support'] != 'nominal_sample_depth_not_whole_layer'):
            raise ValueError('External source identity, unit or depth support mismatch')
        value = Decimal(row['cd_mean_printed'])
        if not value.is_finite() or value < 0:
            raise ValueError('External Cd means must be finite and nonnegative')
    if seen != expected:
        raise ValueError('Incomplete paired external source')
    return sorted(records, key=lambda r: (r['assay'], r['farm_id'], int(r['depth_cm'])))


def contrast_record(assay, observed, values, half_steps, n_hidden=None):
    observed = tuple(sorted(observed)); hidden = tuple(f for f in FARMS if f not in observed)
    expressions = expressions_for(FARMS, observed)
    bounds = {name: box(expression, values, half_steps) for name, expression in expressions.items()}
    if bounds['naive']['center'] != bounds['paired']['center'] + bounds['selection']['center']:
        raise AssertionError('Exact external decomposition identity failed')
    point_signs = {name: _sign(result['center']) for name, result in bounds.items()}
    opposite = lambda a, b: {a, b} == {'positive', 'negative'}
    out = {'study_key': 'guarin2023', 'assay': assay, 'n_farms': 5,
           'n_deep_hidden': len(hidden), 'n_deep_observed': len(observed),
           'hidden_farms': ';'.join(hidden), 'observed_farms': ';'.join(observed),
           'mask': 'all_deep_visible' if not hidden else 'hide_'+str(len(hidden)),
           'hypothetical_mask': bool(hidden),
           'shallow_nominal_depth_cm': 5, 'deep_nominal_depth_cm': 20,
           'same_depth_estimand_as_main': False,
           'rounding_model': ROUNDING_MODEL, 'rounding_precision_verified': False}
    point_keys = {'naive': 'naive_delta_mg_kg', 'paired': 'paired_visible_delta_mg_kg',
                  'selection': 'selection_term_mg_kg', 'baseline': 'full_pairs_delta_mg_kg'}
    for name, result in bounds.items():
        out[point_keys[name]] = float(result['center'])
        out[name+'_reported_sign'] = point_signs[name]
        out[name+'_literal_lower_mg_kg'] = float(result['lower'])
        out[name+'_literal_upper_mg_kg'] = float(result['upper'])
        out[name+'_literal_radius_mg_kg'] = float(result['radius'])
        out[name+'_literal_robust_sign'] = result['sign']
    out['identity_error_mg_kg'] = 0.0
    out['naive_paired_disagree_reported'] = opposite(point_signs['naive'], point_signs['paired'])
    out['flip_while_paired_retains_baseline_reported'] = (
        opposite(point_signs['naive'], point_signs['baseline']) and point_signs['paired']==point_signs['baseline'])
    out['robust_flip_under_literal_token_scenario'] = (
        opposite(bounds['naive']['sign'], bounds['baseline']['sign']) and bounds['paired']['sign']==bounds['baseline']['sign'])
    out['all_farms_decline_at_reported_values'] = all(values[(f,20)] < values[(f,5)] for f in FARMS)
    return out, expressions


def _save(path, rows):
    if not rows:
        raise ValueError('Empty external output')
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_external_analysis(package_dir, output_dir):
    """Compute a standalone farm-summary demonstration; no pooled estimand."""
    source = load_source(package_dir)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    sites, masks, scope, coefficients = [], [], [], []
    witness = None
    for assay in ASSAYS:
        rows = [r for r in source if r['assay']==assay]
        lookup = {(r['farm_id'], int(r['depth_cm'])): r for r in rows}
        values = {key: Fraction(Decimal(r['cd_mean_printed'])) for key,r in lookup.items()}
        halves = {key: literal_half_step(r['cd_mean_printed']) for key,r in lookup.items()}
        for farm in FARMS:
            shallow, deep = lookup[(farm,5)], lookup[(farm,20)]
            sites.append({'study_key': 'guarin2023', 'assay': assay, 'farm_id': farm,
                'shallow_nominal_depth_cm': 5, 'deep_nominal_depth_cm': 20,
                'shallow_cd_printed': shallow['cd_mean_printed'], 'deep_cd_printed': deep['cd_mean_printed'],
                'shallow_cd_mg_kg': float(values[(farm,5)]), 'deep_cd_mg_kg': float(values[(farm,20)]),
                'delta_mg_kg': float(values[(farm,20)]-values[(farm,5)]),
                'declines_at_reported_values': values[(farm,20)]<values[(farm,5)],
                'shallow_literal_half_step_mg_kg': float(halves[(farm,5)]),
                'deep_literal_half_step_mg_kg': float(halves[(farm,20)]),
                'rounding_precision_verified': False, 'rounding_model': ROUNDING_MODEL,
                'source_table': 1, 'source_page': 4,
                'depth_support': 'nominal_sample_depth_not_whole_layer',
                'source_doi': '10.1016/j.envc.2023.100765'})
        assay_masks = []
        for count in range(5):
            for hidden in combinations(FARMS, count):
                observed = tuple(f for f in FARMS if f not in hidden)
                record, expressions = contrast_record(assay, observed, values, halves)
                assay_masks.append(record)
                if assay=='EPA3050B' and hidden==('F2','F4'):
                    witness = dict(record)
                    for metric, expression in expressions.items():
                        for key, coefficient in expression.items():
                            coefficients.append({'assay': assay, 'hidden_farms': 'F2;F4', 'metric': metric,
                                'farm_id': key[0], 'depth_cm': key[1],
                                'coefficient_numerator': coefficient.numerator,
                                'coefficient_denominator': coefficient.denominator,
                                'coefficient': float(coefficient),
                                'cd_mean_printed': lookup[key]['cd_mean_printed'],
                                'literal_half_step_mg_kg': float(halves[key]),
                                'precision_verified': False})
        masks.extend(assay_masks)
        nonbaseline = [r for r in assay_masks if r['n_deep_hidden']]
        scope.append({'study_key': 'guarin2023','assay': assay, 'n_farms': 5,
            'baseline_delta_mg_kg': assay_masks[0]['full_pairs_delta_mg_kg'],
            'all_farms_decline_at_reported_values': assay_masks[0]['all_farms_decline_at_reported_values'],
            'single_masks': 5, 'double_masks': 10, 'triple_masks': 10, 'quadruple_masks': 5,
            'all_nonempty_retained_masks': 31,
            'all_opposite_sign_masks_reported': sum(r['naive_paired_disagree_reported'] for r in nonbaseline),
            'minimum_hidden_for_opposite_signs_reported': min((r['n_deep_hidden'] for r in nonbaseline if r['naive_paired_disagree_reported']),default=None),
            'single_opposite_sign_masks_reported': sum(r['naive_paired_disagree_reported'] for r in nonbaseline if r['n_deep_hidden']==1),
            'double_opposite_sign_masks_reported': sum(r['naive_paired_disagree_reported'] for r in nonbaseline if r['n_deep_hidden']==2),
            'robust_reversal_masks_under_literal_token_scenario': sum(r['robust_flip_under_literal_token_scenario'] for r in nonbaseline),
            'precision_verified': False, 'same_depth_estimand_as_main': False})
    outputs = {'code5_external_site_values.csv': sites, 'code5_external_mask_contrasts.csv': masks,
               'code5_external_scope_summary.csv': scope,
               'code5_external_witness_coefficients.csv': coefficients}
    for filename, rows in outputs.items():
        _save(output_dir/filename, rows)
    report = {'status': 'STANDALONE_PUBLISHED_MEAN_COUNTEREXAMPLE_COMPUTED',
        'study_key': 'guarin2023', 'source_doi': '10.1016/j.envc.2023.100765',
        'source_file': SOURCE_FILE, 'source_sha256': SOURCE_SHA256,
        'source_mean_count': len(source), 'farm_count': 5, 'assay_count': 2,
        'masks_including_baselines': len(masks), 'scope_results': scope,
        'epa_hide_f2_f4_witness': witness,
        'pooled_with_main_analysis': False, 'same_depth_estimand_as_main': False,
        'measurement_precision_or_sampling_inference_performed': False,
        'statistical_validation_of_original_effect': False,
        'rounding_model': ROUNDING_MODEL,
        'literal_token_precision_verified': False,
        'outputs': list(outputs),
        'interpretation': 'Separate published-table arithmetic counterexample at nominal5 and20cm, not replication of the main depth-window effect or a population-level finding.',
        'limitations': [
            'Five farm summaries, not50independent farms; within-farm subsamples and assay methods are not independent studies.',
            'EPA3050B andMehlich3 quantify different operational Cd fractions; no cross-assay pooling.',
            'Mixed printed decimals (EPA shallowF2=0.5 andF5=1) leave trailing-zero convention unresolved.',
            'Literal-token intervals are an explicit stress scenario, not verified reporting precision, measurement error, SE or confidence intervals.',
            'Reported-value all-declining and reversal statements refer only to the published means.',
            'Masks are constructed, not observed historical missingness; enumeration counts are not probabilities or independent tests.',
            'Nominal sample depths5 and20cm do not form the main analysis whole-layer0-30/30-50cm comparison.',
            'External published-value example was identified duringCODE5; exploratory, not prespecified validation.'
        ]}
    json.dumps(report, allow_nan=False)
    return report
