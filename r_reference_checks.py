"""Verify executed base-R references, compare without claiming equivalence.

Windows users need only the existing Python stack. The archived R run is a
reference calculation; this function never says R was executed on their PC.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from depth_models import smooth_decomposition


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_r_reference(data, package_dir, output_dir):
    package_dir, output_dir = Path(package_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = package_dir / 'data' / 'r_reference'
    manifest = json.loads((base / 'reference_manifest.json').read_text(encoding='utf-8'))
    if _hash(package_dir / 'data' / 'wade2022_source.csv') != manifest['source_sha256']:
        raise ValueError('R reference source SHA-256 mismatch')
    for name, expected in manifest['file_sha256'].items():
        if _hash(package_dir / name) != expected:
            raise ValueError('R reference file SHA-256 mismatch: '+name)
    fixture = pd.read_csv(base/'r_input.csv', dtype={'study_id':str,'site_depth_series':str})
    actual = data.sort_values('source_row').reset_index(drop=True)
    fixture = fixture.sort_values('source_row').reset_index(drop=True)
    if len(actual) != len(fixture):
        raise ValueError('R input row count mismatch')
    for col in ('source_row','study_id','site_depth_series'):
        if not np.array_equal(actual[col].astype(str),fixture[col].astype(str)):
            raise ValueError('R input identity mismatch: '+col)
    for col in ('depth_min_cm','depth_max_cm','total_cd_mg_kg'):
        if not np.allclose(actual[col],fixture[col],rtol=0,atol=1e-12,equal_nan=False):
            raise ValueError('R input value mismatch: '+col)
    subsets = {
        'all_rows_upper':fixture,
        'no_arguello_upper':fixture.loc[fixture.lead_author.ne('David Arguello')],
        'conservative_omit_17_27':fixture.loc[~fixture.study_id.isin(['17','27'])],
        'audited_farm_composites_20_31':fixture.loc[fixture.study_id.isin(['20','31'])],
    }
    reference = pd.read_csv(base/'r_loess_reference.csv')
    comparisons, contrasts = [], []
    for (scenario,response), rows in reference.groupby(['scenario','response'],sort=True):
        frame = subsets[scenario].sort_values(['site_depth_series','depth_max_cm']).copy()
        if response == 'flattened':
            frame['total_cd_mg_kg'] = frame.groupby('site_depth_series').total_cd_mg_kg.transform('mean')
        elif response in ('forced_nonincreasing','forced_nondecreasing'):
            if scenario != 'audited_farm_composites_20_31':
                raise ValueError('Unexpected forced-profile reference scope')
            for _, g in frame.groupby('site_depth_series'):
                values = np.sort(g.total_cd_mg_kg.to_numpy())
                if response == 'forced_nonincreasing': values = values[::-1]
                frame.loc[g.index,'total_cd_mg_kg'] = values
        elif response != 'observed':
            raise ValueError('Unrecognized R response')
        rows = rows.sort_values('coordinate_cm').copy()
        fit = smooth_decomposition(frame,axis='upper',span=.75,weighting='rows',grid=rows.coordinate_cm)
        rows['python_fit'] = fit.observed.to_numpy()
        rows['python_minus_r_interpolate'] = rows.python_fit - rows.r_reversed_x
        rows['python_minus_r_direct'] = rows.python_fit - rows.r_direct_reversed_x
        comparisons.append(rows)
        idx = rows.set_index('coordinate_cm')
        for a,b in ((20.,50.),(30.,50.),(30.,80.),(50.,80.)):
            rec = dict(scenario=scenario,response=response,left_coordinate_cm=a,right_coordinate_cm=b)
            for col in ('r_positive_x','r_reversed_x','r_direct_positive_x','r_direct_reversed_x','python_fit'):
                rec[col+'_change'] = float(idx.loc[b,col]-idx.loc[a,col])
            contrasts.append(rec)
    comparisons = pd.concat(comparisons,ignore_index=True)
    contrasts = pd.DataFrame(contrasts)
    comparisons.to_csv(output_dir/'code3_r_comparison.csv',index=False)
    contrasts.to_csv(output_dir/'code3_r_contrasts.csv',index=False)
    focal = contrasts.loc[contrasts.left_coordinate_cm.eq(50)&contrasts.right_coordinate_cm.eq(80)]
    warning_table = pd.read_csv(base/'r_warnings.csv')
    warning_table.to_csv(output_dir/'code3_r_warnings.csv',index=False)
    report = {
        'status':'REFERENCE_HASH_AND_INPUT_VERIFIED',
        'r_executed_during_this_windows_run':False,
        'r_reference_version':manifest['r_version'],
        'reference_input_rows':len(fixture),'reference_prediction_rows':len(reference),
        'r_warning_rows':len(warning_table),
        'reproduction_scope':manifest['reproduction_scope'],
        'max_abs_python_minus_r_interpolate':float(comparisons.python_minus_r_interpolate.abs().max()),
        'max_abs_python_minus_r_direct':float(comparisons.python_minus_r_direct.abs().max()),
        'max_abs_r_interpolation_coordinate_reversal_difference':float(comparisons.reversal_difference.abs().max()),
        'focal_50_to_80':[
            {'scenario':r.scenario,'response':r.response,
             'r_interpolated_reversed':r.r_reversed_x_change,
             'r_direct_reversed':r.r_direct_reversed_x_change,
             'python_change':r.python_fit_change} for r in focal.itertuples()],
        'interpretation':'Actual R reference calculations, not bitwise reproduction of the published figure or a causal test. Numerical differences are reported, not treated as validation failures.',
    }
    (output_dir.parent/'r_reference_status.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report
