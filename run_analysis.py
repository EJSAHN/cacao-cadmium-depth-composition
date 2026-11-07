#!/usr/bin/env python3
"""Recompute the fixed-table cacao cadmium diagnostics without plotting."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
SOURCE_HASH = '09f328a87063373ccf0d15f5159a7b12775dbf8623233b35a74d734ab51a365a'
SOURCE_URL = ('https://raw.githubusercontent.com/jordon-wade/Cacao-cadmium-meta-analysis/'
              'b16cd1def36732550f0c25182e82a71260cc3596/Extracted%20and%20finalized%20data.csv')
FIXTURE_INPUT_HASH = 'a62ad073a4ab3973ce86ecb7de8ad92082332f200b36a0155bfb46bbf52af6a9'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def prepare_inputs(source, out):
    """Copy public fixtures and construct the omitted source-derived R input."""
    from depth_core import read_source, profile_table
    if digest(source) != SOURCE_HASH:
        raise ValueError('Source SHA-256 mismatch; the input has not been analysed')
    package = out / 'inputs'
    shutil.copytree(ROOT/'data', package/'data')
    shutil.copytree(ROOT/'reproduction', package/'reproduction')
    shutil.copyfile(source, package/'data/wade2022_source.csv')
    raw, data, audit = read_source(source)
    fixture = data.copy()
    fixture['lead_author'] = [raw.iloc[int(i)-2]['LeadAuthor.primary'].strip() for i in data.source_row]
    eligible = profile_table(data)
    allowed = set(eligible.loc[eligible.eligible_primary, 'site_depth_series'])
    fixture['permutable'] = data.site_depth_series.isin(allowed)
    path = package/'data/r_reference/r_input.csv'
    fixture.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')
    if digest(path) != FIXTURE_INPUT_HASH:
        raise ValueError('Regenerated R input differs from the archived fixture')
    return package, data, audit


def compare_table(actual_path, reference_path):
    """Compare CSV values and identities, tolerating only numeric roundoff."""
    import numpy as np
    import pandas as pd
    actual = pd.read_csv(actual_path, dtype=str, keep_default_na=False)
    reference = pd.read_csv(reference_path, dtype=str, keep_default_na=False)
    if actual.shape != reference.shape or list(actual.columns) != list(reference.columns):
        raise ValueError('Table shape or columns differ: '+Path(reference_path).name)
    largest = 0.0
    for col in actual:
        a, b = actual[col], reference[col]
        if a.equals(b):
            continue
        av, bv = pd.to_numeric(a, errors='coerce'), pd.to_numeric(b, errors='coerce')
        nonnumeric = av.isna() | bv.isna()
        if not a[nonnumeric].equals(b[nonnumeric]):
            raise ValueError('Nonnumeric identity differs: '+str(reference_path)+' / '+col)
        valid = ~nonnumeric
        if not np.allclose(av[valid], bv[valid], rtol=1e-11, atol=1e-10):
            raise ValueError('Numeric value differs: '+str(reference_path)+' / '+col)
        if valid.any():
            largest = max(largest, float(np.abs(av[valid]-bv[valid]).max()))
    return {'table': Path(reference_path).name, 'rows': len(actual),
            'columns': len(actual.columns), 'max_abs_difference': largest}


def run(source, out):
    import numpy as np
    import pandas as pd
    from code3_stats import run_code3_analysis
    from code4_stats import run_code4_analysis
    from code5_stats import run_code5_analysis
    from external_stats import run_external_analysis
    from r_reference_checks import verify_r_reference
    from source_checks import verify_primary_means
    from summary_tables import make_public_tables
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'status.json', {'status':'RUNNING'})
    try:
        inputs, data, audit = prepare_inputs(source, out)
        tables = out/'tables'; tables.mkdir()
        write_json(out/'input_audit.json', audit)
        data.to_csv(tables/'input_total_cd_records.csv', index=False, encoding='utf-8-sig')
        checked = verify_primary_means(data, inputs/'data')
        checked.to_csv(tables/'primary_means_113_comparison.csv', index=False, encoding='utf-8-sig')
        write_json(out/'smoother_summary.json', run_code3_analysis(data, tables))
        write_json(out/'r_reference_summary.json', verify_r_reference(data, inputs, tables))
        write_json(out/'mask_summary.json', run_code4_analysis(data, tables))
        write_json(out/'rounding_summary.json', run_code5_analysis(data, tables, inputs))
        write_json(out/'external_summary.json', run_external_analysis(inputs, tables))
        make_public_tables(data, audit, out)
        checks = [compare_table(tables/p.name, p) for p in sorted((ROOT/'reference').glob('*.csv'))]
        write_json(out/'reference_comparison.json', {'status':'MATCH', 'comparisons':checks,
                   'rtol':1e-11, 'atol':1e-10})
        write_json(out/'environment.json', {'python':platform.python_version(),
                   'platform':platform.platform(), 'numpy':np.__version__, 'pandas':pd.__version__})
        write_json(out/'provenance.json', {'source_sha256':SOURCE_HASH,
                   'source_url':SOURCE_URL, 'generated_r_input_sha256':FIXTURE_INPUT_HASH,
                   'code_sha256':{p.name:digest(p) for p in sorted(ROOT.glob('*.py'))},
                   'r_executed':False, 'scope':'deterministic numerical diagnostics only',
                   'source_version_discrepancy_resolved':False})
        write_json(out/'status.json', {'status':'COMPLETE', 'tables':len(list(tables.glob('*.csv'))),
                   'elapsed_seconds':time.perf_counter()-started})
    except Exception as exc:
        write_json(out/'status.json', {'status':'FAILED', 'error':str(exc)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', type=Path, help='Local unmodified source CSV; no network use')
    group.add_argument('--download-source', action='store_true', help='Download pinned CSV from its original repository')
    parser.add_argument('--output', type=Path, help='New output directory; an existing directory is refused')
    args = parser.parse_args()
    out = args.output or Path('outputs')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    if out.exists():
        parser.error('Output already exists; choose a new directory')
    if args.download_source:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='cacao_cd_source_') as temp:
            source = Path(temp)/'source.csv'
            request = urllib.request.Request(SOURCE_URL, headers={'User-Agent':'cacao-depth-analysis/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response:
                source.write_bytes(response.read())
            run(source, out)
    else:
        run(args.source, out)
    print('COMPLETE: '+str(out.resolve()))
    print('Numerical tables only. R references are verified, not rerun.')


if __name__ == '__main__':
    main()
