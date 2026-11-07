"""Build manuscript-facing summaries from computed tables, not pasted numbers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from io_support import write_json


def make_public_tables(data, audit, run):
    run = Path(run)
    tables = run / 'tables'
    def read(name):
        return pd.read_csv(tables / name)
    def save(name, frame):
        frame.to_csv(tables / name, index=False)
    sites = read('code5_site_values.csv')
    scopes = read('code5_rounding_scope_summary.csv')
    scenarios = read('code5_rounding_scenarios.csv')
    external = read('code5_external_mask_contrasts.csv')
    ext_sites = read('code5_external_site_values.csv')
    r = read('code3_r_contrasts.csv')
    rc = read('code3_r_comparison.csv')
    r_focal = r.loc[r.left_coordinate_cm.eq(50) & r.right_coordinate_cm.eq(80)].copy()
    support = data.groupby(['study_id', 'country', 'depth_min_cm', 'depth_max_cm'], dropna=False).agg(
        n_rows=('source_row', 'size'), n_series=('site_depth_series','nunique')).reset_index()
    support['primary_means_checked'] = support.study_id.astype(str).isin(['20','31'])
    save('code6_figure1_interval_support.csv', support)
    save('code6_figure1_r_curves.csv', rc)
    save('code6_figure1_r_focal_contrasts.csv', r_focal)
    save('code6_figure2_sites.csv', sites)
    witnesses = scenarios.loc[scenarios['mask'].isin(['hide_Buako','hide_Enchi','hide_Buako_and_Enchi'])].copy()
    save('code6_figure2_witnesses.csv', witnesses)
    epa = ext_sites.loc[ext_sites.assay.eq('EPA3050B')].copy()
    ew = external.loc[external.assay.eq('EPA3050B') & external.hidden_farms.eq('F2;F4')].iloc[0]
    save('code6_figure3_epa_site_values.csv', epa)
    save('code6_figure3_epa_witness.csv', ew.to_frame().T)
    deep = data.loc[data.depth_min_cm.ge(50)]
    coverage = pd.DataFrame([
        ['Public source snapshot',audit['source_rows'],31,11,'Raw rows, not necessarily independent observations'],
        ['Numeric Cd and valid depth',len(data),data.study_id.nunique(),data.country.nunique(),'Source row eligibility; zeros retained'],
        ['Primary-checked Ghana/Ecuador means',113,2,2,'27 candidate site series; composite means, not raw cores'],
        ['Complete common-support source cells',97,2,2,'26 sites; 7 Ghana + 19 Ecuador'],
        ['Whole intervals starting at >=50 cm',len(deep),deep.study_id.nunique(),deep.country.nunique(),'14 rows from 7 Ghana sites in this snapshot only'],
    ], columns=['scope','n_source_rows_or_cells','n_studies','n_countries','interpretation'])
    # Enforce the frozen, audited scope. A changed source requires a new review.
    if len(sites)!=26 or len(deep)!=14 or set(deep.study_id.astype(str))!={'20'}:
        raise ValueError('Evidence scope differs from the primary-audited snapshot')
    save('code6_table1_evidence_scope.csv', coverage)
    save('code6_table2_mask_enumeration.csv', scopes)
    cols=['scope','mask','n_shallow','n_deep_observed','hidden_site_names','naive_delta_mg_kg',
          'naive_lower_mg_kg','naive_upper_mg_kg','paired_visible_delta_mg_kg','paired_lower_mg_kg',
          'paired_upper_mg_kg','selection_term_mg_kg','robust_flip_while_paired_retains_baseline']
    save('code6_table3_selected_witnesses.csv', witnesses[cols])
    save('code6_table4_external_example.csv', external.loc[external.assay.eq('EPA3050B') &
        (external['mask'].eq('all_deep_visible')|external.hidden_farms.eq('F2;F4'))])
    return {'scientific_tables': len(list(tables.glob('*.csv')))}
