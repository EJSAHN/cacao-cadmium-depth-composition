"""Compare pinned extracted values to separately transcribed primary means."""
from pathlib import Path
import pandas as pd


def verify_primary_means(data, reference_directory):
    keys=['study_id','site_name','depth_min_cm','depth_max_cm']
    refs=[]
    for name in ['code2_ghana_primary_cd_check.csv','code2_ecuador_primary_cd_check.csv']:
        ref=pd.read_csv(Path(reference_directory)/name,dtype={'study_id':str,'site_name':str})
        ref['reference_file']=name
        refs.append(ref)
    reference=pd.concat(refs,ignore_index=True)
    if reference.duplicated(keys).any():
        raise ValueError('Duplicate primary transcription key')
    selected=data[data.study_id.astype(str).isin(['20','31'])].copy()
    selected['study_id']=selected.study_id.astype(str)
    merged=reference.merge(selected[keys+['total_cd_mg_kg','source_row']],on=keys,
                           how='outer',validate='one_to_one',indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError('Primary transcription does not match all selected source keys')
    merged['difference_mg_kg']=merged.total_cd_mg_kg-merged.primary_cd_mg_kg
    if merged.difference_mg_kg.isna().any() or (merged.difference_mg_kg.abs()>1e-12).any():
        raise ValueError('Primary transcription value mismatch; no correction applied')
    if len(merged)!=113 or merged.groupby('study_id').size().to_dict()!={'20':37,'31':76}:
        raise ValueError('Unexpected primary transcription coverage')
    return merged.drop(columns=['_merge'])
