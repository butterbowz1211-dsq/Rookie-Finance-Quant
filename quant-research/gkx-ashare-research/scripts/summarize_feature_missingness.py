"""Aggregate pre-ranking feature missingness. Author: Gilmour Jia."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

FEATURES = ['mvel1','turn','std_turn','dolvol','ill','zerotrade','baspread',
            'maxret','retvol','mom1m','mom6m','mom12m','mom36m','chmom',
            'beta','betasq','idiovol','chcsho','dy']
WINDOWS = {'train': ('2010-01','2018-12'), 'valid': ('2019-01','2021-12'),
           'test': ('2022-01','2025-12')}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    columns=FEATURES+['eligible_asof','label_month','ret_fwd']
    panel=pd.read_csv(args.input,usecols=columns)
    eligible=panel.eligible_asof.astype(str).str.lower().isin(['true','1','1.0'])
    labelled=np.isfinite(pd.to_numeric(panel.ret_fwd,errors='coerce'))
    rows=[]
    for split,(start,end) in WINDOWS.items():
        part=panel.loc[eligible & labelled & panel.label_month.between(start,end)]
        for feature in FEATURES:
            values=pd.to_numeric(part[feature],errors='coerce').to_numpy(float)
            finite=int(np.isfinite(values).sum())
            missing=len(part)-finite
            rows.append({'split':split,'feature':feature,'n':len(part),
                         'finite_before_rank':finite,'missing_before_rank':missing,
                         'neutral_rank_imputation_fraction':missing/len(part) if len(part) else None})
    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.output/'feature_missingness.csv',index=False)
    with args.input.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    meta={'author':'Gilmour Jia','input_filename':args.input.name,'input_sha256':digest,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'population':'Eligible rows with finite target returns, split by target month; exact final model population',
          'stage':'Frozen builder output before cross-sectional ranks and train standardization',
          'interpretation':'Nonfinite feature values become neutral rank zero before standardization; observed median ranks are not counted as imputation',
          'ranking_note':'The actual rank reference population additionally includes eligible rows without a future observed label; this table describes only final fitting/scoring rows',
          'windows':WINDOWS,'rows':len(rows),'numpy':np.__version__,'pandas':pd.__version__}
    (args.output/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).groupby('split').agg(n=('n','first'),max_missing_fraction=('neutral_rank_imputation_fraction','max')).to_string())

if __name__=='__main__': main()
