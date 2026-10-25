"""Publishable monthly summaries from private prediction files. Author: Gilmour Jia.

R² columns use fractions; *_percent columns multiply by 100. Pearson IC and
Spearman Rank IC are cross-sectional correlations, not tests of significance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import sys
import numpy as np
import pandas as pd

KEYS = ["code", "month", "label_month"]
COMMON = ["ret_fwd", "market_cap"]


def join_predictions(frames):
    if not frames:
        raise ValueError("No prediction frames supplied")
    joined = frames[0].copy()
    required = set(KEYS + COMMON)
    if not required.issubset(joined):
        raise ValueError("Missing key/label/market_cap fields")
    if joined.duplicated(KEYS).any():
        raise ValueError("Duplicate prediction keys")
    for frame in frames[1:]:
        if not required.issubset(frame):
            raise ValueError("Missing key/label/market_cap fields")
        overlap = (set(joined) & set(frame)) - required
        if overlap:
            raise ValueError(f"Duplicate model names: {sorted(overlap)}")
        merged = joined.merge(frame, on=KEYS, how="outer", validate="one_to_one",
                              suffixes=("", "__check"), indicator=True)
        if not merged["_merge"].eq("both").all():
            raise ValueError("Prediction runs do not have exactly the same stock-month keys")
        for column in COMMON:
            left = merged[column].to_numpy(float)
            right = merged[column+"__check"].to_numpy(float)
            if not np.array_equal(left,right,equal_nan=True):
                raise ValueError(f"Input runs disagree on {column}")
        joined = merged.drop(columns=["_merge"]+[c+"__check" for c in COMMON])
    expected = (pd.PeriodIndex(joined.month,freq="M")+1).astype(str)
    if not np.array_equal(expected.to_numpy(),joined.label_month.astype(str).to_numpy()):
        raise ValueError("A label month is not the next calendar month")
    models = [c for c in joined if c not in required]
    if not models or not np.isfinite(joined[["ret_fwd"]+models].to_numpy(float)).all():
        raise ValueError("Missing models or nonfinite labels/predictions")
    return joined.sort_values(KEYS).reset_index(drop=True), models


def correlation(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    # std() may leave rounding residue even when every float is identical.
    # Test the actual range before computing a correlation.
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:
        return float("nan")
    return float(np.corrcoef(x,y)[0,1])


def summarize(joined,models):
    rows=[]
    for month,group in joined.groupby("label_month",sort=True):
        y=group.ret_fwd.to_numpy(float)
        denominator=float(np.sum(y*y,dtype=np.float64))
        for model in models:
            prediction=group[model].to_numpy(float)
            squared_error=float(np.sum((y-prediction)**2,dtype=np.float64))
            rows.append({"label_month":month,"model":model,"n":len(group),
                         "sum_squared_error":squared_error,"sum_squared_return":denominator,
                         "r2_zero":1-squared_error/denominator if denominator>0 else np.nan,
                         "pearson_ic":correlation(prediction,y),
                         "spearman_rank_ic":correlation(pd.Series(prediction).rank(method="average"),
                                                         pd.Series(y).rank(method="average"))})
    monthly=pd.DataFrame(rows)
    pooled=[]
    for model,group in monthly.groupby("model",sort=False):
        den=group.sum_squared_return.sum()
        value=1-group.sum_squared_error.sum()/den if den>0 else np.nan
        pooled.append({"model":model,"n":int(group.n.sum()),"months":len(group),
                       "pooled_r2_zero":value,"pooled_r2_zero_percent":100*value,
                       "mean_monthly_r2_zero":float(group.r2_zero.mean()),
                       "pearson_ic_valid_months":int(group.pearson_ic.notna().sum()),
                       "spearman_rank_ic_valid_months":int(group.spearman_rank_ic.notna().sum()),
                       "mean_monthly_pearson_ic":float(group.pearson_ic.mean()),
                       "mean_monthly_spearman_rank_ic":float(group.spearman_rank_ic.mean())})
    return monthly,pd.DataFrame(pooled)


def check_reported(pooled,metric_frames,tolerance=1e-10):
    reported=pd.concat(metric_frames,ignore_index=True)
    if reported.model.duplicated().any():
        raise ValueError("Duplicate model in reported metric files")
    checked=pooled.merge(reported[["model","OOS"]],on="model",how="outer",validate="one_to_one",indicator=True)
    if not checked["_merge"].eq("both").all():
        raise ValueError("Reported models differ from predictions")
    checked["difference_percent_points"]=checked.pooled_r2_zero_percent-checked.OOS
    if not np.all(np.abs(checked.difference_percent_points)<=tolerance):
        raise ValueError("Recomputed pooled R² does not agree with reported metrics")
    return checked.drop(columns="_merge")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",nargs="+",required=True,type=pathlib.Path)
    ap.add_argument("--metrics",nargs="+",required=True,type=pathlib.Path)
    ap.add_argument("--output",required=True,type=pathlib.Path)
    args=ap.parse_args()
    joined,models=join_predictions([pd.read_csv(p,dtype={"code":str}) for p in args.input])
    monthly,pooled=summarize(joined,models)
    checked=check_reported(pooled,[pd.read_csv(p) for p in args.metrics])
    if args.output.exists():
        raise FileExistsError("Choose a fresh aggregation output directory")
    args.output.mkdir(parents=True)
    monthly.to_csv(args.output/"monthly_metrics.csv",index=False)
    checked.to_csv(args.output/"pooled_metrics_recomputed.csv",index=False)
    bindings=[]
    for path in args.input+args.metrics:
        with path.open("rb") as stream:
            digest=hashlib.file_digest(stream,"sha256").hexdigest()
        bindings.append({"filename":path.name,"sha256":digest,"bytes":path.stat().st_size})
    manifest={"author":"Gilmour Jia","status":"recomputed_and_matched",
              "python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,
              "stock_month_rows":len(joined),"months":int(joined.label_month.nunique()),"models":models,
              "key_join":"strict one-to-one outer join; identical realized returns and market caps",
              "r2_tolerance_percent_points":1e-10,"inputs":bindings,
              "script_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              "privacy":"Only aggregate monthly and pooled metrics exported; no individual stock predictions",
              "units":{"r2_zero":"fraction","pooled_r2_zero_percent":"percent","ic":"correlation"},
              "limitations":["Same repeatedly examined historical test period; not independent external validation",
                             "No uncertainty interval or multiple-testing adjustment computed",
                             "Undefined correlations for constant forecasts are missing, not zero",
                             "Total-return targets and residual source-data limitations remain"]}
    (args.output/"aggregation_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(checked.to_string(index=False))
    return 0


if __name__=="__main__": raise SystemExit(main())
