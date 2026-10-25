"""A-share research runner. Author: Gilmour Jia.

All splits refer to TARGET return months. Historical outputs are not overwritten.
Inputs must come from the repaired calendar-label feature builder. Returns remain
unwinsorized TOTAL returns, not excess returns. Training/validation stay separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import sklearn

HERE = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(os.environ.get("GKX_DERIVED_DIR", os.environ.get("GKX_DATA_DIR", HERE / "data")))
OUT = pathlib.Path(os.environ.get("GKX_OUT_DIR", HERE / "out_repaired"))
CLEAN = pathlib.Path(os.environ.get("GKX_CLEAN_DIR", HERE.parents[1] / "A股行业时点与ST数据清洗" / "01_data"))
MC = HERE.parent / "A_蒙特卡洛"
sys.path.insert(0, str(MC))
import models as mcmodels
import nn as mcnn
from evaluation import (SPLIT, r2_zero, r2_sample_mean, r2_training_mean,
                        r2_monthly_average, rank_characteristics,
                        split_masks, encode_industry)

FEATS = ["mvel1", "turn", "std_turn", "dolvol", "ill", "zerotrade", "baspread",
         "maxret", "retvol", "mom1m", "mom6m", "mom12m", "mom36m", "chmom",
         "beta", "betasq", "idiovol", "chcsho", "dy"]
FEATS19 = FEATS
FUND_FEATS = ["bm", "ep", "sp", "roa", "lev", "agr", "acc", "cfp", "nincr"]
NEW43 = ["absacc", "aeavol", "age", "bm_ia", "cash", "cashpr", "cfp_ia", "chatoia",
         "chinv", "chpmia", "cinvest", "divi", "divo", "ear", "egr", "gma", "grCAPX",
         "herf", "indmom", "invest", "mve_ia", "operprof", "orgcap", "pchcapx_ia",
         "pchgm_pchsale", "pchsale_pchinvt", "pchsale_pchrect", "pchsale_pchxsga",
         "pchsaleinv", "pctacc", "pricedelay", "roavol", "roeq", "rsup", "salecash",
         "saleinv", "salerec", "sgr", "sin", "std_dolvol", "stdacc", "stdcf", "tang"]
# Known unsafe old cached definitions cannot be re-enabled by a generic waiver.
UNSAFE_CACHED = {"divi", "divo", "herf", "indmom", "sin", "bm_ia", "cfp_ia",
                 "chatoia", "chpmia", "mve_ia", "pchcapx_ia"}
MACROS = ["dp", "ep", "bm", "ntis", "svar"]
r2_paper = r2_zero
r2_demeaned = r2_sample_mean
r2_monthly = r2_monthly_average
cs_rank = rank_characteristics


def cs_rank_inplace(df, cols):
    ranked = rank_characteristics(df, cols)
    df[cols] = ranked[cols]


def standardize_inplace(Xtr, Xva, Xte):
    mu = Xtr.mean(axis=0, dtype=np.float64)
    sd = Xtr.std(axis=0, dtype=np.float64)
    sd[sd < 1e-8] = 1
    for matrix in (Xtr, Xva, Xte):
        matrix -= mu
        matrix /= sd
    return Xtr, Xva, Xte


def _read(path, sources):
    sources.append(path)
    return pd.read_csv(path, dtype={"code": str, "month": str})


def _merge(df, other):
    return df.merge(other, on=["code", "month"], how="left", validate="one_to_one", sort=False)


def _safe_tag(value):
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("tag must contain only letters, digits, underscores or hyphens")
    return value


def prepare_panel(args):
    sources = []
    df = _read(DATA / "features_monthly.csv", sources)
    if df.duplicated(["code", "month"]).any():
        raise ValueError("Duplicate code-month feature rows")
    # Rank the information available at formation time, before inspecting future y.
    required = {"label_month", "eligible_asof", "ret_fwd", "market_cap"}
    if not required.issubset(df.columns):
        raise ValueError(f"Rebuild features with the repaired builder; missing {sorted(required - set(df.columns))}")
    eligibility = df["eligible_asof"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df = df.loc[eligibility].reset_index(drop=True)
    chars = list(FEATS)
    excluded = []
    if args.chars != "19":
        if not args.allow_unverified_features:
            raise ValueError("Financial vintages are unverified. Use --allow-unverified-features only for explicitly exploratory runs.")
        if args.chars == "fund56":
            fund = _read(DATA / "fundamentals_v2_monthly.csv", sources)
            cols = [c for c in fund if c not in ("code", "month", "label_month", "eligible_asof")]
        else:
            choices = [DATA / "fundamentals_monthly_v2.csv", DATA / "fundamentals_monthly.csv"]
            file = next((p for p in choices if p.exists()), choices[0])
            fund = _read(file, sources)
            cols = [c for c in FUND_FEATS if c in fund]
        cols = [c for c in cols if c not in chars and c not in UNSAFE_CACHED]
        df = _merge(df, fund[["code", "month"] + cols])
        chars += cols
        if args.chars == "71":
            factor = _read(DATA / "factors_43_monthly.csv", sources)
            excluded = sorted(UNSAFE_CACHED.intersection(NEW43))
            cols = [c for c in NEW43 if c not in UNSAFE_CACHED]
            df = _merge(df, factor[["code", "month"] + cols])
            chars += cols
    if args.industry:
        ind = _read(CLEAN / "stock_industry_monthly.csv", sources)
        ind["code6"] = ind["code"].str.extract(r"(\d{6})", expand=False)
        if ind.duplicated(["code6", "month"]).any():
            raise ValueError("Industry source has duplicate stock-month rows")
        df["code6"] = df["code"].str.extract(r"(\d{6})", expand=False)
        df = df.merge(ind[["code6", "month", "industry_code"]], on=["code6", "month"],
                      how="left", validate="many_to_one", sort=False)
    ranked = rank_characteristics(df, chars)
    # Only the evaluation/fitting population needs an observed label.
    df = ranked.loc[np.isfinite(pd.to_numeric(ranked.ret_fwd, errors="coerce"))].reset_index(drop=True)
    masks = split_masks(df)
    if any(not m.any() for m in masks.values()):
        raise ValueError("Training, validation and test windows must all contain labelled rows")
    columns = list(chars)
    industry_info = None
    if args.industry:
        dummies, industry_info = encode_industry(df.industry_code, masks["train"], args.industry_min_count)
        df = pd.concat([df, dummies], axis=1)
        columns += list(dummies.columns)
    if args.macro:
        if not args.allow_unverified_features:
            raise ValueError("Macro caches inherit financial-vintage risks; use --allow-unverified-features only for exploratory runs")
        macro = _read(DATA / "macro_monthly.csv", sources)
        available = [m for m in MACROS if m in macro]
        macro = macro[["month"] + available].rename(columns={m: f"mac_{m}" for m in available})
        df = df.merge(macro, on="month", how="left", validate="many_to_one", sort=False)
        new_columns = {}
        for macro_name in available:
            values = df[f"mac_{macro_name}"].to_numpy(float)
            train_values = values[masks["train"]]
            if not np.isfinite(train_values).any():
                raise ValueError(f"No observed training values for {macro_name}")
            mu, sd = np.nanmean(train_values), np.nanstd(train_values)
            z = (np.where(np.isfinite(values), values, mu) - mu) / (sd if sd > 1e-8 else 1)
            # Interact company characteristics only; industry indicators stay separate.
            for char in chars:
                name = f"{char}x{macro_name}"
                new_columns[name] = df[char].to_numpy(float) * z
        df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        columns += list(new_columns)
    return df, columns, masks, sources, {"excluded_cached_features": excluded, "industry": industry_info}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="ridge,lasso,enet")
    ap.add_argument("--chars", choices=["19", "28", "71", "fund56"], default="19")
    ap.add_argument("--fundamentals", action="store_true", help="Legacy alias for --chars 28")
    ap.add_argument("--industry", action="store_true")
    ap.add_argument("--no-industry", action="store_true", help="Compatibility alias; industry is off by default")
    ap.add_argument("--industry-min-count", type=int, default=100)
    ap.add_argument("--macro", action="store_true")
    ap.add_argument("--allow-unverified-features", action="store_true")
    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--rf-leaf", type=int, default=100)
    ap.add_argument("--gb-trees", type=int, default=200)
    ap.add_argument("--gb-leaf", type=int, default=100)
    ap.add_argument("--nn-depth", type=int, default=0, choices=range(6))
    ap.add_argument("--nn-ensemble", type=int, default=10)
    ap.add_argument("--nn-epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="repaired")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fundamentals:
        args.chars = "28"
    if args.no_industry:
        args.industry = False
    tag = _safe_tag(args.tag)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"ashare_{tag}.csv"
    if dest.exists():
        raise FileExistsError(f"Choose a new --tag; will not overwrite {dest}")
    df, columns, masks, sources, preprocessing = prepare_panel(args)
    X = df[columns].to_numpy(dtype=np.float32)
    y = df.ret_fwd.to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite predictors after preprocessing")
    tr, va, te = (masks[name] for name in ("train", "valid", "test"))
    Xtr, Xva, Xte = (X[mask].copy() for mask in (tr, va, te))
    ytr, yva, yte = y[tr], y[va], y[te]
    standardize_inplace(Xtr, Xva, Xte)
    rows, predictions = [], df.loc[te, ["code", "month", "label_month", "ret_fwd", "market_cap"]].copy()
    want = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    if args.nn_depth and "nn" not in want:
        want.append("nn")
    invalid = set(want) - {"ols", "ridge", "lasso", "enet", "pcr", "pls", "rf", "gbrt", "nn"}
    if invalid or not want:
        raise ValueError(f"Invalid model selection: {sorted(invalid)}")
    print(f"Target-month splits: {SPLIT}; rows={tuple(m.sum() for m in (tr, va, te))}; features={len(columns)}", flush=True)
    for name in want:
        if name == "ols":
            predict, info = mcmodels.fit_ols(Xtr, ytr), {}
        elif name == "ridge":
            predict, info = mcmodels.fit_ridge_path(Xtr, ytr, np.logspace(0, 5, 11), Xva, yva)
        elif name in ("lasso", "enet"):
            predict, info = mcmodels.fit_enet(Xtr, ytr, np.logspace(-6, -2, 9),
                                              1.0 if name == "lasso" else 0.5, Xva, yva)
        elif name in ("pcr", "pls"):
            fit = mcmodels.fit_pcr if name == "pcr" else mcmodels.fit_pls
            predict, info = fit(Xtr, ytr, range(1, min(len(columns), len(ytr)-1, 40)+1), Xva, yva)
        elif name == "rf":
            grid = [(d, m) for d in (1, 3, 6) for m in sorted({min(5, len(columns)), min(10, len(columns))})]
            predict, info = mcmodels.fit_rf(Xtr, ytr, grid, Xva, yva, args.seed, args.rf_trees, args.rf_leaf)
        elif name == "gbrt":
            grid = [(d, lr, args.gb_trees) for d in (1, 2) for lr in (0.01, 0.1)]
            predict, info = mcmodels.fit_gbrt(Xtr, ytr, grid, Xva, yva, args.seed, min_samples_leaf=args.gb_leaf)
        else:
            grid = [(lam, lr) for lam in (1e-5, 1e-4, 1e-3) for lr in (0.01, 0.1)]
            predict, info = mcnn.fit_nn(Xtr, ytr, Xva, yva, args.nn_depth or 3, grid,
                                        args.nn_ensemble, args.seed, args.nn_epochs)
            name = f"nn{args.nn_depth or 3}"
        ptr, pva, pte = predict(Xtr), predict(Xva), predict(Xte)
        parameters = {k: v for k, v in info.items() if k not in ("coef", "imp", "mask")}
        row = {"model": name, "n_feat": len(columns), "IS": 100*r2_zero(ytr, ptr),
               "valid": 100*r2_zero(yva, pva), "OOS": 100*r2_zero(yte, pte),
               "OOS_sample_mean": 100*r2_sample_mean(yte, pte),
               "OOS_training_mean": 100*r2_training_mean(yte, pte, ytr.mean()),
               "OOS_equal_month": 100*r2_monthly_average(yte, pte, predictions.label_month),
               "parameters": json.dumps(parameters, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))}
        rows.append(row)
        predictions[name] = pte
        # Checkpoint each completed model; failed later models do not erase evidence.
        pd.DataFrame(rows).to_csv(dest, index=False)
        predictions.to_csv(OUT / f"predictions_{tag}.csv", index=False)
        print(f"{name}: validation={row['valid']:.4f}%, pooled OOS={row['OOS']:.4f}%", flush=True)
    source_metadata = []
    for source in sources:
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        source_metadata.append({"filename": source.name, "bytes": source.stat().st_size, "sha256": digest})
    manifest = {"author": "Gilmour Jia", "status": "exploratory_unverified_financial_vintages" if args.chars != "19" or args.macro else "repaired_price_feature_run",
                "target": "next-calendar-month unmodified total return", "r2_primary": "pooled zero benchmark; CSV values in percent",
                "split_axis": "label_month", "splits": SPLIT,
                "counts": {key: int(mask.sum()) for key, mask in masks.items()},
                "formation_months_test": sorted(predictions.month.unique()), "target_months_test": sorted(predictions.label_month.unique()),
                "feature_columns": columns, "settings": vars(args), "preprocessing": preprocessing,
                "sources": source_metadata, "numpy": np.__version__, "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__, "python": sys.version,
                "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [pathlib.Path(__file__), MC / "evaluation.py", MC / "models.py", MC / "nn.py"]},
                "limitations": ["Original data revisions, survivorship and corporate actions are not fully independently verified",
                                "Total returns are not GKX excess returns", "Fixed holdout; not annual walk-forward",
                                "Prior exploration of this test period prevents calling it a fresh untouched holdout"]}
    (OUT / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

