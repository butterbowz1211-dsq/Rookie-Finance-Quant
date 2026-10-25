"""Gross portfolio diagnostics from repaired prediction files. Author: Gilmour Jia.

This is an equal-weight ranking diagnostic, not an executable short-selling
strategy. Costs, borrow availability and next-session fills remain unmodelled.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "A_蒙特卡洛"))
from evaluation import portfolio_statistics, label_months


def decile_portfolios(predictions, model, min_stocks=20):
    required = {"code", "month", "label_month", "ret_fwd", model}
    if not required.issubset(predictions):
        raise ValueError(f"Missing prediction fields: {sorted(required-set(predictions))}")
    label_months(predictions)
    if predictions.duplicated(["code", "label_month"]).any():
        raise ValueError("Duplicate stock-target-month predictions")
    rows = []
    for month, group in predictions.groupby("label_month", sort=True):
        if len(group) < min_stocks:
            continue
        if not np.isfinite(group[[model, "ret_fwd"]].to_numpy(float)).all():
            raise ValueError("Nonfinite prediction or realized return")
        if group[model].nunique() < 2:
            # A constant forecast contains no decile-ranking information.
            continue
        # Quantile cut points cannot yield 10 deciles with too many prediction ties.
        quantile = pd.qcut(group[model], 10, labels=False, duplicates="drop")
        if quantile.nunique() != 10:
            continue
        high = group.loc[quantile == 9, "ret_fwd"].mean()
        low = group.loc[quantile == 0, "ret_fwd"].mean()
        rows.append({"label_month": month, "high": high, "low": low,
                     "spread": high-low, "market_ew": group.ret_fwd.mean(), "n": len(group)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=pathlib.Path)
    ap.add_argument("--manifest", required=True, type=pathlib.Path)
    ap.add_argument("--model", default="ridge")
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--min-stocks", type=int, default=20)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("target") != "next-calendar-month unmodified total return":
        raise ValueError("Portfolio checks require an explicit unmodified-return manifest")
    predictions = pd.read_csv(args.predictions, dtype={"code": str})
    portfolios = decile_portfolios(predictions, args.model, args.min_stocks)
    if len(portfolios) < 2:
        raise ValueError("Fewer than two complete decile months; no performance summary")
    months = pd.PeriodIndex(portfolios.label_month, freq="M")
    contiguous = np.all(np.diff(months.asi8) == 1)
    stats = {name: portfolio_statistics(portfolios[name]) for name in ("high", "low", "spread", "market_ew")}
    if not contiguous:
        # Compounding disjoint months cannot be called an uninterrupted strategy CAGR.
        for value in stats.values():
            value["cagr"] = None
    result = {"author": "Gilmour Jia", "model": args.model, "gross": True,
              "risk_free_rate_assumed_for_ratio": 0, "consecutive_target_months": bool(contiguous),
              "statistics": stats,
              "limitations": ["No transaction costs, borrow constraints, execution delay or capacity model",
                              "High-minus-low spread compounding is a notional diagnostic, not a specified funded account",
                              "Low portfolio returns are not the realized P&L of an implementable short position",
                              "Mean-over-SD is not a risk-free-adjusted Sharpe ratio for long-only portfolios",
                              "Missing target returns and unavailable deciles can affect sample selection"]}
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = args.output / "portfolios.csv", args.output / "portfolio_diagnostics.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Choose a fresh output directory")
    portfolios.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

