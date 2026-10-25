"""Annotate as-of eligibility without deleting or changing realised returns.

First observed month is a coverage flag, not verified IPO information.
Optional training winsorisation writes a separate ret_train column.
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
from data_contract import DATA, DERIVED, require_unique


def to_prefix(code: str) -> str:
    c = str(code).strip().lower()
    if c[:2] in ("sh", "sz", "bj"):
        return c
    c = c.zfill(6)
    return ("sh" if c[0] in "69" else "sz" if c[0] in "032" else "bj") + c


def trading_days_by_month(code: str) -> pd.Series:
    p = DATA / "prices" / f"{to_prefix(code)}.csv"
    if not p.exists():
        return pd.Series(dtype=int)
    d = pd.read_csv(p, usecols=["date"]).drop_duplicates("date")
    return d["date"].astype(str).str[:7].value_counts().sort_index()


def annotate_returns(df: pd.DataFrame, counts: dict, min_days: int = 10,
                     drop_first: bool = True) -> pd.DataFrame:
    require_unique(df, ["code", "month"])
    out = df.sort_values(["code", "month"]).copy()
    out["ndays"] = [counts.get((c, m), 0) for c, m in zip(out.code, out.month)]
    out["is_first_observed"] = out.groupby("code").cumcount().eq(0)
    out["eligible_asof"] = out.ndays.ge(min_days)
    if drop_first:
        out["eligible_asof"] &= ~out.is_first_observed
    out["invalid_return"] = ~np.isfinite(out.ret) | out.ret.lt(-1)
    return out


def add_training_winsor(df: pd.DataFrame, quantile: float,
                       train_start: str, train_end: str) -> tuple[pd.DataFrame, tuple]:
    if not 0 < quantile < .5:
        raise ValueError("winsor must lie strictly between 0 and 0.5")
    fit = df.loc[df.month.between(train_start, train_end), "ret"].dropna()
    if fit.empty:
        raise ValueError("No observations in the declared training interval")
    lo, hi = fit.quantile([quantile, 1 - quantile])
    out = df.copy()
    out["ret_train"] = np.nan
    train = out.month.between(train_start, train_end)
    out.loc[train, "ret_train"] = out.loc[train, "ret"].clip(lo, hi)
    return out, (float(lo), float(hi))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA / "returns_monthly.csv"))
    ap.add_argument("--out", default=str(DERIVED / "returns_clean.csv"))
    ap.add_argument("--min-days", type=int, default=10)
    ap.add_argument("--winsor", type=float, default=0.0)
    ap.add_argument("--train-start", default="2010-01")
    ap.add_argument("--train-end")
    ap.add_argument("--drop-first", action="store_true", default=True)
    ap.add_argument("--keep-first", dest="drop_first", action="store_false")
    args = ap.parse_args()
    if args.winsor and not args.train_end:
        ap.error("--winsor requires --train-end; full-sample thresholds are forbidden")
    df = pd.read_csv(args.input, dtype={"code": str})
    counts = {(c, m): int(n) for c in df.code.unique()
              for m, n in trading_days_by_month(c).items()}
    out = annotate_returns(df, counts, args.min_days, args.drop_first)
    thresholds = None
    if args.winsor:
        out, thresholds = add_training_winsor(out, args.winsor, args.train_start, args.train_end)
    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    report = (f"# Return eligibility audit\n\nRows retained: {len(out)}. "
              f"As-of eligible rows: {int(out.eligible_asof.sum())}. "
              f"Invalid source returns: {int(out.invalid_return.sum())}.\n\n"
              "Realised ret is unchanged. Eligibility is applied at feature month t, "
              "never by target-month trading days. Missing target returns remain missing.\n\n"
              f"Training-only winsor thresholds: {thresholds}.\n")
    dest.with_name("screen_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
