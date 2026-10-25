"""用不复权日线 + 分红送配表构造含股息的日/月总收益。

原理
----
新浪日线是不复权的，除权除息日当天价格会出现一个"缺口"，这个缺口不是收益。
对除权日前持有 1 股的投资者，除权日后持有 (1 + 送转/10) 股并获得 (派息/10) 元现金，
所以当天的总收益是

    r_t = ( (1 + BONUS_IT_RATIO/10) * P_t + PRETAX_BONUS_RMB/10 ) / P_{t-1} - 1

其余交易日 r_t = P_t / P_{t-1} - 1。把日收益连乘即得月收益。

用法
----
    python build_returns.py --codes 600000,000001
    python build_returns.py --all
输出
----
    data/returns_daily/<code>.csv   （date, ret）
    data/returns_monthly.csv        （code, month, ret）
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
DATA = BASE / "data"
PRICES = DATA / "prices"
RET_D = DATA / "returns_daily"
RET_M = DATA / "returns_monthly.csv"


def load_dividends() -> pd.DataFrame:
    path = DATA / "dividends.csv"
    if not path.exists():
        raise SystemExit(f"缺 {path}，先跑 fetch_dividends.py")
    dv = pd.read_csv(path, dtype={"SECURITY_CODE": str})
    dv["ex_date"] = pd.to_datetime(dv["EX_DIVIDEND_DATE"], errors="coerce")
    dv = dv.dropna(subset=["ex_date"])
    dv["bonus"] = pd.to_numeric(dv["BONUS_IT_RATIO"], errors="coerce").fillna(0.0) / 10.0
    dv["cash"] = pd.to_numeric(dv["PRETAX_BONUS_RMB"], errors="coerce").fillna(0.0) / 10.0
    # 同一除权日可能有多条（中期+年度），合并
    g = dv.groupby(["SECURITY_CODE", "ex_date"], as_index=False).agg(
        bonus=("bonus", "sum"), cash=("cash", "sum"))
    return g


def code6(sym: str) -> str:
    return "".join(ch for ch in sym if ch.isdigit())[-6:]


def daily_returns(px: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    px = px.sort_values("date").reset_index(drop=True)
    close = px["close"].to_numpy(dtype=float)
    prev = np.concatenate([[np.nan], close[:-1]])
    ret = close / prev - 1.0

    if len(div):
        ex = div["ex_date"].values.astype("datetime64[ns]")
        mult = (1.0 + div["bonus"].to_numpy(dtype=float))
        cash = div["cash"].to_numpy(dtype=float)
        lookup = {np.datetime64(d): (m, c) for d, m, c in zip(ex, mult, cash)}
        dates = px["date"].to_numpy(dtype="datetime64[ns]")
        for i, d in enumerate(dates):
            ev = lookup.get(d)
            if ev is not None and i > 0 and np.isfinite(prev[i]) and prev[i] > 0:
                m, c = ev
                ret[i] = (m * close[i] + c) / prev[i] - 1.0
    return pd.DataFrame({"date": px["date"], "ret": ret}).dropna()


def to_monthly(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["code", "month", "ret"])
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    m = d.groupby("month", as_index=False)["ret"].apply(lambda s: float(np.prod(1.0 + s) - 1.0))
    m = m.rename(columns={"ret": "ret"})
    m.insert(0, "code", sym)
    return m


def build_one(sym: str, divs: pd.DataFrame) -> pd.DataFrame | None:
    path = PRICES / f"{sym}.csv"
    if not path.exists():
        return None
    px = pd.read_csv(path, parse_dates=["date"])
    if px.empty:
        return None
    c6 = code6(sym)
    sub = divs[divs["SECURITY_CODE"] == c6]
    d = daily_returns(px, sub)
    if d.empty:
        return None
    RET_D.mkdir(parents=True, exist_ok=True)
    d.to_csv(RET_D / f"{sym}.csv", index=False, encoding="utf-8-sig")
    return to_monthly(d, sym)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", type=str, default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    divs = load_dividends()
    if args.all:
        syms = sorted(p.stem for p in PRICES.glob("*.csv"))
    elif args.codes:
        from fetch_prices import to_sina
        syms = [to_sina(c) for c in args.codes.split(",") if c.strip()]
    else:
        raise SystemExit("需要 --codes 或 --all")

    frames = []
    for s in syms:
        m = build_one(s, divs)
        if m is not None:
            frames.append(m)
    if not frames:
        print("没有可用数据")
        return
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RET_M, index=False, encoding="utf-8-sig")
    print(f"{len(syms)} 只，{len(out)} 条月收益，写到 {RET_M}")


if __name__ == "__main__":
    main()
