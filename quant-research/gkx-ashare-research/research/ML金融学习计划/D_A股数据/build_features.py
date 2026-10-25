"""
构造 A 股月频特征面板（论文 94 个特征中"只用行情+股本就能算"的那一批）。

论文的 94 个特征按频率分：年频 61 + 季频 13 + 月频 20。这一版先做月频的那一类，
因为它们是论文里最重要的信号（价格趋势、流动性、风险），而 A 股的财报数据还要
额外处理"公告日滞后"，留到下一版。

已实现的 19 个特征（论文缩写沿用，方便对照）：

| 缩写 | 含义 | 构造 |
|---|---|---|
| mvel1 | 市值 | log(月末收盘价 × 当期总股本) |
| turn | 换手率 | 当月成交量之和 / 流通 A 股 |
| std_turn | 换手波动 | 月换手率的 3 个月滚动标准差 |
| dolvol | 交易额 | log(当月日均成交额)，再取 3 个月均值 |
| ill | Amihud 非流动性 | 当月 mean(abs(日收益) / 日成交额) |
| zerotrade | 零成交 | 1 − 当月交易日数 / 当月市场交易日数 |
| baspread | 高低价差 | 当月 mean((最高−最低)/收盘)（无报价数据的代理） |
| maxret | 最大日收益 | 当月日收益最大值 |
| retvol | 收益波动 | 当月日收益标准差 |
| mom1m | 1 月动量 | 特征月 t 已实现的总收益（预测 t+1） |
| mom6m | 6 月动量 | 过去 6 个月累计总收益 |
| mom12m | 12 月动量 | 过去 12 个月累计总收益 |
| mom36m | 36 月动量 | 过去 36 个月累计总收益 |
| chmom | 动量变化 | mom6m(t) − mom6m(t−6) |
| beta | 市场贝塔 | 过去 36 个月对等权市场收益回归的斜率 |
| betasq | 贝塔平方 | beta² |
| idiovol | 特质波动 | 上述回归残差的标准差 |
| chcsho | 股本变化 | log(总股本_t / 总股本_{t−12}) |
| dy | 股息率 | 过去 12 个月每股派息之和 / 月末收盘价 |

特征按记录的有效日期和形成月回溯构造，标签是下一自然月收益（`ret_fwd`）。
这不是历史公开版本的认证：股本首次披露日期、供应商回补与全文件覆盖入口仍有限制。
精确定义和未完成时点核验见项目 docs/methods_appendix.md 与 docs/data_bug_fixes.md。

输入：
    data/prices/<code>.csv     日线（新浪，不复权）
    data/shares.csv            股本变动历史（东财）
    data/returns_monthly.csv   未缩尾的原始月总收益
    data/dividends.csv         分红送配（用于股息率）
输出：
    data/repaired/features_monthly.csv   面板（code, month, 特征..., ret_fwd）
    data/repaired/features_report.md         覆盖率与描述统计

用法：
    python build_features.py
    python build_features.py --min-months 24 --limit 200
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from data_contract import ensure_derived_directory

from data_contract import DATA, DERIVED, attach_forward_returns, require_unique
from build_returns import daily_returns
HERE = pathlib.Path(__file__).resolve().parent
PRICES = DATA / "prices"

FEATS = ["mvel1", "turn", "std_turn", "dolvol", "ill", "zerotrade", "baspread",
         "maxret", "retvol", "mom1m", "mom6m", "mom12m", "mom36m", "chmom",
         "beta", "betasq", "idiovol", "chcsho", "dy"]


def load_shares() -> pd.DataFrame:
    sh = pd.read_csv(DATA / "shares.csv", dtype={"SECURITY_CODE": str})
    sh["date"] = pd.to_datetime(sh["END_DATE"])
    return sh[["SECURITY_CODE", "date", "TOTAL_SHARES", "LISTED_A_SHARES"]].dropna(subset=["date"])


def shares_asof(sh: pd.DataFrame, code6: str, dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """对每个日期取最近一次 ≤ 该日的股本；返回 (总股本, 流通A股)。"""
    sub = sh[sh["SECURITY_CODE"] == code6].sort_values("date")
    if sub.empty:
        n = len(dates)
        return np.full(n, np.nan), np.full(n, np.nan)
    idx = np.searchsorted(sub["date"].values, dates.values, side="right") - 1
    tot = np.where(idx >= 0, sub["TOTAL_SHARES"].values[np.clip(idx, 0, None)], np.nan)
    lst = np.where(idx >= 0, sub["LISTED_A_SHARES"].values[np.clip(idx, 0, None)], np.nan)
    return tot, lst


def load_dividends() -> pd.DataFrame:
    p = DATA / "dividends.csv"
    if not p.exists():
        raise FileNotFoundError("dividends.csv is required for corporate-action adjusted daily features")
    d = pd.read_csv(p, dtype={"SECURITY_CODE": str})
    d["date"] = pd.to_datetime(d["EX_DIVIDEND_DATE"], errors="coerce")
    d["cash_per_share"] = pd.to_numeric(d["PRETAX_BONUS_RMB"], errors="coerce") / 10.0
    d["bonus"] = pd.to_numeric(d["BONUS_IT_RATIO"], errors="coerce").fillna(0) / 10.0
    # Exact duplicate source records must not double-count distributions.
    d = d.drop_duplicates().dropna(subset=["date"])
    return d.groupby(["SECURITY_CODE", "date"], as_index=False).agg(
        cash_per_share=("cash_per_share", "sum"), bonus=("bonus", "sum"))


def monthly_features(px: pd.DataFrame, sh: pd.DataFrame, div: pd.DataFrame,
                     code: str, cal_days: pd.Series) -> pd.DataFrame:
    """单只股票的月频特征。px 需按日期升序、含 date/open/high/low/close/volume。"""
    px = px.sort_values("date").reset_index(drop=True)
    require_unique(px, ["date"])
    sub = div.loc[div["SECURITY_CODE"].eq(code[-6:])].rename(
        columns={"date": "ex_date", "cash_per_share": "cash"})
    # Unadjusted close-to-close returns create false volatility on ex dates.
    adjusted = daily_returns(px, sub)
    px = px.merge(adjusted, on="date", how="left", validate="one_to_one")
    px["dollar"] = px["close"] * px["volume"]
    px["month"] = px["date"].dt.strftime("%Y-%m")
    px["hl_ratio"] = (px["high"] - px["low"]) / px["close"]
    px["ill_d"] = px["ret"].abs() * 1e9 / px["dollar"].replace(0, np.nan)

    g = px.groupby("month")
    out = pd.DataFrame({
        "close": g["close"].last(),
        "vol_sum": g["volume"].sum(),
        "dollar_mean": g["dollar"].mean(),
        "maxret": g["ret"].max(),
        "retvol": g["ret"].std(),
        "hl": g["hl_ratio"].mean(),
        "ill": g["ill_d"].mean(),
        "ndays": g["date"].count(),
        "last_date": g["date"].last(),
    }).reset_index()

    # 股本（按月末日期取最近一次）；股本 ≤ 0 视为缺失，否则换手率与市值会出 inf
    tot, lst = shares_asof(sh, code.replace("sh", "").replace("sz", "").replace("bj", ""),
                           out["last_date"])
    tot = np.where(np.asarray(tot) > 0, tot, np.nan)
    lst = np.where(np.asarray(lst) > 0, lst, np.nan)
    out["total_shares"], out["listed_shares"] = tot, lst
    out["market_cap"] = out["close"] * out["total_shares"]
    out["mvel1"] = np.log(out["market_cap"])
    out["turn"] = out["vol_sum"] / out["listed_shares"]
    out["dolvol"] = np.log(out["dollar_mean"].replace(0, np.nan))
    out["zerotrade"] = 1.0 - out["ndays"] / out["month"].map(cal_days).astype(float)
    out["baspread"] = out["hl"]

    out = out.sort_values("month").reset_index(drop=True)
    calendar = pd.period_range(out["month"].min(), out["month"].max(), freq="M").astype(str)
    out = out.set_index("month").reindex(calendar).rename_axis("month").reset_index()
    out["std_turn"] = out["turn"].rolling(3, min_periods=2).std()
    out["dolvol"] = out["dolvol"].rolling(3, min_periods=1).mean()

    # 月收益另从原始总收益面板精确合并。
    return out


def rolling_market_exposure(r, x, window=36, min_obs=24):
    """OLS slope/intercept with consistent covariance/variance normalisation."""
    r, x = np.asarray(r, dtype=float), np.asarray(x, dtype=float)
    beta, idio = np.full(len(r), np.nan), np.full(len(r), np.nan)
    for i in range(len(r)):
        lo = max(0, i - window + 1)
        rr, xx = r[lo:i + 1], x[lo:i + 1]
        ok = np.isfinite(rr) & np.isfinite(xx)
        if ok.sum() < min_obs:
            continue
        rr, xx = rr[ok], xx[ok]
        vx = xx.var(ddof=0)
        if vx <= 0:
            continue
        b = np.cov(rr, xx, ddof=0)[0, 1] / vx
        resid = rr - (rr.mean() - b * xx.mean()) - b * xx
        beta[i], idio[i] = b, resid.std(ddof=1)
    return beta, idio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-months", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-days", type=int, default=10)
    args = ap.parse_args()

    # Label source is never the historical globally winsorised returns_clean cache.
    rets = pd.read_csv(DATA / "returns_monthly.csv", dtype={"code": str})
    require_unique(rets, ["code", "month"])
    if rets["ret"].lt(-1).any():
        raise ValueError("Return below -100%; audit corporate actions before building features")
    rets = rets.sort_values(["code", "month"])
    have = set(rets["code"].unique())
    sh = load_shares()
    div = load_dividends()

    files = sorted(PRICES.glob("*.csv"))
    if args.limit:
        files = files[: args.limit]
    print(f"日线文件 {len(files):,} 个，原始月收益覆盖 {len(have):,} 只")

    # 市场日历：所有股票出现过的交易日的并集 → 每月交易日数
    all_dates = set()
    for p in files:
        all_dates.update(pd.read_csv(p, usecols=["date"])["date"].astype(str))
    cal = pd.Series(sorted(all_dates))
    cal_days = pd.to_datetime(cal).dt.strftime("%Y-%m").value_counts().sort_index()
    print(f"交易日并集 {len(cal):,} 天，覆盖 {len(cal_days)} 个月")

    # 等权市场月收益来自未缩尾原始面板；极值需源数据审计，不在此截断。
    mkt = rets.groupby("month")["ret"].mean().rename("mkt")

    frames = []
    for k, p in enumerate(files, 1):
        code = p.stem
        if code not in have:
            continue
        px = pd.read_csv(p, parse_dates=["date"])
        if len(px) < 60:
            continue
        m = monthly_features(px, sh, div, code, cal_days)
        m["code"] = code
        frames.append(m)
        if k % 500 == 0:
            print(f"  已处理 {k}/{len(files)}", flush=True)

    if not frames:
        raise ValueError("No price files with matching raw monthly returns")
    df = pd.concat(frames, ignore_index=True)
    df = df.merge(rets[["code", "month", "ret"]], on=["code", "month"], how="left")

    # 动量类：自然月网格中的未缩尾总收益（缺失月份不跨越）。
    df = df.sort_values(["code", "month"]).reset_index(drop=True)
    grp = df.groupby("code")["ret"]
    cum = lambda w: grp.transform(lambda s: (1 + s).rolling(w, min_periods=w).apply(np.prod, raw=True) - 1)
    # At month-end t the most recent realised monthly return is r_t.
    df["mom1m"] = df["ret"]
    df["mom6m"] = cum(6)
    df["mom12m"] = cum(12)
    df["mom36m"] = cum(36)
    df["chmom"] = df["mom6m"] - df.groupby("code")["mom6m"].shift(6)

    # beta / betasq / idiovol：过去 36 个月对等权市场收益的回归
    df = df.merge(mkt, on="month", how="left")
    df["beta"] = np.nan
    df["idiovol"] = np.nan
    for _, s in df.groupby("code"):
        beta, idio = rolling_market_exposure(s["ret"], s["mkt"])
        df.loc[s.index, "beta"] = beta
        df.loc[s.index, "idiovol"] = idio
    df["betasq"] = df["beta"] ** 2

    # chcsho：总股本 12 个月变化
    df["chcsho"] = np.log(df["total_shares"] /
                          df.groupby("code")["total_shares"].shift(12))

    # dy：过去 12 个月每股派息 / 月末收盘价
    # 必须在「完整月历」上滚动。若直接在面板上按行 rolling(12)，面板起始的
    # 前 12 个月没有更早的派息行可累加，TTM 会从 0 逐步补到满值（2010 年均值
    # 只有 0.32%，2011 年后约 1.0%），等于给最早的 12 个月注入了系统性低估值。
    if len(div):
        div["ym"] = div["date"].dt.strftime("%Y-%m")
        agg = div.groupby(["SECURITY_CODE", "ym"])["cash_per_share"].sum().reset_index()
        agg["code"] = agg["SECURITY_CODE"].apply(
            lambda c: ("sh" if c[0] in "69" else "sz" if c[0] in "032" else "bj") + c)
        agg = agg.rename(columns={"ym": "month"})

        lo = (pd.Period(df["month"].min(), freq="M") - 13).strftime("%Y-%m")
        cal = pd.period_range(lo, df["month"].max(), freq="M").strftime("%Y-%m")
        grid = pd.MultiIndex.from_product(
            [df["code"].unique(), cal], names=["code", "month"]).to_frame(index=False)
        grid = grid.merge(agg[["code", "month", "cash_per_share"]],
                          on=["code", "month"], how="left")
        grid["cash_per_share"] = grid["cash_per_share"].fillna(0.0)
        grid = grid.sort_values(["code", "month"], kind="mergesort")
        grid["div_ttm"] = grid.groupby("code")["cash_per_share"].transform(
            lambda s: s.rolling(12, min_periods=12).sum())
        df = df.merge(grid[["code", "month", "div_ttm"]], on=["code", "month"], how="left")
        df["dy"] = df["div_ttm"] / df["close"]
        df = df.drop(columns=["div_ttm"])
    else:
        df["dy"] = np.nan

    # 标签：下个月收益
    df = attach_forward_returns(df, rets)

    # 逐月横截面缩尾 1%/99%：`ill` 这类比值特征有极重尾，不缩尾会把回归拉爆
    df[FEATS] = df[FEATS].replace([np.inf, -np.inf], np.nan)
    for f in FEATS:
        g = df.groupby("month")[f]
        lo, hi = g.quantile(0.01), g.quantile(0.99)
        df[f] = df[f].clip(df["month"].map(lo), df["month"].map(hi))

    # Trailing history and tradability are known at t; do not use future coverage.
    observed = df["ndays"].fillna(0).gt(0)
    df["history_months"] = observed.groupby(df["code"]).cumsum()
    df["eligible_asof"] = observed & df["ndays"].ge(args.min_days) & df["history_months"].ge(args.min_months)
    keep = ["code", "month"] + FEATS + ["ret", "ret_fwd", "label_month", "ndays", "close", "market_cap", "history_months", "eligible_asof"]
    df = df[[c for c in keep if c in df.columns]]
    # Keep unknown labels so feature preprocessing need not select on future coverage.
    df = df[observed].copy()
    ensure_derived_directory()
    out = DERIVED / "features_monthly.csv"
    df.to_csv(out, index=False)
    (DERIVED / "features_manifest.json").write_text(json.dumps({
        "schema": "gkx-features/1", "author": "Gilmour Jia",
        "label": "next_calendar_month_raw_total_return", "return_winsorized": False,
        "label_month_explicit": True, "eligibility": "feature_month_asof",
        "rows": len(df), "eligible_rows": int(df.eligible_asof.sum()),
        "remaining_limitations": ["vendor_vintages_unverified", "delisting_coverage_unverified", "rights_issues_not_adjusted", "no_borrow_or_execution_model"]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n面板 {len(df):,} 行 / {df.code.nunique():,} 只 / {df.month.nunique()} 个月")
    print(f"已写出 {out}")

    lines = ["# A 股月频特征面板", "",
             f"面板 {len(df):,} 行，{df.code.nunique():,} 只，"
             f"{df.month.min()} → {df.month.max()}（{df.month.nunique()} 个月）", "",
             "| 特征 | 非缺失率 | 均值 | 标准差 | 5% | 中位数 | 95% |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for f in FEATS:
        if f not in df:
            continue
        s = df[f]
        lines.append(f"| {f} | {s.notna().mean():.3f} | {s.mean():.4f} | {s.std():.4f} | "
                     f"{s.quantile(.05):.4f} | {s.median():.4f} | {s.quantile(.95):.4f} |")
    (DERIVED / "features_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"已写出 {DERIVED / 'features_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
