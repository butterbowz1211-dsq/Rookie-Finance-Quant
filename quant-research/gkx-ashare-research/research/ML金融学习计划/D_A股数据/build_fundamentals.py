"""构造 A 股的财报类月度特征，按公告日对齐（供应商历史版本仍待核验）。

关键约定
--------
1. **一律用 `NOTICE_DATE`（公告日）而不是 `REPORT_DATE`（报告期）对齐。**
   用报告期对齐等于假设投资者在报告期结束当天就知道财报，这是典型的前视偏差。
   A 股年报法定截止日是次年 4 月 30 日，中位数滞后 57 天、75 分位 110 天。
2. **流量科目做 TTM。** 东财的利润表与现金流量表是年初至今累计值，
   所以 TTM = 本期累计 + 上年年报 − 上年同期累计；年报本身就是 TTM。
3. **每个"股票-月"只取该月末之前已公告的最新一份报告**，用 merge_asof 向后匹配。

用法：
    python build_fundamentals.py
    python build_fundamentals.py --allow-unverified-vintages --dump-report
输出：data/fundamentals_monthly.csv
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from data_contract import ensure_derived_directory
from data_contract import (DATA, DERIVED, calendar_report_lag, ttm_values,
                           raw_market_cap, require_financial_vintage_opt_in)

BASE = pathlib.Path(__file__).resolve().parent

# 需要构造的特征（论文 94 个特征里能在 A 股用三大报表复现的那一批）
OUT_FEATURES = ["bm", "ep", "sp", "roa", "lev", "agr", "acc", "cfp", "nincr"]


def load_statements() -> pd.DataFrame:
    st = pd.read_csv(DATA / "statements.csv", dtype={"SECURITY_CODE": str})
    st = st.rename(columns={"SECURITY_CODE": "code"})
    for c in ("REPORT_DATE", "NOTICE_DATE"):
        st[c] = pd.to_datetime(st[c], errors="coerce")
    st = st.dropna(subset=["code", "REPORT_DATE", "NOTICE_DATE"])
    st["code"] = st["code"].str.zfill(6)
    # 同一代码同一报告期可能有多条（更正公告），保留公告日最晚的一条
    st = st.sort_values(["code", "REPORT_DATE", "NOTICE_DATE"])
    st = st.drop_duplicates(subset=["code", "REPORT_DATE"], keep="last")
    # A late revision of an input cannot be used before its recorded publication.
    st["NOTICE_DATE"] = st.groupby("code")["NOTICE_DATE"].cummax()
    return st


def add_ttm(st: pd.DataFrame) -> pd.DataFrame:
    """TTM needs all three dated inputs; missing history remains missing."""
    st = st.sort_values(["code", "REPORT_DATE"]).copy()
    for col in ("PARENT_NETPROFIT", "TOTAL_OPERATE_INCOME", "NETCASH_OPERATE"):
        if col in st.columns:
            values, available = ttm_values(st, col, code_col="code")
            st[col + "_TTM"] = values
            st["NOTICE_DATE"] = pd.concat([st["NOTICE_DATE"], available], axis=1).max(axis=1)
    return st


def add_ratios(st: pd.DataFrame) -> pd.DataFrame:
    """不需要市值就能算的比率。"""
    ta = st.get("TOTAL_ASSETS")
    tl = st.get("TOTAL_LIABILITIES")
    te = st.get("TOTAL_EQUITY")
    ar = st.get("ACCOUNTS_RECE")
    inv = st.get("INVENTORY")
    ap = st.get("ACCOUNTS_PAYABLE")

    if ta is not None and tl is not None:
        st["lev"] = np.where(ta > 0, tl / ta, np.nan)
    if ta is not None and "PARENT_NETPROFIT_TTM" in st:
        st["roa"] = np.where(ta > 0, st["PARENT_NETPROFIT_TTM"] / ta, np.nan)
    if ta is not None:
        # 应计项目：非现金营运资本的年度变化 / 总资产
        wc = None
        if ar is not None:
            wc = ar.fillna(0.0)
        if inv is not None:
            wc = (wc if wc is not None else 0.0) + inv.fillna(0.0)
        if ap is not None:
            wc = (wc if wc is not None else 0.0) - ap.fillna(0.0)
        if wc is not None:
            st["_wc"] = wc
            st["_wc_prev"] = calendar_report_lag(st, st["_wc"], code_col="code")
            st["acc"] = np.where(ta > 0, (st["_wc"] - st["_wc_prev"]) / ta, np.nan)
            st = st.drop(columns=["_wc", "_wc_prev"])
        # 资产增长率（同比）
        st["_ta_prev"] = calendar_report_lag(st, st["TOTAL_ASSETS"], code_col="code")
        st["agr"] = np.where(st["_ta_prev"] > 0, ta / st["_ta_prev"] - 1.0, np.nan)
        st = st.drop(columns=["_ta_prev"])

    # 连续盈利改善的季度数（论文的 nincr，取值 0–4）
    if "PARENT_NETPROFIT_TTM" in st:
        prior = calendar_report_lag(st, st["PARENT_NETPROFIT_TTM"], code_col="code")
        st["_ni_yoy"] = st["PARENT_NETPROFIT_TTM"] - prior
        rise = (st["_ni_yoy"] > 0).astype(int)

        def streak(s: pd.Series) -> pd.Series:
            out = np.zeros(len(s), dtype=int)
            run = 0
            for i, v in enumerate(s.to_numpy()):
                run = run + 1 if v else 0
                out[i] = min(run, 4)
            return pd.Series(out, index=s.index)

        st["nincr"] = rise.groupby(st["code"]).transform(streak) / 4.0
        st = st.drop(columns=["_ni_yoy"])
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-report", action="store_true")
    ap.add_argument("--allow-unverified-vintages", action="store_true")
    args = ap.parse_args()
    require_financial_vintage_opt_in(args.allow_unverified_vintages)
    ensure_derived_directory()

    st = load_statements()
    print(f"报表 {len(st):,} 行 / {st['code'].nunique():,} 个代码 / "
          f"{st['REPORT_DATE'].nunique()} 个报告期")

    st = add_ttm(st)
    st = add_ratios(st)

    # 市值来自 features_monthly 的 mvel1（log 市值）
    feat = pd.read_csv(DERIVED / "features_monthly.csv", dtype={"code": str},
                       usecols=["code", "month", "market_cap"])
    # features_monthly 的 code 带 sh/sz/bj 前缀（如 sh600000），财报表是裸 6 位
    feat["code6"] = feat["code"].str[-6:]
    feat["mc"] = raw_market_cap(feat)
    feat["date"] = pd.to_datetime(feat["month"] + "-01") + pd.offsets.MonthEnd(0)
    print(f"面板 {len(feat):,} 行 / {feat['month'].nunique()} 个月")

    # 市值相关的估值比率，先挂在报告行上（用报告公告月的市值近似）
    st["date"] = st["NOTICE_DATE"]
    st = st.rename(columns={"code": "code6"})

    cols = ["code6", "date", "NOTICE_DATE", "REPORT_DATE",
            "TOTAL_EQUITY", "PARENT_NETPROFIT_TTM", "TOTAL_OPERATE_INCOME_TTM",
            "NETCASH_OPERATE_TTM", "roa", "lev", "agr", "acc", "nincr"]
    cols = [c for c in cols if c in st.columns]
    right = st[cols].sort_values("date").reset_index(drop=True)

    # 先把市值并到面板上，再按公告日向后匹配
    left = feat[["code", "code6", "month", "date", "mc"]].sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(left, right.drop(columns=["date"]),
                           left_on="date", right_on="NOTICE_DATE",
                           by="code6", direction="backward",
                           tolerance=pd.Timedelta(days=400))
    merged = merged.dropna(subset=["NOTICE_DATE"])

    te = merged.get("TOTAL_EQUITY")
    if te is not None:
        merged["bm"] = np.where(merged["mc"] > 0, te / merged["mc"], np.nan)
    if "PARENT_NETPROFIT_TTM" in merged:
        merged["ep"] = np.where(merged["mc"] > 0,
                                merged["PARENT_NETPROFIT_TTM"] / merged["mc"], np.nan)
    if "TOTAL_OPERATE_INCOME_TTM" in merged:
        merged["sp"] = np.where(merged["mc"] > 0,
                                merged["TOTAL_OPERATE_INCOME_TTM"] / merged["mc"], np.nan)
    if "NETCASH_OPERATE_TTM" in merged:
        merged["cfp"] = np.where(merged["mc"] > 0,
                                 merged["NETCASH_OPERATE_TTM"] / merged["mc"], np.nan)

    keep = ["code", "month"] + [f for f in OUT_FEATURES if f in merged.columns]
    out = merged[keep].copy()

    # 逐月横截面缩尾 1%/99%（与行情类特征同一口径）
    for f in OUT_FEATURES:
        if f not in out:
            continue
        q1 = out.groupby("month")[f].transform(lambda s: s.quantile(0.01))
        q99 = out.groupby("month")[f].transform(lambda s: s.quantile(0.99))
        out[f] = out[f].clip(q1, q99)

    out.to_csv(DERIVED / "fundamentals_monthly.csv", index=False, encoding="utf-8-sig")
    (DERIVED / "fundamentals_manifest.json").write_text(json.dumps({
        "schema": "gkx-financials/1", "author": "Gilmour Jia",
        "status": "exploratory_unverified_vendor_vintages",
        "explicit_opt_in": True, "point_in_time_certified": False,
        "remaining_limitations": ["original_statement_vintages_unavailable", "source_field_definitions_need_validation"],
        "rows": len(out)
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {DERIVED / 'fundamentals_monthly.csv'}：{len(out):,} 行 / "
          f"{out['code'].nunique():,} 只 / {out['month'].nunique()} 个月")

    if args.dump_report:
        lines = ["# A 股财报类特征面板", "",
                 f"面板 {len(out):,} 行，{out['code'].nunique():,} 只，"
                 f"{out['month'].min()} → {out['month'].max()}", "",
                 "| 特征 | 非缺失率 | 均值 | 标准差 | 5% | 中位数 | 95% |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for f in OUT_FEATURES:
            if f not in out:
                continue
            s = out[f]
            lines.append(f"| {f} | {s.notna().mean():.3f} | {s.mean():.4f} | {s.std():.4f} | "
                         f"{s.quantile(0.05):.4f} | {s.median():.4f} | {s.quantile(0.95):.4f} |")
        (DERIVED / "fundamentals_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"写出 {DERIVED / 'fundamentals_report.md'}")


if __name__ == "__main__":
    main()
