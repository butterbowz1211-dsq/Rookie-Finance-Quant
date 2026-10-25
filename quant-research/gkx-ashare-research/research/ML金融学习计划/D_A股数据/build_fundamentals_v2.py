"""按论文 Table A.6 的 B 组清单构造 57 个公司特征（第二版）。

与 `build_fundamentals.py` 的区别
--------------------------------
旧版只用 `statements.csv`（东财 DMSK 摘要表，21 个科目），只能做 9 个比率。
新版把 `fetch_statements_full.py` 抓的**完整三张报表 + 主要指标表**接进来：

    full_bal.csv   41 列（含长短借款、在建工程、递延税款、应付薪酬…）
    full_inc.csv   23 列（含 RESEARCH_EXPENSE 研发费用）
    full_cf.csv    18 列（含 FA_IR_DEPR 折旧摊销、PAY_ALL_TAX 实缴税费）
    full_main.csv  18 列（含 STAFF_NUM 员工数、ROIC、TAXRATE）

并保留 `statements.csv` 作为**spine**：它覆盖全部 5,612 个代码（含 97 家银行/券商/保险
——这些公司用 `GBALANCE` 一般工商业模板查不到，只在摘要表里有基本科目）。

口径约定
--------
* 中国财报的利润表与现金流量表是**年内累计**（YTD），必须转成 TTM：
  Q4 即全年；Q1–Q3 用 `TTM = YTD_本期 + 上年年报 − 上年同期 YTD`。
* 资产负债表是**时点值**，不转 TTM。
* 使用保守公告日期映射到月末；行业聚合与市值分母在月度可见截面计算。
* 目前供应商缓存的历史版本未验证，默认拒绝生成；探索性使用须显式 opt-in。
* divi/divo 已停用，等待可核验的时点分红定义与记录。
* 比率的分母若为 0 或缺失，结果为 NaN（不填 0，避免制造假信号）。

用法
----
  python build_fundamentals_v2.py
  python build_fundamentals_v2.py --allow-unverified-vintages --dump-report

输出
----
  data/fundamentals_v2_monthly.csv     code, month, <57 列>
  data/fundamentals_v2_report.md       覆盖率与描述统计
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from data_contract import ensure_derived_directory
from data_contract import (DATA, DERIVED, CLEAN, calendar_report_lag, ttm_values,
                           raw_market_cap, require_financial_vintage_opt_in)

HERE = pathlib.Path(__file__).resolve().parent


# 需要转 TTM 的流量科目（利润表 + 现金流量表）
FLOW_BAL: list[str] = []          # 资产负债表都是时点值
FLOW_INC = [
    "TOTAL_OPERATE_INCOME", "OPERATE_COST", "TOTAL_OPERATE_COST", "OPERATE_TAX_ADD",
    "SALE_EXPENSE", "MANAGE_EXPENSE", "RESEARCH_EXPENSE", "FINANCE_EXPENSE",
    "OPERATE_PROFIT", "TOTAL_PROFIT", "INCOME_TAX", "NETPROFIT", "PARENT_NETPROFIT",
    "DEDUCT_PARENT_NETPROFIT", "INVEST_INCOME", "FE_INTEREST_EXPENSE",
]
FLOW_CF = [
    "NETCASH_OPERATE", "NETCASH_INVEST", "NETCASH_FINANCE", "CONSTRUCT_LONG_ASSET",
    "ASSIGN_DIVIDEND_PORFIT", "PAY_ALL_TAX", "FA_IR_DEPR", "IA_AMORTIZE",
    "PAY_STAFF_CASH", "SALES_SERVICES",
]
FLOW = FLOW_INC + FLOW_CF

# 特征清单（与 特征缺口清单.md 的 B 组对应）
FEATURES = [
    "absacc", "bm_ia", "cash", "cashdebt", "cashpr", "cfp_ia", "chatoia", "chempia",
    "chinv", "chpmia", "chtx", "cinvest", "currat", "depr", "divi", "divo", "egr",
    "gma", "grCAPX", "grltnoa", "herf", "hire", "invest", "lgr", "mve_ia",
    "operprof", "orgcap", "pchcapx_ia", "pchcurrat", "pchdepr", "pchgm_pchsale",
    "pchquick", "pchsale_pchinvt", "pchsale_pchrect", "pchsale_pchxsga",
    "pchsaleinv", "pctacc", "ps", "quick", "rd", "rd_mve", "rd_sale", "realestate",
    "roaq", "roavol", "roeq", "roic", "rsup", "salecash", "saleinv", "salerec",
    "sgr", "stdacc", "stdcf", "tang", "tb",
]


# --------------------------------------------------------------------------- #
# 读取与合并
# --------------------------------------------------------------------------- #
def load_full() -> pd.DataFrame:
    """Merge source snapshots conservatively using the latest component timestamp.

    These caches do not establish original-vintage availability; main() requires
    explicit exploratory opt-in. A NOTICE_DATE from one table cannot certify all.
    """
    parts = []
    keys = ["SECURITY_CODE", "REPORT_DATE"]
    for tag in ("bal", "inc", "cf", "main"):
        df = pd.read_csv(DATA / f"full_{tag}.csv", dtype={"SECURITY_CODE": str})
        for c in ("REPORT_DATE", "NOTICE_DATE"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df = df.dropna(subset=keys + ["NOTICE_DATE"]).sort_values(keys + ["NOTICE_DATE"])
        df = df.drop_duplicates(keys, keep="last")
        df = df.drop(columns=["REPORT_TYPE"], errors="ignore")
        df = df.rename(columns={"NOTICE_DATE": f"_notice_{tag}"})
        parts.append(df)
    out = parts[0]
    for nxt in parts[1:]:
        dup = [c for c in nxt if c in out and c not in keys]
        out = out.merge(nxt.drop(columns=dup), on=keys, how="outer", validate="one_to_one")
    notices = [c for c in out if c.startswith("_notice_")]
    out["NOTICE_DATE"] = out[notices].max(axis=1)
    return out.drop(columns=notices)


def load_spine() -> pd.DataFrame:
    """摘要表作为 spine：覆盖全部代码，含 97 家金融企业。"""
    st = pd.read_csv(DATA / "statements.csv", dtype={"SECURITY_CODE": str})
    st["REPORT_DATE"] = pd.to_datetime(st["REPORT_DATE"])
    st["NOTICE_DATE"] = pd.to_datetime(st["NOTICE_DATE"], errors="coerce")
    return st.dropna(subset=["NOTICE_DATE"]).sort_values("NOTICE_DATE").drop_duplicates(subset=["SECURITY_CODE", "REPORT_DATE"], keep="last")


def combine(spine: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """spine 左连接 full；full 缺失的科目回落到 spine。"""
    overlap = [c for c in spine.columns
               if c in full.columns and c not in ("SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE")]
    df = spine.merge(full, on=["SECURITY_CODE", "REPORT_DATE"], how="left",
                     suffixes=("", "_f"))
    for c in overlap:
        df[c] = df[c].where(df[c].notna(), df[f"{c}_f"])
        df = df.drop(columns=[f"{c}_f"])
    df["NOTICE_DATE"] = df[["NOTICE_DATE", "NOTICE_DATE_f"]].max(axis=1)
    if "NOTICE_DATE_f" in df.columns:
        df = df.drop(columns=["NOTICE_DATE_f"])
    return df


# --------------------------------------------------------------------------- #
# 累计 → TTM
# --------------------------------------------------------------------------- #
def to_ttm(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """TTM with exact periods, missing dependencies, and conservative timestamps."""
    df = df.sort_values(["SECURITY_CODE", "REPORT_DATE"]).copy()
    source = df.copy()
    dates = [df["NOTICE_DATE"]]
    for c in cols:
        df[c], available = ttm_values(source, c)
        dates.append(available)
    df["NOTICE_DATE"] = pd.concat(dates, axis=1).max(axis=1)
    # Lagged fields and rolling quantities also inherit earlier recorded revisions.
    df["NOTICE_DATE"] = df.groupby("SECURITY_CODE")["NOTICE_DATE"].cummax()
    return df


# --------------------------------------------------------------------------- #
# 特征计算
# --------------------------------------------------------------------------- #
def _div(a, b):
    """Safe division, including unavailable optional fields."""
    idx = getattr(a, "index", None)
    if idx is None:
        idx = getattr(b, "index", None)
    if idx is not None:
        a = pd.to_numeric(pd.Series(a, index=idx), errors="coerce")
        b = pd.to_numeric(pd.Series(b, index=idx), errors="coerce")
        return (a / b).where(b.notna() & b.ne(0)).replace([np.inf, -np.inf], np.nan)
    if a is None or b is None or b == 0:
        return np.nan
    return a / b


def _lag(df: pd.DataFrame, col: str, n: int = 4) -> pd.Series:
    """同一股票往前 n 个报告期（默认 4 期 = 一年）。"""
    return calendar_report_lag(df, df[col], quarters=n)


def current_assets(df: pd.DataFrame) -> pd.Series:
    """流动资产合计。

    注意：东财完整表里的 `CURRENT_ASSET_BALANCE` 绝大多数行是 0（占位字段），
    直接用它算流动比率会得到 0/0。所以按科目加总重建。
    """
    parts = ["MONETARYFUNDS", "ACCOUNTS_RECE", "NOTE_ACCOUNTS_RECE", "INVENTORY",
             "OTHER_CURRENT_ASSET"]
    have = [c for c in parts if c in df.columns]
    s = sum(df[c].fillna(0.0) for c in have)
    return s.where(s > 0, np.nan)


def current_liab(df: pd.DataFrame) -> pd.Series:
    """流动负债合计，同样按科目加总。"""
    parts = ["SHORT_LOAN", "ACCOUNTS_PAYABLE", "NOTE_ACCOUNTS_PAYABLE",
             "ADVANCE_RECEIVABLES", "TAX_PAYABLE", "STAFF_SALARY_PAYABLE",
             "NONCURRENT_LIAB_1YEAR", "OTHER_CURRENT_LIAB", "DIVIDEND_PAYABLE"]
    have = [c for c in parts if c in df.columns]
    s = sum(df[c].fillna(0.0) for c in have)
    return s.where(s > 0, np.nan)


def _shift4(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    """按股票把任意序列往回挪 4 个报告期。"""
    return calendar_report_lag(df, s)


def _score(cond: pd.Series, base: pd.Series) -> pd.Series:
    """条件成立记 1、不成立记 0；`base` 缺失时记 NaN（不把缺失当 0）。"""
    s = pd.Series(np.where(cond.fillna(False), 1.0, 0.0), index=cond.index)
    return s.where(pd.Series(base, index=cond.index).notna())


def industry_map() -> pd.DataFrame:
    """Historical monthly classifications; no static-industry fallback."""
    path = CLEAN / "stock_industry_monthly.csv"
    if not path.exists():
        return pd.DataFrame(columns=["code6", "month", "industry_code"])
    ind = pd.read_csv(path, dtype=str)
    ind["code6"] = ind["code"].str.split(".").str[0].str[-6:].str.zfill(6)
    return ind[["code6", "month", "industry_code"]].drop_duplicates(["code6", "month"])


def add_industry_adj(df: pd.DataFrame, col: str, ind, new: str) -> None:
    """Keep the base value; adjust only AFTER publication-date monthly alignment."""
    df[new] = df[col]


def compute(df: pd.DataFrame, mc: pd.DataFrame, ind: pd.Series) -> pd.DataFrame:
    """在报告期层面算出全部特征。mc 是 (code, month) → 市值，用于市值类比率。"""
    TA = df["TOTAL_ASSETS"]
    TA_l = _lag(df, "TOTAL_ASSETS")
    EQ_l = _lag(df, "TOTAL_EQUITY")
    SALE = df["TOTAL_OPERATE_INCOME"]
    SALE_l = _lag(df, "TOTAL_OPERATE_INCOME")
    NI = df["PARENT_NETPROFIT"]
    OI = df["OPERATE_PROFIT"]
    OC = df["OPERATE_COST"]

    # --- 应计与现金流 ---
    CA = current_assets(df)
    CL = current_liab(df)
    df["_ca"] = CA
    df["_cl"] = CL
    d_ca = CA - _shift4(df, CA)
    d_cash = df["MONETARYFUNDS"].fillna(0) - _lag(df, "MONETARYFUNDS").fillna(0)
    d_cl = CL - _shift4(df, CL)
    sl = df.get("SHORT_LOAN", pd.Series(np.nan, index=df.index))
    d_std = sl.fillna(0) - calendar_report_lag(df, sl).fillna(0)
    tp = df.get("TAX_PAYABLE", pd.Series(np.nan, index=df.index))
    d_tp = tp.fillna(0) - calendar_report_lag(df, tp).fillna(0)
    acc = (d_ca - d_cash) - (d_cl - d_std) - d_tp
    df["pctacc"] = _div(acc, TA_l)
    df["absacc"] = df["pctacc"].abs()

    # --- 现金类 ---
    df["cash"] = _div(df["MONETARYFUNDS"], TA)
    debt = df.get("SHORT_LOAN", pd.Series(0.0, index=df.index)).fillna(0) + \
        df.get("LONG_LOAN", pd.Series(0.0, index=df.index)).fillna(0)
    df["cashdebt"] = _div(NI + df["FA_IR_DEPR"], debt.replace(0, np.nan))
    df["cashpr"] = _div(mc + df["TOTAL_LIABILITIES"] - TA, df["MONETARYFUNDS"])

    # --- 流动性与杠杆 ---
    df["currat"] = _div(CA, CL)
    df["quick"] = _div(CA - df["INVENTORY"].fillna(0), CL)
    df["lgr"] = _div(
        (df.get("LONG_LOAN", 0).fillna(0) + df.get("BOND_PAYABLE", 0).fillna(0)),
        (_lag(df, "LONG_LOAN").fillna(0) + _lag(df, "BOND_PAYABLE").fillna(0))
        .replace(0, np.nan)) - 1

    # --- 增长 ---
    df["egr"] = _div(df["TOTAL_PARENT_EQUITY"] if "TOTAL_PARENT_EQUITY" in df
                     else df["TOTAL_EQUITY"], EQ_l) - 1
    cap = df.get("FIXED_ASSET", pd.Series(0.0, index=df.index)).fillna(0) + \
        df.get("CIP", pd.Series(0.0, index=df.index)).fillna(0)
    cap_l = _lag(df, "FIXED_ASSET").fillna(0) + _lag(df, "CIP").fillna(0)
    df["grCAPX"] = _div(cap, cap_l.replace(0, np.nan)) - 1
    df["cinvest"] = _div(cap - cap_l, TA_l)
    df["invest"] = _div(TA - TA_l, TA_l) - _div(
        df["TOTAL_LIABILITIES"] - _lag(df, "TOTAL_LIABILITIES"), TA_l)
    df["sgr"] = _div(SALE, SALE_l) - 1
    ltnoa = TA - df.get("CURRENT_ASSET_BALANCE", 0).fillna(0) - \
        df.get("INVEST_REALESTATE", 0).fillna(0)
    df["grltnoa"] = _div(ltnoa, _lag(df, "TOTAL_ASSETS").fillna(0)) - \
        _div(_lag(df, "TOTAL_ASSETS").fillna(0) - _lag(df, "CURRENT_ASSET_BALANCE").fillna(0)
             - _lag(df, "INVEST_REALESTATE").fillna(0), _lag(df, "TOTAL_ASSETS").fillna(0))

    # --- 盈利 ---
    df["roaq"] = _div(NI, TA_l)
    df["roeq"] = _div(NI, EQ_l)
    df["gma"] = _div(SALE - OC, TA_l)
    sga = df["SALE_EXPENSE"].fillna(0) + df["MANAGE_EXPENSE"].fillna(0)
    df["operprof"] = _div(SALE - OC - sga - df["FE_INTEREST_EXPENSE"].fillna(0), EQ_l)
    df["orgcap"] = _div(sga, TA_l)
    df["depr"] = _div(df["FA_IR_DEPR"], df.get("FIXED_ASSET"))
    df["tb"] = _div(df["TOTAL_PROFIT"] - df["INCOME_TAX"], df["TOTAL_PROFIT"])
    df["roic"] = df.get("ROIC", pd.Series(np.nan, index=df.index))
    if "ROIC" in df.columns:
        df["roic"] = df["ROIC"]
        df = df.drop(columns=["ROIC"])

    # --- 研发 ---
    rd = df["RESEARCH_EXPENSE"]
    df["rd"] = _div(rd, TA_l)
    df["rd_mve"] = _div(rd, mc)
    df["rd_sale"] = _div(rd, SALE)

    # --- 周转 ---
    df["pchsaleinv"] = _div(SALE, df["INVENTORY"])
    df["salecash"] = _div(SALE, df["MONETARYFUNDS"])
    df["saleinv"] = _div(SALE, df["INVENTORY"])
    df["salerec"] = _div(SALE, df["ACCOUNTS_RECE"])
    df["chinv"] = _div(df["INVENTORY"], TA) - _div(_lag(df, "INVENTORY"), TA_l)
    df["realestate"] = _div(df.get("INVEST_REALESTATE"), TA)

    # --- 有形性 ---
    df["tang"] = _div(
        df["MONETARYFUNDS"].fillna(0) + 0.715 * df["ACCOUNTS_RECE"].fillna(0)
        + 0.547 * df["INVENTORY"].fillna(0) + 0.535 * df.get("FIXED_ASSET", 0).fillna(0),
        TA)

    # --- 百分比变化类 ---
    def pch(col: str) -> pd.Series:
        return _div(df[col], _lag(df, col)) - 1

    p_sale = pch("TOTAL_OPERATE_INCOME")
    df["pchdepr"] = pch("FA_IR_DEPR")
    gm = _div(SALE - OC, SALE)
    df["pchgm_pchsale"] = (gm - _shift4(df, gm)) - p_sale
    df["pchsale_pchinvt"] = p_sale - pch("INVENTORY")
    df["pchsale_pchrect"] = p_sale - pch("ACCOUNTS_RECE")
    df["pchsale_pchxsga"] = p_sale - pch("SALE_EXPENSE")
    pcapx = _div(cap, cap_l.replace(0, np.nan)) - 1
    df["pchcapx_ia"] = pcapx
    add_industry_adj(df, "pchcapx_ia", ind, "pchcapx_ia")

    # 流动比率/速动的百分比变化（用已算好的 currat / quick）
    df["pchcurrat"] = _div(df["currat"], _lag(df, "currat")) - 1
    df["pchquick"] = _div(df["quick"], _lag(df, "quick")) - 1

    # --- 行业调整类 ---
    ato = _div(SALE, TA_l)
    df["chatoia"] = ato - _shift4(df, ato)
    add_industry_adj(df, "chatoia", ind, "chatoia")

    df["chpmia"] = gm - _shift4(df, gm)
    add_industry_adj(df, "chpmia", ind, "chpmia")

    df["chempia"] = _div(df.get("STAFF_NUM"), _lag(df, "STAFF_NUM")) - 1
    add_industry_adj(df, "chempia", ind, "chempia")
    df["hire"] = _div(df.get("STAFF_NUM"), _lag(df, "STAFF_NUM")) - 1

    df["chtx"] = _div(df.get("TAX_PAYABLE"), TA) - _div(_lag(df, "TAX_PAYABLE"), TA_l)

    # --- Piotroski F-score（9 项，缺失不记 0） ---
    cfo = _div(df["NETCASH_OPERATE"], TA_l)
    lev_now = _div(df["TOTAL_LIABILITIES"], TA)
    lev_pre = _div(_lag(df, "TOTAL_LIABILITIES"), TA_l)
    sh = df.get("SHARE_CAPITAL", pd.Series(np.nan, index=df.index))
    gm_pre = _shift4(df, gm)
    ato_pre = _shift4(df, ato)
    f = pd.DataFrame(index=df.index)
    f["roa_pos"] = _score(df["roaq"] > 0, df["roaq"])
    f["cfo_pos"] = _score(cfo > 0, cfo)
    f["droa"] = _score(df["roaq"] > _lag(df, "roaq"), _lag(df, "roaq"))
    f["accrual"] = _score(cfo > df["roaq"], df["roaq"])
    f["dlev"] = _score(lev_now < lev_pre, lev_pre)
    f["dcurr"] = _score(df["currat"] > _shift4(df, df["currat"]), _shift4(df, df["currat"]))
    f["shares"] = _score(sh <= _shift4(df, sh), _shift4(df, sh))
    f["dgm"] = _score(gm > gm_pre, gm_pre)
    f["dato"] = _score(ato > ato_pre, ato_pre)
    df["ps"] = f.sum(axis=1, min_count=6)

    # --- 波动类（8 期滚动） ---
    for col, src in (("roavol", "roaq"), ("stdacc", "pctacc"), ("stdcf", "NETCASH_OPERATE")):
        s = df[src] if src != "NETCASH_OPERATE" else _div(df[src], TA_l)
        df[col] = s.groupby(df["SECURITY_CODE"]).transform(
            lambda x: x.rolling(8, min_periods=4).std())

    # --- 收入意外 ---
    df["rsup"] = _div(SALE - SALE_l, SALE_l.abs())

    # --- 股本与分红 ---
    sh = df.get("SHARE_CAPITAL")
    df["chcsho"] = _div(sh, _lag(df, "SHARE_CAPITAL")) - 1 if sh is not None else np.nan

    return df


def add_dividend_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude flags that originally read future-year ex-dividend events."""
    df["divi"] = np.nan
    df["divo"] = np.nan
    return df


def add_herf(df: pd.DataFrame, ind) -> pd.DataFrame:
    # Industry concentration is computed on the available monthly cross-section.
    df["herf"] = np.nan
    return df


def monthly_industry_adjustments(out: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    out = out.merge(ind, on=["code6", "month"], how="left", validate="many_to_one")
    for col in ("pchcapx_ia", "chatoia", "chpmia", "chempia", "bm_ia", "cfp_ia", "mve_ia"):
        med = out.groupby(["month", "industry_code"])[col].transform("median")
        out[col] = out[col] - med
    # Each firm contributes its latest available TTM sales once, not all quarters.
    sale = out["TOTAL_OPERATE_INCOME"].where(out["TOTAL_OPERATE_INCOME"].ge(0))
    group = [out["month"], out["industry_code"]]
    total = sale.groupby(group).transform("sum").replace(0, np.nan)
    out["herf"] = (sale / total).pow(2).groupby(group).transform("sum")
    out.loc[out["industry_code"].isna() | total.isna(), "herf"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-report", action="store_true")
    ap.add_argument("--allow-unverified-vintages", action="store_true")
    args = ap.parse_args()
    require_financial_vintage_opt_in(args.allow_unverified_vintages)
    ensure_derived_directory()

    print("读取数据…")
    spine = load_spine()
    full = load_full()
    df = combine(spine, full)
    print(f"合并后 {len(df):,} 行 / {df.SECURITY_CODE.nunique():,} 个代码 / "
          f"{df.REPORT_DATE.nunique()} 个报告期")

    uni = pd.read_csv(DATA / "universe_a.csv", dtype=str)
    df = df[df["SECURITY_CODE"].isin(set(uni["code"]))].copy()
    print(f"限 A 股池后 {len(df):,} 行 / {df.SECURITY_CODE.nunique():,} 只")

    df = df.sort_values(["SECURITY_CODE", "REPORT_DATE"]).reset_index(drop=True)
    df = to_ttm(df, [c for c in FLOW if c in df.columns])

    # 市值：把月末市值挂到报告期所在月，用于 cashpr / rd_mve 等。
    # 注意 features_monthly 的 code 带 sh/sz/bj 前缀，而财报表是裸 6 位，
    # 必须用 code6 桥接，否则一行也匹配不上（v1 就在这里踩过 0 行输出的坑）。
    feat = pd.read_csv(DERIVED / "features_monthly.csv",
                       usecols=["code", "month", "market_cap"], dtype={"code": str, "month": str})
    feat["mc"] = raw_market_cap(feat)
    feat["code6"] = feat["code"].str[-6:]
    df["code6"] = df["SECURITY_CODE"].str[-6:]
    df["_ym"] = df["REPORT_DATE"].dt.strftime("%Y-%m")
    df = df.merge(feat[["code6", "month", "mc"]].rename(columns={"month": "_ym"}),
                  on=["code6", "_ym"], how="left")
    print(f"  市值匹配率 {df['mc'].notna().mean():.1%}")

    ind = industry_map()
    df = compute(df, df["mc"], ind)
    df = add_dividend_flags(df)
    df = add_herf(df, ind)

    # 行业调整的估值比率（需要市值）
    df["_bm"] = _div(df["TOTAL_EQUITY"], df["mc"])
    df["_cfp"] = _div(df["NETCASH_OPERATE"], df["mc"])
    add_industry_adj(df, "_bm", ind, "bm_ia")
    add_industry_adj(df, "_cfp", ind, "cfp_ia")
    df["mve_ia"] = np.log(df["mc"])

    # 按公告日映射到月度面板（不按报告期，避免前视）。
    # 同样必须用 code6 桥接：面板的 code 带 sh/sz/bj 前缀，财报表是裸 6 位。
    df = df.dropna(subset=["NOTICE_DATE"]).copy()
    df = df.sort_values("NOTICE_DATE")
    months = feat[["code", "code6", "month", "mc"]].rename(columns={"code": "SECURITY_CODE"}).copy()
    months["_d"] = pd.to_datetime(months["month"] + "-01") + pd.offsets.MonthEnd(0)
    months = months.sort_values("_d")
    raw = ["TOTAL_EQUITY", "NETCASH_OPERATE", "MONETARYFUNDS", "TOTAL_LIABILITIES", "TOTAL_ASSETS", "RESEARCH_EXPENSE", "TOTAL_OPERATE_INCOME"]
    out = pd.merge_asof(months, df[["code6", "NOTICE_DATE"] + FEATURES + raw],
                        left_on="_d", right_on="NOTICE_DATE", by="code6",
                        direction="backward", tolerance=pd.Timedelta(days=500))
    out["bm_ia"] = _div(out["TOTAL_EQUITY"], out["mc"])
    out["cfp_ia"] = _div(out["NETCASH_OPERATE"], out["mc"])
    out["mve_ia"] = np.log(out["mc"])
    out["rd_mve"] = _div(out["RESEARCH_EXPENSE"], out["mc"])
    out["cashpr"] = _div(out["mc"] + out["TOTAL_LIABILITIES"] - out["TOTAL_ASSETS"], out["MONETARYFUNDS"])
    out = monthly_industry_adjustments(out, ind)
    out = out[["SECURITY_CODE", "month"] + FEATURES]
    out = out.rename(columns={"SECURITY_CODE": "code"})

    # 探索性预处理：逐月横截面缩尾；此步骤不等同于 GKX 的排名变换。
    # 财报比率有极重尾：`sgr` 在营收接近 0 的公司上能到几千，
    # `saleinv` 在存货接近 0 时同理。不缩尾会把回归直接拉爆。
    print("逐月横截面缩尾 1%/99% …")
    for c in FEATURES:
        s = pd.to_numeric(out[c], errors="coerce")
        q = out.assign(_s=s).groupby("month")["_s"].quantile([0.01, 0.99]).unstack()
        out[c] = s.clip(lower=out["month"].map(q[0.01]), upper=out["month"].map(q[0.99]))

    p = DERIVED / "fundamentals_v2_monthly.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    (DERIVED / "fundamentals_v2_manifest.json").write_text(json.dumps({
        "schema": "gkx-financials/1", "author": "Gilmour Jia",
        "status": "exploratory_unverified_vendor_vintages",
        "explicit_opt_in": True, "point_in_time_certified": False,
        "remaining_limitations": ["original_statement_vintages_unavailable", "source_field_definitions_need_validation"],
        "rows": len(out)
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {p}  {len(out):,} 行 × {len(FEATURES)} 特征")

    rep = ["# 第二版财报特征：覆盖率与描述统计\n", f"面板 {len(out):,} 行 / "
           f"{out.code.nunique():,} 只 / {out.month.nunique()} 个月\n",
           "| 特征 | 非缺失率 | 均值 | 标准差 | 5% | 中位数 | 95% |", "|---|---:|---:|---:|---:|---:|---:|"]
    for c in FEATURES:
        s = pd.to_numeric(out[c], errors="coerce")
        rep.append(f"| {c} | {s.notna().mean():.3f} | {s.mean():.4f} | {s.std():.4f} "
                   f"| {s.quantile(0.05):.4f} | {s.median():.4f} | {s.quantile(0.95):.4f} |")
    (DERIVED / "fundamentals_v2_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("已写出 data/fundamentals_v2_report.md")

    if args.dump_report:
        print("\n".join(rep[:14]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
