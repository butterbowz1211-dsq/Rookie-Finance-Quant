"""构造 A 股的宏观（时间序列）变量，供 §2.1 式 (21) 的 x ⊗ c 设计矩阵使用。

论文的八个宏观变量是 dp、ep、bm、ntis、tbl、tms、dfy、svar。前四个与最后一个
可以用 A 股数据自己算，后三个是债券市场的（期限利差、违约利差、无风险利率），
A 股这边没有对应的公开序列，因此这里只做五个。

构造方式（全部为市值加权，逐月重算）
------------------------------------
    dp_t   = Σ_i MC_it · dy_it  / Σ_i MC_it        （股息率）
    ep_t   = Σ_i MC_it · ep_it  / Σ_i MC_it        （盈利收益率）
    bm_t   = Σ_i MC_it · bm_it  / Σ_i MC_it        （账面市值比）
    ntis_t = Σ_i MC_it · chcsho_it / Σ_i MC_it     （净股本发行）
    svar_t = 过去 12 个月市值加权市场收益的方差      （已实现波动）

前四个用市值加权把个股比率聚合成市场比率，等价于"总量/总量"。
svar 用月度市场收益的 12 个月滚动方差，与论文用日频已实现方差的口径不同，
量级接近但不可直接对齐。

市场特征聚合用当月原始市值；市场收益用前一自然月市值。
缺失特征的股票同时从该变量分子与分母排除。数据版本仍需单独核验。

用法：
    python build_macro.py
输出：data/macro_monthly.csv
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from data_contract import ensure_derived_directory
from data_contract import DATA, DERIVED, raw_market_cap

BASE = pathlib.Path(__file__).resolve().parent

MACROS = ["dp", "ep", "bm", "ntis", "svar"]


def value_weighted(panel: pd.DataFrame, col: str, wcol: str = "mc") -> pd.Series:
    """∑ MC·x / ∑ MC，逐月。"""
    values = pd.to_numeric(panel[col], errors="coerce")
    w = pd.to_numeric(panel[wcol], errors="coerce")
    valid = np.isfinite(values) & np.isfinite(w) & w.gt(0)
    w = w.where(valid)
    num = (panel[col] * w).groupby(panel["month"]).sum(min_count=1)
    den = w.groupby(panel["month"]).sum(min_count=1)
    return num / den


def main() -> None:
    feat = pd.read_csv(DERIVED / "features_monthly.csv", dtype={"code": str},
                       usecols=["code", "month", "market_cap", "dy", "chcsho"])
    feat["mc"] = raw_market_cap(feat)

    fun = pd.read_csv(DERIVED / "fundamentals_monthly.csv", dtype={"code": str})
    panel = feat.merge(fun, on=["code", "month"], how="left")

    out = pd.DataFrame({"month": sorted(panel["month"].unique())}).set_index("month")
    out["dp"] = value_weighted(panel, "dy")
    out["ep"] = value_weighted(panel, "ep")
    out["bm"] = value_weighted(panel, "bm")
    out["ntis"] = value_weighted(panel, "chcsho")

    # 市值加权市场收益：用前一自然月市值权重乘当月未缩尾收益。
    ret = pd.read_csv(DATA / "returns_monthly.csv", dtype={"code": str},
                      usecols=["code", "month", "ret"])
    lag_cap = panel[["code", "month", "mc"]].copy()
    lag_cap["month"] = (pd.PeriodIndex(lag_cap.month, freq="M") + 1).astype(str)
    m = ret.merge(lag_cap, on=["code", "month"], how="left", validate="one_to_one")
    out["mkt_ret"] = value_weighted(m, "ret")

    # 12 个月滚动方差（最少 6 个观测）
    out = out.reindex(pd.period_range(out.index.min(), out.index.max(), freq="M").astype(str))
    out.index.name = "month"
    out["svar"] = out["mkt_ret"].rolling(12, min_periods=6).var()

    out = out.reset_index()
    ensure_derived_directory()
    out.to_csv(DERIVED / "macro_monthly.csv", index=False, encoding="utf-8-sig")

    lines = ["# A 股宏观变量", "",
             f"{len(out)} 个月（{out.month.min()} → {out.month.max()}）", "",
             "| 变量 | 非缺失 | 均值 | 标准差 | 5% | 中位数 | 95% |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for c in MACROS + ["mkt_ret"]:
        s = out[c]
        lines.append(f"| {c} | {s.notna().sum():d} | {s.mean():.4f} | {s.std():.4f} | "
                     f"{s.quantile(0.05):.4f} | {s.median():.4f} | {s.quantile(0.95):.4f} |")
    (DERIVED / "macro_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"写出 {DERIVED / 'macro_monthly.csv'}：{len(out)} 个月，"
          f"{len(out.columns) - 1} 个变量")
    print(out.head(3).to_string(index=False))
    print(out.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
