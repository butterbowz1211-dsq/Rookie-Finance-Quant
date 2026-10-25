"""拉取东财「完整」三张报表与主要指标表（不是 DMSK 摘要表）。

背景
----
`fetch_statements.py` 用的是 `RPT_DMSK_FN_BALANCE/INCOME/CASHFLOW`，那是数据中心的
**摘要**版本，只有 57/46/48 个字段，缺研发费用、折旧摊销、长短借款、员工数等，
导致论文 Table A.6 的 94 个特征里有一大半算不出来。

东财另有完整报表接口：

============  ======  ===========================================
reportName    字段数  说明
============  ======  ===========================================
GBALANCE        319   资产负债表（一般工商业）
GINCOME         203   利润表（含 RESEARCH_EXPENSE）
GCASHFLOW       254   现金流量表（含 FA_IR_DEPR 折旧摊销）
MAINFINADATA    165   主要指标（含 STAFF_NUM 员工数、ROIC、TAXRATE）
============  ======  ===========================================

本脚本按报告期分页抓取，只保留 `build_fundamentals.py` 需要的列（避免几百 MB 的 CSV），
支持 `--resume` 断点续跑，每期落盘一次。

用法
----
  python fetch_statements_full.py --start 2008 --end 2025 --resume
  python fetch_statements_full.py --start 2023 --end 2024 --tables bal,inc     # 小样本调试

输出
----
  data/full_bal.csv   资产负债表
  data/full_inc.csv   利润表
  data/full_cf.csv    现金流量表
  data/full_main.csv  主要指标
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.parse
import urllib.request

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
BASE_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# 每张表保留的列。SECURITY_CODE / REPORT_DATE / NOTICE_DATE 三列所有表都有。
KEEP: dict[str, list[str]] = {
    "bal": [
        "SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE", "REPORT_TYPE",
        "TOTAL_ASSETS", "TOTAL_LIABILITIES", "TOTAL_EQUITY", "TOTAL_PARENT_EQUITY",
        "MINORITY_EQUITY", "CURRENT_ASSET_BALANCE", "CURRENT_LIAB_BALANCE",
        "MONETARYFUNDS", "ACCOUNTS_RECE", "NOTE_ACCOUNTS_RECE", "INVENTORY",
        "ACCOUNTS_PAYABLE", "NOTE_ACCOUNTS_PAYABLE", "ADVANCE_RECEIVABLES",
        "FIXED_ASSET", "CIP", "INTANGIBLE_ASSET", "GOODWILL", "INVEST_REALESTATE",
        "LONG_EQUITY_INVEST", "OTHER_EQUITY_INVEST", "DEFER_TAX_ASSET",
        "DEFER_TAX_LIAB", "SHORT_LOAN", "LONG_LOAN", "BOND_PAYABLE",
        "NONCURRENT_LIAB_1YEAR", "TAX_PAYABLE", "STAFF_SALARY_PAYABLE",
        "DIVIDEND_PAYABLE", "SHARE_CAPITAL", "CAPITAL_RESERVE", "UNASSIGN_RPOFIT",
        "OTHER_CURRENT_ASSET", "OTHER_CURRENT_LIAB", "OTHER_PAYABLE", "OTHER_RECE",
    ],
    "inc": [
        "SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE", "REPORT_TYPE",
        "TOTAL_OPERATE_INCOME", "OPERATE_INCOME", "OPERATE_COST",
        "TOTAL_OPERATE_COST", "OPERATE_TAX_ADD", "SALE_EXPENSE", "MANAGE_EXPENSE",
        "RESEARCH_EXPENSE", "FINANCE_EXPENSE", "OPERATE_PROFIT", "TOTAL_PROFIT",
        "INCOME_TAX", "NETPROFIT", "PARENT_NETPROFIT", "DEDUCT_PARENT_NETPROFIT",
        "BASIC_EPS", "INVEST_INCOME", "OTHER_INCOME", "FE_INTEREST_EXPENSE",
    ],
    "cf": [
        "SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE", "REPORT_TYPE",
        "NETCASH_OPERATE", "NETCASH_INVEST", "NETCASH_FINANCE",
        "CONSTRUCT_LONG_ASSET", "ASSIGN_DIVIDEND_PORFIT", "PAY_ALL_TAX",
        "FA_IR_DEPR", "IA_AMORTIZE", "INVENTORY_REDUCE", "OPERATE_RECE_REDUCE",
        "OPERATE_PAYABLE_ADD", "DEFER_TAX", "PAY_STAFF_CASH", "SALES_SERVICES",
    ],
    "main": [
        "SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE",
        "STAFF_NUM", "ROIC", "TAXRATE", "LD", "SD", "TOTAL_SHARE", "AVG_TOI",
        "TOTAL_ASSETS_PK", "TOTAL_EQUITY_PK", "ROEJQ", "XSMLL", "XSJLL",
        "ZCFZL", "TOTALOPERATEREVE", "PARENTNETPROFIT",
    ],
}

REPORT_NAME = {
    "bal": "RPT_F10_FINANCE_GBALANCE",
    "inc": "RPT_F10_FINANCE_GINCOME",
    "cf": "RPT_F10_FINANCE_GCASHFLOW",
    "main": "RPT_F10_FINANCE_MAINFINADATA",
}

PAGE_SIZE = 500


def quarters(start_year: int, end_year: int) -> list[str]:
    """按报告期生成 2008Q1..2025Q4 的日期串。"""
    out = []
    for y in range(start_year, end_year + 1):
        for md in ("03-31", "06-30", "09-30", "12-31"):
            out.append(f"{y}-{md}")
    return out


def _get(params: dict, tries: int = 4, timeout: int = 25) -> dict:
    """带指数退避的 GET。"""
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - 网络错误种类多，统一退避重试
            last = e
            time.sleep(1.5 * (2 ** i))
    raise RuntimeError(f"请求失败 {tries} 次：{last}")


def fetch_period(table: str, period: str, pause: float = 0.15) -> pd.DataFrame:
    """抓一个报告期的一张表。返回只含 KEEP 列的 DataFrame（可能为空）。"""
    cols = KEEP[table]
    frames: list[pd.DataFrame] = []
    page = 1
    total = None
    while True:
        q = {
            "reportName": REPORT_NAME[table],
            "columns": ",".join(cols),
            "pageNumber": page,
            "pageSize": PAGE_SIZE,
            "filter": f"(REPORT_DATE='{period}')",
            "sortColumns": "SECURITY_CODE",
            "sortTypes": 1,
        }
        j = _get(q)
        res = j.get("result")
        if not res or not res.get("data"):
            break
        frames.append(pd.DataFrame(res["data"]))
        if total is None:
            total = int(res.get("count") or 0)
        if page * PAGE_SIZE >= total:
            break
        page += 1
        time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    # 只保留实际存在的列（接口偶尔会少给字段）
    keep = [c for c in cols if c in df.columns]
    return df[keep]


def load_done(path: pathlib.Path) -> set[str]:
    """读取已完成的报告期。"""
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["REPORT_DATE"], dtype=str)
    return set(df["REPORT_DATE"].dropna().unique())


def run_table(table: str, periods: list[str], resume: bool, pause: float) -> None:
    out = DATA / f"full_{table}.csv"
    done = load_done(out) if resume else set()
    todo = [p for p in periods if p not in done]
    print(f"[{table}] 共 {len(periods)} 期，已完成 {len(done)}，待抓 {len(todo)}", flush=True)
    if not todo:
        return
    if not resume and out.exists():
        out.unlink()

    for i, period in enumerate(todo, 1):
        df = fetch_period(table, period, pause=pause)
        if len(df):
            df.to_csv(out, mode="a", header=not out.exists(), index=False,
                      encoding="utf-8-sig")
        print(f"[{table}] {i}/{len(todo)}  {period}  {len(df):5d} 行", flush=True)
        time.sleep(pause)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2008)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--tables", default="bal,inc,cf,main")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pause", type=float, default=0.15)
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    periods = quarters(args.start, args.end)
    print(f"报告期 {periods[0]} → {periods[-1]}，共 {len(periods)} 期")

    for table in [t.strip() for t in args.tables.split(",") if t.strip()]:
        if table not in REPORT_NAME:
            raise SystemExit(f"未知表名 {table}，可选 {list(REPORT_NAME)}")
        run_table(table, periods, args.resume, args.pause)

    print("\n完成。文件：")
    for table in REPORT_NAME:
        p = DATA / f"full_{table}.csv"
        if p.exists():
            n = sum(1 for _ in p.open("rb")) - 1
            print(f"  {p.name:16s} {p.stat().st_size / 1e6:8.1f} MB  {n:,} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
