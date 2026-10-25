"""按报告期批量抓三大报表的关键科目（东方财富 datacenter）。

三张表分开取，按 (SECURITY_CODE, REPORT_DATE) 合并：
  RPT_DMSK_FN_BALANCE   资产负债表
  RPT_DMSK_FN_INCOME    利润表
  RPT_DMSK_FN_CASHFLOW  现金流量表

只保留构造特征需要的科目——三张表合计 151 个字段，全存会让 CSV 膨胀到几百 MB。

用法：
    python fetch_statements.py --start 2008 --end 2025
    python fetch_statements.py --start 2008 --end 2025 --resume
输出：data/statements.csv
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
DATA = BASE / "data"
API = "https://datacenter-web.eastmoney.com/api/data/v1/get"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://data.eastmoney.com/",
}

TABLES = {
    "balance": ("RPT_DMSK_FN_BALANCE", [
        "SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE",
        "TOTAL_ASSETS", "TOTAL_LIABILITIES", "TOTAL_EQUITY",
        "INVENTORY", "ACCOUNTS_RECE", "ACCOUNTS_PAYABLE",
        "FIXED_ASSET", "MONETARYFUNDS", "DEBT_ASSET_RATIO",
    ]),
    "income": ("RPT_DMSK_FN_INCOME", [
        "SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE",
        "PARENT_NETPROFIT", "DEDUCT_PARENT_NETPROFIT",
        "TOTAL_OPERATE_INCOME", "OPERATE_PROFIT", "TOTAL_PROFIT",
        "OPERATE_COST", "SALE_EXPENSE", "MANAGE_EXPENSE", "FINANCE_EXPENSE",
    ]),
    "cashflow": ("RPT_DMSK_FN_CASHFLOW", [
        "SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE",
        "NETCASH_OPERATE", "NETCASH_INVEST", "NETCASH_FINANCE",
        "CONSTRUCT_LONG_ASSET",
    ]),
}


def _get(url: str, timeout: int = 25, tries: int = 4) -> dict:
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as exc:                      # noqa: BLE001
            last = exc
            time.sleep(1.5 * 2 ** k)
    raise last


def _page(report: str, columns: list[str], report_date: str, page: int,
          page_size: int = 500) -> dict:
    params = {
        "reportName": report,
        "columns": ",".join(columns),
        "filter": f"(REPORT_DATE='{report_date}')",
        "pageNumber": str(page),
        "pageSize": str(page_size),
        "sortColumns": "SECURITY_CODE",
        "sortTypes": "1",
        "source": "WEB",
        "client": "WEB",
    }
    return _get(API + "?" + urllib.parse.urlencode(params))


def fetch_table(report: str, columns: list[str], report_date: str,
                sleep: float = 0.3) -> pd.DataFrame:
    first = _page(report, columns, report_date, 1)
    result = first.get("result") or {}
    rows = list(result.get("data") or [])
    pages = int(result.get("pages") or 1)
    for page in range(2, pages + 1):
        time.sleep(sleep)
        again = _page(report, columns, report_date, page)
        rows.extend((again.get("result") or {}).get("data") or [])
    return pd.DataFrame(rows)


def quarters(start_year: int, end_year: int) -> list[str]:
    return [f"{y}-{md}" for y in range(start_year, end_year + 1)
            for md in ("03-31", "06-30", "09-30", "12-31")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2008)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "statements.csv"
    frames: list[pd.DataFrame] = []
    done: set[str] = set()

    if args.resume and out.exists():
        old = pd.read_csv(out, dtype={"SECURITY_CODE": str})
        frames.append(old)
        done = set(old["REPORT_DATE"].astype(str).str.slice(0, 10).unique())
        print(f"续跑：已有 {len(old)} 行，覆盖 {len(done)} 个报告期")

    periods = quarters(args.start, args.end)
    for i, rd in enumerate(periods, 1):
        if rd in done:
            continue
        merged: pd.DataFrame | None = None
        try:
            for name, (report, cols) in TABLES.items():
                df = fetch_table(report, cols, rd, sleep=args.sleep)
                keep = [c for c in dict.fromkeys(cols) if c in df.columns]
                if merged is None:
                    merged = df[keep].copy()
                else:
                    add = df[keep].copy()
                    add = add.drop(columns=[c for c in add.columns if c.endswith("NOTICE_DATE")], errors="ignore")
                    merged = merged.merge(add, on=["SECURITY_CODE", "REPORT_DATE"], how="outer")
                print(f"  {rd} {name}: {len(df)} 行", flush=True)
                time.sleep(args.sleep)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {rd} 失败：{type(exc).__name__}: {exc}", flush=True)
            continue

        if merged is not None and len(merged):
            frames.append(merged)
        print(f"[{i:3d}/{len(periods)}] {rd}: {0 if merged is None else len(merged)} 家", flush=True)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        done.add(rd)

    if not frames:
        print("没有抓到任何数据")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["SECURITY_CODE", "REPORT_DATE"])
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n合计 {len(df)} 行，{df['SECURITY_CODE'].nunique()} 个代码，"
          f"{df['REPORT_DATE'].nunique()} 个报告期")
    print(f"写到 {out}")


if __name__ == "__main__":
    main()
