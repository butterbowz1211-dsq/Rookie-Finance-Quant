"""按报告期批量抓 A 股分红送配（东方财富 datacenter RPT_SHAREBONUS_DET）。

用途：新浪日线是不复权的，需要用除权除息日 + 送转比例 + 每股派息
自己构造含股息的日总收益（见 build_returns.py）。

单位约定（实测）：
  PRETAX_BONUS_RMB  每 10 股税前派息，如 4.20 表示每股 0.42 元
  BONUS_IT_RATIO    每 10 股送转合计，如 4.5 表示每股送转 0.45 股
  BONUS_RATIO / IT_RATIO  其中送股 / 转增，同样按每 10 股
  EX_DIVIDEND_DATE  除权除息日；EQUITY_RECORD_DATE 股权登记日

用法：
    python fetch_dividends.py --start 2010 --end 2025
    python fetch_dividends.py --start 2010 --end 2025 --resume

输出：data/dividends.csv
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

COLUMNS = ",".join([
    "SECURITY_CODE", "SECURITY_NAME_ABBR",
    "REPORT_DATE", "EX_DIVIDEND_DATE", "EQUITY_RECORD_DATE", "NOTICE_DATE",
    "BONUS_IT_RATIO", "BONUS_RATIO", "IT_RATIO", "PRETAX_BONUS_RMB",
    "IMPL_PLAN_PROFILE", "ASSIGN_PROGRESS", "MARKET_TYPE",
])

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://data.eastmoney.com/",
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


def _page(report_date: str, page: int, page_size: int = 500) -> dict:
    params = {
        "reportName": "RPT_SHAREBONUS_DET",
        "columns": COLUMNS,
        "filter": f"(REPORT_DATE='{report_date}')",
        "pageNumber": str(page),
        "pageSize": str(page_size),
        "sortColumns": "SECURITY_CODE",
        "sortTypes": "1",
        "source": "WEB",
        "client": "WEB",
    }
    return _get(API + "?" + urllib.parse.urlencode(params))


def fetch_period(report_date: str, sleep: float = 0.3) -> list[dict]:
    rows: list[dict] = []
    first = _page(report_date, 1)
    result = first.get("result") or {}
    rows.extend(result.get("data") or [])
    pages = int(result.get("pages") or 1)
    for page in range(2, pages + 1):
        time.sleep(sleep)
        again = _page(report_date, page)
        rows.extend((again.get("result") or {}).get("data") or [])
    return rows


def quarters(start_year: int, end_year: int) -> list[str]:
    """分红方案只挂在半年报和年报上，季报没有，所以只查 06-30 与 12-31。"""
    return [f"{y}-{md}" for y in range(start_year, end_year + 1)
            for md in ("06-30", "12-31")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "dividends.csv"
    done: set[str] = set()
    frames: list[pd.DataFrame] = []

    if args.resume and out.exists():
        old = pd.read_csv(out, dtype={"SECURITY_CODE": str})
        frames.append(old)
        done = set(old["REPORT_DATE"].astype(str).str.slice(0, 10).unique())
        print(f"续跑：已有 {len(old)} 行，覆盖 {len(done)} 个报告期")

    periods = quarters(args.start, args.end)
    for i, rd in enumerate(periods, 1):
        if rd in done:
            continue
        try:
            rows = fetch_period(rd, sleep=args.sleep)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {rd} 失败：{type(exc).__name__}: {exc}", flush=True)
            continue
        if rows:
            frames.append(pd.DataFrame(rows))
        print(f"[{i:3d}/{len(periods)}] {rd}: {len(rows)} 条", flush=True)
        time.sleep(args.sleep)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        done.add(rd)

    if not frames:
        print("没有抓到任何数据")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["SECURITY_CODE", "EX_DIVIDEND_DATE", "REPORT_DATE"])
    df.to_csv(out, index=False, encoding="utf-8-sig")

    has_ex = df["EX_DIVIDEND_DATE"].notna().sum()
    print(f"\n合计 {len(df)} 行，{df['SECURITY_CODE'].nunique()} 个代码，"
          f"其中有除权除息日的 {has_ex} 行")
    print(f"写到 {out}")


if __name__ == "__main__":
    main()
