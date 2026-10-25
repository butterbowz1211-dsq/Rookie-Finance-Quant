"""按报告期批量抓 A 股财务指标（东方财富 datacenter）。

为什么按报告期抓而不是按股票抓：一次请求能拿到某个报告期的 500 家公司，
64 个季度 × 约 24 页 ≈ 1500 次请求即可覆盖全市场十六年，比逐股请求快两个数量级。
另一个作用是顺带拿到**退市公司**的代码——某个历史报告期里有 1 万多个代码，
远超当前的 5,563 只在市公司，差额就是退市与 B 股。

用法：
    python fetch_financials.py --start 2010 --end 2025
    python fetch_financials.py --start 2010 --end 2025 --resume

输出：data/financials.csv（原始行，含 NOTICE_DATE 供做正确的滞后）
      data/universe.csv（全部出现过的代码及其首末报告期）
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
    "SECURITY_CODE", "SECURITY_NAME_ABBR", "TRADE_MARKET",
    "REPORTDATE", "NOTICE_DATE",
    "BASIC_EPS", "DEDUCT_BASIC_EPS", "BPS",
    "TOTAL_OPERATE_INCOME", "PARENT_NETPROFIT",
    "WEIGHTAVG_ROE", "YSTZ", "SJLTZ",
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
        "reportName": "RPT_LICO_FN_CPD",
        "columns": COLUMNS,
        "filter": f"(REPORTDATE='{report_date}')",
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
    out = []
    for y in range(start_year, end_year + 1):
        for md in ("03-31", "06-30", "09-30", "12-31"):
            out.append(f"{y}-{md}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "financials.csv"
    done: set[str] = set()
    frames: list[pd.DataFrame] = []

    if args.resume and out.exists():
        old = pd.read_csv(out, dtype={"SECURITY_CODE": str})
        frames.append(old)
        done = set(old["REPORTDATE"].astype(str).str.slice(0, 10).unique())
        print(f"续跑：已有 {len(old)} 行，覆盖 {len(done)} 个报告期")

    for i, rd in enumerate(quarters(args.start, args.end), 1):
        if rd in done:
            continue
        try:
            rows = fetch_period(rd, sleep=args.sleep)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {rd} 失败：{type(exc).__name__}: {exc}", flush=True)
            continue
        if rows:
            frames.append(pd.DataFrame(rows))
        print(f"[{i:3d}/{len(quarters(args.start, args.end))}] {rd}: {len(rows)} 家", flush=True)
        time.sleep(args.sleep)

        # 每个报告期落一次盘，中途被杀也不丢
        pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        done.add(rd)

    if not frames:
        print("没有抓到任何数据")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["SECURITY_CODE", "REPORTDATE"])
    df.to_csv(out, index=False, encoding="utf-8-sig")

    uni = (df.groupby("SECURITY_CODE")["REPORTDATE"]
             .agg(["min", "max", "count"]).reset_index()
             .rename(columns={"min": "first_report", "max": "last_report", "count": "n_reports"}))
    uni.to_csv(DATA / "universe.csv", index=False, encoding="utf-8-sig")

    print(f"\n合计 {len(df)} 行，{df['SECURITY_CODE'].nunique()} 个代码，"
          f"{df['REPORTDATE'].nunique()} 个报告期")
    print(f"写到 {out}")
    print(f"股票池写到 {DATA / 'universe.csv'}")


if __name__ == "__main__":
    main()
