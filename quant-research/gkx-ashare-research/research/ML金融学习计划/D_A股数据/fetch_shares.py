"""
抓 A 股股本变动历史（东财 `RPT_F10_EH_EQUITY`）。

用途：市值 = 收盘价 × 当期总股本；换手率 = 成交量 / 流通股本。
这张表只记录**股本发生变化**的日期，所以要取"某月市值"必须向前找最近的
`END_DATE`，不能直接按月末日期查。

接口支持按 `END_DATE` 做区间过滤并分页，所以按整个区间一次拉完即可
（约 28.6 万条、573 页）。

字段：
    SECURITY_CODE, END_DATE, CHANGE_REASON,
    TOTAL_SHARES（总股本）, LISTED_A_SHARES（流通 A 股）,
    LIMITED_SHARES（限售股）, UNLIMITED_SHARES（非限售股）

用法：
    python fetch_shares.py --start 2009-01-01 --end 2025-12-31
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
URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT = "RPT_F10_EH_EQUITY"
COLS = ("SECURITY_CODE,END_DATE,CHANGE_REASON,TOTAL_SHARES,TOTAL_SHARES_RATIO,"
        "LISTED_A_SHARES,LIMITED_SHARES,UNLIMITED_SHARES")


def _get(params: dict, timeout: int = 30, tries: int = 5) -> dict:
    url = URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # 网络抖动就退避重试
            last = e
            time.sleep(1.5 ** k)
    raise RuntimeError(f"请求失败：{last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2009-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--out", default=str(DATA / "shares.csv"))
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    filt = f"(END_DATE>='{args.start}')(END_DATE<='{args.end}')"
    base = {"reportName": REPORT, "columns": COLS, "pageSize": args.page_size,
            "sortColumns": "END_DATE", "sortTypes": 1, "source": "WEB",
            "client": "WEB", "filter": filt}

    first = _get({**base, "pageNumber": 1})
    res = first.get("result") or {}
    total, pages = res.get("count"), res.get("pages")
    if not total:
        print("没有数据")
        return 1
    print(f"共 {total:,} 条，{pages} 页（每页 {args.page_size}）")

    frames = [pd.DataFrame(res["data"])]
    for page in range(2, pages + 1):
        d = _get({**base, "pageNumber": page})
        r = d.get("result") or {}
        if r.get("data"):
            frames.append(pd.DataFrame(r["data"]))
        if page % 50 == 0 or page == pages:
            print(f"  {page}/{pages}", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df["END_DATE"] = pd.to_datetime(df["END_DATE"]).dt.strftime("%Y-%m-%d")
    df["SECURITY_CODE"] = df["SECURITY_CODE"].astype(str).str.zfill(6)
    for c in ("TOTAL_SHARES", "LISTED_A_SHARES", "LIMITED_SHARES", "UNLIMITED_SHARES",
              "TOTAL_SHARES_RATIO"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["SECURITY_CODE", "END_DATE"], keep="last")
    df = df.sort_values(["SECURITY_CODE", "END_DATE"])

    out = pathlib.Path(args.out)
    df.to_csv(out, index=False)
    print(f"\n{len(df):,} 行 / {df.SECURITY_CODE.nunique():,} 只，写到 {out}")
    print(df.head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
