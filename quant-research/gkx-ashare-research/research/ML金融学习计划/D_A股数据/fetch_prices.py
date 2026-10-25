"""按股票抓 A 股日线（新浪财经 K 线接口）。

接口：http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/
      CN_MarketData.getKLineData?symbol={sh600000}&scale=240&ma=no&datalen=20000

实测要点（2026-10 于本机）：
  - datalen 没有 1023 上限：20000 会返回该股票全部历史
    （sz000001 得 8,459 根，1991-04-03 起；sh600000 得 6,412 根，1999-11-10 起）
  - 一次请求约 0.4–0.6 秒，返回字段 day/open/high/low/close/volume
  - **价格不复权**（sh600000 2023-05-17 收 7.610，而腾讯前复权为 6.139）
    除权除息的调整在 build_returns.py 里用分红送配表另做
  - 同一时间窗内稳定，未见限流

用法：
    python fetch_prices.py --codes 000001,600000 --start 2010 --end 2025
    python fetch_prices.py --codes-file data/pilot_codes.txt
    python fetch_prices.py --universe data/universe_a.csv --resume

输出：data/prices/<code>.csv（code 为 sh600000 形式），每只一个文件
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
DATA = BASE / "data"
PRICES = DATA / "prices"
API = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://finance.sina.com.cn/",
}

FIELDS = ["date", "open", "high", "low", "close", "volume"]


def to_sina(code: str) -> str:
    """把 600000 / sh600000 / 000001.SZ 统一成 sh600000 或 sz000001。"""
    code = code.strip().lower().replace(".sh", "").replace(".sz", "").replace(".bj", "")
    if code.startswith(("sh", "sz", "bj")):
        return code
    digits = "".join(ch for ch in code if ch.isdigit()).zfill(6)
    if digits.startswith(("60", "68", "51", "58", "11", "90")):
        return "sh" + digits
    if digits.startswith(("00", "30", "20", "12", "15", "16")):
        return "sz" + digits
    if digits.startswith(("43", "83", "87", "92")):
        return "bj" + digits
    return "sz" + digits


def _get(url: str, timeout: int = 25, tries: int = 4) -> bytes:
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:                      # noqa: BLE001
            last = exc
            time.sleep(1.0 * 2 ** k)
    raise last


def fetch_one(code: str, start: int, end: int) -> pd.DataFrame:
    sym = to_sina(code)
    url = f"{API}?symbol={sym}&scale=240&ma=no&datalen=20000"
    raw = _get(url)
    if not raw or raw.strip() in (b"null", b"[]"):
        return pd.DataFrame(columns=FIELDS)
    arr = json.loads(raw)
    if not arr:
        return pd.DataFrame(columns=FIELDS)
    df = pd.DataFrame(arr)
    df = df.rename(columns={"day": "date"})
    df = df[[c for c in FIELDS if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= f"{start}-01-01") & (df["date"] <= f"{end}-12-31")]
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "code", sym)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", type=str, default="")
    ap.add_argument("--codes-file", type=str, default="")
    ap.add_argument("--universe", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    PRICES.mkdir(parents=True, exist_ok=True)

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    elif args.codes_file:
        codes = [c.strip() for c in pathlib.Path(args.codes_file).read_text(encoding="utf-8").split(",") if c.strip()]
    elif args.universe:
        uni = pd.read_csv(args.universe, dtype=str)
        codes = [str(c) for c in uni.iloc[:, 0].tolist()]
    else:
        raise SystemExit("需要 --codes / --codes-file / --universe")

    if args.limit:
        codes = codes[: args.limit]

    t_all = time.time()
    done = skipped = failed = empty = 0
    for i, code in enumerate(codes, 1):
        sym = to_sina(code)
        path = PRICES / f"{sym}.csv"
        if args.resume and path.exists() and path.stat().st_size > 0:
            skipped += 1
            continue
        try:
            df = fetch_one(sym, args.start, args.end)
        except Exception as exc:                      # noqa: BLE001
            print(f"[{i}/{len(codes)}] {sym} 失败 {type(exc).__name__}: {exc}", flush=True)
            failed += 1
            time.sleep(1.0)
            continue
        if df.empty:
            empty += 1
        else:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            done += 1
        if i % 10 == 0 or i == len(codes):
            el = time.time() - t_all
            rate = el / i
            print(f"[{i}/{len(codes)}] {sym} {len(df)} 根 | 完成 {done} 跳过 {skipped} "
                  f"失败 {failed} 空 {empty} | 单只 {rate:.2f}s "
                  f"| 剩余 {max(len(codes)-i,0)*rate/60:.1f} 分钟", flush=True)
        time.sleep(args.sleep)

    print(f"\n完成 {done}，跳过 {skipped}，失败 {failed}，空 {empty}，"
          f"总耗时 {(time.time()-t_all)/60:.1f} 分钟")
    print(f"写到 {PRICES}")


if __name__ == "__main__":
    main()
