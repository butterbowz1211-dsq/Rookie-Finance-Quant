"""抓 A 股公司的行业分类（东财 RPT_F10_BASIC_ORGINFO）。

产出 data/industries.csv，列：
    code, name, ind_csrc, ind_em
其中 ind_csrc 是证监会行业（如「金融业-货币金融服务」），ind_em 是东财行业
（如「金融-银行-股份制与城商行」）。只用 ind_csrc 的一级分类做哑变量时
取「-」前的部分。

用法：
    python fetch_industries.py                # 全量
    python fetch_industries.py --resume       # 已有文件则续
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT = "RPT_F10_BASIC_ORGINFO"
COLUMNS = "SECURITY_CODE,SECURITY_NAME_ABBR,INDUSTRYCSRC1,EM2016"
PAGE_SIZE = 500


def _get(params: dict, tries: int = 4, timeout: int = 25) -> dict:
    q = urllib.parse.urlencode(params)
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(
                f"{URL}?{q}", headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (2**i))
    raise RuntimeError(f"请求失败：{last}")


def fetch_all(pause: float = 0.15, verbose: bool = True) -> pd.DataFrame:
    """分页抓全量，返回 DataFrame。"""
    first = _get(
        {
            "reportName": REPORT,
            "columns": COLUMNS,
            "pageSize": PAGE_SIZE,
            "pageNumber": 1,
        }
    )
    if not first.get("success"):
        raise RuntimeError(f"接口返回失败：{first.get('message')}")
    pages = first["result"]["pages"]
    total = first["result"]["count"]
    if verbose:
        print(f"共 {total} 条，{pages} 页，每页 {PAGE_SIZE}")

    rows = list(first["result"]["data"])
    for p in range(2, pages + 1):
        d = _get(
            {
                "reportName": REPORT,
                "columns": COLUMNS,
                "pageSize": PAGE_SIZE,
                "pageNumber": p,
            }
        )
        if d.get("success") and d["result"]:
            rows.extend(d["result"]["data"])
        if verbose and p % 10 == 0:
            print(f"  {p}/{pages} 页，累计 {len(rows)} 条")
        time.sleep(pause)

    df = pd.DataFrame(rows).rename(
        columns={
            "SECURITY_CODE": "code",
            "SECURITY_NAME_ABBR": "name",
            "INDUSTRYCSRC1": "ind_csrc",
            "EM2016": "ind_em",
        }
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="已有文件则跳过")
    ap.add_argument("--pause", type=float, default=0.15)
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    out = DATA / "industries.csv"
    if args.resume and out.exists():
        print(f"{out} 已存在，跳过（{len(pd.read_csv(out))} 行）")
        return

    df = fetch_all(pause=args.pause)
    df = df.drop_duplicates("code").sort_values("code")
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n已写出 {out}：{len(df)} 行")
    print(f"  证监会一级行业（'-' 前）个数：{df['ind_csrc'].str.split('-').str[0].nunique()}")
    print(f"  东财一级行业（'-' 前）个数：{df['ind_em'].str.split('-').str[0].nunique()}")
    print(df["ind_csrc"].str.split("-").str[0].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
