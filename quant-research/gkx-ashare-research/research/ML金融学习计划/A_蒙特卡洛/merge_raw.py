"""
把分片跑出来的 out/raw_<tag>_s<k>.jsonl 合并成一张表。

分片的做法：把 100 次重复拆成 n 份，每份用不同的 --seed0/--reps 和不同的 --tag 后缀
同时跑，最后用这个脚本合并。重复的 (seed, model) 只保留第一条。

用法：
    python merge_raw.py A1_TREE_pc50_a
    python merge_raw.py A1_TREE_pc50_a --out out/tableA1_TREE_pc50_a.csv
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="分片文件共用的前缀，例如 A1_TREE_pc50_a")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = sorted(OUT.glob(f"raw_{args.tag}*.jsonl"))
    if not paths:
        print(f"没有找到 out/raw_{args.tag}*.jsonl")
        return 1

    seen = set()
    order, agg = [], {}
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = (int(rec["seed"]), rec["model"])
            if key in seen:
                continue
            seen.add(key)
            m = rec["model"]
            if m not in agg:
                agg[m] = ([], [])
                order.append(m)
            agg[m][0].append(rec["is"])
            agg[m][1].append(rec["oos"])

    print(f"合并 {len(paths)} 个文件：" + "、".join(p.name for p in paths))
    seeds = sorted({s for s, _ in seen})
    print(f"共 {len(seeds)} 个种子（{seeds[0]}–{seeds[-1]}）\n")

    print(f"{'Model':12s} {'IS R2(%)':>9s} {'OOS R2(%)':>10s} {'OOS sd':>8s}")
    rows = []
    for m in order:
        iv, ov = np.array(agg[m][0]), np.array(agg[m][1])
        rows.append((m, iv.mean(), ov.mean(), ov.std(ddof=1) if len(ov) > 1 else 0.0))
        print(f"{m:12s} {iv.mean():9.2f} {ov.mean():10.2f} {rows[-1][3]:8.2f}")

    dest = pathlib.Path(args.out) if args.out else OUT / f"tableA1_{args.tag}.csv"
    with dest.open("w", encoding="utf-8") as fh:
        fh.write("model,is_mean,is_sd,oos_mean,oos_sd,n\n")
        for m, im, om, osd in rows:
            iv = np.array(agg[m][0])
            n = len(iv)
            isd = iv.std(ddof=1) if n > 1 else 0.0
            fh.write(f"{m},{im:.4f},{isd:.4f},{om:.4f},{osd:.4f},{n}\n")
    print(f"\n已写出 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
