"""把分散在多个 CSV 里的结果拼成完整的 Table A.1 一列，并与论文并排对照。

线性模型、树模型、神经网络是分批跑的（树和 NN 还分片并行），结果落在不同的
`out/tableA1_*.csv` 里。这个脚本按 (case, Pc) 把它们合并，一次打印整列。

用法：
    python combine_a1.py a 50
    python combine_a1.py b 100
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
REF = HERE.parent / "reference" / "GXK_TableA1.csv"

ORDER = ["OLS", "OLS+H", "Ridge", "Ridge+H", "Lasso", "Lasso+H", "ENet", "ENet+H",
         "PCR", "PLS", "GLM", "GLM+H", "RF", "GBRT", "GBRT+H",
         "NN1", "NN2", "NN3", "NN4", "NN5", "Oracle"]


def load_all(case: str, pc: int) -> pd.DataFrame:
    frames = []
    for name in (f"tableA1_A1_pc{pc}_{case}.csv",
                 f"tableA1_A1_TREE_pc{pc}_{case}.csv",
                 f"tableA1_A1_NN_pc{pc}_{case}.csv",
                 f"tableA1_GLM_pc{pc}_{case}.csv"):
        p = OUT / name
        if p.exists():
            d = pd.read_csv(p)
            d["src"] = name
            frames.append(d)
    if not frames:
        raise SystemExit(f"out/ 下没有 pc{pc}_{case} 的任何结果")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["model"], keep="first")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=("a", "b"))
    ap.add_argument("pc", type=int, choices=(50, 100))
    args = ap.parse_args()

    mine = load_all(args.case, args.pc)
    ref = pd.read_csv(REF)
    ref = ref[(ref["case"] == args.case) & (ref["Pc"] == args.pc)]
    piv = ref.pivot_table(index="model", columns="IS_or_OOS", values="R2_percent")

    rows = []
    for m in ORDER:
        sub = mine[mine["model"] == m]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append((m, float(r["is_mean"]), float(r["oos_mean"]),
                     float(piv.loc[m, "IS"]) if m in piv.index else float("nan"),
                     float(piv.loc[m, "OOS"]) if m in piv.index else float("nan"),
                     float(r["oos_sd"]) if "oos_sd" in r and pd.notna(r["oos_sd"]) else float("nan"),
                     int(r["n"]) if "n" in r and pd.notna(r["n"]) else 0))

    print(f"Table A.1 完整列：DGP ({args.case})，Pc={args.pc}")
    print(f"{'模型':10s} {'IS(我)':>8s} {'IS(论文)':>9s} {'ΔIS':>7s} "
          f"{'OOS(我)':>9s} {'OOS(论文)':>10s} {'ΔOOS':>7s} {'sd':>6s} {'n':>5s}")
    print("-" * 83)
    dis, dos = [], []
    for m, im, om, ir, orr, sd, n in rows:
        di, do = im - ir, om - orr
        dis.append(di)
        dos.append(do)
        ntxt = f"{n:5d}" if n else "    —"
        print(f"{m:10s} {im:8.2f} {ir:9.2f} {di:+7.2f} {om:9.2f} {orr:10.2f} {do:+7.2f} "
              f"{sd:6.2f} {ntxt}")
    print("-" * 82)
    print(f"{'平均偏差':10s} {'':8s} {'':9s} {sum(dis)/len(dis):+7.2f} "
          f"{'':9s} {'':10s} {sum(dos)/len(dos):+7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
