"""
把一次蒙特卡洛结果与论文 Table A.1 并排对照。

用法：
    python compare_A1.py out/tableA1_A1_pc50_a.csv a 50
    python compare_A1.py out/tableA1_A1_pc50_a.csv a 50 --models OLS,Ridge,Lasso

参数依次为：结果 csv、DGP 情形（a/b）、特征个数（50/100）。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REF = BASE / "reference" / "GXK_TableA1.csv"


def load_ref(case: str, pc: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with REF.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["case"] == case and int(row["Pc"]) == pc:
                out.setdefault(row["model"], {})[row["IS_or_OOS"]] = float(row["R2_percent"])
    return out


def load_mine(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            out[row["model"]] = {
                "IS": float(row["is_mean"]),
                "OOS": float(row["oos_mean"]),
                "IS_sd": float(row["is_sd"]),
                "OOS_sd": float(row["oos_sd"]),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("case", choices=("a", "b"))
    ap.add_argument("pc", type=int, choices=(50, 100))
    ap.add_argument("--models", default="")
    args = ap.parse_args()

    ref = load_ref(args.case, args.pc)
    mine = load_mine(args.result)
    only = {m.strip() for m in args.models.split(",")} if args.models else None

    print(f"Table A.1 对照：DGP ({args.case})，Pc={args.pc}，{args.result.name}")
    print(f"{'模型':<10s}{'IS(我)':>9s}{'IS(论文)':>10s}{'ΔIS':>8s}"
          f"{'OOS(我)':>10s}{'OOS(论文)':>11s}{'ΔOOS':>9s}{'OOS sd':>8s}")
    print("-" * 76)
    diffs_is, diffs_oos = [], []
    for name, val in mine.items():
        if only and name not in only:
            continue
        r = ref.get(name, {})
        r_is, r_oos = r.get("IS"), r.get("OOS")
        d_is = val["IS"] - r_is if r_is is not None else None
        d_oos = val["OOS"] - r_oos if r_oos is not None else None
        if d_is is not None:
            diffs_is.append(d_is)
        if d_oos is not None:
            diffs_oos.append(d_oos)
        fmt = lambda v: f"{v:9.2f}" if v is not None else f"{'—':>9s}"
        fmt_d = lambda v: f"{v:+8.2f}" if v is not None else f"{'—':>8s}"
        print(f"{name:<10s}{val['IS']:9.2f}{fmt(r_is)}{fmt_d(d_is)}"
              f"{val['OOS']:10.2f}{fmt(r_oos)}{fmt_d(d_oos)}{val['OOS_sd']:8.2f}")
    if diffs_is:
        print("-" * 76)
        print(f"{'平均偏差':<10s}{'':>9s}{'':>10s}{sum(diffs_is)/len(diffs_is):+8.2f}"
              f"{'':>10s}{'':>11s}{sum(diffs_oos)/len(diffs_oos):+9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
