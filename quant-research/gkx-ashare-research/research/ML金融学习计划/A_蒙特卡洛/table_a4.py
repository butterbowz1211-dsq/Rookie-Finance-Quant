"""
Table A.4：变量重要性（Internet Appendix A）。

两种 VI 口径（论文 §1.9 脚注 27）：
  * 树模型：mean decrease in impurity（sklearn 的 feature_importances_）
  * 神经网络：SSD，即训练集上 (∂g/∂z_j)² 的和（`nn.ssd`）

论文 Table A.4 报的是"平均变量重要性"，各列加总为 100，所以这里把每次重复的
VI 向量归一化到总和 100，再对被关注的 6 个协变量分别平均、对其余协变量取平均。

协变量在 X 中的排列是 [c_1..c_Pc, c_1*x..c_Pc*x]，所以
    c1,c2,c3 → 列 0,1,2      c1*x,c2*x,c3*x → 列 Pc,Pc+1,Pc+2

用法：
    python table_a4.py --pc 50 --model a --reps 20 --methods tree
    python table_a4.py --pc 50 --model a --reps 20 --methods nn
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

import dgp
import models
import nn
from run_mc import build_design, split_three

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"

COV = ["c1", "c2", "c3", "c1*x", "c2*x", "c3*x"]
NN_GRID = [(lam, lr) for lam in (1e-5, 1e-4, 1e-3) for lr in (0.01, 0.1)]


def vi_trees(Xtr, ytr, Xva, yva, seed, args) -> dict[str, np.ndarray]:
    rf_grid = [(d, m) for d in (int(x) for x in args.rf_depth.split(","))
               for m in (int(x) for x in args.rf_mfeat.split(","))]
    gb_grid = [(d, lr, args.gb_trees) for d in (1, 2)
               for lr in (float(x) for x in args.gb_lr.split(","))]
    _, i_rf = models.fit_rf(Xtr, ytr, rf_grid, Xva, yva, seed, n_trees=args.rf_trees)
    _, i_gb = models.fit_gbrt(Xtr, ytr, gb_grid, Xva, yva, seed)
    _, i_gbh = models.fit_gbrt(Xtr, ytr, gb_grid, Xva, yva, seed, huber=True)
    out = {}
    for name, info in (("RF", i_rf), ("GBRT", i_gb), ("GBRT+H", i_gbh)):
        imp = np.asarray(info["imp"], dtype=float)
        out[name] = imp / imp.sum() * 100.0
    return out


def vi_nn(Xtr, ytr, Xva, yva, seed, args) -> dict[str, np.ndarray]:
    """SSD 变量重要性。注意深层网络最后一层只有 1–2 个 ReLU 单元，偶尔会整层
    死亡（激活比例 0），此时 SSD 全为 0，必须跳过，否则归一化会得到 NaN。"""
    out = {}
    for depth in (1, 2, 3, 4, 5):
        _, info, nets = nn.fit_nn(Xtr, ytr, Xva, yva, depth, NN_GRID,
                                  ensemble=args.nn_ensemble, seed=seed,
                                  epochs=args.nn_epochs, return_nets=True)
        shares, dead = [], 0
        for net, _, _, _ in nets:
            s = nn.ssd(net, Xtr)
            tot = float(np.nansum(s))
            if not np.isfinite(tot) or tot <= 0.0:
                dead += 1
                continue
            shares.append(s / tot * 100.0)
        if not shares:
            print(f"    [警告] NN{depth} 的 {args.nn_ensemble} 个网络 SSD 全为零，跳过",
                  flush=True)
            continue
        if dead:
            print(f"    NN{depth}: {dead}/{args.nn_ensemble} 个网络末层死亡，已排除",
                  flush=True)
        out[f"NN{depth}"] = np.mean(shares, axis=0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", type=int, default=50, choices=(50, 100))
    ap.add_argument("--model", default="a", choices=("a", "b"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--theta-scale", type=float, default=1.08)
    ap.add_argument("--methods", default="tree", help="逗号分隔：tree,nn")
    ap.add_argument("--rf-trees", type=int, default=300)
    ap.add_argument("--gb-trees", type=int, default=300)
    ap.add_argument("--rf-depth", default="1,2,3,4,5,6")
    ap.add_argument("--rf-mfeat", default="3,5,10,20,30,50")
    ap.add_argument("--gb-lr", default="0.001,0.01")
    ap.add_argument("--nn-ensemble", type=int, default=10)
    ap.add_argument("--nn-epochs", type=int, default=100)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.reps < 1:
        ap.error("--reps must be positive")

    OUT.mkdir(exist_ok=True)
    which = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not which or not set(which) <= {"tree", "nn"}:
        ap.error("--methods must contain tree and/or nn")
    tag = args.tag or f"pc{args.pc}_{args.model}_{'-'.join(which)}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        ap.error("--tag may contain only letters, numbers, underscores and hyphens")
    raw_path = OUT / f"raw_a4_{tag}.jsonl"
    if raw_path.exists():
        raise FileExistsError("Choose a new A4 tag; existing records cannot be appended implicitly")

    for rep in range(args.reps):
        seed = args.seed0 + rep
        theta = [t * args.theta_scale for t in dgp.THETA[args.model]]
        s = dgp.simulate(args.pc, args.model, seed, theta=theta)
        X, y, t_eff = build_design(s)
        tr, va, te = split_three(t_eff)
        Xtr, Xva, Xte = models._standardize(X[tr], X[va], X[te])
        ytr, yva = y[tr], y[va]

        rec: dict[str, np.ndarray] = {}
        if "tree" in which:
            rec.update(vi_trees(Xtr, ytr, Xva, yva, seed, args))
        if "nn" in which:
            rec.update(vi_nn(Xtr, ytr, Xva, yva, seed, args))

        with raw_path.open("a", encoding="utf-8") as fh:
            for name, imp in rec.items():
                fh.write(json.dumps({"seed": seed, "method": name,
                                     "imp": [round(float(v), 4) for v in imp]}) + "\n")
        print(f"[rep {rep+1}/{args.reps}] seed={seed} done: " + ", ".join(rec), flush=True)

    agg, order = {}, []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        m = r["method"]
        if m not in agg:
            agg[m] = []
            order.append(m)
        agg[m].append(r["imp"])

    p = 2 * args.pc
    idx6 = [0, 1, 2, args.pc, args.pc + 1, args.pc + 2]
    noise_idx = [j for j in range(p) if j not in idx6]

    print(f"\n=== Table A.4  Pc={args.pc}  ({args.model})  reps={len(next(iter(agg.values())))} ===")
    print(f"{'Method':10s} " + " ".join(f"{c:>6s}" for c in COV) + f" {'Noise':>6s}")
    rows = []
    for m in order:
        arr = np.array(agg[m], dtype=float)
        f6 = arr[:, idx6].mean(axis=0)
        fn = arr[:, noise_idx].mean()
        rows.append((m, f6, fn))
        print(f"{m:10s} " + " ".join(f"{v:6.2f}" for v in f6) + f" {fn:6.2f}")

    out = OUT / f"tableA4_{tag}.json"
    out.write_text(json.dumps([{"method": m, "six": list(map(float, f6)), "noise": float(fn)}
                               for m, f6, fn in rows], indent=1), encoding="utf-8")
    print(f"\n已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
