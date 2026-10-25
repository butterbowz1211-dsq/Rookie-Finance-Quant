"""
Table A.3：变量选择频率（Internet Appendix A）。

对每次重复，用 Lasso / ENet / GLM（及其 Huber 版）在训练集上拟合、在验证集上选超参，
然后记录每个协变量是否被选中（系数非零 / 整组非零）。最后统计 6 个被关注的协变量
各自的选中频率，以及其余 P-6 个「噪声」协变量的平均选中频率。

协变量在 X 中的排列是 [c_1..c_Pc, c_1*x..c_Pc*x]，所以
    c1,c2,c3        → 列 0,1,2
    c1*x,c2*x,c3*x  → 列 Pc,Pc+1,Pc+2

用法：
    python table_a3.py --pc 50 --model a --reps 50 --models l1
    python table_a3.py --pc 50 --model a --reps 30 --models glm --resume
"""
from __future__ import annotations

import argparse
import json
import pathlib
import hashlib
import re

import numpy as np

import dgp
import models
from run_mc import build_design, split_three

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"

COV = ["c1", "c2", "c3", "c1*x", "c2*x", "c3*x"]
METHOD_GROUPS = {"l1": {"Lasso", "Lasso+H", "ENet", "ENet+H"},
                 "glm": {"GLM", "GLM+H"}}


def read_completed(path, signature):
    """Reject corrupt, duplicated, legacy, or differently configured records."""
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("run_signature") != signature:
            raise ValueError("Cannot resume legacy or differently configured A3 records; use a new tag")
        seed, method = int(rec["seed"]), rec["method"]
        if method in done.setdefault(seed, set()):
            raise ValueError("Duplicate seed/method in A3 records")
        done[seed].add(method)
    return done


def fit_all(Xtr, ytr, Xva, yva, which):
    """返回 {方法名: 布尔数组（长度 P，表示该协变量是否被选中）}。"""
    out = {}
    alphas = np.logspace(-4, -1, 10)
    if "l1" in which:
        for name, l1, hub in (("Lasso", 1.0, False), ("Lasso+H", 1.0, True),
                              ("ENet", 0.5, False), ("ENet+H", 0.5, True)):
            _, info = models.fit_enet(Xtr, ytr, alphas, l1, Xva, yva, huber=hub)
            out[name] = info["coef"] != 0
    if "glm" in which:
        glams = np.logspace(-4, -0.5, 25)
        for name, hub in (("GLM", False), ("GLM+H", True)):
            _, info = models.fit_glm(Xtr, ytr, glams, Xva, yva, huber=hub)
            out[name] = info["mask"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", type=int, default=50, choices=(50, 100))
    ap.add_argument("--model", default="a", choices=("a", "b"))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--theta-scale", type=float, default=1.08)
    ap.add_argument("--models", default="l1", help="l1 或 glm，或 l1,glm")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.reps < 1:
        ap.error("--reps must be positive")

    OUT.mkdir(exist_ok=True)
    tag = args.tag or f"pc{args.pc}_{args.model}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        ap.error("--tag may contain only letters, numbers, underscores and hyphens")
    raw_path = OUT / f"raw_a3_{tag}.jsonl"
    which = {item.strip() for item in args.models.split(",") if item.strip()}
    if not which or not which <= METHOD_GROUPS.keys():
        ap.error("--models must contain l1 and/or glm")
    requested = set().union(*(METHOD_GROUPS[group] for group in which))
    signature = {"pc": args.pc, "model": args.model, "theta_scale": args.theta_scale,
                 "source_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                   for name in ("table_a3.py", "dgp.py", "models.py", "run_mc.py")}}
    if raw_path.exists() and not args.resume:
        raise FileExistsError("Choose a new A3 tag or explicit --resume")

    done: dict[int, set] = {}
    if args.resume and raw_path.exists():
        done = read_completed(raw_path, signature)
        print(f"  续跑：已有 {len(done)} 个种子的记录", flush=True)

    for rep in range(args.reps):
        seed = args.seed0 + rep
        if requested <= done.get(seed, set()):
            continue
        theta = [t * args.theta_scale for t in dgp.THETA[args.model]]
        s = dgp.simulate(args.pc, args.model, seed, theta=theta)
        X, y, t_eff = build_design(s)
        tr, va, te = split_three(t_eff)
        Xtr, Xva, Xte = models._standardize(X[tr], X[va], X[te])
        sel = fit_all(Xtr, y[tr], Xva, y[va], which)
        with raw_path.open("a", encoding="utf-8") as fh:
            for name, mask in sel.items():
                if name in done.get(seed, set()):
                    continue
                fh.write(json.dumps({"seed": seed, "method": name,
                                     "run_signature": signature,
                                     "sel": [int(v) for v in mask]}) + "\n")
        print(f"[rep {rep+1}/{args.reps}] seed={seed} done", flush=True)

    # ---- 汇总
    agg: dict[str, list] = {}
    order = []
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            m = rec["method"]
            if m not in agg:
                agg[m] = []
                order.append(m)
            agg[m].append(rec["sel"])

    p = 2 * args.pc
    idx6 = [0, 1, 2, args.pc, args.pc + 1, args.pc + 2]
    noise_idx = [j for j in range(p) if j not in idx6]

    rows = []
    print(f"\n=== Table A.3  Pc={args.pc}  ({args.model})  reps={len(next(iter(agg.values())))} ===")
    print(f"{'Method':10s} " + " ".join(f"{c:>6s}" for c in COV) + f" {'Noise':>6s}")
    for m in order:
        arr = np.array(agg[m], dtype=float)
        f6 = arr[:, idx6].mean(axis=0)
        fn = arr[:, noise_idx].mean()
        rows.append((m, f6, fn))
        print(f"{m:10s} " + " ".join(f"{v:6.2f}" for v in f6) + f" {fn:6.2f}")

    out = OUT / f"tableA3_{tag}.json"
    out.write_text(json.dumps([{"method": m, "six": list(map(float, f6)), "noise": float(fn)}
                               for m, f6, fn in rows], indent=1), encoding="utf-8")
    print(f"\n已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
