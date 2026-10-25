"""
A 层 · 蒙特卡洛驱动：对每个重复，生成 DGP → 训练/验证/检验三段 → 估计 → 算 IS/OOS R²。

样本划分：T = 180 → 3 个连续等长子样本，各 60 期（IA p1）。
评估口径（IA p2）：IS 与 OOS 的 R² 都相对于「基于样本内均值的估计量」，
    分母一律用训练样本均值做中心化。

用法示例：
    python run_mc.py --pc 50 --model a --reps 20
    python run_mc.py --pc 50 --model a --reps 100 --models lin,tree
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

import dgp
import models

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(os.environ.get("GKX_MC_OUT_DIR", HERE / "out_repaired"))

BIG = 1e9


def build_design(s: dict, horizon: int = 1):
    """返回 (X, y, T_eff)。X 为 z_{i,t} = (1, x_t)' ⊗ c_{i,t}；
    y 为 t 期起、连续 horizon 期的累计收益（horizon=1 即 r_{i,t+1}）。"""
    c = s["c"]                  # (T, N, Pc)
    x = s["x"]                  # (T,)
    r = s["r"]                  # (T, N)  r[t] 就是 t+1 期收益
    if horizon < 1 or horizon > c.shape[0]:
        raise ValueError("horizon must be in 1..T")
    if c.ndim != 3 or r.shape != c.shape[:2] or x.shape != (c.shape[0],):
        raise ValueError("DGP arrays have incompatible shapes")
    n_stocks = c.shape[1]
    t_eff = c.shape[0] - horizon + 1
    if horizon == 1:
        y = r[:t_eff]
    else:
        y = np.stack([r[k:k + horizon].sum(axis=0) for k in range(t_eff)], axis=0)
    cc = c[:t_eff]
    xx = x[:t_eff]
    X = np.concatenate([cc, cc * xx[:, None, None]], axis=2).reshape(t_eff * n_stocks, 2 * c.shape[2])
    return X, y.reshape(t_eff * n_stocks), t_eff


def split_three(t_eff: int, horizon: int = 1, n_stocks: int | None = None):
    """Formation-time thirds, purging h-1 tail origins at train/valid boundaries.

    A response at origin t uses shocks r[t:t+h].  The last retained response
    of an earlier block must finish before the next block's first origin.
    This is the repaired research protocol, not a claim about GKX's exact
    multi-horizon simulation implementation. h=1 retains the old 60/60/60 split.
    """
    t = t_eff // 3
    n_stocks = dgp.N if n_stocks is None else n_stocks
    purge = horizon - 1
    if horizon < 1 or t <= purge or n_stocks < 1:
        raise ValueError("Insufficient periods for nonoverlapping horizon labels")
    idx = np.arange(t_eff * n_stocks).reshape(t_eff, n_stocks)
    return idx[:t-purge].ravel(), idx[t:2*t-purge].ravel(), idx[2*t:].ravel()


def r2_relative_to_train_mean(ytr, ptr, yte, pte):
    mu = ytr.mean()
    sse_is = float(np.sum((ytr - ptr) ** 2))
    sst_is = float(np.sum((ytr - mu) ** 2))
    sse_oos = float(np.sum((yte - pte) ** 2))
    sst_oos = float(np.sum((yte - mu) ** 2))
    return 100 * (1 - sse_is / sst_is), 100 * (1 - sse_oos / sst_oos)


# ------------------------------------------------------------------ 各模型

def run_linear(Xtr, ytr, Xva, yva, Xte, yte, seed):
    """返回 {模型名: (pred_tr, pred_te)} 与超参记录。"""
    res, info = {}, {}

    f = models.fit_ols(Xtr, ytr); res["OLS"] = (f(Xtr), f(Xte))
    f = models.fit_huber_ols(Xtr, ytr); res["OLS+H"] = (f(Xtr), f(Xte))

    lams = np.logspace(0, 5, 11)
    f, i = models.fit_ridge_path(Xtr, ytr, lams, Xva, yva); res["Ridge"] = (f(Xtr), f(Xte)); info["Ridge"] = i

    f, i = models.fit_ridge_huber(Xtr, ytr, lams, Xva, yva)
    res["Ridge+H"] = (f(Xtr), f(Xte)); info["Ridge+H"] = i

    alphas = np.logspace(-4, -1, 10)
    f, i = models.fit_enet(Xtr, ytr, alphas, 1.0, Xva, yva); res["Lasso"] = (f(Xtr), f(Xte)); info["Lasso"] = i
    f, i = models.fit_enet(Xtr, ytr, alphas, 1.0, Xva, yva, huber=True); res["Lasso+H"] = (f(Xtr), f(Xte)); info["Lasso+H"] = i
    f, i = models.fit_enet(Xtr, ytr, alphas, 0.5, Xva, yva); res["ENet"] = (f(Xtr), f(Xte)); info["ENet"] = i
    f, i = models.fit_enet(Xtr, ytr, alphas, 0.5, Xva, yva, huber=True); res["ENet+H"] = (f(Xtr), f(Xte)); info["ENet+H"] = i

    ks = list(range(1, 21))
    f, i = models.fit_pcr(Xtr, ytr, ks, Xva, yva); res["PCR"] = (f(Xtr), f(Xte)); info["PCR"] = i
    f, i = models.fit_pls(Xtr, ytr, ks, Xva, yva); res["PLS"] = (f(Xtr), f(Xte)); info["PLS"] = i
    return res, info


def run_glm(Xtr, ytr, Xva, yva, Xte, yte):
    res, info = {}, {}
    glams = np.logspace(-4, -0.5, 25)
    f, i = models.fit_glm(Xtr, ytr, glams, Xva, yva); res["GLM"] = (f(Xtr), f(Xte)); info["GLM"] = i
    f, i = models.fit_glm(Xtr, ytr, glams, Xva, yva, huber=True); res["GLM+H"] = (f(Xtr), f(Xte)); info["GLM+H"] = i
    return res, info


def run_trees(Xtr, ytr, Xva, yva, Xte, yte, seed, rf_trees=300, gb_trees=300,
              rf_depths=(1, 3, 6), rf_mfeats=(30,), gb_depths=(1, 2),
              gb_lrs=(0.01, 0.1)):
    res, info = {}, {}
    grid = [(d, m) for d in rf_depths for m in rf_mfeats]
    f, i = models.fit_rf(Xtr, ytr, grid, Xva, yva, seed, n_trees=rf_trees)
    res["RF"] = (f(Xtr), f(Xte)); info["RF"] = i
    ggrid = [(d, lr, gb_trees) for d in gb_depths for lr in gb_lrs]
    f, i = models.fit_gbrt(Xtr, ytr, ggrid, Xva, yva, seed); res["GBRT"] = (f(Xtr), f(Xte)); info["GBRT"] = i
    f, i = models.fit_gbrt(Xtr, ytr, ggrid, Xva, yva, seed, huber=True); res["GBRT+H"] = (f(Xtr), f(Xte)); info["GBRT+H"] = i
    return res, info


def run_nn(Xtr, ytr, Xva, yva, Xte, yte, seed, depths, ensemble=10, epochs=100):
    import nn
    res, info = {}, {}
    grid = [(l, r) for l in (1e-5, 1e-4, 1e-3) for r in (0.01, 0.1)]
    for d in depths:
        f, i = nn.fit_nn(Xtr, ytr, Xva, yva, d, grid, ensemble=ensemble, seed=seed, epochs=epochs)
        res[f"NN{d}"] = (f(Xtr), f(Xte)); info[f"NN{d}"] = i
    return res, info


def run_oracle(c, x, y, model, tr, te):
    F = models.oracle_features(c, x, model)
    f = models.fit_ols(F[tr], y[tr])
    return {"Oracle": (f(F[tr]), f(F[te]))}, {}


# ------------------------------------------------------------------ 主循环

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", type=int, default=50, choices=(50, 100))
    ap.add_argument("--model", default="a", choices=("a", "b"))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--models", default="lin,oracle")
    ap.add_argument("--theta-scale", type=float, default=1.0)
    ap.add_argument("--nndepth", default="")
    ap.add_argument("--nnes", type=int, default=10)
    ap.add_argument("--nep", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=1,
                    help="预测期限：1=月度，3=季度，6=半年，12=年度（Table A.2）")
    ap.add_argument("--rf-trees", type=int, default=300)
    ap.add_argument("--gb-trees", type=int, default=300)
    ap.add_argument("--rf-depth", default="1,3,6",
                    help="随机森林深度网格，论文 Table A.5 是 1,2,3,4,5,6")
    ap.add_argument("--rf-mfeat", default="30",
                    help="每次分裂的候选特征数，论文是 3,5,10,20,30,50")
    ap.add_argument("--gb-depth", default="1,2")
    ap.add_argument("--gb-lr", default="0.01,0.1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true",
                    help="从 out/raw_<tag>.jsonl 续跑，跳过已完成的种子")
    args = ap.parse_args()

    want = set(args.models.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"pc{args.pc}_{args.model}_h{args.horizon}"
    raw_path = OUT / f"raw_{tag}.jsonl"
    split_protocol = {"name": "formation_thirds_purged", "horizon": args.horizon,
                      "purge_origins_at_each_boundary": args.horizon - 1}
    if raw_path.exists() and not args.resume:
        raise FileExistsError("Choose a new tag or explicit --resume; existing simulations are not overwritten")

    done = set()
    if args.resume and raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("split_protocol") != split_protocol:
                raise ValueError("Cannot resume legacy or different-horizon simulation records")
            done.add(int(rec["seed"]))
        print(f"  续跑：已完成 {len(done)} 个种子，跳过", flush=True)

    acc: dict[str, list] = {}
    t0 = time.time()

    for rep in range(args.reps):
        seed = args.seed0 + rep
        if seed in done:
            continue
        kw = {}
        if args.theta_scale != 1.0:
            kw["theta"] = [t * args.theta_scale for t in dgp.THETA[args.model]]
        s = dgp.simulate(args.pc, args.model, seed, **kw)
        X, y, t_eff = build_design(s, args.horizon)
        tr, va, te = split_three(t_eff, args.horizon, s["c"].shape[1])

        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]
        Xte, yte = X[te], y[te]
        Xtr, Xva, Xte = models._standardize(Xtr, Xva, Xte)

        preds, info = {}, {}
        if "ols" in want:
            f = models.fit_ols(Xtr, ytr)
            preds["OLS"] = (f(Xtr), f(Xte))
        if "lin" in want:
            p, i = run_linear(Xtr, ytr, Xva, yva, Xte, yte, seed)
            preds.update(p); info.update(i)
        if "tree" in want:
            p, i = run_trees(Xtr, ytr, Xva, yva, Xte, yte, seed,
                             rf_trees=args.rf_trees, gb_trees=args.gb_trees,
                             rf_depths=tuple(int(x) for x in args.rf_depth.split(",")),
                             rf_mfeats=tuple(int(x) for x in args.rf_mfeat.split(",")),
                             gb_depths=tuple(int(x) for x in args.gb_depth.split(",")),
                             gb_lrs=tuple(float(x) for x in args.gb_lr.split(",")))
            preds.update(p); info.update(i)
        if "oracle" in want:
            cflat = s["c"][:t_eff].reshape(-1, s["c"].shape[2])
            xflat = np.repeat(s["x"][:t_eff], dgp.N)
            p, i = run_oracle(cflat, xflat, y, args.model, tr, te)
            preds.update(p)
        if "nn" in want:
            depths = tuple(int(x) for x in args.nndepth.split(",")) if args.nndepth else (1, 2, 3, 4, 5)
            p, i = run_nn(Xtr, ytr, Xva, yva, Xte, yte, seed, depths,
                          ensemble=args.nnes, epochs=args.nep)
            preds.update(p); info.update(i)
        if "glm" in want:
            p, i = run_glm(Xtr, ytr, Xva, yva, Xte, yte)
            preds.update(p); info.update(i)

        for name, (ptr, pte) in preds.items():
            is_r2, oos_r2 = r2_relative_to_train_mean(ytr, ptr, yte, pte)
            acc.setdefault(name, []).append((is_r2, oos_r2))

        # 逐次落盘：即便进程被杀，已完成的重复也不会丢
        with raw_path.open("a", encoding="utf-8") as fh:
            for name in preds:
                is_r2, oos_r2 = acc[name][-1]
                fh.write(json.dumps({"seed": seed, "model": name,
                                     "split_protocol": split_protocol,
                                     "sample_counts": {"train": len(tr), "valid": len(va), "test": len(te)},
                                     "is": round(float(is_r2), 4),
                                     "oos": round(float(oos_r2), 4)}) + "\n")

        if rep == 0 and info:
            brief = {k: {kk: round(float(vv), 6) if isinstance(vv, (int, float, np.floating)) else vv
                         for kk, vv in v.items() if kk not in ("coef", "mask", "imp")}
                     for k, v in info.items()}
            print("  选中超参：" + "  ".join(f"{k}={v}" for k, v in brief.items()), flush=True)
        print(f"[rep {rep+1}/{args.reps}] seed={seed} done  ({time.time()-t0:.0f}s)", flush=True)

    order, agg = [], {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            m = rec["model"]
            if m not in agg:
                agg[m] = ([], [])
                order.append(m)
            agg[m][0].append(rec["is"])
            agg[m][1].append(rec["oos"])

    rows = []
    for name in order:
        iv, ov = np.array(agg[name][0]), np.array(agg[name][1])
        rows.append((name, iv.mean(), iv.std(ddof=1) if len(iv) > 1 else 0.0,
                     ov.mean(), ov.std(ddof=1) if len(ov) > 1 else 0.0))
    nrep = max(len(v[0]) for v in agg.values()) if agg else 0

    print(f"\n=== Pc={args.pc}  model=({args.model})  horizon={args.horizon}  reps={nrep} ===")
    print(f"{'Model':10s} {'IS R2(%)':>10s} {'OOS R2(%)':>10s}")
    for name, is_m, is_s, oos_m, oos_s in rows:
        print(f"{name:10s} {is_m:10.2f} {oos_m:10.2f}   (sd {is_s:.2f}/{oos_s:.2f})")

    out = OUT / f"tableA1_{tag}.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("model,is_mean,is_sd,oos_mean,oos_sd\n")
        for name, is_m, is_s, oos_m, oos_s in rows:
            f.write(f"{name},{is_m:.4f},{is_s:.4f},{oos_m:.4f},{oos_s:.4f}\n")
    print(f"\n已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
