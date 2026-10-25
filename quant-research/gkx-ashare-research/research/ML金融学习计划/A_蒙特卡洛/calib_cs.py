"""检查论文"截面 R² = 50%"这句话对应的是哪个量，以及它是否约束 θ0。

论文 Internet Appendix A 有两条校准要求：
  (1) "their variances are calibrated so that the average time series R2 is 40%
       and the average annualized volatility is 30%."
  (2) "We calibrate the values of θ0 such that the cross-sectional R2 is 50%,
       and the predictive R2 is 5%."

本脚本枚举"截面 R²"的几种候选口径，并给出纯噪声基准，用来说明
(2) 里的 50% 在自然读法下有一半是自由度、且对 θ0 很不敏感。

用法：
    python calib_cs.py                 # 默认 5 个种子
    python calib_cs.py --seeds 10
"""
from __future__ import annotations

import argparse

import numpy as np

import dgp

THETA = {"a": [0.02, 0.02, 0.02], "b": [0.04, 0.03, 0.012]}


def design(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """z = [c, x ⊗ c]，与 run_mc.build_design 的前半段一致。"""
    return np.concatenate([c, c * x[:, None, None]], axis=2)


def cs_r2(r: np.ndarray, z: np.ndarray) -> float:
    """逐期横截面 OLS 的 R²，对 t 平均。r, z 都是 T×N(×P)。"""
    T, N = r.shape
    vals = []
    for t in range(T):
        X = np.column_stack([np.ones(N), z[t]])
        y = r[t]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = X @ beta
        vals.append(1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return 100.0 * float(np.mean(vals))


def paired_cs_r2(y: np.ndarray, f: np.ndarray) -> float:
    """逐期把 y 对单个回归元 f 做横截面回归，返回平均 R²。"""
    T, N = y.shape
    vals = []
    for t in range(T):
        slope, intercept = np.polyfit(f[t], y[t], 1)
        pred = slope * f[t] + intercept
        vals.append(1.0 - ((y[t] - pred) ** 2).sum() / ((y[t] - y[t].mean()) ** 2).sum())
    return 100.0 * float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("一、候选口径（情形 (a)/(b)，θ0 用印出值，5 个种子平均）\n")
    header = f"{'候选口径':34s} {'(a) Pc=50':>12s} {'(a) Pc=100':>12s} {'(b) Pc=50':>12s} {'(b) Pc=100':>12s}"
    print(header)
    print("-" * len(header))
    rows: dict[str, dict[tuple[str, int], list[float]]] = {
        "r ~ z 逐期横截面 OLS": {},
        "r ~ g* 逐期横截面": {},
        "g* ~ z 逐期横截面 OLS": {},
    }
    for case in ("a", "b"):
        for pc in (50, 100):
            acc = {k: [] for k in rows}
            for seed in range(args.seeds):
                s = dgp.simulate(pc, case, seed)
                z = design(s["c"], s["x"])
                acc["r ~ z 逐期横截面 OLS"].append(cs_r2(s["r"], z))
                acc["r ~ g* 逐期横截面"].append(paired_cs_r2(s["r"], s["g"]))
                acc["g* ~ z 逐期横截面 OLS"].append(cs_r2(s["g"], z))
            for k in rows:
                rows[k][(case, pc)] = acc[k]
    for name, d in rows.items():
        cells = [f"{np.mean(d[(c, p)]):9.2f}% ±{np.std(d[(c, p)]):4.2f}" for c in ("a", "b") for p in (50, 100)]
        print(f"{name:34s} " + " ".join(f"{c:>12s}" for c in cells))

    print("\n二、纯噪声基准：把 r 换成与 z 无关的正态随机数\n")
    for pc in (50, 100):
        vals = []
        for seed in range(args.seeds):
            s = dgp.simulate(pc, "a", seed)
            z = design(s["c"], s["x"])
            noise = rng.normal(0.0, 0.12, size=(z.shape[0], z.shape[1]))
            vals.append(cs_r2(noise, z))
        print(f"  Pc={pc:3d}  {np.mean(vals):6.2f}% ±{np.std(vals):4.2f}   "
              f"（理论值 Pc/(N-2) = {pc}/{dgp.N - 2} = {100.0 * pc / (dgp.N - 2):.2f}%）")

    print("\n三、对 θ0 的敏感性（倍数作用于印出的 θ0）\n")
    print(f"{'':22s} {'Pc=50':>12s} {'Pc=100':>12s}")
    for case in ("a", "b"):
        for mult in (0.5, 1.0, 1.08, 2.0):
            theta = [t * mult for t in THETA[case]]
            cells = []
            for pc in (50, 100):
                vals = []
                for seed in range(args.seeds):
                    s = dgp.simulate(pc, case, seed, theta=theta)
                    vals.append(cs_r2(s["r"], design(s["c"], s["x"])))
                cells.append(f"{np.mean(vals):6.2f}% ±{np.std(vals):4.2f}")
            print(f"({case}) θ×{mult:<4.2f}          {cells[0]:>12s} {cells[1]:>12s}")


if __name__ == "__main__":
    main()
