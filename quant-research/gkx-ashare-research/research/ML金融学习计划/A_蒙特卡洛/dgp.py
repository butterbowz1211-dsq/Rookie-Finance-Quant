"""
A 层 · 数据生成过程（GXK 2020, Internet Appendix A）

严格按 IA p1 的式子实现：

    r_{i,t+1} = g*(z_{i,t}) + e_{i,t+1}
    e_{i,t+1} = beta_{i,t} v_{t+1} + eps_{i,t+1}
    z_{i,t}   = (1, x_t)' ⊗ c_{i,t}
    beta_{i,t}= (c_{i1,t}, c_{i2,t}, c_{i3,t})

    v_{t+1} ~ N(0, 0.05^2 I_3)
    eps_{i,t+1} ~ t_5(0, 0.05^2)          # 0.05 是 scale，方差 = 0.05^2 * 5/3

    c_{ij,t} = 2/(N+1) * CSrank(cbar_{ij,t}) - 1        (A.1)
    cbar_{ij,t} = rho_j cbar_{ij,t-1} + e_{ij,t}
    rho_j ~ U[0.9, 1],  e_{ij,t} ~ N(0, 1 - rho_j^2)

    x_t = rho x_{t-1} + u_t,  u_t ~ N(0, 1-rho^2),  rho = 0.95   (A.2)

    (a) g* = (c1, c2, c3*x) . (0.02, 0.02, 0.02)
    (b) g* = (c1^2, c1*c2, sgn(c3*x)) . (0.04, 0.03, 0.012)

固定 N = 200, T = 180, Px = 2, Pc ∈ {100, 50}。
"""
from __future__ import annotations

import numpy as np

N = 200
T = 180
PX = 2
RHO_X = 0.95
V_SCALE = 0.05
EPS_SCALE = 0.05
EPS_DF = 5

THETA = {
    "a": np.array([0.02, 0.02, 0.02]),
    "b": np.array([0.04, 0.03, 0.012]),
}


def _cs_rank_to_unit_interval(panel: np.ndarray) -> np.ndarray:
    """
    panel: (L, N, Pc)，逐 t、逐 j 做横截面秩变换：2*rank/(N+1) - 1 ∈ (-1, 1)。
    秩从 1 到 N，取值对称分布在 0 两侧。
    """
    L, n, k = panel.shape
    ranks = panel.argsort(axis=1).argsort(axis=1) + 1.0     # (L, N, Pc), 1..N
    return 2.0 * ranks / (n + 1.0) - 1.0


def simulate_characteristics(rng: np.random.Generator, pc: int) -> np.ndarray:
    """返回 c, 形状 (T+1, N, Pc)，取值在 [-1, 1]。"""
    rho = rng.uniform(0.9, 1.0, size=pc)                    # (Pc,)
    sd = np.sqrt(1.0 - rho ** 2)                            # (Pc,)

    cbar = np.empty((T + 1, N, pc))
    cbar[0] = rng.standard_normal((N, pc))
    shocks = rng.standard_normal((T, N, pc))
    for t in range(1, T + 1):
        cbar[t] = rho * cbar[t - 1] + sd * shocks[t - 1]

    return _cs_rank_to_unit_interval(cbar)


def simulate_macro(rng: np.random.Generator) -> np.ndarray:
    """返回 x, 形状 (T+1,)，单位方差 AR(1)。"""
    x = np.empty(T + 1)
    x[0] = rng.standard_normal()
    u = rng.standard_normal(T) * np.sqrt(1.0 - RHO_X ** 2)
    for t in range(1, T + 1):
        x[t] = RHO_X * x[t - 1] + u[t - 1]
    return x


def signal(c: np.ndarray, x: np.ndarray, model: str) -> np.ndarray:
    """g*(z_{i,t})，形状 (T+1, N)。"""
    c1, c2, c3 = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    xt = x[:, None]
    th = THETA[model]

    if model == "a":
        feats = np.stack([c1, c2, c3 * xt], axis=-1)
    elif model == "b":
        feats = np.stack([c1 ** 2, c1 * c2, np.sign(c3 * xt)], axis=-1)
    else:
        raise ValueError(model)

    return feats @ th


def simulate(pc: int, model: str, seed: int, **overrides) -> dict:
    """生成一个蒙特卡洛样本。overrides 可改 v_scale / eps_scale / theta。

    返回的 c / x 是 t = 0..T-1 的特征，r 是与之对应的 r_{i,t+1}，
    即 r 的第 t 行由第 t 行的 c/x 与第 t+1 期的冲击共同决定。
    """
    rng = np.random.default_rng(seed)
    v_scale = overrides.get("v_scale", V_SCALE)
    eps_scale = overrides.get("eps_scale", EPS_SCALE)

    c_all = simulate_characteristics(rng, pc)               # (T+1, N, Pc)
    x_all = simulate_macro(rng)                             # (T+1,)

    c = c_all[:T]                                           # 解释变量取 t = 0..T-1
    x = x_all[:T]

    g = signal(c, x, model) if "theta" not in overrides else _signal_with(c, x, model, overrides["theta"])

    beta = c[:, :, :3]                                      # (T, N, 3)，t 期
    v = rng.standard_normal((T, 3)) * v_scale               # v_{t+1}
    factor = np.einsum("tnk,tk->tn", beta, v)               # beta_{i,t} v_{t+1}

    eps = rng.standard_t(EPS_DF, size=(T, N)) * eps_scale   # eps_{t+1}

    r = g + factor + eps                                    # r_{i,t+1}

    return dict(c=c, x=x, g=g, beta=beta, v=v, factor=factor, eps=eps, r=r)


def _signal_with(c, x, model, theta):
    c1, c2, c3 = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    xt = x[:, None]
    feats = (np.stack([c1, c2, c3 * xt], -1) if model == "a"
             else np.stack([c1 ** 2, c1 * c2, np.sign(c3 * xt)], -1))
    return feats @ np.asarray(theta)


def diagnostics(s: dict) -> dict:
    """把四种候选"R²"和波动率都算出来，供人工判断口径。"""
    g, factor, eps, r = s["g"], s["factor"], s["eps"], s["r"]
    x = s["x"]
    out = {}

    # 去掉最后一行（t+1 用到 t=T 的 c，但 r 只有 T+1 行；保守起见两组都报）
    def flat(a):
        return a.reshape(-1)

    var_r = flat(r).var()
    out["var_r"] = var_r
    out["ann_vol_pooled"] = np.sqrt(var_r) * np.sqrt(12)

    # 平均"每只股票的时序波动率"再年化
    per_stock_sd = r.std(axis=0)
    out["ann_vol_avg_i"] = per_stock_sd.mean() * np.sqrt(12)
    out["ann_vol_med_i"] = np.median(per_stock_sd) * np.sqrt(12)

    # 口径 1：g* 占总方差（= 论文说的 predictive R²）
    out["r2_gstar_share"] = g.var() / var_r

    # 口径 2：三因子成分占总方差（每只股票 r 对三因子的时序回归 R²，再对 i 平均）
    r2_factor_i = 1.0 - (r - factor).var(axis=0) / r.var(axis=0)
    out["r2_factor_share_avg_i"] = float(np.nanmean(r2_factor_i))
    out["r2_factor_share_pooled"] = factor.var() / var_r

    # 口径 3：把 g* 也当作"信号"，看 3 因子 + g* 解释多少
    out["r2_gstar_plus_factor"] = (g + factor).var() / var_r

    return out


if __name__ == "__main__":
    for pc in (50, 100):
        for model in ("a", "b"):
            acc = []
            for seed in range(5):
                d = diagnostics(simulate(pc, model, seed))
                acc.append(d)
            keys = acc[0].keys()
            print(f"\n=== Pc={pc}  model=({model})   seeds 0-4 ===")
            for k in keys:
                vals = np.array([a[k] for a in acc])
                print(f"  {k:24s} mean={vals.mean():10.6f}   sd={vals.std(ddof=1):10.6f}")
