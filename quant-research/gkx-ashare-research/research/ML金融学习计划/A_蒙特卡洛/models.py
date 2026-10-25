"""
A 层 · 估计器集合（对应 IA Table A.1 的 21 行）

Table A.1 的行与正文 §1 的 13 个模型并不完全相同：它额外含 Ridge、Lasso，
且没有 OLS-3。本文件实现的映射：

    OLS / OLS+H          -> 全部协变量的池化 OLS（H 用 Huber-IRLS）
    Ridge / Ridge+H      -> l2 惩罚（SVD 一次，整条 λ 路径共享）
    Lasso / Lasso+H      -> l1 惩罚
    ENet  / ENet+H       -> 弹性网，ρ = 0.5
    PCR / PLS            -> 主成分 / 偏最小二乘回归，成分数在验证集上选
    GLM   / GLM+H        -> 每个协变量展开成 3 结点的样条基，按组 lasso 惩罚
    RF                   -> 随机森林
    GBRT / GBRT+H        -> 梯度提升树
    NN1..NN5             -> 见 nn.py（本文件不含）
    Oracle               -> 只对真实协变量做池化 OLS

调参一律在验证样本上按 MSE 选。评估用「相对样本内均值」的口径：
    R2_IS  = 1 - SSE(train) / Σ_{train} (r - mean_train)^2
    R2_OOS = 1 - SSE(test)  / Σ_{test}  (r - mean_train)^2
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import SplineTransformer


# ---------------------------------------------------------------- 基础工具

def _center(X: np.ndarray, y: np.ndarray):
    xm = X.mean(axis=0)
    ym = y.mean()
    return X - xm, y - ym, xm, ym


def _standardize(Xtr, Xva, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd


def _mse(y, p):
    return float(np.mean((y - p) ** 2))


# ---------------------------------------------------------------- OLS / Huber

def fit_ols(Xtr, ytr):
    Xc, yc, xm, ym = _center(Xtr, ytr)
    beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    return lambda Z: (Z - xm) @ beta + ym


def fit_huber_ols(Xtr, ytr, xi_q=0.999, iters=30, tol=1e-9):
    """IRLS：|r| <= ξ 权重 1，|r| > ξ 权重 ξ/|r|；ξ 取残差绝对值的 99.9% 分位。"""
    Xc, yc, xm, ym = _center(Xtr, ytr)
    beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    for _ in range(iters):
        resid = yc - Xc @ beta
        xi = np.quantile(np.abs(resid), xi_q)
        if xi <= 0:
            break
        w = np.where(np.abs(resid) <= xi, 1.0, xi / np.maximum(np.abs(resid), 1e-12))
        sw = np.sqrt(w)
        new = np.linalg.lstsq(Xc * sw[:, None], yc * sw, rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return lambda Z: (Z - xm) @ beta + ym


# ---------------------------------------------------------------- 惩罚线性

def fit_ridge_path(Xtr, ytr, lams, Xva, yva):
    Xc, yc, xm, ym = _center(Xtr, ytr)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uy = U.T @ yc
    best, best_mse = None, np.inf
    Xva_c = Xva - xm
    for lam in lams:
        coef = Vt.T @ (s / (s ** 2 + lam) * Uy)
        m = _mse(yva, Xva_c @ coef + ym)
        if m < best_mse:
            best_mse, best = m, (lam, coef)
    lam, coef = best
    return (lambda Z: (Z - xm) @ coef + ym), {"lam": lam}


def fit_enet(Xtr, ytr, lams, l1_ratio, Xva, yva, huber=False):
    Xc, yc, xm, ym = _center(Xtr, ytr)
    best, best_mse = None, np.inf
    for lam in lams:
        model = ElasticNet(alpha=lam, l1_ratio=l1_ratio, fit_intercept=False,
                           max_iter=5000, tol=1e-5)
        model.fit(Xc, yc)
        m = _mse(yva, (Xva - xm) @ model.coef_ + ym)
        if m < best_mse:
            best_mse, best = m, (lam, model.coef_)
    lam, coef = best
    pred = lambda Z: (Z - xm) @ coef + ym
    if not huber:
        return pred, {"lam": lam, "nnz": int(np.sum(coef != 0)), "coef": coef}

    # Huber 变体：用上一步的 λ 作为起点，在加权最小二乘上重解一次 λ 网格
    def weighted_enet_target(yw, Xw, lam):
        mdl = ElasticNet(alpha=lam, l1_ratio=l1_ratio, fit_intercept=False,
                         max_iter=5000, tol=1e-5)
        mdl.fit(Xw, yw)
        return mdl.coef_

    beta = coef
    for _ in range(10):
        resid = yc - Xc @ beta
        xi = np.quantile(np.abs(resid), 0.999)
        if xi <= 0:
            break
        w = np.where(np.abs(resid) <= xi, 1.0, xi / np.maximum(np.abs(resid), 1e-12))
        sw = np.sqrt(w)
        beta_new = weighted_enet_target(yc * sw, Xc * sw[:, None], lam)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return (lambda Z: (Z - xm) @ beta + ym), {"lam": lam, "nnz": int(np.sum(beta != 0)),
                                              "coef": beta}


# ---------------------------------------------------------------- 降维

def fit_pcr(Xtr, ytr, ks, Xva, yva):
    Xc, yc, xm, ym = _center(Xtr, ytr)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uy = U.T @ yc
    tolerance = np.finfo(s.dtype).eps * max(Xc.shape) * (s[0] if len(s) else 0)
    rank = int(np.sum(s > tolerance))
    if rank == 0:
        return (lambda Z: np.full(len(Z), ym)), {"k": 0, "rank": 0}
    ks = sorted({min(int(k), rank) for k in ks if int(k) > 0})
    if not ks:
        raise ValueError("PCR requires at least one positive candidate component count")
    best, best_mse = None, np.inf
    Xva_c = Xva - xm
    for k in ks:
        coef = Vt[:k].T @ (Uy[:k] / s[:k])
        m = _mse(yva, Xva_c @ coef + ym)
        if m < best_mse:
            best_mse, best = m, (k, coef)
    k, coef = best
    return (lambda Z: (Z - xm) @ coef + ym), {"k": k}


def fit_pls(Xtr, ytr, ks, Xva, yva):
    Xc, yc, xm, ym = _center(Xtr, ytr)
    best, best_mse = None, np.inf
    for k in ks:
        model = PLSRegression(n_components=k, scale=False)
        model.fit(Xc, yc)
        m = _mse(yva, model.predict(Xva - xm).ravel() + ym)
        if m < best_mse:
            best_mse, best = m, (k, model)
    k, model = best
    return (lambda Z: model.predict(Z - xm).ravel() + ym), {"k": k}


# ---------------------------------------------------------------- GLM（组 lasso）

def _group_soft(v, thr):
    norm = np.linalg.norm(v)
    return np.zeros_like(v) if norm <= thr else v * (1.0 - thr / norm)


def fit_glm(Xtr, ytr, lams, Xva, yva, n_knots=3, iters=400, huber=False):
    """每个原始协变量 → 3 结点的三次样条基；每组一个 l2 范数惩罚（组 lasso）。

    样条基的相邻列高度共线，直接在原基上做近端梯度会因条件数过大而不收敛
    （实测条件数 1.7e5，300 步后 ||beta|| 仍近 0）。这里对每组做 QR 正交化，
    在正交基上求解组 lasso，再通过 R^{-1} 映射回原基做预测。
    注意：组惩罚定义在标准化正交基上，不与任意原始样条基的组惩罚等价。
    """
    n_grp = Xtr.shape[1]
    spl = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False,
                            extrapolation="linear")
    A_tr = spl.fit_transform(Xtr)
    A_va = spl.transform(Xva)
    g = A_tr.shape[1] // n_grp
    Qtr = np.empty_like(A_tr)
    Qva = np.empty_like(A_va)
    mus, Rs = [], []
    # QR 出来的 Q 列是单位范数；乘 sqrt(n) 使其列方差为 1，与其它模型的标准化口径一致，
    # 否则 X'X/n = I/n，组阈值 lam/L 会被放大 n 倍，导致所有组被压成 0。
    s = np.sqrt(A_tr.shape[0])
    for j in range(n_grp):
        sl = slice(j * g, (j + 1) * g)
        mu = A_tr[:, sl].mean(axis=0)
        Q, R = np.linalg.qr(A_tr[:, sl] - mu)
        Qtr[:, sl] = s * Q
        # Q_new = (A_new - mu) @ R^{-1}，即解 R^T X^T = (A_new-mu)^T
        Qva[:, sl] = s * np.linalg.solve(R.T, (A_va[:, sl] - mu).T).T
        mus.append(mu)
        Rs.append(R)

    n, p = Qtr.shape
    ym = float(ytr.mean())
    yc = ytr - ym
    QtQ = Qtr.T @ Qtr / n  # 组内已正交，块对角为单位阵

    def group_cd(lam, ytarget, weights=None, sweeps=60):
        """组坐标下降。每组已 QR 正交化，组内 Gram 为单位阵，更新有闭式：
        b_g = z_g * max(0, 1 - lam/||z_g||)，z_g = X_g'r/n。"""
        beta = np.zeros(p)
        if weights is None:
            Xw, target = Qtr, ytarget
        else:
            # Weighted least squares uses sqrt(w) on BOTH X and y.  After
            # weighting, block Gram matrices are no longer identity matrices.
            sqrt_w = np.sqrt(weights)
            Xw, target = Qtr * sqrt_w[:, None], ytarget * sqrt_w
        r = target.copy()
        lipschitz = [max(float(np.linalg.eigvalsh(
            Xw[:, j*g:(j+1)*g].T @ Xw[:, j*g:(j+1)*g] / n)[-1]), 1e-12)
            for j in range(n_grp)]
        for _ in range(sweeps):
            maxdiff = 0.0
            for j in range(n_grp):
                sl = slice(j * g, (j + 1) * g)
                Xg = Xw[:, sl]
                old = beta[sl].copy()
                z = old + Xg.T @ r / (n * lipschitz[j])
                nz = np.linalg.norm(z)
                threshold = lam / lipschitz[j]
                b = np.zeros(g) if nz <= threshold else z * (1.0 - threshold / nz)
                maxdiff = max(maxdiff, float(np.max(np.abs(b - beta[sl]))))
                beta[sl] = b
                r = r + Xg @ (old - b)
            if maxdiff < 1e-9:
                break
        return beta

    best, best_mse = None, np.inf
    for lam in lams:
        beta = group_cd(lam, yc)
        m = _mse(yva, Qva @ beta + ym)
        if m < best_mse:
            best_mse, best = m, (lam, beta)
    lam, beta = best

    if huber:
        for _ in range(8):
            resid = yc - Qtr @ beta
            xi = np.quantile(np.abs(resid), 0.999)
            if xi <= 0:
                break
            w = np.where(np.abs(resid) <= xi, 1.0, xi / np.maximum(np.abs(resid), 1e-12))
            beta = group_cd(lam, yc, w)

    nonzero_groups = int(np.sum(np.abs(beta.reshape(-1, g)).sum(axis=1) > 1e-10))
    mask = np.abs(beta.reshape(-1, g)).sum(axis=1) > 1e-10

    def predict(Z):
        B = spl.transform(Z)
        out = np.empty_like(B)
        for j in range(n_grp):
            sl = slice(j * g, (j + 1) * g)
            out[:, sl] = s * np.linalg.solve(Rs[j].T, (B[:, sl] - mus[j]).T).T
        return out @ beta + ym

    return predict, {"lam": lam, "groups": nonzero_groups, "mask": mask}


# ---------------------------------------------------------------- 树

def fit_rf(Xtr, ytr, grid, Xva, yva, seed=0, n_trees=300, min_samples_leaf=5):
    best, best_mse = None, np.inf
    for depth, mfeat in grid:
        model = RandomForestRegressor(
            n_estimators=n_trees, max_depth=depth, max_features=mfeat,
            min_samples_leaf=min_samples_leaf, n_jobs=1, random_state=seed)
        model.fit(Xtr, ytr)
        m = _mse(yva, model.predict(Xva))
        if m < best_mse:
            best_mse, best = m, (depth, mfeat, model)
    depth, mfeat, model = best
    return (lambda Z: model.predict(Z)), {"depth": depth, "mfeat": mfeat,
                                          "best": {"depth": depth, "max_features": mfeat},
                                          "n_trees": n_trees, "min_samples_leaf": min_samples_leaf,
                                          "seed": seed,
                                          "imp": model.feature_importances_}


def fit_gbrt(Xtr, ytr, grid, Xva, yva, seed=0, huber=False, min_samples_leaf=1):
    best, best_mse = None, np.inf
    # 注意：sklearn 的 huber 损失里 alpha 是「残差绝对值的分位数」，不是 δ 的绝对值本身。
    # 论文口径是「ξ 取残差绝对值的 99.9% 分位」，对应 alpha=0.999。
    for depth, lr, ntree in grid:
        model = GradientBoostingRegressor(
            n_estimators=ntree, max_depth=depth, learning_rate=lr,
            loss="huber" if huber else "squared_error", alpha=0.999,
            subsample=1.0, random_state=seed, min_samples_leaf=min_samples_leaf)
        model.fit(Xtr, ytr)
        m = _mse(yva, model.predict(Xva))
        if m < best_mse:
            best_mse, best = m, (depth, lr, ntree, model)
    depth, lr, ntree, model = best
    return (lambda Z: model.predict(Z)), {"depth": depth, "lr": lr, "ntree": ntree,
                                          "best": {"depth": depth, "learning_rate": lr, "n_trees": ntree},
                                          "min_samples_leaf": min_samples_leaf, "seed": seed,
                                          "imp": model.feature_importances_}


def fit_ridge_huber(Xtr, ytr, lams, Xva, yva, iters=15):
    """Huber-IRLS 下的 ridge：每步解加权正规方程 (X'WX + λI)β = X'Wy。"""
    Xc, yc, xm, ym = _center(Xtr, ytr)
    Xva_c = Xva - xm
    p = Xc.shape[1]
    eye = np.eye(p)
    best, best_mse = None, np.inf
    for lam in lams:
        beta = np.zeros(p)
        for _ in range(iters):
            resid = yc - Xc @ beta
            xi = np.quantile(np.abs(resid), 0.999)
            if xi <= 0:
                break
            w = np.where(np.abs(resid) <= xi, 1.0, xi / np.maximum(np.abs(resid), 1e-12))
            Xw = Xc * w[:, None]
            A = Xw.T @ Xc + lam * eye
            b = Xw.T @ yc
            new = np.linalg.solve(A, b)
            if np.max(np.abs(new - beta)) < 1e-9:
                beta = new
                break
            beta = new
        m = _mse(yva, Xva_c @ beta + ym)
        if m < best_mse:
            best_mse, best = m, (lam, beta)
    lam, beta = best
    return (lambda Z: (Z - xm) @ beta + ym), {"lam": lam}


# ---------------------------------------------------------------- Oracle

def oracle_features(c: np.ndarray, x: np.ndarray, model: str) -> np.ndarray:
    """只用真实协变量：模型 (a) 用 (c1,c2,c3*x)；(b) 用 (c1^2, c1c2, sgn(c3*x))。"""
    c = np.asarray(c)
    x = np.asarray(x)
    xt = x[:, None] if c.ndim == 3 else x.reshape(-1)
    c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2]
    if model == "a":
        return np.stack([c1, c2, c3 * xt], axis=-1)
    return np.stack([c1 ** 2, c1 * c2, np.sign(c3 * xt)], axis=-1)
