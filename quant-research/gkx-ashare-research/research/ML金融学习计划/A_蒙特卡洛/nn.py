"""
A 层 · 神经网络 NN1–NN5（纯 NumPy 实现）

结构（IA Table A.1 注）：NN1=32，NN2=32-16，NN3=32-16-8，NN4=32-16-8-4，
NN5=32-16-8-4-2，隐藏层 ReLU。五个正则化手段全部实现：
    1. L1 惩罚（λ1 ∈ {1e-5, 1e-4, 1e-3}）
    2. 学习率网格选择（Adam，lr ∈ {0.01, 0.1}；不声称实现逐轮衰减）
    3. 早停（patience = 5 个 epoch）
    4. 批归一化（每个隐藏层，γ、β 可学）
    5. 多随机种子集成（默认 10 个）

注意：论文的神经网络**没有 dropout**。正文出现的 "dropout" 指的是随机森林
每次分裂随机抽取一部分预测变量，不是神经网络里的 dropout。
"""
from __future__ import annotations

import numpy as np

HIDDEN = {1: (32,), 2: (32, 16), 3: (32, 16, 8), 4: (32, 16, 8, 4), 5: (32, 16, 8, 4, 2)}
BN_EPS = 1e-8


class Net:
    def __init__(self, n_in: int, hidden: tuple, rng: np.random.Generator):
        dims = (n_in,) + hidden
        self.W, self.b, self.g, self.be = [], [], [], []
        for i in range(len(hidden)):
            fan_in = dims[i]
            self.W.append(rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, hidden[i])))
            self.b.append(np.zeros(hidden[i]))
            self.g.append(np.ones(hidden[i]))
            self.be.append(np.zeros(hidden[i]))
        self.wo = rng.normal(0, np.sqrt(2.0 / hidden[-1]), hidden[-1])
        self.bo = 0.0

    def params(self):
        return self.W + self.b + self.g + self.be + [self.wo, np.array(self.bo)]

    def forward(self, X, train=True):
        self.cache = []
        H = X
        for i in range(len(self.W)):
            Z = H @ self.W[i] + self.b[i]
            if train:
                mu = Z.mean(axis=0)
                var = Z.var(axis=0)
                Zh = (Z - mu) / np.sqrt(var + BN_EPS)
            else:
                mu, var = self.running[i]
                Zh = (Z - mu) / np.sqrt(var + BN_EPS)
            Zb = self.g[i] * Zh + self.be[i]
            Hn = np.maximum(Zb, 0.0)
            self.cache.append((H, Z, Zh, Zb, Hn, mu, var))
            H = Hn
        return H @ self.wo + self.bo

    def backward(self, dy, lam1, n):
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.W)
        grads_g = [None] * len(self.W)
        grads_be = [None] * len(self.W)
        H_last = self.cache[-1][4]
        g_wo = H_last.T @ dy + lam1 * np.sign(self.wo)
        g_bo = dy.sum()
        dH = np.outer(dy, self.wo)
        for i in reversed(range(len(self.W))):
            Hprev, Z, Zh, Zb, Hn, mu, var = self.cache[i]
            dZb = dH * (Zb > 0)
            grads_g[i] = (dZb * Zh).sum(axis=0)
            grads_be[i] = dZb.sum(axis=0)
            dZh = dZb * self.g[i]
            dvar = (dZh * (Z - mu) * -0.5 * (var + BN_EPS) ** -1.5).sum(axis=0)
            dmu = (dZh * -1.0 / np.sqrt(var + BN_EPS)).sum(axis=0) + dvar * (-2.0 * (Z - mu)).sum(axis=0) / len(Z)
            dZ = dZh / np.sqrt(var + BN_EPS) + dvar * 2.0 * (Z - mu) / len(Z) + dmu / len(Z)
            grads_W[i] = Hprev.T @ dZ + lam1 * np.sign(self.W[i])
            grads_b[i] = dZ.sum(axis=0)
            dH = dZ @ self.W[i].T
        return grads_W, grads_b, grads_g, grads_be, g_wo, g_bo


def train_one(Xtr, ytr, Xva, yva, hidden, lam1, lr, seed, epochs=100, patience=5,
              batch=10000, beta1=0.9, beta2=0.999, eps=1e-8, bn_mom=0.9):
    """训练一个网络，返回验证 MSE 最好的一组权重。"""
    if len(ytr) < 2 or not len(yva) or epochs < 1 or patience < 1 or batch < 2:
        raise ValueError("NN requires training >=2, validation >=1, and positive epochs/patience")
    if not all(np.isfinite(a).all() for a in (Xtr, ytr, Xva, yva)):
        raise ValueError("NN inputs must be finite")
    rng = np.random.default_rng(seed)
    ym, ys = ytr.mean(), ytr.std() or 1.0
    yt = (ytr - ym) / ys
    yv = (yva - ym) / ys

    net = Net(Xtr.shape[1], hidden, rng)
    net.running = [(np.zeros(h), np.ones(h)) for h in hidden]
    flat = net.params()
    m = [np.zeros_like(p) for p in flat]
    v = [np.zeros_like(p) for p in flat]
    best_mse, best_state = np.inf, None
    bad, step, n = 0, 0, len(yt)

    def set_params(params):
        nW = len(net.W)
        net.W = params[:nW]
        net.b = params[nW:2 * nW]
        net.g = params[2 * nW:3 * nW]
        net.be = params[3 * nW:4 * nW]
        net.wo = params[-2]
        net.bo = float(params[-1])

    best_epoch = None
    for epoch in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            set_params(flat)
            pred = net.forward(Xtr[b], train=True)
            for i in range(len(net.W)):
                _, _, _, _, _, mu, var = net.cache[i]
                net.running[i] = (bn_mom * net.running[i][0] + (1 - bn_mom) * mu,
                                  bn_mom * net.running[i][1] + (1 - bn_mom) * var)
            dy = 2.0 * (pred - yt[b]) / len(b)
            gW, gb, gg, gbe, gwo, gbo = net.backward(dy, lam1, len(b))
            gl = gW + gb + gg + gbe + [gwo, np.array(gbo)]
            step += 1
            for k in range(len(flat)):
                m[k] = beta1 * m[k] + (1 - beta1) * gl[k]
                v[k] = beta2 * v[k] + (1 - beta2) * gl[k] ** 2
                mh = m[k] / (1 - beta1 ** step)
                vh = v[k] / (1 - beta2 ** step)
                flat[k] = flat[k] - lr * mh / (np.sqrt(vh) + eps)

        set_params(flat)
        # 用整个训练集重算一遍批归一化的滑动统计量，避免小批数下 running 统计没收敛
        net.forward(Xtr, train=True)
        for i in range(len(net.W)):
            net.running[i] = (net.cache[i][5].copy(), net.cache[i][6].copy())
        vm = float(np.mean((net.forward(Xva, train=False) - yv) ** 2))
        if vm < best_mse - 1e-12:
            best_mse = vm
            best_state = ([p.copy() for p in flat],
                          [(a.copy(), b.copy()) for a, b in net.running])
            bad = 0
            best_epoch = epoch + 1
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise FloatingPointError("NN did not produce a finite validation checkpoint")
    set_params(best_state[0])
    net.running = [(a.copy(), b.copy()) for a, b in best_state[1]]
    net.best_epoch = best_epoch
    return net, best_mse, ym, ys


def fit_nn(Xtr, ytr, Xva, yva, depth, grid, ensemble=10, seed=0, epochs=100, patience=5,
           return_nets=False):
    """用验证集上两种子平均预测的 MSE 调参，再训练完整集成。"""
    if depth not in HIDDEN or ensemble < 1:
        raise ValueError("NN depth must be 1..5 and ensemble must be positive")
    hidden = HIDDEN[depth]
    best, best_mse = None, np.inf
    for lam1, lr in grid:
        tuning_predictions = []
        for e in range(2):
            net, vm, ym, ys = train_one(Xtr, ytr, Xva, yva, hidden, lam1, lr,
                                        seed * 1000 + e, epochs, patience)
            tuning_predictions.append(net.forward(Xva, train=False) * ys + ym)
        mse = float(np.mean((np.mean(tuning_predictions, axis=0) - yva) ** 2))
        if mse < best_mse:
            best_mse, best = mse, (lam1, lr)
    if best is None:
        raise ValueError("No finite candidate in NN validation grid")
    lam1, lr = best

    models = [train_one(Xtr, ytr, Xva, yva, hidden, lam1, lr,
                        seed * 1000 + 100 + e, epochs, patience)
              for e in range(ensemble)]

    def predict(Z):
        return np.mean([net.forward(Z, train=False) * ys + ym for net, _, ym, ys in models], axis=0)

    info = {"lam1": lam1, "lr": lr, "ens": ensemble,
            "tuning_seeds": [seed * 1000 + e for e in range(2)],
            "ensemble_seeds": [seed * 1000 + 100 + e for e in range(ensemble)],
            "best_epochs": [net.best_epoch for net, _, _, _ in models],
            "selection": "validation MSE of two-seed mean prediction"}
    if return_nets:
        return predict, info, models
    return predict, info


def input_grads(net: "Net", X: np.ndarray) -> np.ndarray:
    """输出对输入的梯度 dg/dX，形状 (n, n_in)，用 running 统计量走推理模式。"""
    net.forward(X, train=False)
    dH = np.repeat(net.wo[None, :], len(X), axis=0)
    for i in reversed(range(len(net.W))):
        _, Z, _, Zb, _, _, var = net.cache[i]
        dZb = dH * (Zb > 0)
        dZh = dZb * net.g[i]
        dZ = dZh / np.sqrt(var + BN_EPS)
        dH = dZ @ net.W[i].T
    return dH


def ssd(net: "Net", X: np.ndarray) -> np.ndarray:
    """论文 §1.9 脚注 27 的 SSD：训练集上 (∂g/∂z_j)² 之和。"""
    g = input_grads(net, X)
    return (g ** 2).sum(axis=0)
