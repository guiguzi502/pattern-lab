"""支持向量机：软间隔 + 核技巧 + SMO 求解。

这是模式识别里最「高级」的一块，也是手写实现最能体现功底的地方。
不用 sklearn，SMO 从零写。

核心优化问题：
    min  ½‖w‖² + C Σ ξ_i
    s.t. y_i(wᵀφ(x_i) + b) >= 1 - ξ_i,  ξ_i >= 0

对偶形式：
    max  Σ α_i - ½ ΣΣ α_i α_j y_i y_j K(x_i, x_j)
    s.t. 0 <= α_i <= C,  Σ α_i y_i = 0

SMO 每次挑两个 α 做解析解，反复直到收敛。
"""
from __future__ import annotations

import numpy as np

TOL = 1e-6


# ══════════════════════════════════════════════════════════
# 核函数
# ══════════════════════════════════════════════════════════


def kernel_linear(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T


def kernel_rbf(A: np.ndarray, B: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    a2 = (A ** 2).sum(axis=1)[:, None]
    b2 = (B ** 2).sum(axis=1)[None, :]
    d2 = np.maximum(a2 + b2 - 2 * A @ B.T, 0.0)
    return np.exp(-gamma * d2)


def kernel_poly(A: np.ndarray, B: np.ndarray, degree: int = 3, coef0: float = 1.0) -> np.ndarray:
    return (A @ B.T + coef0) ** degree


def make_kernel(name: str, **kw):
    name = name.lower()
    if name == "linear":
        return lambda A, B: kernel_linear(A, B)
    if name == "rbf":
        g = float(kw.get("gamma", 1.0))
        return lambda A, B: kernel_rbf(A, B, g)
    if name == "poly":
        d = int(kw.get("degree", 3))
        c = float(kw.get("coef0", 1.0))
        return lambda A, B: kernel_poly(A, B, d, c)
    raise ValueError(f"未知核函数: {name}")


# ══════════════════════════════════════════════════════════
# SMO
# ══════════════════════════════════════════════════════════


class SVM:
    """支持向量机（SMO 求解，支持软间隔与核技巧）。"""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 200,
        seed: int = 0,
    ):
        self.C = float(C)
        self.kernel = kernel
        self.gamma = float(gamma)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.rng = np.random.default_rng(seed)

        self.X_ = None
        self.y_ = None       # ±1
        self.alpha_ = None
        self.b_ = 0.0
        self.classes_ = None
        self.K_ = None
        self.history_ = []
        self._scale = 1.0

    def _kfun(self):
        return make_kernel(self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM 只处理两类问题")

        self.X_ = X
        self.y_ = np.where(y == self.classes_[1], 1.0, -1.0)
        n = X.shape[0]

        kfun = self._kfun()
        self.K_ = kfun(X, X)

        C = self.C
        tol = self.tol

        alpha = np.zeros(n)
        b = 0.0

        # 误差缓存 E_i = f(x_i) - y_i，初始 f=0 故 E = -y
        E = -self.y_.copy()

        self.history_ = []
        it = 0
        # 只要还有 KKT 违背就继续迭代（总轮数由 max_iter 兜底），
        # stall 只用来防止「完全动不了」时白转
        stall = 0
        max_stall = 30

        while it < self.max_iter and stall < max_stall:
            num_changed = 0

            # 内层：反复扫描，直到这一遍没有任何 α 被更新
            inner = 0
            while inner < 50:
                changed_this_pass = 0
                # 按 |E| 降序扫描，误差大的样本优先（Platt 的最大步长启发式）
                for i in np.argsort(-np.abs(E)):
                    yi = self.y_[i]
                    Ei = float(E[i])
                    ai_old = alpha[i]

                    # KKT 违背判定（两个方向都算）
                    viol_low = (yi * Ei < -tol) and (alpha[i] < C - 1e-12)
                    viol_high = (yi * Ei > tol) and (alpha[i] > 1e-12)
                    if not (viol_low or viol_high):
                        continue

                    # 选 j：按 |E_i - E_j| 降序尝试，第一个能产生有效步长的胜出。
                    # 只取 argmax 会在 j 已顶到边界时白白卡住（经典 SMO 陷阱）。
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    cand = np.flatnonzero(mask)
                    if cand.size == 0:
                        continue
                    cand = cand[np.argsort(-np.abs(E[cand] - Ei))]

                    updated = False
                    for j in cand:
                        j = int(j)
                        Ej = float(E[j])
                        aj_old = alpha[j]
                        yj = self.y_[j]

                        # 上下界
                        if yi != yj:
                            L = max(0.0, aj_old - ai_old)
                            H = min(C, C + aj_old - ai_old)
                        else:
                            L = max(0.0, ai_old + aj_old - C)
                            H = min(C, ai_old + aj_old)
                        if L >= H - 1e-12:
                            continue

                        eta = self.K_[i, i] + self.K_[j, j] - 2.0 * self.K_[i, j]
                        if eta <= 1e-12:
                            continue

                        aj_new = float(np.clip(aj_old + yj * (Ei - Ej) / eta, L, H))
                        if abs(aj_new - aj_old) < 1e-12:
                            continue
                        ai_new = float(ai_old + yi * yj * (aj_old - aj_new))

                        alpha[i], alpha[j] = ai_new, aj_new

                        # 偏置更新
                        b1 = b - Ei - yi * (ai_new - ai_old) * self.K_[i, i] \
                             - yj * (aj_new - aj_old) * self.K_[i, j]
                        b2 = b - Ej - yi * (ai_new - ai_old) * self.K_[i, j] \
                             - yj * (aj_new - aj_old) * self.K_[j, j]
                        if 0 < ai_new < C:
                            b = b1
                        elif 0 < aj_new < C:
                            b = b2
                        else:
                            b = 0.5 * (b1 + b2)

                        updated = True
                        break

                    if not updated:
                        continue

                    # 重算误差缓存
                    E = (alpha * self.y_) @ self.K_ + b - self.y_

                    changed_this_pass += 1
                    num_changed += 1

                inner += 1
                if changed_this_pass == 0:
                    break

            it += 1
            stall = 0 if num_changed > 0 else stall + 1

            # KKT 违背统计，严格按三分类判据：
            #   α=0      → y·f ≥ 1
            #   0<α<C    → y·f = 1
            #   α=C      → y·f ≤ 1
            yf = self.y_ * (E + self.y_)      # E = f - y  ⇒  f = E + y
            n_viol = int(np.sum(
                ((alpha <= 1e-9) & (yf < 1 - tol)) |
                ((alpha >= C - 1e-9) & (yf > 1 + tol)) |
                ((alpha > 1e-9) & (alpha < C - 1e-9) & (np.abs(yf - 1) > tol))
            ))
            self.history_.append({
                "iter": it,
                "changed": num_changed,
                "violations": n_viol,
                "alpha_norm": float(np.sum(alpha * alpha)),
            })
            if n_viol == 0:
                break

        # 用自由支持向量（0<α<C）重算偏置：这通常比循环里累加的 b 更准，
        # 但个别数据集会反而变差，所以「谁违背少用谁」，保证绝不回退
        free = np.flatnonzero((alpha > 1e-9) & (alpha < C - 1e-9))
        if free.size > 0:
            b_free = float(np.mean(self.y_[free] - (alpha * self.y_) @ self.K_[:, free]))

            def _nviol(bb):
                yf = self.y_ * ((alpha * self.y_) @ self.K_ + bb)
                return int(np.sum(
                    ((alpha <= 1e-9) & (yf < 1 - tol)) |
                    ((alpha >= C - 1e-9) & (yf > 1 + tol)) |
                    ((alpha > 1e-9) & (alpha < C - 1e-9) & (np.abs(yf - 1) > tol))
                ))

            if _nviol(b_free) <= _nviol(b):
                b = b_free

        self.alpha_ = alpha
        self.b_ = float(b)

        # 用训练集决策值的中位绝对偏差当 sigmoid 缩放尺度，
        # 让概率输出在 (0,1) 上有合理的渐变层次（margin 可能很大或很小）
        f_tr = self.decision_function(X)
        spread = float(np.median(np.abs(f_tr)))
        self._scale = spread if spread > 1e-6 else 1.0
        return self

    def _f(self, i: int, alpha: np.ndarray, b: float) -> float:
        """决策函数值 f(x_i) = Σ α_j y_j K(x_j, x_i) + b。"""
        return float(np.sum(alpha * self.y_ * self.K_[:, i]) + b)

    def decision_function(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        kfun = self._kfun()
        Kx = kfun(X, self.X_)
        return Kx @ (self.alpha_ * self.y_) + self.b_

    def predict_proba(self, X):
        """把决策函数值映射成概率（Platt sigmoid 缩放）。

        SVM 本身输出的是到超平面的有符号距离，不是概率。
        Platt 的做法是用一个 logistic 函数压到 (0,1)：
            P(y=1 | f) = 1 / (1 + exp(A·f + B))
        这里做自适应缩放：用训练集决策值的标准差当特征尺度，
        否则 margin 动辄几十，sigmoid 会瞬间饱和，决策区域就没渐变层次了。
        严格校准仍需在独立验证集上拟合 A/B，教学场景不必。
        """
        f = np.asarray(self.decision_function(X), dtype=float)
        if not hasattr(self, "_scale") or self._scale is None:
            self._scale = 1.0
        z = np.clip(f / self._scale, -500, 500)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        s = self.decision_function(X)
        return np.where(s >= 0, self.classes_[1], self.classes_[0])

    @property
    def support_vectors(self):
        """支持向量索引（α > 0）。"""
        if self.alpha_ is None:
            return np.array([], dtype=int)
        return np.flatnonzero(self.alpha_ > 1e-6)

    def decision_boundary_2d(self, xlim, ylim, resolution: int = 200):
        """二维决策边界网格（用于画等高线）。"""
        xs = np.linspace(xlim[0], xlim[1], resolution)
        ys = np.linspace(ylim[0], ylim[1], resolution)
        XX, YY = np.meshgrid(xs, ys)
        grid = np.column_stack([XX.ravel(), YY.ravel()])
        Z = self.decision_function(grid).reshape(XX.shape)
        return XX, YY, Z
