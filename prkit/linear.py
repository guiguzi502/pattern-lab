"""线性与近邻方法：Fisher 判别、LDA、感知器、kNN。

这几个是模式识别第二、三章的主角，
共同特点：决策面是线性的（kNN 除外，但它足够重要）。
"""
from __future__ import annotations

import numpy as np

TOL = 1e-12


# ══════════════════════════════════════════════════════════
# Fisher 线性判别
# ══════════════════════════════════════════════════════════


def fisher_lda(X: np.ndarray, y: np.ndarray) -> dict:
    """Fisher 线性判别分析（两类或多类）。

    核心思想：找一个投影方向 w，让类间散度最大、类内散度最小。
        J(w) = wᵀ S_b w / wᵀ S_w w
    最优 w 是 S_w^{-1} S_b 最大特征值对应的特征向量。
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y)
    n, d = X.shape
    classes = np.unique(y)
    K = len(classes)
    if K < 2:
        raise ValueError("至少需要两个类别")

    mu_all = X.mean(axis=0)
    means = {c: X[y == c].mean(axis=0) for c in classes}

    # 类间散度 S_b
    Sb = np.zeros((d, d))
    # 类内散度 S_w
    Sw = np.zeros((d, d))
    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        diff = (means[c] - mu_all).reshape(-1, 1)
        Sb += nc * (diff @ diff.T)
        if nc > 1:
            Xc_c = Xc - means[c]
            Sw += Xc_c.T @ Xc_c

    Sw += 1e-6 * np.eye(d)  # 正则化，避免奇异

    # 求 S_w^{-1} S_b 的特征分解
    try:
        M = np.linalg.solve(Sw, Sb)
    except np.linalg.LinAlgError:
        M = np.linalg.pinv(Sw) @ Sb

    eigvals, eigvecs = np.linalg.eig(M)
    order = np.argsort(-np.real(eigvals))
    eigvals = np.real(eigvals)[order]
    eigvecs = np.real(eigvecs)[:, order]

    # 最多取 K-1 个判别方向
    n_comp = min(K - 1, d)
    W = eigvecs[:, :n_comp]

    # 归一化每个方向
    for j in range(n_comp):
        norm = np.linalg.norm(W[:, j])
        if norm > TOL:
            W[:, j] /= norm

    return {
        "W": W,
        "eigenvalues": eigvals[:n_comp],
        "Sb": Sb,
        "Sw": Sw,
        "means": means,
        "n_components": n_comp,
    }


def fisher_project(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """把数据投影到判别方向上。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return X @ np.asarray(W, dtype=float)


class FisherClassifier:
    """Fisher 判别分类器：投影后按类均值最近 + 阈值分割。

    两类时是单阈值；多类时用投影空间里的最近类均值。
    """

    def __init__(self):
        self.W_ = None
        self.classes_ = None
        self.proj_means_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y)
        res = fisher_lda(X, y)
        self.W_ = res["W"]
        self.classes_ = np.unique(y)
        Z = fisher_project(X, self.W_)
        for c in self.classes_:
            self.proj_means_[c] = Z[y == c].mean(axis=0)
            self.priors_[c] = float(np.mean(y == c))
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Z = fisher_project(X, self.W_)
        dists = np.zeros((Z.shape[0], len(self.classes_)))
        for j, c in enumerate(self.classes_):
            dists[:, j] = np.linalg.norm(Z - self.proj_means_[c], axis=1)
        return self.classes_[np.argmin(dists, axis=1)]

    def predict_proba(self, X):
        """软化的后验概率。

        Fisher 判别本身只给「离哪个类均值更近」，不给概率。
        这里把负距离做 softmax，得到一个可用的软输出——
        严格来说不是概率密度的比值，但单调关系正确，可用于画 ROC / 决策区域。
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Z = fisher_project(X, self.W_)
        d = np.zeros((Z.shape[0], len(self.classes_)))
        for j, c in enumerate(self.classes_):
            d[:, j] = np.linalg.norm(Z - self.proj_means_[c], axis=1)

        scale = float(d.std()) if d.std() > 1e-9 else 1.0
        logits = -d / scale
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)


# ══════════════════════════════════════════════════════════
# 感知器
# ══════════════════════════════════════════════════════════


class Perceptron:
    """经典感知器（Rosenblatt）。

    只对线性可分数据保证收敛，这是它最大的局限，
    也是后面要讲 SVM 的动机。
    """

    def __init__(self, lr: float = 0.1, n_epochs: int = 100, shuffle: bool = True, seed: int = 0):
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.w_ = None
        self.b_ = 0.0
        self.classes_ = None
        self.history_ = []

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("感知器只处理两类问题")

        # 标签映射到 ±1
        t = np.where(y == self.classes_[1], 1.0, -1.0)
        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        self.history_ = []

        for epoch in range(self.n_epochs):
            idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
            errors = 0
            for i in idx:
                if t[i] * (X[i] @ self.w_ + self.b_) <= 0:
                    self.w_ += self.lr * t[i] * X[i]
                    self.b_ += self.lr * t[i]
                    errors += 1
            self.history_.append({"epoch": epoch, "errors": errors,
                                  "w": self.w_.copy(), "b": self.b_})
            if errors == 0:
                break
        return self

    def decision_function(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X @ self.w_ + self.b_

    def predict(self, X):
        s = self.decision_function(X)
        return np.where(s >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        """感知器只有符号输出，用 sigmoid 软化得到可画 ROC 的连续分数。"""
        s = np.asarray(self.decision_function(X), dtype=float)
        scale = float(np.median(np.abs(s))) if np.median(np.abs(s)) > 1e-9 else 1.0
        p1 = 1.0 / (1.0 + np.exp(-np.clip(s / scale, -500, 500)))
        return np.column_stack([1.0 - p1, p1])


# ══════════════════════════════════════════════════════════
# k 近邻
# ══════════════════════════════════════════════════════════


class KNN:
    """k 近邻分类器。

    没有训练过程（惰性学习），预测时才算距离。
    这是「参数方法 vs 非参数方法」对比的另一极。
    """

    def __init__(self, k: int = 5, weights: str = "uniform", metric: str = "euclidean"):
        self.k = int(k)
        self.weights = weights   # uniform | distance
        self.metric = metric
        self.X_ = None
        self.y_ = None

    def fit(self, X, y):
        self.X_ = np.atleast_2d(np.asarray(X, dtype=float))
        self.y_ = np.asarray(y)
        return self

    def _dist(self, X):
        """计算 (n_query, n_train) 距离矩阵。"""
        if self.metric == "manhattan":
            return np.abs(X[:, None, :] - self.X_[None, :, :]).sum(axis=2)
        # 欧氏：用展开式避免三维张量
        x2 = (X ** 2).sum(axis=1)[:, None]
        t2 = (self.X_ ** 2).sum(axis=1)[None, :]
        d2 = x2 + t2 - 2 * X @ self.X_.T
        return np.sqrt(np.maximum(d2, 0.0))

    def predict_proba(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        classes = np.unique(self.y_)
        D = self._dist(X)
        k = min(self.k, self.X_.shape[0])

        out = np.zeros((X.shape[0], len(classes)))
        for i in range(X.shape[0]):
            idx = np.argpartition(D[i], k - 1)[:k]
            if self.weights == "distance":
                w = 1.0 / np.maximum(D[i, idx], 1e-9)
            else:
                w = np.ones(k)
            for j, c in enumerate(classes):
                out[i, j] = w[self.y_[idx] == c].sum()
        s = out.sum(axis=1, keepdims=True)
        s[s < TOL] = TOL
        return out / s

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.unique(self.y_)[np.argmax(proba, axis=1)]


# ══════════════════════════════════════════════════════════
# 降维：PCA
# ══════════════════════════════════════════════════════════


def pca(X: np.ndarray, n_components: int = 2) -> dict:
    """主成分分析（用 SVD，不用协方差特征分解，数值更稳）。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu = X.mean(axis=0)
    Xc = X - mu

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # 解释方差比
    var = (S ** 2) / max(X.shape[0] - 1, 1)
    total = var.sum()
    ratio = var / total if total > TOL else np.zeros_like(var)

    k = min(int(n_components), Vt.shape[0])
    W = Vt[:k].T
    Z = Xc @ W
    return {
        "components": W,
        "explained_variance_ratio": ratio[:k],
        "cumulative_ratio": np.cumsum(ratio)[:k],
        "mean": mu,
        "projected": Z,
        "all_ratio": ratio,
    }
