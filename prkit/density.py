"""概率密度估计：参数方法与非参数方法。

只依赖 numpy。这是 Bayes 分类器的上游——
分类器本身只是把 p(x|w) 和先验拼起来，真正难的是估出 p(x|w)。
"""
from __future__ import annotations

import numpy as np

TOL = 1e-12


# ══════════════════════════════════════════════════════════
# 参数方法
# ══════════════════════════════════════════════════════════


def mle_gaussian(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """最大似然估计多元高斯参数。

    返回 (mu, Sigma)，Sigma 用有偏估计（除以 n，不是 n-1）——
    这是 MLE 的正确定义，教材多数也这么写。
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / max(n, 1)
    # 检查奇异：样本数少于维度时必然奇异
    return mu, Sigma


def regularize_cov(Sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """给协方差加一点对角扰动，避免奇异导致求逆失败。"""
    d = Sigma.shape[0]
    return Sigma + eps * np.eye(d)


def gaussian_log_pdf(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """多元高斯对数密度。

    用 Cholesky 分解求 log|Σ| 和二次型，比 det/inv 数值稳定。
    维度高时直接算 det 会溢出。
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu = np.asarray(mu, dtype=float).ravel()
    d = mu.size

    Sigma = regularize_cov(np.asarray(Sigma, dtype=float), 1e-9)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        Sigma = regularize_cov(Sigma, 1e-3)
        L = np.linalg.cholesky(Sigma)

    Xc = X - mu
    # 解 L z = Xc^T，得 z = L^{-1} Xc^T
    z = np.linalg.solve(L, Xc.T)
    quad = np.sum(z ** 2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (quad + log_det + d * np.log(2 * np.pi))


# ══════════════════════════════════════════════════════════
# 非参数方法：Parzen 窗与 kNN
# ══════════════════════════════════════════════════════════


def kernel_gaussian(u: np.ndarray) -> np.ndarray:
    """标准正态核 φ(u) = (2π)^{-1/2} exp(-u²/2)。"""
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


def kernel_uniform(u: np.ndarray) -> np.ndarray:
    """方窗核：|u| <= 0.5 时为 1。"""
    return (np.abs(u) <= 0.5).astype(float)


def kernel_epanechnikov(u: np.ndarray) -> np.ndarray:
    """Epanechnikov 核：|u| <= 1 时为 0.75(1-u²)，均方误差意义下最优。"""
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    m = np.abs(u) <= 1.0
    out[m] = 0.75 * (1.0 - u[m] ** 2)
    return out


KERNELS = {
    "gaussian": kernel_gaussian,
    "uniform": kernel_uniform,
    "epanechnikov": kernel_epanechnikov,
}


def parzen_pdf(
    X_train: np.ndarray,
    X_query: np.ndarray,
    h: float = 1.0,
    kernel: str = "gaussian",
) -> np.ndarray:
    """Parzen 窗密度估计。

    p(x) = 1/(n·h^d) Σ_i φ((x - x_i)/h)

    高维时 h^d 会让密度迅速趋零，所以内部改在对数域算，
    但这里为了教学清晰仍返回线性密度（界面会限制维度）。
    """
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
    n, d = X_train.shape
    h = max(float(h), 1e-6)
    phi = KERNELS.get(kernel, kernel_gaussian)

    # 逐点算距离，避免构造 n×m×d 大张量
    out = np.zeros(X_query.shape[0])
    for j, x in enumerate(X_query):
        diff = (x - X_train) / h          # (n, d)
        # 各维核乘积
        k_prod = np.prod(phi(diff), axis=1) if d > 1 else phi(diff.ravel())
        out[j] = k_prod.sum() / (n * h ** d)
    return out


def knn_pdf(
    X_train: np.ndarray,
    X_query: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """k 近邻密度估计。

    p(x) = k / (n · V_k)，V_k 是包含 k 个样本的最小超球体积。

    注意：当查询点与某个训练点完全重合时 r_k=0，理论密度是无穷大。
    这里返回 np.inf 而不是造一个假的有限值——让调用方看见「重合」这件事，
    而不是被 1e-9 的钳位糊弄出一个荒谬的天文数字。

    返回 (密度, 第 k 近邻距离)。
    """
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
    n, d = X_train.shape
    k = int(np.clip(k, 1, n))

    # 单位球体积 V_d = π^(d/2) / Γ(d/2 + 1)
    from math import gamma, pi

    unit_vol = pi ** (d / 2.0) / gamma(d / 2.0 + 1.0)

    dens = np.zeros(X_query.shape[0])
    rk = np.zeros(X_query.shape[0])
    for j, x in enumerate(X_query):
        dist = np.linalg.norm(X_train - x, axis=1)
        r = float(np.sort(dist)[k - 1])
        rk[j] = r
        if r <= 1e-12:
            # 查询点与训练点重合，密度理论无穷
            dens[j] = np.inf
        else:
            vol = unit_vol * r ** d
            dens[j] = k / (n * vol)
    return dens, rk


# ══════════════════════════════════════════════════════════
# 直方图（一维/二维，用于直观展示）
# ══════════════════════════════════════════════════════════


def hist_pdf_1d(x: np.ndarray, bins: int = 30, smooth_h: float = 0.0):
    """一维直方图密度，可选 Parzen 平滑。"""
    x = np.asarray(x, dtype=float).ravel()
    counts, edges = np.histogram(x, bins=int(bins), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if smooth_h > 0 and x.size > 1:
        dens = parzen_pdf(x.reshape(-1, 1), centers.reshape(-1, 1), h=smooth_h)
        return centers, dens
    return centers, counts
