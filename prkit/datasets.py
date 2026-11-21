"""数据集生成：经典二维形状 + 高斯团。

都用固定种子，保证每次打开界面看到的数据一样，
方便对照不同算法的决策边界。
"""
from __future__ import annotations

import numpy as np


def make_moons(n_per_class: int = 100, noise: float = 0.15, seed: int = 42):
    """两个交错的半月形。线性不可分的经典例子。"""
    rng = np.random.default_rng(seed)
    n = int(n_per_class)

    t0 = np.linspace(0, np.pi, n)
    x0 = np.column_stack([np.cos(t0), np.sin(t0)])

    t1 = np.linspace(0, np.pi, n)
    x1 = np.column_stack([1 - np.cos(t1), 0.5 - np.sin(t1)])

    X = np.vstack([x0, x1]) + rng.normal(0, noise, (2 * n, 2))
    y = np.array([0] * n + [1] * n)
    return X, y


def make_circles(n_per_class: int = 100, noise: float = 0.1, factor: float = 0.4, seed: int = 42):
    """同心圆。必须用 RBF 核才能分开。"""
    rng = np.random.default_rng(seed)
    n = int(n_per_class)

    t_out = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_out = np.column_stack([np.cos(t_out), np.sin(t_out)])

    t_in = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_in = factor * np.column_stack([np.cos(t_in), np.sin(t_in)])

    X = np.vstack([x_out, x_in]) + rng.normal(0, noise, (2 * n, 2))
    y = np.array([0] * n + [1] * n)
    return X, y


def make_blobs(n_samples: int = 200, n_classes: int = 3, spread: float = 1.0, seed: int = 42):
    """各向同性高斯团。Bayes 分类器的最优场景。"""
    rng = np.random.default_rng(seed)
    n = int(n_samples)
    K = int(n_classes)

    # 类别中心均匀分布在一个圆上
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)]) * 2.5

    per = max(1, n // K)
    Xs, ys = [], []
    for i in range(K):
        Xi = rng.normal(centers[i], spread, (per, 2))
        Xs.append(Xi)
        ys.append(np.full(per, i))
    return np.vstack(Xs), np.concatenate(ys)


def make_anisotropic(n_per_class: int = 120, seed: int = 42):
    """各向异性高斯：协方差椭圆被拉长且方向不同。

    这个数据集专门用来对比 full / diag / shared 三种协方差假设——
    用 shared（LDA）会明显分不对，用 full（QDA）才对。
    """
    rng = np.random.default_rng(seed)
    n = int(n_per_class)

    # 类 0：沿 45° 拉长
    A0 = np.array([[2.0, 0.0], [0.0, 0.3]])
    R0 = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                   [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    S0 = R0 @ A0 @ R0.T

    # 类 1：沿 -45° 拉长
    A1 = np.array([[2.0, 0.0], [0.0, 0.3]])
    R1 = np.array([[np.cos(-np.pi / 4), -np.sin(-np.pi / 4)],
                   [np.sin(-np.pi / 4), np.cos(-np.pi / 4)]])
    S1 = R1 @ A1 @ R1.T

    X0 = rng.multivariate_normal([-1.2, 1.2], S0, n)
    X1 = rng.multivariate_normal([1.2, -1.2], S1, n)
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def make_xor(n_per_class: int = 100, spread: float = 0.35, seed: int = 42):
    """异或分布：四个象限对角相同类。线性完全不可分。"""
    rng = np.random.default_rng(seed)
    n = int(n_per_class)

    # 类 0 在 (+,+) 和 (-,-)
    X0 = np.vstack([
        rng.normal([1.2, 1.2], spread, (n // 2, 2)),
        rng.normal([-1.2, -1.2], spread, (n - n // 2, 2)),
    ])
    # 类 1 在 (+,-) 和 (-,+)
    X1 = np.vstack([
        rng.normal([1.2, -1.2], spread, (n // 2, 2)),
        rng.normal([-1.2, 1.2], spread, (n - n // 2, 2)),
    ])
    X = np.vstack([X0, X1])
    y = np.array([0] * X0.shape[0] + [1] * X1.shape[0])
    return X, y


def make_overlapping(n_per_class: int = 150, seed: int = 42):
    """高度重叠的两类。用来展示「理论上限」——再好的分类器也就 75% 左右。"""
    rng = np.random.default_rng(seed)
    n = int(n_per_class)
    X0 = rng.normal([0, 0], 1.0, (n, 2))
    X1 = rng.normal([1.0, 1.0], 1.0, (n, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def make_linearly_separable(n_per_class: int = 80, margin: float = 0.4, seed: int = 42):
    """严格线性可分：两类被一条宽间隔分开。

    感知器会收敛，SVM 会找到最大间隔面。
    """
    rng = np.random.default_rng(seed)
    n = int(n_per_class)

    X0 = rng.uniform([-2, -2], [0 - margin, 2], (n, 2))
    X1 = rng.uniform([0 + margin, -2], [2, 2], (n, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


DATASETS = {
    "blobs": {"fn": make_blobs, "name": "高斯团（3 类）", "desc": "各向同性高斯，Bayes 最优场景", "n_class": 3},
    "moons": {"fn": make_moons, "name": "双半月", "desc": "线性不可分，需要核方法", "n_class": 2},
    "circles": {"fn": make_circles, "name": "同心圆", "desc": "RBF 核才能分开，线性完全失效", "n_class": 2},
    "anisotropic": {"fn": make_anisotropic, "name": "各向异性高斯", "desc": "对比 full/diag/shared 协方差假设", "n_class": 2},
    "xor": {"fn": make_xor, "name": "异或分布", "desc": "四个象限对角同类，线性绝无可能", "n_class": 2},
    "overlapping": {"fn": make_overlapping, "name": "高重叠两类", "desc": "用来理解贝叶斯错误率下界", "n_class": 2},
    "separable": {"fn": make_linearly_separable, "name": "线性可分", "desc": "感知器能收敛，SVM 找最大间隔", "n_class": 2},
}


def get_dataset(key: str, **kw):
    if key not in DATASETS:
        raise KeyError(f"未知数据集: {key}，可用: {list(DATASETS)}")
    return DATASETS[key]["fn"](**kw)


def list_datasets():
    return [{"key": k, "name": v["name"], "desc": v["desc"], "n_class": v["n_class"]}
            for k, v in DATASETS.items()]


def train_test_split(X, y, test_size: float = 0.3, seed: int = 0, stratify: bool = True):
    """简单划分，尽量保持类别比例。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if stratify:
        tr_idx, te_idx = [], []
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            n_te = max(1, int(round(len(idx) * test_size)))
            te_idx.extend(idx[:n_te])
            tr_idx.extend(idx[n_te:])
        tr_idx = np.array(tr_idx)
        te_idx = np.array(te_idx)
        rng.shuffle(tr_idx)
        rng.shuffle(te_idx)
    else:
        perm = rng.permutation(n)
        n_te = max(1, int(round(n * test_size)))
        te_idx = perm[:n_te]
        tr_idx = perm[n_te:]

    return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
