"""Bayes 分类器：三种决策规则 + 判别函数。

模式识别第一课的完整链路：
    p(x|w) 已知 → 后验 p(w|x) = p(x|w)P(w) / p(x)
    → 决策规则 → 分类结果

三种规则的区别只在「误判代价」：
    最小错误率   代价 0-1 对称，退化为取最大后验
    最小风险     任意代价矩阵，贝叶斯期望风险最小
    Neyman-Pearson 固定一类错误率，最小化另一类
"""
from __future__ import annotations

import numpy as np

from .density import gaussian_log_pdf, mle_gaussian, parzen_pdf, regularize_cov

TOL = 1e-12


class BayesClassifier:
    """高斯假设下的 Bayes 分类器。

    支持三种协方差约束，这是教材里最重要的一组对比：
        full     每类独立完整协方差 -> 二次判别 QDA
        diag     对角协方差         -> 朴素贝叶斯的高斯版
        shared   各类共用一个协方差 -> 线性判别 LDA
    """

    def __init__(self, cov_type: str = "full", reg: float = 1e-6):
        if cov_type not in ("full", "diag", "shared"):
            raise ValueError(f"cov_type 必须是 full/diag/shared，收到 {cov_type}")
        self.cov_type = cov_type
        self.reg = float(reg)
        self.classes_: np.ndarray | None = None
        self.priors_: dict = {}
        self.means_: dict = {}
        self.covs_: dict = {}
        self.feature_names_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None) -> "BayesClassifier":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y)
        n, d = X.shape
        self.classes_ = np.unique(y)
        self.feature_names_ = list(feature_names) if feature_names else [f"x{i+1}" for i in range(d)]

        # 先验：用频率估计
        self.priors_ = {c: float(np.mean(y == c)) for c in self.classes_}

        # 各类高斯参数
        for c in self.classes_:
            Xc = X[y == c]
            if Xc.shape[0] < 2:
                # 单样本：均值用样本，方差给一个兜底值
                self.means_[c] = Xc.mean(axis=0) if Xc.shape[0] else np.zeros(d)
                self.covs_[c] = np.eye(d)
            else:
                mu, Sigma = mle_gaussian(Xc)
                self.means_[c] = mu
                self.covs_[c] = Sigma

        # shared：所有样本共用协方差（去掉类内均值后的合并协方差）
        if self.cov_type == "shared":
            Sw = np.zeros((d, d))
            for c in self.classes_:
                Xc = X[y == c]
                if Xc.shape[0] < 2:
                    continue
                Xc_c = Xc - self.means_[c]
                Sw += Xc_c.T @ Xc_c
            Sw /= max(n - len(self.classes_), 1)
            for c in self.classes_:
                self.covs_[c] = Sw.copy()

        # diag：把非对角元素清零
        if self.cov_type == "diag":
            for c in self.classes_:
                self.covs_[c] = np.diag(np.diag(self.covs_[c]))

        for c in self.classes_:
            self.covs_[c] = regularize_cov(self.covs_[c], self.reg)
        return self

    # ---------- 判别函数 ----------
    def discriminant(self, X: np.ndarray) -> np.ndarray:
        """返回 (n_samples, n_classes) 的判别函数值。

        高斯假设下 g_i(x) = ln p(x|w_i) + ln P(w_i)，
        取对数是单调变换，不改变 argmax，但避免下溢。
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.zeros((X.shape[0], len(self.classes_)))
        for j, c in enumerate(self.classes_):
            log_lik = gaussian_log_pdf(X, self.means_[c], self.covs_[c])
            out[:, j] = log_lik + np.log(max(self.priors_[c], TOL))
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """后验概率 p(w|x)，按行归一化。"""
        g = self.discriminant(X)
        # log-sum-exp 稳定归一化
        gmax = g.max(axis=1, keepdims=True)
        e = np.exp(g - gmax)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray, cost: np.ndarray | None = None) -> np.ndarray:
        """分类。

        cost 为 (K, K) 代价矩阵，cost[i,j] = 真实类 i 判成 j 的代价。
        给 None 就是最小错误率（等价于 0-1 代价）。
        """
        proba = self.predict_proba(X)
        if cost is None:
            idx = np.argmax(proba, axis=1)
        else:
            cost = np.asarray(cost, dtype=float)
            K = len(self.classes_)
            if cost.shape != (K, K):
                raise ValueError(f"代价矩阵必须是 {K}×{K}")
            # 期望风险 R(j|x) = Σ_i cost[i,j] · P(w_i|x)
            risk = proba @ cost
            idx = np.argmin(risk, axis=1)
        return self.classes_[idx]

    def decision_boundary_mesh(
        self, xlim, ylim, resolution: int = 200
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成二维决策区域网格，用于画背景色块。"""
        xs = np.linspace(xlim[0], xlim[1], resolution)
        ys = np.linspace(ylim[0], ylim[1], resolution)
        XX, YY = np.meshgrid(xs, ys)
        grid = np.column_stack([XX.ravel(), YY.ravel()])
        proba = self.predict_proba(grid)
        idx = np.argmax(proba, axis=1)
        # 返回类别索引网格 + 最大后验概率（用于画渐变）
        return XX, YY, idx.reshape(XX.shape), proba.max(axis=1).reshape(XX.shape)


class BayesParzenClassifier:
    """非参数 Bayes 分类器：p(x|w) 用 Parzen 窗估。

    这是参数方法的对照组——不需要假设高斯，
    但需要更多样本，且要选窗宽 h。
    """

    def __init__(self, h: float = 0.5, kernel: str = "gaussian"):
        self.h = float(h)
        self.kernel = kernel
        self.classes_: np.ndarray | None = None
        self.samples_: dict = {}
        self.priors_: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesParzenClassifier":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        for c in self.classes_:
            self.samples_[c] = X[y == c]
            self.priors_[c] = float(np.mean(y == c))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.zeros((X.shape[0], len(self.classes_)))
        for j, c in enumerate(self.classes_):
            p = parzen_pdf(self.samples_[c], X, h=self.h, kernel=self.kernel)
            out[:, j] = p * max(self.priors_[c], TOL)
        s = out.sum(axis=1, keepdims=True)
        s[s < TOL] = TOL
        return out / s

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ══════════════════════════════════════════════════════════
# Neyman-Pearson（两类专用）
# ══════════════════════════════════════════════════════════


def neyman_pearson_threshold(
    scores0: np.ndarray, scores1: np.ndarray, alpha: float = 0.05
) -> dict:
    """Neyman-Pearson 判决门限。

    设 H0: 属于类 0，H1: 属于类 1。
    约束 P(判为 H1 | H0 为真) <= alpha（虚警率受限），
    在此约束下最小化漏检率。

    返回门限、实际虚警率、检出率。
    """
    scores0 = np.asarray(scores0, dtype=float).ravel()
    scores1 = np.asarray(scores1, dtype=float).ravel()

    # 门限取类 0 分数的上 alpha 分位点
    tau = float(np.quantile(scores0, 1.0 - alpha))

    # 实际虚警率 = 类 0 中被判为类 1 的比例
    far = float(np.mean(scores0 > tau))
    # 检出率 = 类 1 中被正确判出的比例
    pd = float(np.mean(scores1 > tau))

    return {"threshold": tau, "false_alarm_rate": far, "detection_rate": pd, "alpha": float(alpha)}


def roc_curve(scores0: np.ndarray, scores1: np.ndarray, n_points: int = 200) -> dict:
    """ROC 曲线：遍历门限，算 (虚警率, 检出率)。

    AUC 用梯形法积分，衡量两类可分性。
    """
    scores0 = np.asarray(scores0, dtype=float).ravel()
    scores1 = np.asarray(scores1, dtype=float).ravel()
    lo = float(min(scores0.min(), scores1.min()))
    hi = float(max(scores0.max(), scores1.max()))
    taus = np.linspace(hi + 1e-9, lo - 1e-9, int(n_points))

    far = np.array([np.mean(scores0 > t) for t in taus])
    pd = np.array([np.mean(scores1 > t) for t in taus])

    # 按 FAR 升序排，保证积分单调
    order = np.argsort(far)
    far_s, pd_s = far[order], pd[order]
    auc = float(np.trapezoid(pd_s, far_s))
    return {"far": far_s, "pd": pd_s, "auc": auc, "thresholds": taus[order]}


# ══════════════════════════════════════════════════════════
# 评价指标
# ══════════════════════════════════════════════════════════


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def classification_report(y_true, y_pred, labels=None) -> dict:
    """准确率 + 每类精确率/召回率/F1。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    cm = confusion_matrix(y_true, y_pred, labels)
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else 0.0

    per_class = {}
    for i, c in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[str(c)] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i, :].sum())}

    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per_class, "confusion": cm, "labels": list(labels)}
