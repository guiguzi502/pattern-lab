"""prkit 单元测试。

原则：能用解析解验证的，绝不用「跑通了」当成功。
"""
import numpy as np
import pytest

from prkit import (
    mle_gaussian,
    gaussian_log_pdf,
    parzen_pdf,
    knn_pdf,
    BayesClassifier,
    BayesParzenClassifier,
    neyman_pearson_threshold,
    roc_curve,
    confusion_matrix,
    classification_report,
    fisher_lda,
    FisherClassifier,
    Perceptron,
    KNN,
    pca,
    SVM,
    get_dataset,
    list_datasets,
    train_test_split,
    make_linearly_separable,
    make_moons,
    make_blobs,
    make_anisotropic,
)


# ══════════════════════════════════════════════════════════
# 密度估计
# ══════════════════════════════════════════════════════════


class TestDensity:
    def test_mle_gaussian_mean(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mu, S = mle_gaussian(X)
        assert np.allclose(mu, [3.0, 4.0])

    def test_mle_gaussian_cov_known(self):
        # 手算：X = [[0],[2]]，均值 1，方差 = ((0-1)^2+(2-1)^2)/2 = 1
        X = np.array([[0.0], [2.0]])
        mu, S = mle_gaussian(X)
        assert mu[0] == pytest.approx(1.0)
        assert S[0, 0] == pytest.approx(1.0)

    def test_gaussian_log_pdf_1d_standard(self):
        # 标准正态在 x=0 处 pdf = 1/sqrt(2π)，log = -0.5*ln(2π)
        # 注意：内部会加 1e-9 对角正则保证 Cholesky 稳定，故容差放宽到 1e-6
        X = np.array([[0.0]])
        lp = gaussian_log_pdf(X, np.array([0.0]), np.array([[1.0]]))
        assert lp[0] == pytest.approx(-0.5 * np.log(2 * np.pi), abs=1e-6)

    def test_gaussian_log_pdf_integrates_to_one(self):
        # 数值积分应得 1
        xs = np.linspace(-8, 8, 4001)
        X = xs.reshape(-1, 1)
        p = np.exp(gaussian_log_pdf(X, np.array([0.5]), np.array([[2.0]])))
        area = np.trapezoid(p, xs)
        assert area == pytest.approx(1.0, abs=1e-6)

    def test_gaussian_log_pdf_2d(self):
        # 二维单位协方差，在原点 log pdf = -ln(2π)
        X = np.array([[0.0, 0.0]])
        lp = gaussian_log_pdf(X, np.zeros(2), np.eye(2))
        assert lp[0] == pytest.approx(-np.log(2 * np.pi), abs=1e-6)

    def test_parzen_integrates_to_one(self):
        rng = np.random.default_rng(0)
        Xtr = rng.normal(0, 1, (200, 1))
        xs = np.linspace(-10, 10, 2001)
        p = parzen_pdf(Xtr, xs.reshape(-1, 1), h=0.5)
        area = np.trapezoid(p, xs)
        assert area == pytest.approx(1.0, abs=0.02)

    def test_parzen_peak_at_data(self):
        # 密度峰值应在样本密集处
        Xtr = np.array([[0.0], [0.1], [-0.1], [5.0]])
        q = np.array([[0.0], [5.0]])
        p = parzen_pdf(Xtr, q, h=0.5)
        assert p[0] > p[1]

    def test_parzen_small_h_sharper(self):
        """小 h 应让峰更尖：峰值更高、拖尾更短。

        不能直接比较某个固定点的密度——如果该点恰好没数据，
        小 h 反而给出接近 0 的值，那是正确行为。
        """
        Xtr = np.linspace(-2, 2, 20).reshape(-1, 1)
        grid = np.linspace(-3, 3, 601).reshape(-1, 1)

        p_small = parzen_pdf(Xtr, grid, h=0.05)
        p_large = parzen_pdf(Xtr, grid, h=2.0)

        # 峰值：小 h 更高
        assert p_small.max() > p_large.max()
        # 拖尾：小 h 在远离数据中心处衰减更快
        far = np.abs(grid.ravel()) > 2.5
        assert p_small[far].mean() < p_large[far].mean()

    def test_knn_pdf_formula(self):
        # 一维，查询点与训练点不重合
        Xtr = np.array([[0.0], [1.0], [2.0], [3.0]])
        q = np.array([[0.5]])
        dens, rk = knn_pdf(Xtr, q, k=1)
        # 最近邻是 0 或 1，距离 0.5；一维体积 = 2r = 1.0
        assert rk[0] == pytest.approx(0.5)
        assert dens[0] == pytest.approx(1 / (4 * 1.0))

    def test_knn_pdf_coincident_is_inf(self):
        # 查询点与训练点重合，密度理论无穷，应返回 inf 而不是假的天文数字
        Xtr = np.array([[0.0], [1.0]])
        dens, rk = knn_pdf(Xtr, np.array([[0.0]]), k=1)
        assert rk[0] == pytest.approx(0.0)
        assert np.isinf(dens[0])


# ══════════════════════════════════════════════════════════
# Bayes 分类器
# ══════════════════════════════════════════════════════════


class TestBayes:
    def _two_blobs(self, seed=0, n=200):
        rng = np.random.default_rng(seed)
        X0 = rng.normal([-2, 0], 0.8, (n, 2))
        X1 = rng.normal([2, 0], 0.8, (n, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * n + [1] * n)
        return X, y

    def test_separable_blobs_high_accuracy(self):
        X, y = self._two_blobs()
        clf = BayesClassifier(cov_type="full").fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.95

    def test_proba_sums_to_one(self):
        X, y = self._two_blobs()
        clf = BayesClassifier().fit(X, y)
        p = clf.predict_proba(X[:10])
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_priors_sum_to_one(self):
        X, y = self._two_blobs()
        clf = BayesClassifier().fit(X, y)
        assert sum(clf.priors_.values()) == pytest.approx(1.0)

    def test_shared_cov_equals_lda_shape(self):
        X, y = self._two_blobs()
        clf = BayesClassifier(cov_type="shared").fit(X, y)
        # 各类协方差应完全相同
        assert np.allclose(clf.covs_[0], clf.covs_[1])

    def test_diag_cov_is_diagonal(self):
        X, y = self._two_blobs()
        clf = BayesClassifier(cov_type="diag").fit(X, y)
        S = clf.covs_[0]
        off = S - np.diag(np.diag(S))
        assert np.allclose(off, 0.0, atol=1e-12)

    def test_full_cov_has_offdiag_on_anisotropic(self):
        X, y = make_anisotropic()
        clf = BayesClassifier(cov_type="full").fit(X, y)
        S = clf.covs_[0]
        off = S - np.diag(np.diag(S))
        assert np.abs(off).max() > 0.01

    def test_cost_matrix_shifts_decision(self):
        """代价矩阵必须能改变判决方向。

        cost[i,j] = 真实类 i 判成 j 的代价。
        若把「判成类 0」的代价设得极高，分类器应更少判类 0。
        """
        X, y = self._two_blobs()
        clf = BayesClassifier().fit(X, y)
        base = clf.predict(X)
        n_base_0 = int(np.sum(base == 0))

        # 真实为类 1 却判成类 0，代价 100（极怕漏检）
        # 真实为类 0 却判成类 1，代价 1
        cost = np.array([[0.0, 1.0],
                         [100.0, 0.0]])
        biased = clf.predict(X, cost=cost)
        n_biased_0 = int(np.sum(biased == 0))
        # 判为类 0 的数量应减少（因为判 0 更"贵"）
        assert n_biased_0 < n_base_0

    def test_three_class(self):
        X, y = make_blobs(n_samples=300, n_classes=3)
        clf = BayesClassifier().fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.9


class TestNeymanPearson:
    def test_threshold_in_range(self):
        """5% 虚警率、两类中心相距 2σ 时，检出率理论值约 64%。

        计算：门限 τ = 1.645σ（标准正态 95% 分位）
              检出率 = P(s1 > τ) = P(Z > 1.645 - 2) = Φ(0.355) ≈ 0.639
        """
        rng = np.random.default_rng(0)
        s0 = rng.normal(0, 1, 2000)
        s1 = rng.normal(2, 1, 2000)
        r = neyman_pearson_threshold(s0, s1, alpha=0.05)
        assert r["false_alarm_rate"] <= 0.08   # 允许离散误差
        assert 0.55 < r["detection_rate"] < 0.75

    def test_threshold_separable_high_detection(self):
        """两类完全分开时，检出率应接近 1。"""
        rng = np.random.default_rng(0)
        s0 = rng.normal(0, 0.5, 1000)
        s1 = rng.normal(8, 0.5, 1000)
        r = neyman_pearson_threshold(s0, s1, alpha=0.05)
        assert r["detection_rate"] > 0.99

    def test_roc_auc_separable(self):
        rng = np.random.default_rng(0)
        s0 = rng.normal(0, 0.5, 500)
        s1 = rng.normal(5, 0.5, 500)
        r = roc_curve(s0, s1)
        assert r["auc"] > 0.99

    def test_roc_auc_overlapping_near_half(self):
        rng = np.random.default_rng(0)
        s0 = rng.normal(0, 1, 2000)
        s1 = rng.normal(0, 1, 2000)
        r = roc_curve(s0, s1)
        assert 0.4 < r["auc"] < 0.6


# ══════════════════════════════════════════════════════════
# 评价指标
# ══════════════════════════════════════════════════════════


class TestMetrics:
    def test_confusion_matrix_perfect(self):
        y = np.array([0, 0, 1, 1])
        cm = confusion_matrix(y, y)
        assert cm[0, 0] == 2 and cm[1, 1] == 2
        assert cm[0, 1] == 0

    def test_confusion_matrix_known(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        cm = confusion_matrix(y_true, y_pred)
        assert cm[0, 0] == 1 and cm[0, 1] == 1
        assert cm[1, 0] == 1 and cm[1, 1] == 2

    def test_report_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        r = classification_report(y_true, y_pred)
        assert r["accuracy"] == pytest.approx(0.75)

    def test_report_precision_recall(self):
        # y_true=[0,0,1,1,1], y_pred=[0,1,1,1,1]
        # 类 1: tp=3(fp=第2个样本), fp=1, fn=0
        #   精确率 = 3/4 = 0.75，召回率 = 3/3 = 1.0
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1])
        r = classification_report(y_true, y_pred)
        assert r["per_class"]["1"]["precision"] == pytest.approx(0.75)
        assert r["per_class"]["1"]["recall"] == pytest.approx(1.0)
        # 类 0: tp=1, fp=0, fn=1 -> prec=1.0, rec=0.5
        assert r["per_class"]["0"]["precision"] == pytest.approx(1.0)
        assert r["per_class"]["0"]["recall"] == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════
# 线性方法
# ══════════════════════════════════════════════════════════


class TestFisher:
    def test_projection_direction_2d(self):
        # 两类沿 x 轴分开，判别方向应接近 x 轴
        rng = np.random.default_rng(0)
        X0 = rng.normal([-3, 0], [0.5, 1.0], (100, 2))
        X1 = rng.normal([3, 0], [0.5, 1.0], (100, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 100 + [1] * 100)
        res = fisher_lda(X, y)
        w = res["W"][:, 0]
        # |w_x| 应显著大于 |w_y|
        assert abs(w[0]) > abs(w[1]) * 2

    def test_separation_after_projection(self):
        X, y = make_blobs(n_samples=200, n_classes=2)
        res = fisher_lda(X, y)
        Z = X @ res["W"]
        m0 = Z[y == 0, 0].mean()
        m1 = Z[y == 1, 0].mean()
        s0 = Z[y == 0, 0].std()
        s1 = Z[y == 1, 0].std()
        # 类间距离应大于类内标准差（Fisher 的目标）
        assert abs(m1 - m0) > max(s0, s1)

    def test_classifier_accuracy(self):
        X, y = make_blobs(n_samples=300, n_classes=2)
        clf = FisherClassifier().fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.9


class TestPerceptron:
    def test_converges_on_separable(self):
        X, y = make_linearly_separable(margin=0.8)
        clf = Perceptron(lr=0.1, n_epochs=200).fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc == 1.0

    def test_history_recorded(self):
        X, y = make_linearly_separable()
        clf = Perceptron(n_epochs=50).fit(X, y)
        assert len(clf.history_) > 0
        assert "errors" in clf.history_[0]

    def test_misclassifies_on_xor(self):
        # 异或线性不可分，感知器必然做不到 100%
        X, y = get_dataset("xor")
        clf = Perceptron(n_epochs=50).fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc < 0.9


class TestKNN:
    def test_perfect_on_separable_k1(self):
        X, y = make_linearly_separable(margin=0.8)
        clf = KNN(k=1).fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc == 1.0

    def test_moons_good_accuracy(self):
        X, y = make_moons(n_per_class=100, noise=0.1)
        clf = KNN(k=5).fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.92

    def test_proba_sums_to_one(self):
        X, y = make_moons()
        clf = KNN(k=5).fit(X, y)
        p = clf.predict_proba(X[:20])
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_manhattan_metric_works(self):
        X, y = make_linearly_separable()
        clf = KNN(k=3, metric="manhattan").fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.9


class TestPCA:
    def test_explained_variance_sums(self):
        X, _ = make_blobs(n_samples=200, n_classes=3)
        r = pca(X, n_components=2)
        assert r["explained_variance_ratio"].sum() <= 1.0 + 1e-9

    def test_first_component_max_variance(self):
        # PCA 第一主成分方向上投影方差应最大
        rng = np.random.default_rng(0)
        X = rng.normal(0, [3.0, 0.5], (500, 2))
        r = pca(X, n_components=2)
        v0 = r["projected"][:, 0].var()
        v1 = r["projected"][:, 1].var()
        assert v0 > v1

    def test_projection_shape(self):
        X, _ = make_blobs(n_samples=100, n_classes=2)
        r = pca(X, n_components=1)
        assert r["projected"].shape == (100, 1)


# ══════════════════════════════════════════════════════════
# SVM
# ══════════════════════════════════════════════════════════


class TestSVM:
    def test_linear_separable_perfect(self):
        X, y = make_linearly_separable(margin=1.0)
        clf = SVM(C=10.0, kernel="linear", max_iter=50).fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc == 1.0

    def test_moons_rbf_high_accuracy(self):
        X, y = make_moons(n_per_class=80, noise=0.12)
        clf = SVM(C=5.0, kernel="rbf", gamma=2.0, max_iter=50).fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        assert acc > 0.93

    def test_circles_rbf_works_linear_fails(self):
        X, y = get_dataset("circles")
        rbf = SVM(C=10.0, kernel="rbf", gamma=3.0, max_iter=50).fit(X, y)
        lin = SVM(C=10.0, kernel="linear", max_iter=50).fit(X, y)
        acc_rbf = np.mean(rbf.predict(X) == y)
        acc_lin = np.mean(lin.predict(X) == y)
        assert acc_rbf > 0.9
        assert acc_rbf > acc_lin + 0.1

    def test_alpha_bounded_by_C(self):
        X, y = make_moons(n_per_class=60)
        C = 1.0
        clf = SVM(C=C, kernel="rbf", gamma=1.0, max_iter=30).fit(X, y)
        assert clf.alpha_.max() <= C + 1e-9
        assert clf.alpha_.min() >= -1e-9

    def test_support_vectors_exist(self):
        X, y = make_linearly_separable()
        clf = SVM(C=1.0, kernel="linear", max_iter=50).fit(X, y)
        assert len(clf.support_vectors) > 0
        assert len(clf.support_vectors) < X.shape[0]

    def test_decision_function_shape(self):
        X, y = make_moons()
        clf = SVM(kernel="rbf", max_iter=30).fit(X, y)
        d = clf.decision_function(X)
        assert d.shape == (X.shape[0],)

    def test_poly_kernel_runs(self):
        X, y = get_dataset("xor")
        clf = SVM(C=5.0, kernel="poly", degree=2, max_iter=40).fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        # 二次多项式核应该能解异或
        assert acc > 0.85

    def test_predict_proba_shape_and_normalized(self):
        """SVM 没有原生概率输出，Platt 缩放后必须满足概率公理。"""
        X, y = make_moons(n_per_class=60)
        clf = SVM(C=1.0, kernel="rbf", gamma=2.0, max_iter=40).fit(X, y)
        p = clf.predict_proba(X)
        assert p.shape == (X.shape[0], 2)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_predict_proba_direction_matches_predict(self):
        """概率最大的那一类必须和 predict 的结果一致。"""
        X, y = make_moons(n_per_class=60)
        clf = SVM(C=1.0, kernel="rbf", gamma=2.0, max_iter=40).fit(X, y)
        p = clf.predict_proba(X)
        pred_from_proba = np.where(p[:, 1] >= 0.5, clf.classes_[1], clf.classes_[0])
        assert np.array_equal(pred_from_proba, clf.predict(X))

    def test_predict_proba_has_gradient(self):
        """概率应该有渐变层次，不能全是 0/1 两个值——否则决策区域画不出层次。"""
        X, y = make_moons(n_per_class=60)
        clf = SVM(C=1.0, kernel="rbf", gamma=2.0, max_iter=40).fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        # 至少要有 5 个不同的取值
        assert len(np.unique(np.round(p, 6))) > 5

    def test_smo_satisfies_kkt(self):
        """SMO 必须真正收敛：所有样本满足 KKT 条件。

        这是最容易悄悄退化的地方——早先的实现只更新 6 个 α 就误判收敛，
        留下 79 个违背 KKT 的样本，训练准确率虚高但解不是最优的。
        """
        X, y = make_moons(n_per_class=80, noise=0.15)
        C = 1.0
        tol = 1e-3
        clf = SVM(C=C, kernel="rbf", gamma=2.0, tol=tol, max_iter=100).fit(X, y)

        f = clf.decision_function(X)
        ys = np.where(y == clf.classes_[1], 1.0, -1.0)
        yf = ys * f

        n_viol = int(np.sum(
            ((clf.alpha_ < 1e-9) & (yf < 1 - tol)) |
            ((clf.alpha_ > C - 1e-9) & (yf > 1 + tol)) |
            ((clf.alpha_ >= 1e-9) & (clf.alpha_ <= C - 1e-9) & (np.abs(yf - 1) > tol))
        ))
        assert n_viol == 0, f"仍有 {n_viol} 个样本违背 KKT"

    def test_equality_constraint_exact(self):
        """对偶问题的等式约束 Σα_i y_i = 0 必须精确成立。"""
        X, y = make_moons(n_per_class=60)
        clf = SVM(C=1.0, kernel="rbf", gamma=2.0, max_iter=60).fit(X, y)
        assert abs(float(np.sum(clf.alpha_ * clf.y_))) < 1e-8

    def test_dual_objective_improves(self):
        """对偶目标值应显著优于全零起点（全零时目标 = 0）。"""
        X, y = make_moons(n_per_class=60)
        clf = SVM(C=1.0, kernel="rbf", gamma=2.0, max_iter=60).fit(X, y)
        a, ys, K = clf.alpha_, clf.y_, clf.K_
        dual = float(np.sum(a) - 0.5 * np.sum((a * ys)[:, None] * (a * ys)[None, :] * K))
        assert dual > 5.0

    def test_official_accuracy_high_after_convergence(self):
        """收敛后训练准确率应明显高于早先 bug 版本的 94.3%。"""
        X, y = make_moons(n_per_class=80, noise=0.15)
        clf = SVM(C=1.0, kernel="rbf", gamma=2.0, max_iter=100).fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.97


# ══════════════════════════════════════════════════════════
# 概率输出与 ROC 的正确性
# ══════════════════════════════════════════════════════════


class TestProbabilitiesAndROC:
    """所有分类器都必须提供可用的 predict_proba，否则 ROC 面板会崩。

    这里同时锁住一个曾经出过的严重 bug：
    把同一批样本的两列概率当成两组分数喂给 roc_curve，
    导致 AUC 退化成 ~0.5（看起来像随机猜测，实际是参数传错）。
    """

    MODELS = {
        "bayes_full": lambda: BayesClassifier(cov_type="full"),
        "bayes_diag": lambda: BayesClassifier(cov_type="diag"),
        "bayes_shared": lambda: BayesClassifier(cov_type="shared"),
        "parzen": lambda: BayesParzenClassifier(h=0.35),
        "knn": lambda: KNN(k=5),
        "fisher": lambda: FisherClassifier(),
        "perceptron": lambda: Perceptron(n_epochs=50),
        "svm_rbf": lambda: SVM(C=1.0, kernel="rbf", gamma=2.0, max_iter=80),
    }

    def test_every_model_has_calibrated_proba(self):
        X, y = make_moons(n_per_class=60, noise=0.15)
        for name, mk in self.MODELS.items():
            m = mk().fit(X, y)
            p = m.predict_proba(X)
            assert p.shape == (X.shape[0], 2), f"{name} 概率形状错"
            assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9), f"{name} 概率未归一"
            assert p.min() >= 0.0 and p.max() <= 1.0, f"{name} 概率越界"

    def test_roc_auc_not_degenerate(self):
        """用正确的分组方式算 AUC，所有算法都应远离 0.5。"""
        X, y = make_moons(n_per_class=80, noise=0.15)
        cls = np.unique(y)
        for name, mk in self.MODELS.items():
            m = mk().fit(X, y)
            score = m.predict_proba(X)[:, 1]
            r = roc_curve(score[y == cls[0]], score[y == cls[1]])
            assert r["auc"] > 0.85, f"{name} 的 AUC 只有 {r['auc']:.3f}，疑似分组传错"

    def test_roc_auc_near_one_for_svm(self):
        """SVM-RBF 在双半月上应该接近完美。"""
        X, y = make_moons(n_per_class=80, noise=0.15)
        cls = np.unique(y)
        m = SVM(C=1.0, kernel="rbf", gamma=2.0, max_iter=100).fit(X, y)
        score = m.predict_proba(X)[:, 1]
        r = roc_curve(score[y == cls[0]], score[y == cls[1]])
        assert r["auc"] > 0.98

    def test_neyman_pearson_respects_alpha(self):
        """N-P 准则的实际虚警率必须受 α 约束。"""
        rng = np.random.default_rng(0)
        s0 = rng.normal(0, 1, 2000)
        s1 = rng.normal(1.5, 1, 2000)
        for a in [0.01, 0.05, 0.10, 0.20]:
            r = neyman_pearson_threshold(s0, s1, alpha=a)
            assert r["false_alarm_rate"] <= a + 0.02, f"α={a} 时虚警率超标"


# ══════════════════════════════════════════════════════════
# 数据集
# ══════════════════════════════════════════════════════════


class TestDatasets:
    def test_all_loadable(self):
        for item in list_datasets():
            X, y = get_dataset(item["key"])
            assert X.ndim == 2 and X.shape[1] == 2
            assert y.ndim == 1 and len(y) == X.shape[0]

    def test_two_class_datasets(self):
        for k in ["moons", "circles", "xor", "anisotropic"]:
            X, y = get_dataset(k)
            assert len(np.unique(y)) == 2

    def test_split_preserves_total(self):
        X, y = make_moons()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3)
        assert Xtr.shape[0] + Xte.shape[0] == X.shape[0]
        assert len(ytr) + len(yte) == len(y)

    def test_split_stratified_ratio(self):
        X, y = make_moons(n_per_class=100)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=True)
        # 测试集里两类应大致各半
        r = np.mean(yte == 1)
        assert 0.4 < r < 0.6

    def test_moons_two_classes_balanced(self):
        X, y = make_moons(n_per_class=100)
        assert np.sum(y == 0) == 100
        assert np.sum(y == 1) == 100
