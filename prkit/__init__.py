"""prkit — 手写模式识别工具箱。

全部算法从零实现，不用 sklearn。
目标不是替代 sklearn，而是把课本上的公式一行一行落到代码里，
每一步都能画出来看。

模块：
    density   概率密度估计（MLE 高斯、Parzen 窗、kNN、直方图）
    bayes     Bayes 分类器（最小错误率/最小风险/Neyman-Pearson）+ 评价指标
    linear    Fisher 判别、LDA、感知器、kNN、PCA
    svm       SVM + SMO（线性/RBF/多项式核）
    datasets  经典二维数据集
"""
from .density import (
    mle_gaussian,
    gaussian_log_pdf,
    regularize_cov,
    parzen_pdf,
    knn_pdf,
    hist_pdf_1d,
    KERNELS,
)
from .bayes import (
    BayesClassifier,
    BayesParzenClassifier,
    neyman_pearson_threshold,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from .linear import (
    fisher_lda,
    fisher_project,
    FisherClassifier,
    Perceptron,
    KNN,
    pca,
)
from .svm import SVM, make_kernel, kernel_linear, kernel_rbf, kernel_poly
from .datasets import (
    DATASETS,
    get_dataset,
    list_datasets,
    train_test_split,
    make_moons,
    make_circles,
    make_blobs,
    make_anisotropic,
    make_xor,
    make_overlapping,
    make_linearly_separable,
)

__version__ = "0.1.0"

__all__ = [
    # density
    "mle_gaussian", "gaussian_log_pdf", "regularize_cov",
    "parzen_pdf", "knn_pdf", "hist_pdf_1d", "KERNELS",
    # bayes
    "BayesClassifier", "BayesParzenClassifier",
    "neyman_pearson_threshold", "roc_curve",
    "confusion_matrix", "classification_report",
    # linear
    "fisher_lda", "fisher_project", "FisherClassifier",
    "Perceptron", "KNN", "pca",
    # svm
    "SVM", "make_kernel", "kernel_linear", "kernel_rbf", "kernel_poly",
    # datasets
    "DATASETS", "get_dataset", "list_datasets", "train_test_split",
    "make_moons", "make_circles", "make_blobs", "make_anisotropic",
    "make_xor", "make_overlapping", "make_linearly_separable",
]
