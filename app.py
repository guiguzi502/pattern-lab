"""模式识别实验台 — Streamlit 界面。

设计原则：
1. 每个算法都要能看见「为什么这样分」——决策边界、支持向量、投影方向
2. 参数一动图就变，用滑动条而不是输入框
3. 指标与图并排，不藏在页面底下
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from prkit import (
    BayesClassifier,
    BayesParzenClassifier,
    FisherClassifier,
    KNN,
    Perceptron,
    SVM,
    classification_report,
    get_dataset,
    list_datasets,
    neyman_pearson_threshold,
    pca,
    roc_curve,
    train_test_split,
    fisher_lda,
)
from prkit.density import parzen_pdf

st.set_page_config(
    page_title="模式识别实验台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- 样式 ----------------
st.markdown(
    """
<style>
    .metric-card {
        background: #F4F7F9; border-left: 4px solid #2E86AB;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 10px;
    }
    .metric-label { font-size: 13px; color: #5A6B7B; margin-bottom: 2px; }
    .metric-value { font-size: 23px; font-weight: 600; color: #1A1A1A; }
    .warn { background: #FFF4E5; border-left-color: #E8A33D; }
    .bad  { background: #FDECEC; border-left-color: #D9534F; }
    .good { background: #EEF7EE; border-left-color: #4CAF50; }
    .note { font-size: 13px; color: #5A6B7B; line-height: 1.7; }
</style>
""",
    unsafe_allow_html=True,
)

PALETTE = ["#2E86AB", "#D9534F", "#4CAF50", "#E8A33D", "#7B5EA7", "#26A69A"]


def card(label: str, value: str, kind: str = "") -> str:
    cls = f"metric-card {kind}".strip()
    return f'<div class="{cls}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'


def color_for(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


# ══════════════════════════════════════════════════════════
# 侧栏
# ══════════════════════════════════════════════════════════
st.sidebar.title("🔬 实验设置")

datasets = list_datasets()
ds_names = [d["name"] for d in datasets]
ds_key_map = {d["name"]: d["key"] for d in datasets}
ds_choice = st.sidebar.selectbox("数据集", ds_names, index=1)
ds_key = ds_key_map[ds_choice]
ds_info = next(d for d in datasets if d["key"] == ds_key)
st.sidebar.caption(ds_info["desc"])

n_per = st.sidebar.slider("每类样本数", 30, 300, 100, 10)
noise = st.sidebar.slider("噪声 / 离散度", 0.0, 0.6, 0.15, 0.01)
seed = st.sidebar.number_input("随机种子", 0, 9999, 42, 1)

# 按数据集类型构造
if ds_key == "blobs":
    X, y = get_dataset(ds_key, n_samples=n_per * 3, n_classes=3, spread=0.3 + noise * 2)
elif ds_key in ("moons",):
    X, y = get_dataset(ds_key, n_per_class=n_per, noise=noise)
elif ds_key == "circles":
    X, y = get_dataset(ds_key, n_per_class=n_per, noise=noise)
elif ds_key == "xor":
    X, y = get_dataset(ds_key, n_per_class=n_per, spread=0.15 + noise)
elif ds_key == "anisotropic":
    X, y = get_dataset(ds_key, n_per_class=n_per)
elif ds_key == "overlapping":
    X, y = get_dataset(ds_key, n_per_class=n_per)
else:
    X, y = get_dataset(ds_key, n_per_class=n_per, margin=0.2 + noise)

test_size = st.sidebar.slider("测试集比例", 0.1, 0.5, 0.3, 0.05)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, seed=int(seed))

st.sidebar.divider()
st.sidebar.subheader("算法")

ALGOS = [
    "Bayes（高斯）",
    "Bayes（Parzen 窗）",
    "Fisher 判别",
    "k 近邻",
    "感知器",
    "支持向量机 SVM",
]
algo = st.sidebar.selectbox("选择算法", ALGOS, index=5)

# ---- 各算法参数 ----
if algo == "Bayes（高斯）":
    st.sidebar.caption("g(x) = ln p(x|ω) + ln P(ω)")
    cov_type = st.sidebar.selectbox(
        "协方差假设",
        ["full", "diag", "shared"],
        format_func=lambda x: {"full": "full — 各类独立（QDA）",
                               "diag": "diag — 对角（朴素贝叶斯）",
                               "shared": "shared — 共用（LDA）"}[x],
    )
    use_cost = st.sidebar.checkbox("启用代价矩阵（最小风险）", value=False)
    cost_matrix = None
    if use_cost:
        st.sidebar.caption("cost[i,j] = 真类 i 判成 j 的代价")
        K = len(np.unique(ytr))
        cols = st.sidebar.columns(K)
        cost_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i != j:
                    cost_matrix[i, j] = cols[j].number_input(
                        f"{i}→{j}", 0.0, 100.0, 10.0 if i != j else 0.0, 1.0,
                        key=f"cost_{i}_{j}", label_visibility="collapsed"
                    )

elif algo == "Bayes（Parzen 窗）":
    parzen_h = st.sidebar.slider("窗宽 h", 0.05, 2.0, 0.35, 0.05)
    parzen_kernel = st.sidebar.selectbox("核函数", ["gaussian", "uniform", "epanechnikov"])

elif algo == "Fisher 判别":
    st.sidebar.caption("J(w) = wᵀS_b w / wᵀS_w w")

elif algo == "k 近邻":
    k_val = st.sidebar.slider("k", 1, 25, 5, 1)
    knn_weights = st.sidebar.selectbox("权重", ["uniform", "distance"],
                                       format_func=lambda x: "等权" if x == "uniform" else "按距离加权")
    knn_metric = st.sidebar.selectbox("距离", ["euclidean", "manhattan"],
                                      format_func=lambda x: "欧氏" if x == "euclidean" else "曼哈顿")

elif algo == "感知器":
    perc_lr = st.sidebar.slider("学习率", 0.01, 1.0, 0.1, 0.01)
    perc_epochs = st.sidebar.slider("最大迭代轮数", 5, 200, 50, 5)

else:  # SVM
    svm_C = st.sidebar.slider("C 惩罚系数", 0.01, 50.0, 1.0, 0.01)
    svm_kernel = st.sidebar.selectbox("核函数", ["rbf", "linear", "poly"])
    if svm_kernel == "rbf":
        svm_gamma = st.sidebar.slider("γ (RBF 宽度)", 0.05, 10.0, 2.0, 0.05)
    elif svm_kernel == "poly":
        svm_degree = st.sidebar.slider("多项式次数", 2, 5, 3, 1)
        svm_gamma = 1.0
    else:
        svm_gamma = 1.0
    svm_iter = st.sidebar.slider("SMO 最大轮数", 10, 300, 100, 10)

st.sidebar.divider()
show_boundary = st.sidebar.checkbox("显示决策边界", value=True)
show_test = st.sidebar.checkbox("显示测试集", value=True)

# ══════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════
classes = np.unique(ytr)
n_class = len(classes)

# 坐标范围（留边）
pad = 0.6
xlim = (X[:, 0].min() - pad, X[:, 0].max() + pad)
ylim = (X[:, 1].min() - pad, X[:, 1].max() + pad)

model = None
extra = {}

try:
    if algo == "Bayes（高斯）":
        model = BayesClassifier(cov_type=cov_type).fit(Xtr, ytr)
        y_pred_te = model.predict(Xte, cost=cost_matrix)
        y_pred_tr = model.predict(Xtr, cost=cost_matrix)
    elif algo == "Bayes（Parzen 窗）":
        model = BayesParzenClassifier(h=parzen_h, kernel=parzen_kernel).fit(Xtr, ytr)
        y_pred_te = model.predict(Xte)
        y_pred_tr = model.predict(Xtr)
    elif algo == "Fisher 判别":
        model = FisherClassifier().fit(Xtr, ytr)
        y_pred_te = model.predict(Xte)
        y_pred_tr = model.predict(Xtr)
        extra["fisher"] = fisher_lda(Xtr, ytr)
    elif algo == "k 近邻":
        model = KNN(k=k_val, weights=knn_weights, metric=knn_metric).fit(Xtr, ytr)
        y_pred_te = model.predict(Xte)
        y_pred_tr = model.predict(Xtr)
    elif algo == "感知器":
        if n_class != 2:
            st.error("感知器只处理两类问题，请换个数据集")
            st.stop()
        model = Perceptron(lr=perc_lr, n_epochs=perc_epochs).fit(Xtr, ytr)
        y_pred_te = model.predict(Xte)
        y_pred_tr = model.predict(Xtr)
    else:
        if n_class != 2:
            st.error("SVM 只处理两类问题，请换个数据集")
            st.stop()
        model = SVM(C=svm_C, kernel=svm_kernel, gamma=svm_gamma,
                    degree=svm_degree if svm_kernel == "poly" else 3,
                    max_iter=svm_iter).fit(Xtr, ytr)
        y_pred_te = model.predict(Xte)
        y_pred_tr = model.predict(Xtr)
except Exception as e:
    st.error(f"训练失败：{e}")
    st.stop()

report_tr = classification_report(ytr, y_pred_tr, classes)
report_te = classification_report(yte, y_pred_te, classes)

# ══════════════════════════════════════════════════════════
# 标题 + 指标
# ══════════════════════════════════════════════════════════
st.title("🔬 模式识别实验台")
st.caption("全部算法从零手写，不用 sklearn。参数一动，决策边界跟着变。")

c1, c2, c3, c4 = st.columns(4)
with c1:
    acc = report_te["accuracy"]
    kind = "good" if acc > 0.9 else ("warn" if acc > 0.75 else "bad")
    st.markdown(card("测试集准确率", f"{acc*100:.1f}%", kind), unsafe_allow_html=True)
with c2:
    tr_acc = report_tr["accuracy"]
    gap = tr_acc - acc
    kind = "warn" if gap > 0.12 else "good"
    st.markdown(card("训练集准确率", f"{tr_acc*100:.1f}%", kind), unsafe_allow_html=True)
with c3:
    st.markdown(card("泛化间隙", f"{gap*100:+.1f}%", "warn" if gap > 0.12 else ""), unsafe_allow_html=True)
with c4:
    st.markdown(card("宏平均 F1", f"{report_te['macro_f1']:.3f}"), unsafe_allow_html=True)

if gap > 0.12:
    st.warning(f"⚠️ 训练集 {tr_acc*100:.1f}% 但测试集只有 {acc*100:.1f}%，差 {gap*100:.1f} 个百分点 —— 过拟合了。试着减小模型复杂度（k 调大、C 调小、γ 调小）。")

# ══════════════════════════════════════════════════════════
# 主图：散点 + 决策边界
# ══════════════════════════════════════════════════════════
st.subheader("🗺️ 决策区域")

fig = go.Figure()

# 决策边界背景
if show_boundary:
    RES = 160
    xs = np.linspace(xlim[0], xlim[1], RES)
    ys = np.linspace(ylim[0], ylim[1], RES)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])

    if algo == "Bayes（高斯）":
        _, _, idx_grid, conf = model.decision_boundary_mesh(xlim, ylim, RES)
    elif algo.startswith("支持向量机"):
        # SVM 用 decision_function 画等高线
        _, _, Z = model.decision_boundary_2d(xlim, ylim, RES)
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=Z,
            contours=dict(start=Z.min(), end=Z.max(),
                          size=(Z.max() - Z.min()) / 12,
                          coloring="lines", showlabels=False),
            line=dict(width=0.6, color="rgba(120,140,160,0.35)"),
            showscale=False, hoverinfo="skip", name="决策函数等值线",
        ))
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=Z, contours=dict(type="constraint", operation="=", value=0),
            line=dict(width=3.0, color="#1A1A1A"),
            showscale=False, hoverinfo="skip", name="决策面 f(x)=0",
        ))
        idx_grid = None
    else:
        proba = model.predict_proba(grid)
        idx_grid = np.argmax(proba, axis=1).reshape(XX.shape)

    if idx_grid is not None:
        # 用半透明色块表示决策区域
        colorscale = []
        for i in range(len(classes)):
            c = color_for(i)
            colorscale.append([i / len(classes), c])
            colorscale.append([(i + 1) / len(classes), c])
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=idx_grid,
            colorscale=colorscale, opacity=0.16,
            showscale=False, hoverinfo="skip", name="决策区域",
        ))

# 训练点
for i, c in enumerate(classes):
    m = ytr == c
    fig.add_trace(go.Scatter(
        x=Xtr[m, 0], y=Xtr[m, 1], mode="markers",
        marker=dict(size=8, color=color_for(i), line=dict(width=1, color="white")),
        name=f"类 {c}（训练）",
    ))

# 测试点
if show_test:
    for i, c in enumerate(classes):
        m = yte == c
        if np.any(m):
            fig.add_trace(go.Scatter(
                x=Xte[m, 0], y=Xte[m, 1], mode="markers",
                marker=dict(size=10, color=color_for(i), symbol="diamond",
                            line=dict(width=1.5, color="#333")),
                name=f"类 {c}（测试）",
            ))

# 支持向量
if algo == "支持向量机 SVM":
    sv = model.support_vectors
    if len(sv):
        fig.add_trace(go.Scatter(
            x=Xtr[sv, 0], y=Xtr[sv, 1], mode="markers",
            marker=dict(size=17, color="rgba(0,0,0,0)", line=dict(width=2.2, color="#1A1A1A")),
            name=f"支持向量（{len(sv)} 个）",
        ))

# Fisher 投影方向
if algo == "Fisher 判别" and "fisher" in extra:
    W = extra["fisher"]["W"][:, 0]
    mu = Xtr.mean(axis=0)
    scale = (xlim[1] - xlim[0]) * 0.28
    fig.add_trace(go.Scatter(
        x=[mu[0] - W[0] * scale, mu[0] + W[0] * scale],
        y=[mu[1] - W[1] * scale, mu[1] + W[1] * scale],
        mode="lines",
        line=dict(width=3.2, color="#1A1A1A"),
        name="判别方向 w",
    ))

fig.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="x₁", yaxis_title="x₂",
    legend=dict(orientation="h", y=1.1),
    dragmode="pan",
)
fig.update_xaxes(range=xlim)
fig.update_yaxes(range=ylim, scaleanchor="x", scaleratio=1)
st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# 算法专属面板
# ══════════════════════════════════════════════════════════
if algo == "Bayes（高斯）":
    st.subheader("📐 高斯参数与协方差椭圆")
    st.caption("三种协方差假设的区别，看椭圆形状就够了：full 各自倾斜、diag 轴对齐、shared 形状相同。")
    col1, col2 = st.columns([1, 1])

    with col1:
        for i, c in enumerate(classes):
            mu = model.means_[c]
            S = model.covs_[c]
            st.markdown(
                f"**类 {c}** &nbsp; 先验 P(ω)={model.priors_[c]:.3f} &nbsp; "
                f"μ=({mu[0]:.2f}, {mu[1]:.2f})"
            )
            st.code(f"Σ = \n{np.array2string(S, precision=3, suppress_small=True)}", language=None)

    with col2:
        # 画协方差椭圆
        fig_e = go.Figure()
        for i, c in enumerate(classes):
            mu = model.means_[c]
            S = model.covs_[c]
            vals, vecs = np.linalg.eigh(S)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            t = np.linspace(0, 2 * np.pi, 100)
            # 1σ 和 2σ 椭圆
            for k_sigma, dash in [(1.0, "solid"), (2.0, "dot")]:
                circle = np.array([np.cos(t), np.sin(t)]) * k_sigma
                ell = vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-12))) @ circle
                ell = ell + mu.reshape(2, 1)
                fig_e.add_trace(go.Scatter(
                    x=ell[0], y=ell[1], mode="lines",
                    line=dict(color=color_for(i), width=2, dash=dash),
                    name=f"类 {c} {k_sigma:g}σ", showlegend=(k_sigma == 1.0),
                ))
            fig_e.add_trace(go.Scatter(
                x=[mu[0]], y=[mu[1]], mode="markers",
                marker=dict(size=10, color=color_for(i), symbol="x"),
                showlegend=False,
            ))
        fig_e.update_layout(
            height=330, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="x₁", yaxis_title="x₂",
            legend=dict(orientation="h", y=1.12),
        )
        fig_e.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig_e, use_container_width=True)

elif algo == "Bayes（Parzen 窗）":
    st.subheader("🌊 Parzen 窗密度曲面")
    st.caption(f"窗宽 h={parzen_h}，核 {parzen_kernel}。h 太小过拟合（尖刺一堆），太大过平滑（糊成一片）。")
    RES = 60
    xs = np.linspace(xlim[0], xlim[1], RES)
    ys = np.linspace(ylim[0], ylim[1], RES)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])

    fig_p = make_subplots(rows=1, cols=n_class, subplot_titles=[f"p(x|类 {c})" for c in classes])
    for i, c in enumerate(classes):
        Xc = Xtr[ytr == c]
        p = parzen_pdf(Xc, grid, h=parzen_h, kernel=parzen_kernel).reshape(XX.shape)
        fig_p.add_trace(
            go.Surface(x=xs, y=ys, z=p, colorscale="Blues", showscale=False, opacity=0.9),
            row=1, col=i + 1,
        )
    fig_p.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_p, use_container_width=True)

elif algo == "Fisher 判别" and "fisher" in extra:
    st.subheader("📏 Fisher 投影")
    col1, col2 = st.columns([1, 1])
    W = extra["fisher"]["W"]
    with col1:
        st.markdown("**判别方向**")
        for j in range(W.shape[1]):
            st.code(f"w{j+1} = [{W[0,j]:.4f}, {W[1,j]:.4f}]", language=None)
        st.markdown("**特征值（类间/类内散度比）**")
        for j, ev in enumerate(extra["fisher"]["eigenvalues"]):
            st.markdown(f"λ{j+1} = {ev:.4f}")

    with col2:
        Z = Xtr @ W[:, 0]
        fig_h = go.Figure()
        for i, c in enumerate(classes):
            vals = Z[ytr == c]
            fig_h.add_trace(go.Histogram(
                x=vals, name=f"类 {c}", opacity=0.65,
                marker_color=color_for(i), nbinsx=25,
            ))
        fig_h.update_layout(
            height=330, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="投影值 z = wᵀx", yaxis_title="频次",
            barmode="overlay", legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_h, use_container_width=True)
        st.caption("两类分布重叠越少，判别能力越强。")

elif algo == "感知器":
    st.subheader("📉 感知器收敛过程")
    hist = model.history_
    if hist:
        epochs = [h["epoch"] for h in hist]
        errs = [h["errors"] for h in hist]
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=epochs, y=errs, mode="lines+markers",
            line=dict(color="#2E86AB", width=2.4),
            marker=dict(size=6),
        ))
        fig_c.update_layout(
            height=300, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="迭代轮数", yaxis_title="误分类数",
        )
        st.plotly_chart(fig_c, use_container_width=True)

        if errs[-1] > 0:
            st.warning(
                f"最后一轮仍有 {errs[-1]} 个误分类。感知器**只对线性可分数据保证收敛**——"
                "这个数据集线性不可分，所以它永远停不下来。这正是 SVM 要引入核技巧的原因。"
            )
        else:
            st.success(f"第 {epochs[-1]+1} 轮收敛（误分类归零）。")

elif algo == "支持向量机 SVM":
    st.subheader("⚙️ SMO 求解过程")
    col1, col2 = st.columns([1, 1])
    hist = model.history_
    with col1:
        if hist:
            iters = [h["iter"] for h in hist]
            viols = [h.get("violations", 0) for h in hist]
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=iters, y=viols, mode="lines+markers",
                line=dict(color="#D9534F", width=2.4),
                marker=dict(size=6),
                name="KKT 违背数",
            ))
            fig_c.update_layout(
                height=300, margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="SMO 迭代", yaxis_title="KKT 违背样本数",
            )
            st.plotly_chart(fig_c, use_container_width=True)
            final_viol = viols[-1] if viols else 0
            if final_viol == 0:
                st.success(f"收敛：{len(iters)} 轮外层扫描后全部样本满足 KKT 条件。")
            else:
                st.warning(f"仍有 {final_viol} 个样本违背 KKT，可加大 SMO 轮数。")

    with col2:
        n_sv = len(model.support_vectors)
        st.markdown(card("支持向量数", f"{n_sv} / {Xtr.shape[0]}"), unsafe_allow_html=True)
        st.markdown(card("偏置 b", f"{model.b_:.4f}"), unsafe_allow_html=True)
        # 有多少 α 顶到边界 C
        at_C = int(np.sum(model.alpha_ >= svm_C - 1e-6))
        st.markdown(card("处于边界 C 的 α", f"{at_C} 个（越界样本）"), unsafe_allow_html=True)
        if at_C > 0:
            st.caption("α=C 说明该样本落在间隔内部或分错，是软间隔在起作用。")

# ══════════════════════════════════════════════════════════
# 混淆矩阵 + 每类指标
# ══════════════════════════════════════════════════════════
st.subheader("🎯 混淆矩阵与分类指标")
col_a, col_b = st.columns([1, 1])

with col_a:
    cm = report_te["confusion"]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=[f"判为 {c}" for c in classes], y=[f"真实 {c}" for c in classes],
        colorscale="Blues", text=cm, texttemplate="%{text}",
        textfont=dict(size=16), showscale=False,
    ))
    fig_cm.update_layout(
        height=340, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_b:
    st.markdown("**每类指标（测试集）**")
    rows = []
    for c in classes:
        m = report_te["per_class"][str(c)]
        rows.append({
            "类别": str(c),
            "精确率": f"{m['precision']:.3f}",
            "召回率": f"{m['recall']:.3f}",
            "F1": f"{m['f1']:.3f}",
            "样本数": m["support"],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# ROC（两类专用）
# ══════════════════════════════════════════════════════════
if n_class == 2:
    st.subheader("📈 ROC 曲线与 Neyman-Pearson")
    # 关键：ROC 要的是「同一个评分」，按真实标签分成两组。
    # 早先的错误是拿同一批样本的两列概率当两组分数，AUC 会退化成 ~0.5。
    proba_te = model.predict_proba(Xte)
    score = proba_te[:, 1]                      # 越大越像类 1
    s0 = score[yte == classes[0]]               # 真实为类 0 的样本得分
    s1 = score[yte == classes[1]]               # 真实为类 1 的样本得分

    col_x, col_y = st.columns([1, 1])
    with col_x:
        alpha = st.slider("虚警率上限 α", 0.01, 0.5, 0.05, 0.01)
        np_res = neyman_pearson_threshold(s0, s1, alpha=alpha)
        roc = roc_curve(s0, s1)

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=roc["far"], y=roc["pd"], mode="lines",
            line=dict(color="#2E86AB", width=2.6),
            name=f"ROC（AUC={roc['auc']:.3f}）",
        ))
        fig_r.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#C8D3DC", dash="dash", width=1.2),
            name="随机猜测",
        ))
        fig_r.add_trace(go.Scatter(
            x=[np_res["false_alarm_rate"]], y=[np_res["detection_rate"]],
            mode="markers",
            marker=dict(size=13, color="#D9534F", symbol="star"),
            name="N-P 工作点",
        ))
        fig_r.update_layout(
            height=380, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="虚警率 P(判1|类0)", yaxis_title="检出率 P(判1|类1)",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_y:
        st.markdown(card("AUC", f"{roc['auc']:.4f}",
                         "good" if roc["auc"] > 0.9 else ("warn" if roc["auc"] > 0.75 else "bad")),
                    unsafe_allow_html=True)
        st.markdown(card("N-P 门限", f"{np_res['threshold']:.4f}"), unsafe_allow_html=True)
        st.markdown(card(f"约束 α={alpha:.2f}", f"实际虚警 {np_res['false_alarm_rate']:.3f}"),
                    unsafe_allow_html=True)
        st.markdown(card("检出率", f"{np_res['detection_rate']:.3f}"), unsafe_allow_html=True)
        st.caption(
            "Neyman-Pearson 的思路：先卡死一类错误率上限，再在这个约束下尽量提高检出率。"
            "医疗筛查、雷达检测都用这个准则。"
        )

# ══════════════════════════════════════════════════════════
# 算法说明
# ══════════════════════════════════════════════════════════
with st.expander("📖 这个算法在做什么（点开看公式与直觉）"):
    notes = {
        "Bayes（高斯）": """
**判别函数**：$g_i(x) = \\ln p(x|\\omega_i) + \\ln P(\\omega_i)$

取对数不改变 argmax，但避免概率连乘下溢。

**三种协方差假设的差别**：

| 假设 | 决策面 | 何时用 |
|---|---|---|
| full | 二次曲面 | 各类形状确实不同（各向异性数据） |
| diag | 各轴独立的二次 | 特征不相关，样本少时更稳 |
| shared | **超平面**（线性） | 各类形状相近，只是中心不同 |

shared 就是 LDA——因为二次项 $x^T\\Sigma^{-1}x$ 在两类里相同，相减时消掉了。

**代价矩阵**：最小错误率默认 0-1 代价，最小风险允许不对称。比如漏诊比误诊严重得多，就该给漏诊高代价。
""",
        "Bayes（Parzen 窗）": """
**密度估计**：$p(x) = \\frac{1}{nh^d}\\sum_i \\varphi\\left(\\frac{x-x_i}{h}\\right)$

不假设任何分布形状，直接把每个样本当成一个小核函数的中心堆起来。

**窗宽 h 的取舍**（这就是偏差-方差权衡）：

- h 太小 → 每个点一个尖峰，过拟合
- h 太大 → 糊成一团，欠拟合

理论上 $h \\propto n^{-1/(d+4)}$，但实践中靠看图形调。
""",
        "Fisher 判别": """
**目标**：$J(w) = \\dfrac{w^T S_b w}{w^T S_w w}$

让投影后类间散度大、类内散度小。

**解**：$w^* = S_w^{-1}(\\mu_1 - \\mu_2)$，多类时取 $S_w^{-1}S_b$ 的前 K-1 个特征向量。

**局限**：只用一个方向，两类均值重合但方差不同的情况（如同心圆）无能为力。
""",
        "k 近邻": """
**思想**：物以类聚。看最近的 k 个邻居怎么投票。

**没有训练过程**（惰性学习），预测时才计算距离，所以大数据集上很慢。

**k 的影响**：

- k=1 → 决策边界极其破碎，对噪声敏感
- k 大 → 边界平滑，但会抹掉细节
- k 一般取 $\\sqrt{n}$ 附近，且用奇数避免平票
""",
        "感知器": """
**更新规则**（误分类时）：$w \\leftarrow w + \\eta y_i x_i$

**感知器收敛定理**：若数据线性可分，有限步内必然收敛。

**致命局限**：线性不可分就永远震荡。1969 年 Minsky 指出这点，直接导致神经网络研究停滞十几年。这也是 SVM 用核技巧的动机。
""",
        "支持向量机 SVM": """
**原始问题**：
$$\\min_{w,b,\\xi}\\ \\tfrac12\\|w\\|^2 + C\\sum\\xi_i \\quad \\text{s.t.}\\ y_i(w^T\\phi(x_i)+b) \\ge 1-\\xi_i$$

**对偶问题**：
$$\\max_\\alpha\\ \\sum\\alpha_i - \\tfrac12\\sum\\sum \\alpha_i\\alpha_j y_i y_j K(x_i,x_j)$$
$$\\text{s.t.}\\ 0 \\le \\alpha_i \\le C,\\ \\sum\\alpha_i y_i = 0$$

**三个关键点**：

1. **最大间隔**：$\\|w\\|$ 最小化等于间隔 $2/\\|w\\|$ 最大化
2. **核技巧**：只需内积，不用显式升维。RBF 核对应无穷维特征空间
3. **支持向量**：只有 $\\alpha_i>0$ 的样本影响决策面，其余全部无关

**SMO 算法**：每次挑两个 $\\alpha$ 做解析解，因为等式约束 $\\sum\\alpha_i y_i=0$ 恰好让两个变量的子问题可解。重复直到 KKT 条件满足。

**C 的作用**：C 大 → 严惩违反间隔，容易过拟合；C 小 → 允许更多越界，边界更平滑。
""",
    }
    st.markdown(notes.get(algo, ""))

st.divider()
st.caption("prkit v0.1 · 全部算法手写实现，零 sklearn 依赖 · MIT License")
