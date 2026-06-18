import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os
from utils.data_generator import generate_concentric_circles
from utils.svm_utils import fit_svm_model, get_decision_grid

# ------------------------------------------------------------------------------
# 頁面配置與樣式設定
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SVM 核心技巧 (Kernel Trick) 3D 互動展示系統",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂進階 CSS 樣式：隱藏 stAppToolbar 且應用美化排版
st.markdown("""
<style>
    /* 隱藏 Streamlit 工具列、頁首及頁尾 */
    header[data-testid="stHeader"] {
        visibility: hidden;
        display: none;
        height: 0px;
    }
    [data-testid="stAppToolbar"] {
        visibility: hidden;
        display: none;
        height: 0px;
    }
    footer {
        visibility: hidden;
        display: none;
    }
    
    /* 字型與版面配置 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700;900&family=Space+Grotesk:wght@400;700&display=swap');
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        font-family: 'Noto Sans TC', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Noto Sans TC', sans-serif;
        font-weight: 700;
    }
    
    /* 標題橫幅樣式 */
    .title-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .title-banner h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 900;
    }
    .title-banner p {
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
        opacity: 0.9;
    }
    
    /* 自訂概念警告卡片 */
    .warning-card {
        background-color: rgba(234, 179, 8, 0.08);
        border-left: 5px solid #eab308;
        padding: 1rem;
        border-radius: 6px;
        margin: 1.2rem 0;
    }
    .warning-card-title {
        font-weight: bold;
        color: #d97706;
        font-size: 1.05rem;
        margin-bottom: 0.25rem;
    }
    .warning-card-text {
        font-size: 0.95rem;
        line-height: 1.5;
        color: #1e293b;
    }
    
    /* 特色卡片 */
    .feature-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .feature-card h3 {
        color: #1e3a8a;
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-size: 1.15rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 標題橫幅
# ------------------------------------------------------------------------------
st.markdown("""
<div class="title-banner">
    <h1>⚛️ SVM 核心技巧 (Kernel Trick) 3D 互動展示系統</h1>
    <p>一個以視覺化探討非線性決策邊界的機器學習互動式數學教學平台。</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 側邊欄參數控制台
# ------------------------------------------------------------------------------
st.sidebar.header("🛠️ 參數設定面板")

st.sidebar.subheader("資料集設定")
num_points = st.sidebar.slider("樣本點數量", min_value=50, max_value=500, value=200, step=25)
noise_level = st.sidebar.slider("雜訊程度 (Noise)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
random_seed = st.sidebar.number_input("隨機種子 (Random Seed)", min_value=0, max_value=99999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("SVM 模型設定")

kernel = st.sidebar.selectbox(
    "核心函數 (Kernel)",
    options=["rbf", "linear", "poly"],
    format_func=lambda x: "徑向基底函數 (RBF)" if x == "rbf" else ("線性核心 (Linear)" if x == "linear" else "多項式核心 (Polynomial)")
)

# 正則化參數 C
c_scale = st.sidebar.radio("C 調整模式", ["標準調整", "對數調整"], horizontal=True)
if c_scale == "標準調整":
    C = st.sidebar.slider("C (正則化參數)", min_value=0.01, max_value=100.0, value=1.0, step=0.5)
else:
    C_exp = st.sidebar.slider("C 參數大小 (10^x)", min_value=-2, max_value=3, value=0, step=1)
    C = float(10**C_exp)
    st.sidebar.text(f"當前實際 C 值: {C:.2f}")

# Gamma 參數 (RBF / Poly)
gamma = "scale"
if kernel in ["rbf", "poly"]:
    gamma_type = st.sidebar.radio("Gamma 類型", ["scale", "auto", "自訂數值"], horizontal=True)
    if gamma_type == "自訂數值":
        gamma = st.sidebar.slider("Gamma (核心寬度參數)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    else:
        gamma = gamma_type

# 階數參數 (Poly)
degree = 3
if kernel == "poly":
    degree = st.sidebar.slider("多項式階數 (Degree)", min_value=1, max_value=6, value=3, step=1)

# 圖表視覺效果設定
st.sidebar.markdown("---")
st.sidebar.subheader("圖表視覺效果設定")

enable_animation = st.sidebar.checkbox(
    "啟用 3D 空間拉升/變形動畫",
    value=True,
    help="在 3D 視圖中啟用點與曲面的動態拉升/變形過渡動畫。關閉則顯示靜態 3D 視圖。"
)

# 生成 concentric circles 資料集
X, y = generate_concentric_circles(noise=noise_level, n_samples=num_points, random_seed=random_seed)

# 訓練 SVM 模型
model = fit_svm_model(X, y, kernel=kernel, C=C, gamma=gamma, degree=degree)

# 使用固定邊界生成決策網格，避免隨參數調整發生版面位移
fixed_bounds = (-2.3, 2.3, -2.3, 2.3)
xx, yy, Z = get_decision_grid(model, X, grid_resolution=100, bounds=fixed_bounds)

# 取得支援向量
support_vectors = model.support_vectors_

# 色彩樣式設定 (固定使用經典藍紅 Modern Blue/Red 佈局)
color_class_0 = '#3b82f6'       # 藍色
color_class_1 = '#ef4444'       # 紅色
color_sv = '#eab308'            # 金黃色 (支援向量)
color_sv_bg = 'rgba(234, 179, 8, 0.15)'
color_boundary = '#eab308'      # 決策邊界
colorscale_bg = [
    [0.0, '#3b82f6'],
    [0.5, 'rgba(255, 255, 255, 0.85)'],
    [1.0, '#ef4444']
]
colorscale_surf = 'RdBu'
reversescale_surf = True

# ------------------------------------------------------------------------------
# 主要顯示標籤頁
# ------------------------------------------------------------------------------
tab_2d, tab_3d, tab_manim = st.tabs([
    "📐 2D 投影平面檢視", 
    "🧊 3D 特徵空間 / 決策曲面", 
    "🎬 Manim 概念教學動畫"
])

# ==============================================================================
# 標籤頁 1: 2D 投影平面檢視
# ==============================================================================
with tab_2d:
    st.subheader("2D 分類決策邊界與間距 (Margin)")
    st.markdown(
        "此視圖展示了原始 2D 輸入空間的分類狀態。背景色彩強度代表分類的信心指標。<br>"
        "**黃色實線**代表決策邊界：f(x, y) = 0。<br>"
        "**灰色虛線**代表邊界間距限制：f(x, y) = 1 與 f(x, y) = -1。",
        unsafe_allow_html=True
    )
    
    col_plot, col_info = st.columns([3, 1])
    
    with col_plot:
        fig_2d = go.Figure()
        
        # 1. 信心指數背景等高線圖
        fig_2d.add_trace(go.Contour(
            x=xx[0, :],
            y=yy[:, 0],
            z=Z,
            colorscale=colorscale_bg,
            opacity=0.45,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text='f(x, y) 信心值',
                    side='right',
                    font=dict(size=12)
                )
            ),
            contours=dict(coloring='heatmap', showlines=False),
            hoverinfo='skip'
        ))
        
        # 2. 決策邊界線 (f(x, y) = 0)
        fig_2d.add_trace(go.Contour(
            x=xx[0, :],
            y=yy[:, 0],
            z=Z,
            showscale=False,
            contours=dict(
                start=0.0,
                end=0.0,
                coloring='none',
                showlines=True
            ),
            line=dict(color=color_boundary, width=4),
            hoverinfo='skip',
            name='決策邊界 f(x,y)=0'
        ))
        
        # 3. 間距線 (f(x, y) = ±1) - 灰色虛線
        fig_2d.add_trace(go.Contour(
            x=xx[0, :],
            y=yy[:, 0],
            z=Z,
            showscale=False,
            contours=dict(
                start=-1.0,
                end=1.0,
                size=2,
                coloring='none',
                showlines=True
            ),
            line=dict(color='#475569', width=1.5, dash='dash'),
            hoverinfo='skip',
            name='間距邊界 f(x,y)=±1'
        ))
        
        # 4. 繪製類別 0 點 (內圈)
        fig_2d.add_trace(go.Scatter(
            x=X[y == 0, 0],
            y=X[y == 0, 1],
            mode='markers',
            marker=dict(
                color=color_class_0,
                size=8,
                line=dict(color='#1e293b', width=1.0)
            ),
            name='類別 0 (內圈)'
        ))
        
        # 5. 繪製類別 1 點 (外圈)
        fig_2d.add_trace(go.Scatter(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            mode='markers',
            marker=dict(
                color=color_class_1,
                size=8,
                line=dict(color='#1e293b', width=1.0)
            ),
            name='類別 1 (外圈)'
        ))
        
        # 6. 支援向量高亮框 (黃金環)
        fig_2d.add_trace(go.Scatter(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            mode='markers',
            marker=dict(
                color=color_sv_bg,
                size=14,
                line=dict(color=color_sv, width=2.0)
            ),
            name='支援向量 (Support Vectors)'
        ))
        
        # 2D 圖形排版調整 - 固定坐標範圍，確保調整參數時不位移
        fig_2d.update_layout(
            width=600,
            height=600,
            xaxis=dict(
                title='X1 軸', 
                range=[-2.2, 2.2], 
                showgrid=True, 
                gridcolor='#e2e8f0',
                constrain="domain"  # 強制約束繪圖區比例
            ),
            yaxis=dict(
                title='X2 軸', 
                range=[-2.2, 2.2], 
                showgrid=True, 
                gridcolor='#e2e8f0',
                scaleanchor="x",
                scaleratio=1,
                constrain="domain"  # 強制約束繪圖區比例
            ),
            plot_bgcolor='white',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1.0
            ),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        st.plotly_chart(fig_2d, width='stretch')
        
    with col_info:
        st.markdown("### 📊 資料集統計資訊")
        st.write(f"**樣本點總數:** {num_points}")
        st.write(f"**雜訊強度:** {noise_level}")
        st.write(f"**支援向量點數:** {len(support_vectors)}")
        st.write(f"**支援向量佔比:** {len(support_vectors) / num_points:.1%}")
        
        st.markdown("### ⚡ 參數作用解析")
        if kernel == "rbf":
            gamma_val_str = f"{model.gamma:.4f}" if isinstance(model.gamma, float) else str(model.gamma)
            st.markdown(f"**當前 Gamma 值:** `{gamma_val_str}`")
            st.info(
                "👉 **Gamma 的影響：**\n"
                "- **高 Gamma**：每個樣本點的控制範圍較小，決策邊界會高度扭曲以包裹單個資料點，容易導致過擬合 (Overfitting)。\n"
                "- **低 Gamma**：每個點的控制半徑較大，決策邊界會趨於平緩、圓滑，可能導致欠擬合 (Underfitting)。"
            )
        elif kernel == "poly":
            st.info(
                f"👉 **多項式階數 (Degree = {degree}) 的影響：**\n"
                "階數越高，決策線的彎曲度和複雜度越高，但運算開銷會增加，且在邊緣處容易產生不穩定的震盪。"
            )
        else:
            st.info(
                "👉 **線性核心 (Linear) 的侷限：**\n"
                "可以觀察到線性決策線無法對同心圓進行有效的包圍式分割，因此訓練誤差極高。"
            )
            
        st.info(
            f"👉 **正則化係數 C 的影響：**\n"
            f"- **高 C** (目前 C = {C})：對分類錯誤處罰嚴格，極力想把每個點分對，會導致間距縮小 (Narrow Margin)。\n"
            "- **低 C**：允許部分點分類錯誤以換取最大的分類間距，邊界更具平滑與概括性 (Wide Margin)。"
        )

# ==============================================================================
# 標籤頁 2: 3D 特徵空間 / 決策曲面
# ==============================================================================
with tab_3d:
    st.subheader("3D 特徵空間對比與分析")
    
    # 點出 conceptual mapping 與 true RBF 差異
    st.markdown("""
    <div class="warning-card">
        <div class="warning-card-title">⚠️ 重要數學視覺化概念說明</div>
        <div class="warning-card-text">
            <b>顯示映射 z = x² + y²</b> 是一種為了方便<b>教學與直覺理解</b>而特別設計的 3D 空間轉換投影。<br>
            需要特別注意的是：<b>RBF 核心 (Radial Basis Function)</b> 在數學原理上並不是簡單對應到 3D 空間，而是將特徵投射到一個<b>無限維度的希爾伯特空間 (Infinite-dimensional Hilbert space)</b>。下面展示的 RBF 3D 決策曲面，實際上是 SVM 決策信賴度函數的 3D 信心高度地形圖 <b>z = f(x, y)</b>。
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    view_type = st.radio(
        "選擇 3D 視覺化投影模式",
        options=["教學用顯式 3D 特徵映射 (拋物面 Paraboloid)", "真實 SVM 3D 決策曲面 z = f(x, y)"],
        horizontal=True
    )
    
    fig_3d = go.Figure()
    
    if view_type == "教學用顯式 3D 特徵映射 (拋物面 Paraboloid)":
        st.markdown(
            "此視圖將所有 2D 資料點透過公式 z = x² + y² 向上投影到 3D 拋物面上。 "
            "黃色半透明平面代表 z = 0.5 的 3D 分割超平面 (Hyperplane)。"
        )
        
        # 1. 產生拋物面網格
        u = np.linspace(-1.8, 1.8, 40)
        v = np.linspace(-1.8, 1.8, 40)
        uu, vv = np.meshgrid(u, v)
        zz_paraboloid = uu**2 + vv**2
        
        # 繪製拋物面
        fig_3d.add_trace(go.Surface(
            x=u, y=v, z=zz_paraboloid,
            colorscale='Blues',
            opacity=0.35,
            showscale=False,
            name='投影拋物面'
        ))
        
        # 2. 繪製水平切面 (決策超平面，高度設定為 0.5)
        plane_height = 0.5
        zz_plane = np.ones_like(uu) * plane_height
        
        fig_3d.add_trace(go.Surface(
            x=u, y=v, z=zz_plane,
            colorscale=[[0, 'rgba(234, 179, 8, 0.4)'], [1, 'rgba(234, 179, 8, 0.4)']],
            opacity=0.45,
            showscale=False,
            name='分割超平面'
        ))
        
        # 3. 繪製相交邊界圓 (半徑為 0.5 的平方根)
        t_circle = np.linspace(0, 2*np.pi, 100)
        circle_r = np.sqrt(plane_height)
        cx = circle_r * np.cos(t_circle)
        cy = circle_r * np.sin(t_circle)
        cz = np.ones_like(t_circle) * plane_height
        
        fig_3d.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz,
            mode='lines',
            line=dict(color=color_sv, width=5),
            name='交會相割邊界線'
        ))
        
        # 4. 繪製已投影的類別 0 點
        X_0 = X[y == 0]
        z_0 = X_0[:, 0]**2 + X_0[:, 1]**2
        fig_3d.add_trace(go.Scatter3d(
            x=X_0[:, 0], y=X_0[:, 1], z=z_0,
            mode='markers',
            marker=dict(color=color_class_0, size=5, line=dict(color='#1e293b', width=1)),
            name='類別 0 (內圈)'
        ))
        
        # 5. 繪製已投影的類別 1 點
        X_1 = X[y == 1]
        z_1 = X_1[:, 0]**2 + X_1[:, 1]**2
        fig_3d.add_trace(go.Scatter3d(
            x=X_1[:, 0], y=X_1[:, 1], z=z_1,
            mode='markers',
            marker=dict(color=color_class_1, size=5, line=dict(color='#1e293b', width=1)),
            name='類別 1 (外圈)'
        ))
        
        # 啟用 3D 空間拉升動畫
        if enable_animation:
            frames = []
            steps = np.linspace(0.0, 1.0, 11)
            for i, val in enumerate(steps):
                frames.append(go.Frame(
                    data=[
                        go.Surface(z=val * zz_paraboloid),
                        go.Surface(z=val * zz_plane),
                        go.Scatter3d(z=val * cz),
                        go.Scatter3d(z=val * z_0),
                        go.Scatter3d(z=val * z_1)
                    ],
                    name=f"frame_{i}"
                ))
            fig_3d.frames = frames
            
            fig_3d.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    x=0.1, y=-0.1,
                    buttons=[
                        dict(label="播放 (Play)", method="animate", args=[None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True}]),
                        dict(label="暫停 (Pause)", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
                    ]
                )],
                sliders=[dict(
                    steps=[
                        dict(
                            method='animate',
                            args=[[f'frame_{i}'], {'frame': {'duration': 150, 'redraw': True}, 'mode': 'immediate'}],
                            label=f'{int(val*100)}%'
                        )
                        for i, val in enumerate(steps)
                    ],
                    x=0.25, y=-0.1, len=0.75,
                    currentvalue=dict(font=dict(size=12), prefix='拉升比例: ', visible=True, xanchor='right')
                )]
            )
            
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='X1 軸',
                yaxis_title='X2 軸',
                zaxis_title='Z 高度 (x1² + x2²)',
                xaxis=dict(range=[-2.2, 2.2]),
                yaxis=dict(range=[-2.2, 2.2]),
                zaxis=dict(range=[0, 3.5])
            )
        )
        
    else:
        st.markdown(
            "此視圖呈現真實的 SVM 信心分數地形圖。Z 軸代表決策信賴值 <b>z = f(x1, x2)</b>。 "
            "灰色半透明平面代表 f(x1, x2) = 0 的基準面，交界處的黃色曲線即為決策邊界。"
        )
        
        # 1. 繪製真實的決策曲面
        fig_3d.add_trace(go.Surface(
            x=xx[0, :],
            y=yy[:, 0],
            z=Z,
            colorscale=colorscale_surf,
            reversescale=reversescale_surf,
            opacity=0.6,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text='決策函數值 z=f(x,y)',
                    side='right'
                )
            ),
            # 高亮黃色交界線與投影到地面
            contours=dict(
                z=dict(
                    show=True,
                    start=0.0,
                    end=0.0,
                    size=1.0,
                    color=color_boundary,
                    width=4,
                    project=dict(z=True)
                )
            ),
            name='決策曲面'
        ))
        
        # 2. 繪製 z = 0 平面
        u_lim = np.linspace(xx.min(), xx.max(), 10)
        v_lim = np.linspace(yy.min(), yy.max(), 10)
        uu_lim, vv_lim = np.meshgrid(u_lim, v_lim)
        zz_zero = np.zeros_like(uu_lim)
        
        fig_3d.add_trace(go.Surface(
            x=u_lim, y=v_lim, z=zz_zero,
            colorscale=[[0, 'rgba(100, 116, 139, 0.2)'], [1, 'rgba(100, 116, 139, 0.2)']], # 灰色對照平面
            opacity=0.3,
            showscale=False,
            name='決策基準面 (z=0)'
        ))
        
        # 3. 繪製點在曲面上的對應位置高度
        Z_0 = model.decision_function(X[y == 0])
        fig_3d.add_trace(go.Scatter3d(
            x=X[y == 0, 0], y=X[y == 0, 1], z=Z_0,
            mode='markers',
            marker=dict(color=color_class_0, size=4, line=dict(color='#1e293b', width=1)),
            name='類別 0 (內圈)'
        ))
        
        Z_1 = model.decision_function(X[y == 1])
        fig_3d.add_trace(go.Scatter3d(
            x=X[y == 1, 0], y=X[y == 1, 1], z=Z_1,
            mode='markers',
            marker=dict(color=color_class_1, size=4, line=dict(color='#1e293b', width=1)),
            name='類別 1 (外圈)'
        ))
        
        # 支援向量
        Z_sv = model.decision_function(support_vectors)
        fig_3d.add_trace(go.Scatter3d(
            x=support_vectors[:, 0], y=support_vectors[:, 1], z=Z_sv,
            mode='markers',
            marker=dict(color=color_sv_bg, size=7, line=dict(color=color_sv, width=2)),
            name='支援向量 (Support Vectors)'
        ))
        
        # 啟用 True RBF 3D 空間地形變形動畫
        if enable_animation:
            frames = []
            steps = np.linspace(0.0, 1.0, 11)
            for i, val in enumerate(steps):
                frames.append(go.Frame(
                    data=[
                        go.Surface(z=val * Z),
                        go.Surface(z=zz_zero),
                        go.Scatter3d(z=val * Z_0),
                        go.Scatter3d(z=val * Z_1),
                        go.Scatter3d(z=val * Z_sv)
                    ],
                    name=f"frame_{i}"
                ))
            fig_3d.frames = frames
            
            fig_3d.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    x=0.1, y=-0.1,
                    buttons=[
                        dict(label="播放 (Play)", method="animate", args=[None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True}]),
                        dict(label="暫停 (Pause)", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
                    ]
                )],
                sliders=[dict(
                    steps=[
                        dict(
                            method='animate',
                            args=[[f'frame_{i}'], {'frame': {'duration': 150, 'redraw': True}, 'mode': 'immediate'}],
                            label=f'{int(val*100)}%'
                        )
                        for i, val in enumerate(steps)
                    ],
                    x=0.25, y=-0.1, len=0.75,
                    currentvalue=dict(font=dict(size=12), prefix='變形比例: ', visible=True, xanchor='right')
                )]
            )
            
        # 設定合理的高度範圍
        z_bound = max(abs(Z.min()), abs(Z.max()))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='X1 軸',
                yaxis_title='X2 軸',
                zaxis_title='Z 信心高度 f(x1, x2)',
                xaxis=dict(range=[-2.2, 2.2]),
                yaxis=dict(range=[-2.2, 2.2]),
                zaxis=dict(range=[-z_bound, z_bound])
            )
        )

    # 3D 畫布佈局通用調整
    fig_3d.update_layout(
        width=900,
        height=650,
        margin=dict(l=0, r=0, b=50, t=30),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.98,
            xanchor='right',
            x=1.0
        )
    )
    
    st.plotly_chart(fig_3d, width='stretch')

# ==============================================================================
# 標籤頁 3: Manim 概念教學動畫
# ==============================================================================
with tab_manim:
    st.subheader("🎥 Manim 概念動畫影片展示")
    st.markdown(
        "以下是由 Manim 引擎生成的 3D 動畫影片。影片詳細示範了二維同心圓如何被映射至三維空間中，"
        "並在此特徵空間中被超平面線性分割，最後投影回二維平面形成圓形決策邊界的過程。"
    )
    
    # 檢查預設路徑是否存在已生成的 MP4 影片檔案 (支援從根目錄或 src 目錄下啟動)
    manim_paths = [
        "media/videos/phase1_manim_kernel_trick/1080p60/SVMKernelTrick3D.mp4",
        "../media/videos/phase1_manim_kernel_trick/1080p60/SVMKernelTrick3D.mp4",
        "SVMKernelTrick3D.mp4"
    ]
    
    video_path = None
    for path in manim_paths:
        if os.path.exists(path):
            video_path = path
            break
            
    if video_path is not None:
        st.video(video_path)
        st.success(f"🎬 系統已成功自動加載概念展示動畫影片")
    else:
        st.warning("⚠️ 尚未偵測到已生成的概念教學影片。")
        st.markdown("""
        請在終端機中執行以下指令進行影片渲染：
        ```bash
        manim -pql src/phase1_manim_kernel_trick.py SVMKernelTrick3D
        ```
        影片生成後重新整理網頁，此處將會自動顯示播放器。
        """)

# ------------------------------------------------------------------------------
# 底部學術與原理參考
# ------------------------------------------------------------------------------
st.markdown("---")
st.header("📖 數學理論與公式參考")

col_math1, col_math2 = st.columns(2)

with col_math1:
    st.markdown("""
    <div class="feature-card">
        <h3>🔑 對偶問題與核心技巧 (Kernel Trick)</h3>
        <p>在支援向量機 (SVM) 中，我們透過拉格朗日乘子法求解對偶問題，決策函數僅依賴於輸入向量之間的點積：</p>
        <p style="text-align: center; font-weight: bold; background-color: #f8fafc; padding: 0.75rem; border-radius: 6px;">
            f(x) = Sum_i [ alpha_i * y_i * K(x_i, x) ] + b
        </p>
        <p>其中 <b>K(x_i, x)</b> 即為核心函數。核心技巧的核心思想是在不顯式計算高維映射 phi(x) 的情況下，直接計算高維特徵空間中的內積：K(x, x') = phi(x) 點積 phi(x')，從而避免了「維度災難」。</p>
    </div>
    """, unsafe_allow_html=True)

with col_math2:
    st.markdown("""
    <div class="feature-card">
        <h3>📝 核心函數類型與公式</h3>
        <ul>
            <li><b>線性核心 (Linear Kernel):</b><br>
                <code style="background-color: #f1f5f9; padding: 2px 6px; border-radius: 4px;">K(x, x') = xᵀ * x'</code>
            </li>
            <li style="margin-top: 10px;"><b>徑向基底函數核心 (RBF Kernel):</b><br>
                <code style="background-color: #f1f5f9; padding: 2px 6px; border-radius: 4px;">K(x, x') = exp(-gamma * ||x - x'||²)</code><br>
                <small style="color: #64748b;">用來度量點與點之間的相似度，距離越遠相似度呈指數級衰減趨於 0。</small>
            </li>
            <li style="margin-top: 10px;"><b>多項式核心 (Polynomial Kernel):</b><br>
                <code style="background-color: #f1f5f9; padding: 2px 6px; border-radius: 4px;">K(x, x') = (gamma * xᵀ * x' + r)ᵈ</code><br>
                <small style="color: #64748b;">計算代表所有組合至 d 階數的多項式特徵空間內積。</small>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
