import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_generator import generate_concentric_circles
from utils.svm_utils import fit_svm_model, get_decision_grid

def main():
    # 1. 建立 outputs 資料夾 (如果不存在)
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 設定模型參數 (RBF SVM: C=10.0, gamma=1.0)
    C_val = 10.0
    gamma_val = 1.0
    kernel_type = 'rbf'
    noise_level = 0.05
    n_samples = 200
    seed = 42
    
    print("=" * 60)
    print("Phase 2: SVM RBF 決策面數學驗證與測試")
    print(f"資料集參數: 樣本數={n_samples}, 雜訊={noise_level}, 隨機種子={seed}")
    print(f"SVM 模型參數: 核心函數={kernel_type}, C={C_val}, gamma={gamma_val}")
    print("=" * 60)
    
    # 3. 生成同心圓資料集
    X, y = generate_concentric_circles(noise=noise_level, n_samples=n_samples, random_seed=seed)
    
    # 4. 訓練 RBF SVM 模型
    model = fit_svm_model(X, y, kernel=kernel_type, C=C_val, gamma=gamma_val)
    
    # 5. 建立網格並計算決策函數信心值
    xx, yy, Z = get_decision_grid(model, X, grid_resolution=200, padding=0.2)
    
    # 6. 開始繪圖
    plt.figure(figsize=(9, 8))
    
    # 繪製信心強度背景 (使用藍-紅 coolwarm 漸變，加上透明度)
    bg_contour = plt.contourf(
        xx, yy, Z,
        levels=np.linspace(Z.min(), Z.max(), 50),
        cmap='coolwarm',
        alpha=0.3
    )
    plt.colorbar(bg_contour, label='決策函數信賴值 f(x, y)')
    
    # 繪製資料點: 類別 0 (內圈 - 藍色), 類別 1 (外圈 - 紅色)
    plt.scatter(
        X[y == 0, 0], X[y == 0, 1],
        color='blue',
        edgecolors='k',
        s=40,
        label='類別 0 (內圈 - 藍點)'
    )
    plt.scatter(
        X[y == 1, 0], X[y == 1, 1],
        color='red',
        edgecolors='k',
        s=40,
        label='類別 1 (外圈 - 紅點)'
    )
    
    # 繪製黃色實線決策邊界 (f(x, y) = 0)
    plt.contour(
        xx, yy, Z,
        levels=[0.0],
        colors=['yellow'],
        linewidths=[3.0],
        linestyles=['solid']
    )
    # 用於顯示在圖例中的隱藏虛擬線
    plt.plot([], [], color='yellow', linewidth=3, label='決策邊界 f(x,y)=0')
    
    # 繪製邊界間距限制線 (f(x, y) = -1.0 與 f(x, y) = 1.0) - 灰色虛線
    plt.contour(
        xx, yy, Z,
        levels=[-1.0, 1.0],
        colors=['dimgrey'],
        linewidths=[1.5],
        linestyles=['dashed']
    )
    plt.plot([], [], color='dimgrey', linestyle='--', linewidth=1.5, label='邊界間距 f(x,y)=±1')
    
    # 高亮支援向量點 (金黃色圈)
    sv = model.support_vectors_
    plt.scatter(
        sv[:, 0], sv[:, 1],
        s=120,
        facecolors='none',
        edgecolors='gold',
        linewidths=1.8,
        label='支援向量 (Support Vectors)'
    )
    
    # 設定圖表標題及軸標籤
    # 注意: matplotlib 在預設環境下不一定支援中文顯示。
    # 為了防範中文亂碼 (Matplotlib Font Warning)，我們保留英文標籤，但用最精簡的文字，或添加通用語系設定。
    plt.title(
        f"RBF SVM Decision Space (C={C_val}, gamma={gamma_val})",
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    plt.xlabel("X1", fontsize=12)
    plt.ylabel("X2", fontsize=12)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 設定畫布極限
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    # 存檔
    save_path = os.path.join(output_dir, 'verification_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[成功] 驗證圖表已儲存至: {save_path}")
    print("\n" + "=" * 60)
    print("數學原理解析:")
    print("1. RBF 核心函數公式:")
    print("   K(x, x') = exp(-gamma * ||x - x'||^2)")
    print("\n2. 參數對擬合的影響:")
    print("   - gamma (核心寬度):")
    print(f"     當前設定 gamma = {gamma_val}。")
    print("     Gamma 較小時，邊界傾向平滑與圓潤 (可能欠擬合)；")
    print("     Gamma 較大時，邊界傾向圍繞各別資料點扭曲 (可能過擬合)。")
    print("   - C (正則化係數):")
    print(f"     當前設定 C = {C_val}。")
    print("     C 較小時，允許更多分類誤差，追求最大邊界寬度；")
    print("     C 較大時，對錯誤分類處罰嚴格，迫使邊界緊繃，甚至導致過擬合。")
    print("=" * 60)

if __name__ == '__main__':
    main()
