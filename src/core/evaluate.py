import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# 导入我们定义的网络结构
from model_pinn import FBG_CNN_Base

# ==========================================
# 1. 绘图核心函数 (对标 SCI 期刊风格)
# ==========================================
def plot_regression_result(y_true, y_pred, model_name, color='b', marker='o', save_name='fig.png'):
    """
    绘制真实值 vs 预测值的回归散点图，并计算 R² 和拟合线
    """
    # 设置全局学术字体和线宽
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    
    # 1. 计算决定系数 R² 和 RMSE
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # 2. 计算拟合线 (y = ax + b)
    lr = LinearRegression()
    lr.fit(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
    y_fitted = lr.predict(y_true.reshape(-1, 1))
    
    # 3. 开始绘图
    fig = plt.figure(figsize=(6, 5), dpi=300) # 300 DPI 保证插入 Word 不模糊
    ax = fig.add_subplot(111)
    
    # 绘制理想参考线 (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Ideal: y=x')
    
    # 绘制实际预测散点
    ax.scatter(y_true, y_pred, s=25, c=color, marker=marker, alpha=0.7, edgecolors='k', linewidths=0.5, label='Predicted Data')
    
    # 绘制最佳拟合线
    ax.plot(y_true, y_fitted, c='r', linewidth=2, label=f'Best Fit (R²={r2:.4f})')
    
    # 坐标轴与标签设置
    ax.set_xlabel('Measured Temperature ΔT (°C)', fontweight='bold')
    ax.set_ylabel(f'{model_name} Prediction ΔT (°C)', fontweight='bold')
    ax.set_title(f'Demodulation Linearity: {model_name}')
    
    # 添加 RMSE 文本框
    text_str = f'RMSE = {rmse:.2f} °C\n$R^2$ = {r2:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.axis('equal') # 保证 X 轴和 Y 轴比例一致，视觉上 y=x 才是 45 度角
    
    # 保存图片
    save_path = os.path.join("results", "figures", save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 图表已保存: {save_path}")
    
    return fig

# ==========================================
# 2. 主执行流程：加载 -> 推理 -> 绘图
# ==========================================
if __name__ == "__main__":
    print("🔍 正在加载测试集数据与模型权重...")
    
    # 1. 加载数据 (我们只取测试集部分，也就是最后 20%)
    data = np.load("data/processed/uwfbg_dataset.npz")
    X_all = data['X']
    Y_dT_all = data['Y_dT']
    split_idx = int(0.8 * len(X_all))
    
    X_test = torch.tensor(X_all[split_idx:], dtype=torch.float32).unsqueeze(1)
    Y_test_np = Y_dT_all[split_idx:]
    
    # 2. 初始化网络并加载权重
    model_cnn = FBG_CNN_Base(input_size=401)
    model_pinn = FBG_CNN_Base(input_size=401)
    
    model_cnn.load_state_dict(torch.load("results/models/cnn_baseline.pth"))
    model_pinn.load_state_dict(torch.load("results/models/pinn_semi_supervised.pth"))
    
    model_cnn.eval()
    model_pinn.eval()
    
    # 3. 运行前向推理获取预测值
    with torch.no_grad():
        cnn_pred_dT, _ = model_cnn(X_test)
        pinn_pred_dT, _ = model_pinn(X_test)
        
        cnn_pred_np = cnn_pred_dT.numpy()
        pinn_pred_np = pinn_pred_dT.numpy()
        
    print("🎨 开始绘制对比图表...")
    
    # 4. 绘制并保存两张图
    fig1 = plot_regression_result(Y_test_np, cnn_pred_np, model_name="Data-driven CNN", 
                                  color='#9467bd', marker='x', save_name="fig6a_baseline_cnn.png")
    
    fig2 = plot_regression_result(Y_test_np, pinn_pred_np, model_name="Semi-supervised PINN", 
                                  color='#2ca02c', marker='s', save_name="fig6b_our_pinn.png")
    
    plt.show() # 在窗口中同时展示这两张图