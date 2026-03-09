import torch
import torch.nn as nn

# ==========================================
# 1. 核心网络架构 (复刻原论文 8 层 CNN)
# ==========================================
class FBG_CNN_Base(nn.Module):
    def __init__(self, input_size=401):
        super().__init__()
        # 1D CNN 提取畸变光谱特征
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2)
        )
        
        # 自动计算展平后的维度 (401 经过 3次 MaxPool1d(2) 后变为 50)
        # 64 channels * 50 length = 3200
        
        # 全连接层：输出预测的 温度(dT) 和 波长漂移(dlam)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3200, 128), nn.ReLU(),
            nn.Linear(128, 2) # 输出层维度为 2: [dT_pred, dlam_pred]
        )

    def forward(self, x):
        features = self.features(x)
        out = self.regressor(features)
        return out[:, 0], out[:, 1] # 分离出 dT 和 dlam

# ==========================================
# 2. 半监督物理信息损失函数 (Semi-supervised PINN Loss)
# ==========================================
class PINNLoss(nn.Module):
    def __init__(self, alpha=1.0, K_T=0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha  # 物理约束的权重
        self.K_T = K_T      # 物理先验：热光系数

    def forward(self, dT_pred, dlam_pred, dT_true=None):
        # [核心]：无论是否有标签，预测的波长和温度必须符合热膨胀物理定律
        loss_physics = self.mse(dlam_pred, dT_pred * self.K_T)
        
        # 如果传入了真实标签 -> 计算: 数据误差 + 物理误差
        if dT_true is not None:
            loss_data = self.mse(dT_pred, dT_true)
            return loss_data + self.alpha * loss_physics
        # 如果没有标签 -> 计算: 纯物理自监督误差
        else:
            return self.alpha * loss_physics
            