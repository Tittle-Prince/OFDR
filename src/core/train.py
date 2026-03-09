import torch
import torch.nn as nn

# ==========================================
# [新增模块] 1D SE 注意力机制 (Squeeze-and-Excitation)
# 作用：动态评估各个通道提取到的光谱特征，抑制相邻光栅串扰带来的假峰特征
# ==========================================
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # 全局平均池化，获取全局感受野
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # 输出 0~1 的权重，给重要特征加权，给串扰噪声降权
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# ==========================================
# 核心网络：PI-ACNN (物理信息注意力卷积网络)
# ==========================================
class FBG_CNN_Base(nn.Module):
    def __init__(self, input_size=401):
        super().__init__()
        
        # 1. 浅层特征提取 (捕捉光谱基础边缘和峰位)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 2. 中层特征提取 + 注意力机制 (开始辨别真假峰和串扰)
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            SEBlock1D(channel=64), # 插入注意力机制
            nn.MaxPool1d(2)
        )
        
        # 3. 深层大感受野提取 (使用空洞卷积 dilation=2，捕捉全局畸变展宽包络)
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            SEBlock1D(channel=128), # 再次插入注意力机制
            nn.AdaptiveAvgPool1d(1) # [杀招] 无论前面多长，这里强行汇聚成全局特征，彻底抛弃单纯寻峰
        )
        
        # 4. 物理双分支输出头 (不变，依然输出 温度 和 波长漂移)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2) # [dT_pred, dlam_pred]
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        features = self.layer3(x)
        out = self.regressor(features)
        return out[:, 0], out[:, 1]

# ==========================================
# 半监督物理信息损失函数 (保持不变)
# ==========================================
class PINNLoss(nn.Module):
    def __init__(self, alpha=1.0, K_T=0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha  
        self.K_T = K_T      

    def forward(self, dT_pred, dlam_pred, dT_true=None):
        loss_physics = self.mse(dlam_pred, dT_pred * self.K_T)
        if dT_true is not None:
            loss_data = self.mse(dT_pred, dT_true)
            return loss_data + self.alpha * loss_physics
        else:
            return self.alpha * loss_physics