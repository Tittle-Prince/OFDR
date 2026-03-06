import torch
import torch.nn as nn       # <--- 加上这一行
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model_pinn import FBG_CNN_Base, PINNLoss

# ... 下面的代码保持不变 ...

def load_data(filepath="data/processed/uwfbg_dataset.npz"):
    data = np.load(filepath)
    X = data['X']        
    Y_dT = data['Y_dT']  
    
    # 转换为 PyTorch Tensor (增加 Channel 维度)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1) 
    Y_tensor = torch.tensor(Y_dT, dtype=torch.float32)
    
    # 80% 训练，20% 测试
    split_idx = int(0.8 * len(X))
    train_dataset = TensorDataset(X_tensor[:split_idx], Y_tensor[:split_idx])
    test_dataset = TensorDataset(X_tensor[split_idx:], Y_tensor[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader, X_tensor[split_idx:], Y_tensor[split_idx:]

def train_model(train_loader, use_pinn=False, epochs=40, label_ratio=0.05):
    model = FBG_CNN_Base(input_size=401)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    alpha_weight = 10.0 if use_pinn else 0.0
    criterion = PINNLoss(alpha=alpha_weight, K_T=0.01)
    
    mode_name = "半监督 PINN (物理驱动)" if use_pinn else "普通 CNN (纯数据驱动)"
    print(f"\n🚀 开始训练 {mode_name}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            dT_pred, dlam_pred = model(batch_X)
            
            # 【杀招】：人为制造标签稀缺，每个 batch 只取前 label_ratio 有标签
            split_idx = max(1, int(label_ratio * len(batch_X)))
            
            if use_pinn:
                # PINN：少部分算有监督Loss，大部分算无监督物理Loss
                loss_labeled = criterion(dT_pred[:split_idx], dlam_pred[:split_idx], batch_Y[:split_idx])
                loss_unlabeled = criterion(dT_pred[split_idx:], dlam_pred[split_idx:], dT_true=None)
                loss = loss_labeled + loss_unlabeled
            else:
                # 普通 CNN：无标签数据直接作废，只能用一小撮数据算 MSE
                loss = nn.MSELoss()(dT_pred[:split_idx], batch_Y[:split_idx])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
            
    return model

def evaluate_model(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        dT_pred, _ = model(X_test)
        rmse = torch.sqrt(torch.mean((dT_pred - Y_test)**2)).item()
    return rmse

if __name__ == "__main__":
    train_loader, test_loader, X_test, Y_test = load_data()
    
    # 设置极端的标签比例 (仅仅 5%)
    LABEL_RATIO = 0.05
    print(f"⚠️ 极限测试：仅提供 {LABEL_RATIO*100}% 的温度标签进行训练！")
    
    model_cnn = train_model(train_loader, use_pinn=False, epochs=50, label_ratio=LABEL_RATIO)
    rmse_cnn = evaluate_model(model_cnn, X_test, Y_test)
    
    model_pinn = train_model(train_loader, use_pinn=True, epochs=50, label_ratio=LABEL_RATIO)
    rmse_pinn = evaluate_model(model_pinn, X_test, Y_test)
    
    print("\n" + "="*50)
    print(f"📊 极端少标签 + 高噪声场景下的测试集 RMSE:")
    print(f"普通 8 层 CNN: {rmse_cnn:.4f} °C")
    print(f"半监督 PINN:   {rmse_pinn:.4f} °C")
    print(f"性能提升幅度:   {(rmse_cnn - rmse_pinn) / rmse_cnn * 100:.2f}%")
    print("="*50)
    torch.save(model_cnn.state_dict(), "results/models/cnn_baseline.pth")
    torch.save(model_pinn.state_dict(), "results/models/pinn_semi_supervised.pth")
    print("💾 模型权重已保存至 results/models/ 目录！")