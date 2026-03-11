import numpy as np
import yaml
import os
from tqdm import tqdm  # 用于显示进度条

# ==========================================
# 1. 配置加载模块
# ==========================================
def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# ==========================================
# 2. 物理仿真引擎 (TMM)
# ==========================================
def tmm_uwfbg(wavelengths, T_array, params):
    """
    核心 TMM 算法：输入波长网格和沿光栅的温度分布数组 T_array，输出反射光谱
    """
    L, dn, neff, K_T, lambda_B0 = params['L'], params['dn'], params['neff'], params['K_T'], params['lambda_B0']
    dz = params['dz']
    N_steps = len(T_array)
    
    R_spectrum = np.zeros(len(wavelengths))
    
    for i, lam in enumerate(wavelengths):
        M_total = np.eye(2, dtype=complex)
        lam_m = lam * 1e-9  # 转换为米
        
        for k in range(N_steps):
            dT = T_array[k]
            local_lambda_B = (lambda_B0 + K_T * dT) * 1e-9
            
            kappa = np.pi * dn / lam_m
            delta = 2 * np.pi * neff * (1 / lam_m - 1 / local_lambda_B)
            gamma = np.sqrt(kappa**2 - delta**2 + 0j)
            
            # 传输矩阵元
            sinh_g_dz = np.sinh(gamma * dz)
            cosh_g_dz = np.cosh(gamma * dz)
            M11 = cosh_g_dz - 1j * (delta / gamma) * sinh_g_dz
            M12 = -1j * (kappa / gamma) * sinh_g_dz
            M21 = 1j * (kappa / gamma) * sinh_g_dz
            M22 = cosh_g_dz + 1j * (delta / gamma) * sinh_g_dz
            
            M_seg = np.array([[M11, M12], [M21, M22]])
            M_total = np.dot(M_total, M_seg)
            
        R_spectrum[i] = np.abs(M_total[1, 0] / M_total[0, 0])**2
        
    return R_spectrum

# ==========================================
# 3. 数据集批量生成器 (刻意制造畸变陷阱)
# ==========================================
def generate_dataset(config):
    f_params = config['fbg_params']
    s_params = config['sim_params']
    d_params = config['dataset']
    
    wavelengths = np.linspace(s_params['wl_start'], s_params['wl_end'], s_params['num_points'])
    N_steps = int(f_params['L'] / s_params['dz'])
    z_axis = np.linspace(0, f_params['L'], N_steps)
    
    num_samples = d_params['num_samples']
    X_data = np.zeros((num_samples, s_params['num_points']))
    Y_dT = np.zeros(num_samples)
    Y_dlam = np.zeros(num_samples)
    
    print(f"🚀 开始生成 {num_samples} 组 uwFBG 畸变光谱数据...")
    
    for i in tqdm(range(num_samples), desc="Generating Data"):
        # [核心设计]：随机混合三种温度场，给纯 CNN 挖坑
        profile_type = np.random.choice(['uniform', 'linear', 'quadratic'])
        base_T = np.random.uniform(0, 60) # 基础升温 0-60度
        
        if profile_type == 'uniform':
            T_array = np.full(N_steps, base_T)
        elif profile_type == 'linear':
            grad = np.random.uniform(-20, 20) # 首尾温差
            T_array = base_T + grad * (z_axis / f_params['L'])
        else: # quadratic (局部热点或冷点)
            hotspot_intensity = np.random.uniform(-30, 30)
            center = np.random.uniform(0.2 * f_params['L'], 0.8 * f_params['L'])
            # 构造高斯型局部温度突变
            T_array = base_T + hotspot_intensity * np.exp(-((z_axis - center) / (0.2 * f_params['L']))**2)
            
        # 1. 计算理论平均温度和期望的波长漂移 (这是 PINN 要回归的 Ground Truth)
        mean_dT = np.mean(T_array)
        Y_dT[i] = mean_dT
        Y_dlam[i] = mean_dT * f_params['K_T']
        
        # 2. 调用 TMM 引擎生成纯净畸变光谱
        pure_params = {**f_params, 'dz': s_params['dz']}
        clean_spectrum = tmm_uwfbg(wavelengths, T_array, pure_params)
        
        # 3. [核心设计]：加入剧烈高斯白噪声，淹没真实的微小特征
        # 将光谱归一化到 0-1 方便加噪和网络训练
        max_val = np.max(clean_spectrum)
        if max_val > 0:
            clean_spectrum = clean_spectrum / max_val
            
        noise = np.random.normal(0, d_params['noise_level'], s_params['num_points'])
        noisy_spectrum = clean_spectrum + noise
        
        X_data[i] = noisy_spectrum

    # 确保保存目录存在
    os.makedirs(os.path.dirname(d_params['save_path']), exist_ok=True)
    
    # 打包保存为 .npz 格式
    np.savez(d_params['save_path'], X=X_data, Y_dT=Y_dT, Y_dlam=Y_dlam, wavelengths=wavelengths)
    print(f"✅ 数据集生成完毕，已保存至: {d_params['save_path']}")

if __name__ == "__main__":
    # 执行生成
    config = load_config()
    generate_dataset(config)