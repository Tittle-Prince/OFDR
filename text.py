import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# =========================
# Step 0: 参数设置
# =========================
N = 4096                     # 采样点数
lambda_start = 1549.0        # nm
lambda_end = 1551.0          # nm
lambdas = np.linspace(lambda_start, lambda_end, N)

# 光栅位置（模拟3个FBG，制造重叠）
fbg_positions = [1550.0, 1550.02, 1550.04]
amplitudes = [1.0, 0.8, 0.6]
widths = [0.01, 0.015, 0.02]

# =========================
# Step 1: 构造“光谱”（真实反射）
# =========================
def gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

spectrum = np.zeros_like(lambdas)
for mu, A, w in zip(fbg_positions, amplitudes, widths):
    spectrum += gaussian(lambdas, mu, w, A)

# =========================
# Step 2: 模拟OFDR拍频信号（干涉信号）
# =========================
# 实际系统更复杂，这里用简化模型
k = 2 * np.pi / lambdas

# 构造干涉信号（多个反射点叠加）
signal = np.zeros_like(k)
for i, (mu, A) in enumerate(zip(fbg_positions, amplitudes)):
    z = (mu - 1550.0) * 1e3   # 简单映射到空间（单位随意）
    signal += A * np.cos(2 * k * z)

# 加一点噪声
signal += 0.05 * np.random.randn(N)

# =========================
# Step 3: k空间重采样（关键）
# =========================
k_uniform = np.linspace(k.min(), k.max(), N)
interp_func = interp1d(k, signal, kind='linear', fill_value="extrapolate")
signal_k = interp_func(k_uniform)

# =========================
# Step 4: FFT → 空间域
# =========================
R_z = np.fft.fft(signal_k)
R_z_abs = np.abs(R_z)

# 只取前一半（实际距离）
R_z_abs = R_z_abs[:N//2]

# =========================
# Step 5: 找峰（光栅位置）
# =========================
peaks, _ = find_peaks(R_z_abs, height=np.max(R_z_abs)*0.2)

# =========================
# Step 6: 截取局部窗口（CNN输入）
# =========================
window_size = 50
local_spectra = []

for p in peaks:
    if p - window_size > 0 and p + window_size < len(R_z_abs):
        local = R_z_abs[p - window_size : p + window_size]
        local_spectra.append(local)

local_spectra = np.array(local_spectra)

# =========================
# Step 7: 可视化（理解用）
# =========================
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.title("Simulated spectrum (wavelength domain)")
plt.plot(lambdas, spectrum)

plt.subplot(3,1,2)
plt.title("Simulated beat signal")
plt.plot(signal_k)

plt.subplot(3,1,3)
plt.title("Spatial domain (after FFT)")
plt.plot(R_z_abs)
plt.scatter(peaks, R_z_abs[peaks], color='r')

plt.tight_layout()
plt.show()

# =========================
# Step 8: CNN输入示例
# =========================
print("CNN输入形状:", local_spectra.shape)