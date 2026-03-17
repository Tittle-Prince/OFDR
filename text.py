import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 生成示例数据（你后面替换成自己的真实数据）
# =========================
np.random.seed(42)

def gaussian(x, mu, sigma, amp=1.0):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def make_example_spectrum(x, true_pos, shift_left=-0.015, shift_right=0.018,
                          sigma_center=0.010, sigma_neighbor=0.012,
                          amp_center=1.0, amp_neighbor=0.55, asym=0.0):
    """
    构造一个带邻峰干扰和轻微不对称的示例谱
    """
    y_center = gaussian(x, true_pos, sigma_center, amp_center)
    y_left   = gaussian(x, true_pos + shift_left, sigma_neighbor, amp_neighbor)
    y_right  = gaussian(x, true_pos + shift_right, sigma_neighbor * (1 + 0.2*asym), amp_neighbor * 0.9)

    y = y_center + y_left + y_right

    # 加一点 baseline 和轻微起伏
    baseline = 0.02 + 0.01 * np.sin((x - x.min()) / (x.max() - x.min()) * 2*np.pi)
    y = y + baseline

    # 加轻微噪声
    y = y + np.random.normal(0, 0.002, size=len(x))

    # 归一化
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    return y

# 横轴：局部波长窗口（示例）
x = np.linspace(1549.92, 1550.08, 512)

# 构造 3 个示例样本
samples = []

# sample 1: G明显更好
true1 = 1550.000
y1 = make_example_spectrum(x, true1, shift_left=-0.017, shift_right=0.016, sigma_center=0.010, asym=0.4)
pred_A_1 = 1550.0065
pred_G_1 = 1550.0018
samples.append(("Sample 1", x, y1, true1, pred_A_1, pred_G_1))

# sample 2: 线宽更大，结构性偏移更明显
true2 = 1550.012
y2 = make_example_spectrum(x, true2, shift_left=-0.020, shift_right=0.014, sigma_center=0.013, asym=0.8)
pred_A_2 = 1550.0182
pred_G_2 = 1550.0135
samples.append(("Sample 2", x, y2, true2, pred_A_2, pred_G_2))

# sample 3: 邻峰侵入较明显
true3 = 1549.988
y3 = make_example_spectrum(x, true3, shift_left=-0.012, shift_right=0.012, sigma_center=0.011, amp_neighbor=0.70, asym=0.5)
pred_A_3 = 1549.9815
pred_G_3 = 1549.9868
samples.append(("Sample 3", x, y3, true3, pred_A_3, pred_G_3))


# =========================
# 2. 绘图函数
# =========================
def plot_typical_cases(samples, save_path=None):
    """
    samples: list of tuples
      (title, x, y_raw, true_pos, pred_A, pred_G)
    """
    n = len(samples)
    fig, axes = plt.subplots(
        2, n,
        figsize=(5.2 * n, 6.2),
        sharex=False
    )

    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, (title, x, y_raw, true_pos, pred_A, pred_G) in enumerate(samples):
        # 一阶导数
        d1 = np.gradient(y_raw, x)
        d1 = d1 / (np.max(np.abs(d1)) + 1e-12)  # 归一化，便于显示

        # ---------- 上图：原始谱 ----------
        ax_top = axes[0, i]
        ax_top.plot(x, y_raw, linewidth=2.0, label="Raw spectrum")
        ax_top.axvline(true_pos, linestyle="--", linewidth=1.8, label="Ground truth")
        ax_top.axvline(pred_A, linestyle="-.", linewidth=1.8, label="A prediction")
        ax_top.axvline(pred_G, linestyle=":", linewidth=2.2, label="G prediction")

        ax_top.set_title(title, fontsize=13)
        ax_top.set_ylabel("Normalized amplitude", fontsize=11)
        ax_top.grid(alpha=0.25)

        err_A = abs(pred_A - true_pos)
        err_G = abs(pred_G - true_pos)
        ax_top.text(
            0.02, 0.95,
            f"|A-gt| = {err_A:.4f} nm\n|G-gt| = {err_G:.4f} nm",
            transform=ax_top.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.15)
        )

        # ---------- 下图：一阶导 ----------
        ax_bot = axes[1, i]
        ax_bot.plot(x, d1, linewidth=2.0, label="1st derivative")
        ax_bot.axvline(true_pos, linestyle="--", linewidth=1.8)
        ax_bot.axvline(pred_A, linestyle="-.", linewidth=1.8)
        ax_bot.axvline(pred_G, linestyle=":", linewidth=2.2)

        ax_bot.set_xlabel("Wavelength (nm)", fontsize=11)
        ax_bot.set_ylabel("Normalized d1", fontsize=11)
        ax_bot.grid(alpha=0.25)

    # 统一图例（只取第一个子图的 handles）
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=11, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# 3. 运行绘图
# =========================
plot_typical_cases(samples, save_path="typical_cases_example.png")