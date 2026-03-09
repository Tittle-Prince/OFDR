# OFDR PINN Demodulation Project

全同UWFBG + CNN 解调项目执行方案（浓缩版）

一、总体原则 1. 先验证最小可行方案（MVP），再逐步增加复杂度。 
2.先证明“光谱 → CNN → 解调量”成立，再接入 OFDR。 3.
每一阶段只解决一个问题，保证结果可解释、可复现。

  -----------------------------------------
  Phase 0：问题定义（必须先完成）
  -----------------------------------------
  输入： -
  单个局部光谱窗口（固定长度，如512点） -
  归一化处理

  输出： - 主输出：Δλ（波长漂移） -
  温度由物理公式换算：ΔT = Δλ / K_T

  样本定义： - 一个样本 =
  某位置光栅在某温度下的局部光谱

  数据划分： - Train / Validation / Test
  按温度点或实验轮次划分

  Baseline方法： 1.
  Cross‑correlation（互相关解调） 2.
  Parametric fitting（参数拟合） 3. MLP 4.
  1D CNN
  -----------------------------------------

Phase 1：最小可行数据集（Dataset A）

数据特征： - 理想局部光谱 - 少量噪声 - 不包含OFDR截窗畸变

目标： 验证 CNN 能否从光谱预测 Δλ。

验证指标： - RMSE - MAE - R²

  --------------------------------------
  Phase 2：建立 baseline 对比体系
  --------------------------------------
  运行以下方法： 1. Cross‑correlation 2.
  Parametric fitting 3. MLP 4. 1D CNN

  输出核心表格：

  Method | RMSE | MAE | R² —— | —- | —-
  | —- Cross-correlation Parametric
  fitting MLP 1D CNN
  --------------------------------------

Phase 3：轻度真实数据（Dataset B）

加入扰动： - 白噪声 - 基线漂移 - 幅值变化 - 轻微谱形畸变

结构改进测试： 1. CNN + Dilated Conv 2. CNN + SE Attention 3. CNN +
Dilated + SE

目的：验证改进结构是否提升鲁棒性。

  --------------------------------------
  Phase 4：OFDR风格数据（Dataset C）
  --------------------------------------
  仿真流程： 扫频干涉 → FFT → 空间窗截取
  → IFFT

  得到： -
  含串扰和能量泄漏的局部畸变光谱

  任务： 重新运行所有 baseline 与 CNN
  模型。

  生成论文主结果。
  --------------------------------------

Phase 5：加入物理约束

推荐方式：

模型输出：Δλ

温度计算： ΔT = Δλ / K_T

避免过早引入复杂PINN训练。

  --------------------------------------
  Phase 6：标签依赖实验
  --------------------------------------
  标签比例：

  100% 50% 20% 10% 5%

  绘制曲线：

  Performance vs Label Ratio
  --------------------------------------

Phase 7：OFDR实测接入

步骤：

1.  直接迁移测试（仿真训练 → 实测数据）
2.  少量样本微调
3.  输出预测曲线与误差统计

论文描述：
仿真训练模型对真实OFDR光谱具有一定迁移能力，少量校准可进一步提升精度。

  ----------------
  最终论文贡献点
  ----------------

1.  提出一种基于CNN的全同UWFBG光谱解调框架。
2.  在OFDR空间截窗导致的串扰和谱形畸变条件下，相比传统方法具有更好的鲁棒性。
3.  在较低标签比例下仍保持稳定精度。

  --------------------
  立即执行的四个任务
  --------------------

1.  固定样本输入格式（512点光谱 + Δλ标签）。
2.  生成 Dataset_A（理想光谱数据）。
3.  训练 MLP 与 1D CNN。
4.  实现 Cross‑correlation baseline。

完成以上步骤后，再逐步推进后续阶段。

全同 UWFBG + CNN 解调项目

---

## 📁 文件结构规范

本项目采用严格的目录结构，所有文件应按照以下规范存放。

### 1. `src/` - 源代码目录

**核心文件**（必须放在 `src/core/`）：
| 文件 | 说明 |
|------|------|
| `data_generation.py` | 仿真数据生成脚本（TMM物理仿真） |
| `model_pinn.py` | PINN 模型定义 |
| `train.py` | 训练入口脚本 |
| `evaluate.py` | 模型评估脚本 |
| `phase1_pipeline.py` | Phase1 数据处理流水线 |

**分阶段代码**（按版本存放）：
```
src/
├── core/           # 核心文件（数据生成、模型定义、训练脚本）
├── phase1/         # Phase1: 数据处理流水线
├── phase2/         # Phase2: 基线方法（MLP、CNN1D、互相关法、参数拟合）
├── phase3/         # Phase3: 鲁棒CNN模型（加噪数据训练）
└── phase4a/        # Phase4: FBG阵列解调应用
```

---

### 2. `results/` - 结果目录

**分类存放规则**：

| 子目录 | 存放内容 | 示例文件 |
|--------|----------|----------|
| `results/models/` | 训练好的模型文件 | `*.pth`, `*.pt` |
| `results/datasets/` | 生成的数据集文件 | `*.npz` |
| `results/figures/` | 图表文件 | `*.png`, `*.pdf` |
| `results/metrics/` | 指标文件 | `*.json`, `*.csv` |

**模型文件组织**（按Phase分类）：
```
results/models/
├── pinn_semi_supervised.pth    # 主模型
├── cnn_baseline.pth            # CNN基线模型
├── phase1/                     # Phase1 模型
│   ├── phase1_cnn1d.pt
│   └── phase1_mlp.pt
├── phase2/                     # Phase2 模型
│   ├── mlp/
│   ├── cnn1d/
│   ├── parametric_fitting/
│   └── cross_correlation/
└── phase3/                     # Phase3 模型
    ├── cnn_baseline/
    ├── cnn_dilated/
    ├── cnn_se/
    └── cnn_dilated_se/
```

**数据集文件命名**：
- `uwfbg_dataset.npz` - 原始仿真数据
- `dataset_a_phase1.npz` - Phase1 数据集
- `dataset_b_phase3.npz` - Phase3 数据集
- `dataset_c_phase4a.npz` - Phase4 阵列数据集

---

### 3. `scripts/` - 脚本目录

**分类存放规则**：

| 子目录 | 存放内容 | 示例文件 |
|--------|----------|----------|
| `scripts/figures/` | 画图脚本（生成论文图表） | `generate_fig*.py` |
| `scripts/analysis/` | 分析脚本（数据处理、结果汇总） | `finalize_*.py` |

**画图脚本命名规范**：
```
scripts/figures/
├── generate_fig1_system_overview.py
├── generate_fig3_raw_spectrum.py
├── generate_fig4_spectrum_distortion_refined.py
├── generate_fig5_dataset_construction.py
├── generate_fig6_training_curve.py
├── generate_fig7_model_influence.py
├── generate_fig8_main_results.py
├── generate_fig9_ablation_*.py
└── generate_distortion_robustness_figure.py
```

---

### 4. `config/` - 配置文件目录

```
config/
├── config.yaml         # 数据生成配置
├── phase1.yaml        # Phase1 配置
├── phase2.yaml        # Phase2 配置
├── phase3.yaml        # Phase3 配置
├── phase4a.yaml       # Phase4 配置
└── phase4_array.yaml  # 阵列配置
```

---

## 🚀 快速开始

### 1. 生成仿真数据
```bash
python src/core/data_generation.py
```

### 2. 训练模型
```bash
# Phase3 训练示例
python src/phase3/run_cnn_dilated_se.py
```

### 3. 生成图表
```bash
# 运行画图脚本
python scripts/figures/generate_fig8_main_results.py
```

---

## 📊 各 Phase 说明

| Phase | 目录 | 主要内容 |
|-------|------|----------|
| Phase1 | `src/phase1/` | 数据处理流水线，构建基础数据集 |
| Phase2 | `src/phase2/` | 基线方法对比（MLP、CNN1D、互相关、参数拟合） |
| Phase3 | `src/phase3/` | 鲁棒CNN模型训练（加噪、畸变数据） |
| Phase4 | `src/phase4a/` | FBG阵列解调应用验证 |

---

## ⚠️ 注意事项

1. **不要在根目录放置任何代码文件** - 所有代码必须放在 `src/` 目录下
2. **结果文件必须分类存放** - 模型、数据、图表、指标分别放在对应子目录
3. **保持目录整洁** - 删除空的文件夹，不要留下临时文件
4. **脚本命名规范** - 画图脚本使用 `generate_fig*.py` 命名格式

---

## 📞 联系方式

如有问题，请联系项目维护者。
