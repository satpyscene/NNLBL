# NNLBL: Neural Network Line-By-Line Absorption Calculator
**基于神经网络的高速逐线积分气体吸收截面计算程序**

NNLBL is a high-speed tool for calculating atmospheric gas absorption cross-sections. By replacing traditional Voigt profile convolutions with an optimized Neural Network (NN) and leveraging GPU acceleration, it achieves significant speedups while maintaining high precision compared to HAPI (HITRAN API).

NNLBL 是一个用于快速计算大气气体分子吸收截面的工具。它利用神经网络替代传统的 Voigt 线型卷积计算，并针对 GPU 进行优化，在保持高精度的前提下显著提升了计算速度。

---

## ✨ Key Features | 核心特性

* **High Precision (高精度):** Max relative error < 1% compared to HAPI; average error << 1%.
* **H2O Specialization (水汽特化):** Full support for **Self-Broadening** and **MT-CKD Continuum Absorption**.
* **Hybrid Architecture (混合架构):** Automatically switches between High-Pressure (HP) and Low-Pressure (LP) models.
* **Dual Modes (双模式):** Supports both single-layer (**SINGLE**) and vertical profile (**PROFILE**) calculations.
* **GPU Accelerated (硬件加速):** Automatically detects and uses CUDA if available.

---

## 📂 Project Structure | 目录结构

```text
NNLBL_Project/
├── NNLBL_TEST.py                # [Entry] User configuration & execution | 用户配置入口
├── NNLBL_src/                   # [Source] Core algorithm modules | 核心源码
│   ├── NNLBL_main.py            # Main API logic | 接口主逻辑
│   ├── mt_ckd_h2o.py            # Pythonized MT-CKD H2O continuum | 水汽连续吸收模块
│   └── hapi.py                  # Embedded HAPI (v1.2.2.4)
├── NNmodel&stats/               # [Models] Pre-trained weights & normalization stats | 预训练模型
├── data/                        # [Data] Physical constants & HITRAN cache | 物理常数与缓存
├── atmospheric_profile_for_testing/ # [Input] Sample profiles | 大气廓线输入示例
└── sigma_output_filefold/       # [Output] Calculated cross-sections (.h5) | 结果输出目录
```

---

## 🚀 Quick Start | 快速开始
1. Configuration | 配置参数
Edit NNLBL_TEST.py. The configuration is divided into four main parts:
编辑 NNLBL_TEST.py，配置主要分为以下四个部分：
- Target & Spectra | 目标与光谱
    ```python
    TARGET_ISO_LIST = [1, 2]   # Target isotope IDs (must be from the same molecule)
    ENABLE_CONTINUUM = False   # Enable MT-CKD (Recommended for H2O only)

    SPECTRAL_CONFIG = {
        "min": 4800.0, "max": 5200.0, "step": 0.01, # Unit: cm⁻¹
    }
    ```
- Run Mode | 运行模式
Select SINGLE for laboratory points or PROFILE for atmospheric columns.
选择 SINGLE（单层）或 PROFILE（廓线）。
  - Mode: SINGLE
    ```python
    SINGLE_PARAMS = {
        "p_hpa": 1013.25, "t_k": 296.0, "vmr_ppmv": 40000.0
    }
    ```
    - Mode: PROFILE
    ```python
    PROFILE_PARAMS = {
        "dir": "path/to/profile",
        "p_file": "pres.txt", "p_unit": "hPa",  # Support: hPa, Pa
        "t_file": "temp.txt", "t_unit": "K",    # Support: K, C
        "vmr_file": "h2o.txt", "vmr_unit": "ppmv",
        "name_tag": "US_STD_100"
    }
    ```
2. Run | 执行
```bash
python example_config_NNLBL.py
```

---

## 📊 Output & Data | 结果与数据说明
- Output Format | 输出格式
```text
Results are saved in .h5 format.
结果以 .h5 格式存储。
```
- Vertical Ordering (廓线层级说明):

  - Layer 000: Top of atmosphere (High altitude). Spectrum lines are sharp/dense.

  - Layer 100: Near surface (Low altitude). Spectrum lines are broad/sparse due to pressure broadening.

  - 第 000 层：高空（气压低），谱线尖锐且密集。

  - 第 100 层：近地面（气压高），受压力加宽影响，谱线宽且稀疏。

---

## 🛠 Requirements | 环境依赖
```text
See environment.yml for details. Key dependencies include torch, numpy, h5py, and netCDF4.
请参考 environment.yml。核心依赖包括 torch, numpy, h5py 和 netCDF4。
```